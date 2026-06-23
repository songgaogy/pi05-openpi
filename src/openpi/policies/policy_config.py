import dataclasses
import logging
import os
import pathlib
from typing import Any, Literal

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class RealtimeConfig:
    """Server-side real-time chunking (RTC) settings. See `openpi.policies.policy.RealtimePolicy`."""

    # Number of actions the client executes between inference calls; should match the client's actions-per-chunk.
    execute_horizon: int
    # Number of leading actions that are already committed (frozen) while a new chunk is being computed.
    inference_delay: int = 0
    # "auto" selects "hard" if the model was trained with simulated delay, otherwise "pinv".
    method: Literal["auto", "none", "pinv", "hard"] = "auto"
    prefix_attention_schedule: Literal["linear", "exp", "ones", "zeros"] = "exp"
    max_guidance_weight: float = 5.0


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    realtime: RealtimeConfig | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # Load the single norm_stats.json saved under checkpoint assets/, regardless of train config asset_id.
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets")

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    common_kwargs = dict(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )

    if realtime is not None:
        if is_pytorch:
            raise ValueError("Real-time chunking is only supported for JAX models.")
        method = realtime.method
        if method == "auto":
            # Pair the inference method with how the model was trained.
            simulated_delay = getattr(train_config.model, "rtc_simulated_delay", None)
            method = "hard" if simulated_delay is not None else "pinv"
        logging.info(
            "Enabling real-time chunking: method=%s, execute_horizon=%d, inference_delay=%d",
            method,
            realtime.execute_horizon,
            realtime.inference_delay,
        )
        return _policy.RealtimePolicy(
            model,
            execute_horizon=realtime.execute_horizon,
            inference_delay=realtime.inference_delay,
            method=method,
            prefix_attention_schedule=realtime.prefix_attention_schedule,
            max_guidance_weight=realtime.max_guidance_weight,
            **common_kwargs,
        )

    return _policy.Policy(
        model,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
        **common_kwargs,
    )
