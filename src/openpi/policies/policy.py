from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _restore_hard_prefix(
    actions: jax.Array, prev_action_chunk: jax.Array, inference_delay: int
) -> jax.Array:
    """Restore the known clean prefix before exposing a hard-RTC chunk to synchronous clients."""
    freeze_mask = jnp.arange(actions.shape[1])[None, :] < inference_delay
    return jnp.where(freeze_mask[..., None], prev_action_chunk, actions)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class RealtimePolicy(Policy):
    """Stateful policy wrapper that runs Real-Time Chunking (RTC) inference.

    The server keeps the previously returned action chunk. On each subsequent inference it transforms that chunk back
    into model space relative to the latest observation, shifts it by `execute_horizon` (the number of actions the client
    is assumed to have executed since the last call), and conditions the new chunk on it via `model.sample_actions_rtc`,
    producing smooth chunk transitions.

    This assumes the client requests a new chunk every `execute_horizon` steps (e.g. via the action-chunk broker), so
    the server's `execute_horizon` should match the client's actions-per-chunk. RTC is JAX-only.
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        execute_horizon: int,
        inference_delay: int = 0,
        method: str = "pinv",
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        if self._is_pytorch_model:
            raise ValueError("RealtimePolicy only supports JAX models.")
        if not hasattr(model, "sample_actions_rtc"):
            raise ValueError(f"Model {type(model).__name__} does not support real-time chunking (sample_actions_rtc).")

        self._action_horizon = int(model.action_horizon)
        if not 0 < execute_horizon <= self._action_horizon:
            raise ValueError(f"execute_horizon must be in (0, {self._action_horizon}], got {execute_horizon}")
        if not 0 <= inference_delay <= execute_horizon:
            raise ValueError(f"inference_delay must be in [0, execute_horizon={execute_horizon}], got {inference_delay}")

        self._execute_horizon = execute_horizon
        self._inference_delay = inference_delay
        self._prefix_attention_horizon = self._action_horizon - execute_horizon
        self._method = method
        self._prefix_attention_schedule = prefix_attention_schedule
        self._max_guidance_weight = float(max_guidance_weight)

        self._sample_actions_rtc = nnx_utils.module_jit(
            model.sample_actions_rtc,
            static_argnames=("num_steps", "method", "prefix_attention_schedule"),
        )
        # Previously returned output-space chunk, or None on the first call. Re-encoding it on the next call is required
        # for policies such as ARX that model delta actions relative to the latest observed state.
        self._prev_actions: np.ndarray | None = None

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)
        if self._prev_actions is not None:
            # Input transforms may mutate actions in place (e.g. DeltaActions), so do not expose the cached output array.
            inputs["actions"] = np.array(self._prev_actions, copy=True)
        inputs = self._input_transform(inputs)
        prev_actions = inputs.pop("actions", None)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_arr = jnp.asarray(noise)
            if noise_arr.ndim == 2:
                noise_arr = noise_arr[None, ...]
            sample_kwargs["noise"] = noise_arr

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if prev_actions is None or self._method == "none":
            # First call (no history) falls back to vanilla sampling; RTC needs a previous chunk to condition on.
            actions = self._sample_actions(sample_rng, observation, **sample_kwargs)
        else:
            # Shift the stored chunk so index 0 aligns with the first action this call will generate.
            prev_chunk = jnp.asarray(prev_actions)[np.newaxis, ...]
            prev = jnp.concatenate(
                [
                    prev_chunk[:, self._execute_horizon :],
                    jnp.zeros_like(prev_chunk[:, : self._execute_horizon]),
                ],
                axis=1,
            )
            actions = self._sample_actions_rtc(
                sample_rng,
                observation,
                prev,
                self._inference_delay,
                self._prefix_attention_horizon,
                method=self._method,
                prefix_attention_schedule=self._prefix_attention_schedule,
                max_guidance_weight=self._max_guidance_weight,
                **sample_kwargs,
            )
            if self._method == "hard":
                # The model only learns to denoise the suffix: training masks the prefix loss, and the sampler's final
                # velocity update can move the clean prefix. The synchronous client executes the returned chunk from
                # index 0, so expose the known continuation instead of those unconstrained prefix predictions.
                actions = _restore_hard_prefix(actions, prev, self._inference_delay)
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        # Cache exactly what the client receives. The next input-transform pass converts it back to the latest model-space
        # coordinates before RTC conditioning.
        self._prev_actions = np.array(outputs["actions"], copy=True)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @override
    def reset(self) -> None:
        self._prev_actions = None


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
