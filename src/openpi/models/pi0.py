import logging
from typing import Literal, TypeAlias

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# Schedules for how strongly each prefix action is pinned to the previous action chunk during real-time chunking.
PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]

# Inference-time real-time-chunking method:
# - "none": vanilla flow sampling (no conditioning on the previous chunk).
# - "pinv": soft guidance via the pseudoinverse-corrected velocity (Real-Time Chunking, https://arxiv.org/abs/2506.07339).
# - "hard": hard masking that freezes the prefix to the previous chunk (pairs with a model trained with simulated delay).
RealtimeMethod: TypeAlias = Literal["none", "pinv", "hard"]


def get_prefix_weights(
    start: int | jax.Array, end: int | jax.Array, total: int, schedule: PrefixAttentionSchedule
) -> jax.Array:
    """Port of the kinetix reference (third_party/real-time-chunking-kinetix/src/model.py).

    With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. If start == 0, then the entire chunk is allowed to change. If end == total, then the
    entire prefix is attended to. `end` takes precedence over `start`: if `end < start`, `start` is pushed down to `end`.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.rtc_simulated_delay = config.rtc_simulated_delay
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]. `timestep` may be a
        # single scalar per batch element (`[b]`) or one timestep per action token (`[b, ah]`); the latter is used by
        # real-time chunking, where the (frozen) prefix actions are conditioned at a different timestep than the suffix.
        per_token_time = timestep.ndim == 2
        if per_token_time:
            time_emb = posemb_sincos(
                timestep.reshape(-1), self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
            )
            time_emb = time_emb.reshape(*timestep.shape, -1)
        else:
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            if per_token_time:
                time_tokens = time_emb
            else:
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        u_t = noise - actions

        if self.rtc_simulated_delay is None:
            # Standard flow matching: a single timestep per batch element. (pi05 convention: t=1 is noise, t=0 is the
            # target, so x_t = t * noise + (1 - t) * actions.)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            time_arg = time
            loss_mask = None
        else:
            # Training-time RTC: freeze a randomly-sized prefix at the clean timestep (t=0) and mask its loss. Follows
            # the kinetix reference, adapted to the pi05 time convention (clean is t=0 instead of t=1).
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            w = jnp.exp(jnp.arange(0, self.rtc_simulated_delay)[::-1].astype(jnp.float32))
            w = w / jnp.sum(w)
            delay = jax.random.choice(delay_rng, self.rtc_simulated_delay, batch_shape, p=w)
            # `mask` marks the frozen prefix actions.
            mask = jnp.arange(self.action_horizon)[None, :] < delay[..., None]
            # frozen prefix -> clean (t=0); rest -> sampled timestep, broadcast to a per-token timestep.
            time_arg = jnp.where(mask, 0.0, time[..., None])
            time_expanded = time_arg[..., None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            # per-token keep mask (1 for the free suffix, 0 for the frozen prefix)
            loss_mask = jnp.logical_not(mask).astype(jnp.float32)

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time_arg)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        per_token_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if loss_mask is None:
            return per_token_loss
        # Zero out the frozen prefix tokens so they contribute no gradient (downstream takes the mean over tokens).
        return per_token_loss * loss_mask

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prev_action_chunk: at.Float[at.Array, "b ah ad"],
        inference_delay: int | at.Int[at.Array, ""],
        prefix_attention_horizon: int | at.Int[at.Array, ""],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        method: RealtimeMethod = "pinv",
        prefix_attention_schedule: PrefixAttentionSchedule = "exp",
        max_guidance_weight: float = 5.0,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Real-time chunking sampling, conditioned on the previously generated action chunk.

        Port of `realtime_action` from third_party/real-time-chunking-kinetix/src/model.py, adapted to pi05's reversed
        flow-matching time convention (t=1 is noise, t=0 is the target).

        Args:
            prev_action_chunk: the previous chunk, already shifted so index 0 aligns with the first action to generate.
            inference_delay: number of leading actions that are already committed (frozen) while this call runs.
            prefix_attention_horizon: index (exclusive) past which the previous chunk is ignored. Typically
                `action_horizon - execute_horizon`.
            method: "none" (vanilla), "pinv" (soft pseudoinverse guidance), or "hard" (hard prefix masking).
        """
        if method == "none":
            return self.sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix (constant across flow steps)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)

        def model_velocity(x_t: jax.Array, time: jax.Array) -> jax.Array:
            # `time` may be `[b]` (one timestep per batch element) or `[b, ah]` (one per action token, for hard masking).
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_local = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_local, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        if method == "hard":
            # Hard masking: at every flow step, overwrite the frozen prefix with the previous chunk and condition those
            # tokens at the clean timestep (t=0 in pi05). This is the inference path that pairs with training-time RTC.
            freeze_mask = jnp.arange(self.action_horizon)[None, :] < inference_delay  # [1, ah]

            def step(carry):
                x_t, time = carry
                x_t = jnp.where(freeze_mask[..., None], prev_action_chunk, x_t)
                time_chunk = jnp.where(freeze_mask, 0.0, jnp.broadcast_to(time, (batch_size, self.action_horizon)))
                v_t = model_velocity(x_t, time_chunk)
                return x_t + dt * v_t, time + dt

        else:  # "pinv"
            # Soft guidance: nudge each flow step toward the previous chunk on the (weighted) frozen prefix using the
            # pseudoinverse of the denoiser Jacobian. Weights decay from the frozen prefix into the free suffix.
            weights = get_prefix_weights(
                inference_delay, prefix_attention_horizon, self.action_horizon, prefix_attention_schedule
            )  # [ah]

            def step(carry):
                x_t, time = carry
                t_pi = time  # pi05 time in [0, 1]; t=1 noise, t=0 clean
                t_k = 1.0 - t_pi  # equivalent kinetix time (t=1 clean) so we can reuse the reference formulas

                def denoiser(x: jax.Array):
                    # convert pi05 velocity (noise - action) to the "toward-clean" velocity used by the reference
                    v_clean = -model_velocity(x, jnp.broadcast_to(t_pi, batch_size))
                    x_clean = x + v_clean * (1.0 - t_k)  # == x - t_pi * v_pi : predicted clean action
                    return x_clean, v_clean

                x_clean, vjp_fun, v_clean = jax.vjp(denoiser, x_t, has_aux=True)
                error = (prev_action_chunk - x_clean) * weights[None, :, None]
                pinv_correction = vjp_fun(error)[0]
                # guidance weight constants from the paper (in kinetix time t_k)
                inv_r2 = (t_k**2 + (1 - t_k) ** 2) / ((1 - t_k) ** 2)
                c = jnp.nan_to_num((1 - t_k) / t_k, posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                corrected_v_clean = v_clean + guidance_weight * pinv_correction
                # convert back to a pi05 velocity and take a standard pi05 step (dt < 0)
                v_pi = -corrected_v_clean
                return x_t + dt * v_pi, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
