"""Tests for the real-time chunking (RTC) additions to the pi05 model.

These mirror the kinetix reference implementation in
`third_party/real-time-chunking-kinetix/src/model.py` where applicable.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0 as _pi0
from openpi.models import pi0_config


def _kinetix_get_prefix_weights(start, end, total, schedule):
    """Verbatim port of the kinetix reference, used as a ground-truth oracle."""
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


@pytest.mark.parametrize("schedule", ["linear", "exp", "ones", "zeros"])
@pytest.mark.parametrize(("start", "end"), [(0, 8), (2, 6), (1, 1), (4, 2), (0, 0), (3, 8)])
def test_get_prefix_weights_matches_kinetix(schedule, start, end):
    total = 8
    ours = _pi0.get_prefix_weights(start, end, total, schedule)
    ref = _kinetix_get_prefix_weights(start, end, total, schedule)
    np.testing.assert_allclose(np.asarray(ours), np.asarray(ref), rtol=1e-6, atol=1e-6)


def test_get_prefix_weights_docstring_example():
    # From the docstring: start=2, end=6, total=10 -> 1 1 4/5 3/5 2/5 1/5 0 0 0 0
    w = np.asarray(_pi0.get_prefix_weights(2, 6, 10, "linear"))
    expected = np.array([1, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 0, 0, 0, 0])
    np.testing.assert_allclose(w, expected, rtol=1e-6, atol=1e-6)


def _make_model(rtc_simulated_delay=None):
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, action_horizon=8, rtc_simulated_delay=rtc_simulated_delay)
    return config, config.create(key)


def test_per_token_time_matches_scalar_time():
    """A per-token timestep where every token shares the same value must match the scalar-time path."""
    config, model = _make_model()
    obs = config.fake_obs(1)
    x = jax.random.normal(jax.random.key(1), (1, model.action_horizon, model.action_dim))

    t_scalar = jnp.full((1,), 0.7)
    t_per_token = jnp.full((1, model.action_horizon), 0.7)

    _, _, _, cond_scalar = model.embed_suffix(obs, x, t_scalar)
    _, _, _, cond_per_token = model.embed_suffix(obs, x, t_per_token)

    assert cond_scalar.ndim == 2
    assert cond_per_token.ndim == 3
    # Each token's conditioning should equal the scalar conditioning.
    np.testing.assert_allclose(
        np.asarray(cond_per_token), np.broadcast_to(np.asarray(cond_scalar)[:, None], cond_per_token.shape), rtol=2e-3, atol=2e-3
    )


def test_rtc_none_matches_sample_actions():
    """method='none' must be numerically identical to the plain sampler given the same noise."""
    _, model = _make_model()
    obs = pi0_config.Pi0Config(pi05=True, action_horizon=8).fake_obs(1)
    key = jax.random.key(2)
    noise = jax.random.normal(jax.random.key(3), (1, model.action_horizon, model.action_dim))
    prev = jnp.zeros((1, model.action_horizon, model.action_dim))

    base = model.sample_actions(key, obs, num_steps=4, noise=noise)
    rtc = model.sample_actions_rtc(
        key, obs, prev, inference_delay=1, prefix_attention_horizon=6, num_steps=4, method="none", noise=noise
    )
    np.testing.assert_allclose(np.asarray(base), np.asarray(rtc), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["pinv", "hard"])
def test_rtc_sampling_shapes(method):
    _, model = _make_model()
    obs = pi0_config.Pi0Config(pi05=True, action_horizon=8).fake_obs(1)
    noise = jax.random.normal(jax.random.key(5), (1, model.action_horizon, model.action_dim))
    prev = jax.random.normal(jax.random.key(6), (1, model.action_horizon, model.action_dim))

    out = model.sample_actions_rtc(
        jax.random.key(7),
        obs,
        prev,
        inference_delay=1,
        prefix_attention_horizon=6,
        num_steps=4,
        method=method,
        noise=noise,
    )
    assert out.shape == (1, model.action_horizon, model.action_dim)
    assert jnp.all(jnp.isfinite(out))


def test_rtc_hard_freezes_prefix_input():
    """With hard masking the frozen prefix is overwritten by the previous chunk at every step's *input*, so the noise in
    that prefix region must have no effect whatsoever on the sampled output (matching the kinetix reference, where the
    final prefix is not pinned but its input is)."""
    _, model = _make_model()
    obs = pi0_config.Pi0Config(pi05=True, action_horizon=8).fake_obs(1)
    prev = jax.random.normal(jax.random.key(9), (1, model.action_horizon, model.action_dim))
    delay = 3

    noise_a = jax.random.normal(jax.random.key(8), (1, model.action_horizon, model.action_dim))
    # noise_b differs from noise_a only inside the frozen prefix region.
    noise_b = noise_a.at[:, :delay].set(jax.random.normal(jax.random.key(80), (1, delay, model.action_dim)))

    kwargs = dict(inference_delay=delay, prefix_attention_horizon=6, num_steps=4, method="hard")
    out_a = model.sample_actions_rtc(jax.random.key(10), obs, prev, noise=noise_a, **kwargs)
    out_b = model.sample_actions_rtc(jax.random.key(10), obs, prev, noise=noise_b, **kwargs)
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), rtol=1e-5, atol=1e-5)


def test_training_time_rtc_loss_shape():
    config, model = _make_model(rtc_simulated_delay=4)
    obs = config.fake_obs(2)
    act = config.fake_act(2)
    loss = model.compute_loss(jax.random.key(11), obs, act, train=True)
    assert loss.shape == (2, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))
