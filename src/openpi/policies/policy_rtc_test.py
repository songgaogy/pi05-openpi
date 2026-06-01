# ruff: noqa: SLF001

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.policies import policy as _policy


@pytest.mark.parametrize("inference_delay", [0, 2, 4])
def test_restore_hard_prefix(inference_delay):
    prev = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)
    sampled = jnp.full((1, 4, 2), -1.0)

    restored = np.asarray(_policy._restore_hard_prefix(sampled, prev, inference_delay))

    np.testing.assert_array_equal(restored[:, :inference_delay], np.asarray(prev[:, :inference_delay]))
    np.testing.assert_array_equal(restored[:, inference_delay:], np.asarray(sampled[:, inference_delay:]))


def test_realtime_policy_restores_and_caches_hard_prefix():
    policy = _policy.RealtimePolicy.__new__(_policy.RealtimePolicy)

    def input_transform(inputs):
        if "actions" in inputs:
            inputs["actions"] -= inputs["state"][None]
        return inputs

    def output_transform(outputs):
        return {"actions": outputs["actions"] + outputs["state"][None]}

    policy._input_transform = input_transform
    policy._output_transform = output_transform
    policy._sample_kwargs = {}
    policy._rng = jax.random.key(0)
    policy._prev_actions = np.arange(8, dtype=np.float32).reshape(4, 2)
    policy._execute_horizon = 2
    policy._inference_delay = 1
    policy._prefix_attention_horizon = 2
    policy._method = "hard"
    policy._prefix_attention_schedule = "exp"
    policy._max_guidance_weight = 5.0
    policy._sample_actions_rtc = lambda *args, **kwargs: jnp.full((1, 4, 2), -1.0)

    outputs = policy.infer({"image": {}, "image_mask": {}, "state": np.array([3.0, 3.0], dtype=np.float32)})

    expected = np.array([[4.0, 5.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    np.testing.assert_array_equal(outputs["actions"], expected)
    np.testing.assert_array_equal(policy._prev_actions, expected)
