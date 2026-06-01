# ruff: noqa: SLF001

import pytest

from openpi.serving import policy_warmup


class _FakePolicy:
    def __init__(self) -> None:
        self.inferred_observations = []
        self.reset_count = 0

    def infer(self, observation):
        self.inferred_observations.append(observation)
        observation["value"].append("mutated")
        return {"actions": []}

    def reset(self) -> None:
        self.reset_count += 1


def test_warm_up_policy_runs_until_stable_and_resets_state(monkeypatch):
    clock = iter([0.0, 10.0, 20.0, 26.0, 30.0, 30.08, 40.0, 40.082])
    monkeypatch.setattr(policy_warmup.time, "monotonic", lambda: next(clock))
    policy = _FakePolicy()
    observation = {"value": []}

    result = policy_warmup.warm_up_policy(policy, observation)

    assert result.stabilized
    assert result.latencies_ms == pytest.approx((10_000.0, 6_000.0, 80.0, 82.0))
    assert policy.reset_count == 2
    assert observation == {"value": []}
    assert all(obs == {"value": ["mutated"]} for obs in policy.inferred_observations)


def test_warmup_config_rejects_invalid_iterations():
    with pytest.raises(ValueError, match="stable_window"):
        policy_warmup.WarmupConfig(min_iterations=1, max_iterations=2, stable_window=2)
