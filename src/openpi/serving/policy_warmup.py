import copy
import dataclasses
import logging
import time

from openpi_client import base_policy as _base_policy


@dataclasses.dataclass(frozen=True)
class WarmupConfig:
    min_iterations: int = 3
    max_iterations: int = 6
    stable_window: int = 2
    relative_tolerance: float = 0.25
    absolute_tolerance_ms: float = 10.0

    def __post_init__(self) -> None:
        if not 0 < self.stable_window <= self.min_iterations <= self.max_iterations:
            raise ValueError("warmup iterations must satisfy 0 < stable_window <= min_iterations <= max_iterations")
        if self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be non-negative")
        if self.absolute_tolerance_ms < 0:
            raise ValueError("absolute_tolerance_ms must be non-negative")


@dataclasses.dataclass(frozen=True)
class WarmupResult:
    latencies_ms: tuple[float, ...]
    stabilized: bool


def _latencies_are_stable(latencies_ms: list[float], config: WarmupConfig) -> bool:
    if len(latencies_ms) < config.min_iterations:
        return False
    window = latencies_ms[-config.stable_window :]
    reference_ms = sum(window) / len(window)
    tolerance_ms = max(config.absolute_tolerance_ms, reference_ms * config.relative_tolerance)
    return max(window) - min(window) <= tolerance_ms


def warm_up_policy(
    policy: _base_policy.BasePolicy,
    observation: dict,
    *,
    config: WarmupConfig | None = None,
) -> WarmupResult:
    """Run dummy inferences before serving traffic and reset any stateful policy history afterwards."""
    config = config or WarmupConfig()
    latencies_ms: list[float] = []
    stabilized = False
    policy.reset()
    try:
        for iteration in range(1, config.max_iterations + 1):
            start = time.monotonic()
            policy.infer(copy.deepcopy(observation))
            latency_ms = (time.monotonic() - start) * 1000
            latencies_ms.append(latency_ms)
            logging.info("Policy warmup inference %d: %.1f ms", iteration, latency_ms)
            if _latencies_are_stable(latencies_ms, config):
                stabilized = True
                break
    finally:
        # RTC and other stateful wrappers must not expose dummy rollout history to the first real client.
        policy.reset()

    if stabilized:
        logging.info("Policy warmup stabilized after %d inferences", len(latencies_ms))
    else:
        logging.warning("Policy warmup did not stabilize after %d inferences; starting server anyway", len(latencies_ms))
    return WarmupResult(tuple(latencies_ms), stabilized)
