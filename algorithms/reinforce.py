"""REINFORCE with an optional batch baseline: the simplest honest control."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from rl import PolicyGradient, Rollout

BASELINES = ("batch_mean", "none")


@dataclass
class REINFORCE(PolicyGradient):
    baseline: str = "batch_mean"
    name: str = "reinforce"

    def __post_init__(self) -> None:
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")

    def coefficients(self, rollouts: list[Rollout]) -> list[float]:
        if not rollouts or self.baseline == "none":
            return [item.reward for item in rollouts]
        offset = fmean(item.reward for item in rollouts)
        return [item.reward - offset for item in rollouts]
