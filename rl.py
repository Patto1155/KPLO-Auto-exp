"""Minimal common interface for verifiable-reward RL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Rollout:
    logprob: float
    reward: float
    reference_logprob: float = 0.0
    group: int = 0


class RLAlgorithm(Protocol):
    name: str

    def losses(self, rollouts: list[Rollout]) -> list[float]: ...
