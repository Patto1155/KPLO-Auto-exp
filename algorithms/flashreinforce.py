"""REINFORCE with whitened advantages computed in one fused pass over the batch.

This is the reference semantics only. Replace the pass with kernels after profiling
shows the gradient assembly, rather than the environment, is the bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl import PolicyGradient, Rollout


@dataclass
class FlashREINFORCE(PolicyGradient):
    epsilon: float = 1e-8
    name: str = "flashreinforce"

    def coefficients(self, rollouts: list[Rollout]) -> list[float]:
        if not rollouts:
            return []
        count = len(rollouts)
        total = 0.0
        total_square = 0.0
        for item in rollouts:
            total += item.reward
            total_square += item.reward * item.reward
        mean = total / count
        variance = max(total_square / count - mean * mean, 0.0)
        scale = variance**0.5 + self.epsilon
        return [(item.reward - mean) / scale for item in rollouts]
