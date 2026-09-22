"""Group-relative policy optimisation.

Advantages are standardised inside each group of rollouts that share a problem, and
the reference policy is held nearby with a fixed sampled KL penalty. The contrast with
KLPO is deliberate: fixed coefficient and sampled estimator here, budgeted coefficient
and analytic gradient there.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev

from rl import PolicyGradient, Rollout, grouped, sequence_kl


@dataclass
class GRPO(PolicyGradient):
    epsilon: float = 1e-8
    kl_coefficient: float = 0.02
    kl_estimator: str = "k3"
    name: str = "grpo"

    def coefficients(self, rollouts: list[Rollout]) -> list[float]:
        groups = grouped(rollouts)
        advantages = []
        for item in rollouts:
            rewards = [member.reward for member in groups[item.group]]
            advantage = (item.reward - fmean(rewards)) / (pstdev(rewards) + self.epsilon)
            if self.kl_coefficient:
                advantage -= self.kl_coefficient * sequence_kl(item, self.kl_estimator)
            advantages.append(advantage)
        return advantages
