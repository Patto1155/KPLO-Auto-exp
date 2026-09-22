"""KL-penalised policy optimisation.

KLPO keeps a policy-gradient objective close to the reference checkpoint under an
explicit divergence budget. Two things are research axes rather than fixed choices:

* **Which KL estimator.** `exact` sums the analytic per-step divergence over the
  action vocabulary and differentiates it directly, which is affordable here because
  the vocabulary is ten digits. `k1`, `k2` and `k3` are the usual sampled estimators
  built from the sequence log-ratio and enter through the reward channel instead.
  `k3` is the non-negative, unbiased one; `k1` is unbiased but signed; `k2` is biased
  but low variance.
* **How the penalty is weighted.** With `adaptive` the coefficient chases a KL target
  the way an adaptive-penalty PPO controller does, so the budget rather than the
  coefficient is the thing held fixed across experiments.

KLPO is experiment zero, not a privileged answer. It is expected to lose to something
an agent writes later.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev

from rl import PolicyGradient, Rollout, analytic_kl, grouped, sequence_kl

ESTIMATORS = ("exact", "k1", "k2", "k3")
BASELINES = ("group_mean", "group_norm", "batch_mean", "none")


@dataclass
class KLPO(PolicyGradient):
    kl_coefficient: float = 0.01
    kl_target: float = 0.02
    estimator: str = "exact"
    adaptive: bool = True
    horizon: float = 1.5
    increase: float = 1.25
    decrease: float = 0.8
    baseline: str = "group_mean"
    min_coefficient: float = 1e-4
    max_coefficient: float = 0.5
    epsilon: float = 1e-8
    name: str = "klpo"
    measured_kl: float = 0.0

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")

    def coefficients(self, rollouts: list[Rollout]) -> list[float]:
        advantages = self._advantages(rollouts)
        if self.estimator == "exact":
            return advantages
        return [
            advantage - self.kl_coefficient * sequence_kl(item, self.estimator)
            for advantage, item in zip(advantages, rollouts)
        ]

    def kl_gradient_scale(self) -> float:
        return self.kl_coefficient if self.estimator == "exact" else 0.0

    def observe(self, rollouts: list[Rollout]) -> None:
        self.measured_kl = self.divergence(rollouts)
        if not self.adaptive:
            return
        if self.measured_kl > self.kl_target * self.horizon:
            self.kl_coefficient = min(self.kl_coefficient * self.increase, self.max_coefficient)
        elif self.measured_kl < self.kl_target / self.horizon:
            self.kl_coefficient = max(self.kl_coefficient * self.decrease, self.min_coefficient)

    def divergence(self, rollouts: list[Rollout]) -> float:
        """Mean KL(pi || pi_ref) per sequence, analytic when step detail is available."""
        if not rollouts:
            return 0.0
        if any(item.steps for item in rollouts):
            return fmean(analytic_kl(item) for item in rollouts if item.steps)
        return fmean(sequence_kl(item, "k3") for item in rollouts)

    def _advantages(self, rollouts: list[Rollout]) -> list[float]:
        if self.baseline == "none":
            return [item.reward for item in rollouts]
        if self.baseline == "batch_mean":
            offset = fmean(item.reward for item in rollouts)
            return [item.reward - offset for item in rollouts]
        groups = grouped(rollouts)
        advantages = []
        for item in rollouts:
            rewards = [member.reward for member in groups[item.group]]
            advantage = item.reward - fmean(rewards)
            if self.baseline == "group_norm":
                advantage /= pstdev(rewards) + self.epsilon
            advantages.append(advantage)
        return advantages
