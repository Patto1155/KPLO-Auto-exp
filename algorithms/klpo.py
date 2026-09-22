from dataclasses import dataclass

from rl import Rollout


@dataclass
class KLPO:
    kl_coefficient: float = 0.05
    baseline: float = 0.0
    name: str = "klpo"

    def losses(self, rollouts: list[Rollout]) -> list[float]:
        return [
            -r.logprob * (r.reward - self.baseline)
            + self.kl_coefficient * (r.logprob - r.reference_logprob) ** 2
            for r in rollouts
        ]
