from dataclasses import dataclass

from rl import Rollout


@dataclass
class REINFORCE:
    baseline: float = 0.0
    name: str = "reinforce"

    def losses(self, rollouts: list[Rollout]) -> list[float]:
        return [-r.logprob * (r.reward - self.baseline) for r in rollouts]
