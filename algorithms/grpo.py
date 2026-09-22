from dataclasses import dataclass
from statistics import fmean, pstdev

from rl import Rollout


@dataclass
class GRPO:
    epsilon: float = 1e-8
    name: str = "grpo"

    def losses(self, rollouts: list[Rollout]) -> list[float]:
        groups: dict[int, list[Rollout]] = {}
        for rollout in rollouts:
            groups.setdefault(rollout.group, []).append(rollout)
        losses = []
        for rollout in rollouts:
            rewards = [item.reward for item in groups[rollout.group]]
            advantage = (rollout.reward - fmean(rewards)) / (pstdev(rewards) + self.epsilon)
            losses.append(-rollout.logprob * advantage)
        return losses
