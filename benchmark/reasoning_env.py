"""Protected verifiable-reward reasoning environment.

The task is extreme selection: given four digits, emit the smallest then the largest.
It is exactly verifiable, it needs comparison across positions rather than memorisation
of a lookup table, and a small policy can make partial progress on it quickly, which
leaves a wide frontier between a first working baseline and a perfect score.

Ground truth is computed from the problem rather than stored, so there is no label file
a candidate could read, and the train/held-out partition is a protected hash of the
prompt. Mutable training code can only reach the train split, and every reward query is
charged against a fixed rollout budget.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass

TASK_NAME = "extreme-selection"
TASK_VERSION = "1.0.0"
SPLIT_SALT = "nrl-reasoning-v1"
HELDOUT_MODULUS = 5
DIGIT_BASE = 10
PROMPT_SLOTS = 4
ANSWER_DIGITS = 2
CONTEXT_SLOTS = PROMPT_SLOTS + ANSWER_DIGITS - 1
PAD_TOKEN = DIGIT_BASE
VOCAB_SIZE = DIGIT_BASE + 1
ACTION_SIZE = DIGIT_BASE


class RolloutBudgetExceeded(RuntimeError):
    """Raised when training asks for more verified rollouts than the budget allows."""


class HeldoutAccessError(RuntimeError):
    """Raised when training tries to score a held-out problem."""


@dataclass(frozen=True)
class Problem:
    prompt: tuple[int, ...]

    @property
    def answer(self) -> tuple[int, ...]:
        return (min(self.prompt), max(self.prompt))


def all_problems() -> list[Problem]:
    return [Problem(prompt) for prompt in itertools.product(range(DIGIT_BASE), repeat=PROMPT_SLOTS)]


def is_heldout(problem: Problem) -> bool:
    key = ",".join(str(digit) for digit in problem.prompt)
    digest = hashlib.sha256(f"{SPLIT_SALT}:{key}".encode()).hexdigest()
    return int(digest[:8], 16) % HELDOUT_MODULUS == 0


def context(prompt: tuple[int, ...], emitted: tuple[int, ...]) -> list[int]:
    """Policy input slots: the prompt digits then the answer digits decided so far."""
    history = list(emitted[: ANSWER_DIGITS - 1])
    history += [PAD_TOKEN] * (ANSWER_DIGITS - 1 - len(history))
    return list(prompt) + history


def digit_matches(problem: Problem, emitted: tuple[int, ...]) -> int:
    return sum(int(a == b) for a, b in zip(problem.answer, emitted))


def dense_reward(problem: Problem, emitted: tuple[int, ...]) -> float:
    """Verifiable partial credit: the fraction of answer digits that are correct."""
    return digit_matches(problem, emitted) / ANSWER_DIGITS


def solved(problem: Problem, emitted: tuple[int, ...]) -> bool:
    return tuple(emitted) == problem.answer


def train_split() -> list[Problem]:
    return [problem for problem in all_problems() if not is_heldout(problem)]


def heldout_split() -> list[Problem]:
    return [problem for problem in all_problems() if is_heldout(problem)]


def split_digest() -> str:
    """Fingerprint of the held-out partition, logged with every evaluation."""
    payload = ";".join(",".join(map(str, problem.prompt)) for problem in heldout_split())
    return hashlib.sha256(payload.encode()).hexdigest()


def heldout_sample(count: int, seed: int) -> list[Problem]:
    problems = heldout_split()
    if count >= len(problems):
        return problems
    return random.Random(seed ^ 0xE7A1).sample(problems, count)


class TrainingEnvironment:
    """Train-split problem source that meters every verified rollout.

    Mutable research code owns the curriculum and the rollout strategy; it does not
    own the reward, the split, or the budget.
    """

    def __init__(self, rollout_budget: int, seed: int) -> None:
        self.rollout_budget = int(rollout_budget)
        self.seed = seed
        self._problems = train_split()
        self._used = 0
        self._reward_total = 0.0
        self._solved = 0

    @property
    def rollouts_used(self) -> int:
        return self._used

    @property
    def rollouts_remaining(self) -> int:
        return self.rollout_budget - self._used

    @property
    def train_reward_mean(self) -> float:
        return self._reward_total / self._used if self._used else 0.0

    @property
    def train_solve_rate(self) -> float:
        return self._solved / self._used if self._used else 0.0

    def sample_problem(self, rng: random.Random) -> Problem:
        return rng.choice(self._problems)

    def sample_problems(self, count: int, rng: random.Random) -> list[Problem]:
        return [rng.choice(self._problems) for _ in range(count)]

    def score(self, problem: Problem, emitted: tuple[int, ...]) -> float:
        """Charge one rollout and return the verifiable reward for an attempt."""
        if self._used >= self.rollout_budget:
            raise RolloutBudgetExceeded(f"rollout budget of {self.rollout_budget} exhausted")
        if is_heldout(problem):
            raise HeldoutAccessError(f"{problem.prompt} belongs to the held-out split")
        self._used += 1
        reward = dense_reward(problem, tuple(emitted))
        self._reward_total += reward
        self._solved += int(solved(problem, tuple(emitted)))
        return reward
