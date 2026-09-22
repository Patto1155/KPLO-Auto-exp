"""Mutable RL training loop for the verifiable-reasoning profile.

The protected benchmark hands this module a starting checkpoint and a metered
environment, then measures the returned policy on held-out problems. Everything here
is a research knob: the optimiser, the batch and group shape, the rollout strategy,
the entropy bonus, and which algorithm assembles the gradient.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from benchmark.reasoning_env import ANSWER_DIGITS, Problem, TrainingEnvironment, context
from model import Policy
from sampling import categorical, temper

ALGORITHM = "klpo"
GROUP_SIZE = 8
BATCH_PROBLEMS = 4
LEARNING_RATE = 0.02
TEMPERATURE = 1.0
ENTROPY_BONUS = 0.0
GRADIENT_CLIP = 5.0
ANCHOR_REFRESH = 25


@dataclass(frozen=True)
class Step:
    context: list[int]
    action: int
    probabilities: list[float]
    reference_probabilities: list[float]
    cache: dict


@dataclass(frozen=True)
class Rollout:
    logprob: float
    reward: float
    reference_logprob: float = 0.0
    group: int = 0
    steps: tuple[Step, ...] = field(default=())

    @property
    def logratio(self) -> float:
        """log pi(y) - log pi_ref(y), the sampled quantity every KL estimator uses."""
        return self.logprob - self.reference_logprob


class RLAlgorithm(Protocol):
    name: str

    def coefficients(self, rollouts: list[Rollout]) -> list[float]:
        """Per-rollout weight on the score-function gradient."""

    def kl_gradient_scale(self) -> float:
        """Weight of the analytic KL gradient, or 0.0 for sampled-only penalties."""

    def observe(self, rollouts: list[Rollout]) -> None:
        """Hook for adaptive controllers; called once per optimiser step."""


class PolicyGradient:
    """Defaults shared by every algorithm: no analytic KL and no adaptive state."""

    name = "policy-gradient"

    def coefficients(self, rollouts: list[Rollout]) -> list[float]:
        raise NotImplementedError

    def kl_gradient_scale(self) -> float:
        return 0.0

    def observe(self, rollouts: list[Rollout]) -> None:
        return None

    def losses(self, rollouts: list[Rollout]) -> list[float]:
        """Scalar view of the objective; the trainer differentiates it directly."""
        return [-item.logprob * weight for item, weight in zip(rollouts, self.coefficients(rollouts))]


def grouped(rollouts: list[Rollout]) -> dict[int, list[Rollout]]:
    groups: dict[int, list[Rollout]] = {}
    for item in rollouts:
        groups.setdefault(item.group, []).append(item)
    return groups


def sequence_kl(item: Rollout, estimator: str) -> float:
    """Sampled KL(pi || pi_ref) estimators evaluated on one rollout."""
    logratio = item.logratio
    if estimator == "k1":
        return logratio
    if estimator == "k2":
        return 0.5 * logratio * logratio
    if estimator == "k3":
        return math.exp(-logratio) - 1.0 + logratio
    raise ValueError(f"unknown sampled KL estimator {estimator!r}")


def analytic_kl(item: Rollout) -> float:
    """Exact per-step KL(pi || pi_ref) summed over the sequence."""
    total = 0.0
    for step in item.steps:
        for probability, reference in zip(step.probabilities, step.reference_probabilities):
            total += probability * math.log(max(probability, 1e-12) / max(reference, 1e-12))
    return total


class Adam:
    def __init__(self, tensors, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        self.moment = [[[0.0] * len(row) for row in tensor] for tensor in tensors]
        self.velocity = [[[0.0] * len(row) for row in tensor] for tensor in tensors]

    def step(self, tensors, gradients) -> None:
        self.step_count += 1
        bias1 = 1.0 - self.beta1**self.step_count
        bias2 = 1.0 - self.beta2**self.step_count
        for tensor, gradient, moment, velocity in zip(tensors, gradients, self.moment, self.velocity):
            for row, grad_row, moment_row, velocity_row in zip(tensor, gradient, moment, velocity):
                for index, value in enumerate(grad_row):
                    moment_row[index] = self.beta1 * moment_row[index] + (1.0 - self.beta1) * value
                    velocity_row[index] = self.beta2 * velocity_row[index] + (1.0 - self.beta2) * value * value
                    corrected = moment_row[index] / bias1
                    scale = math.sqrt(velocity_row[index] / bias2) + self.eps
                    row[index] -= self.learning_rate * corrected / scale


def make_algorithm(name: str) -> RLAlgorithm:
    from algorithms import FlashREINFORCE, GRPO, KLPO, REINFORCE

    table = {
        "reinforce": REINFORCE,
        "grpo": GRPO,
        "klpo": KLPO,
        "flashreinforce": FlashREINFORCE,
    }
    if name not in table:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(table)}")
    return table[name]()


class _ReferencePolicy:
    """The KL anchor, memoised because contexts repeat constantly.

    Refreshing the anchor periodically turns the KL penalty into a proximal step-size
    constraint. Pinning it to the random starting checkpoint instead would make every
    KL-penalised algorithm optimise against its own objective.
    """

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._cache: dict[tuple[int, ...], list[float]] = {}

    def refresh(self, policy: Policy) -> None:
        self.policy = policy.clone()
        self._cache.clear()

    def distribution(self, slots: list[int]) -> list[float]:
        key = tuple(slots)
        cached = self._cache.get(key)
        if cached is None:
            cached = self.policy.distribution(slots)
            self._cache[key] = cached
        return cached


def rollout(policy: Policy, reference: _ReferencePolicy, problem: Problem, group: int, rng: random.Random) -> tuple[Rollout, tuple[int, ...]]:
    prompt = problem.prompt
    emitted: list[int] = []
    steps: list[Step] = []
    logprob = 0.0
    reference_logprob = 0.0
    for _ in range(ANSWER_DIGITS):
        slots = context(prompt, tuple(emitted))
        probabilities, cache = policy.forward(slots)
        probabilities = temper(probabilities, TEMPERATURE)
        reference_probabilities = reference.distribution(slots)
        action = categorical(probabilities, rng)
        logprob += math.log(max(probabilities[action], 1e-12))
        reference_logprob += math.log(max(reference_probabilities[action], 1e-12))
        steps.append(Step(slots, action, probabilities, reference_probabilities, cache))
        emitted.append(action)
    return Rollout(logprob, 0.0, reference_logprob, group, tuple(steps)), tuple(emitted)


def _accumulate_rollout(policy: Policy, item: Rollout, coefficient: float, kl_scale: float, gradients) -> None:
    for step in item.steps:
        probabilities = step.probabilities
        reference = step.reference_probabilities
        dlogits = [coefficient * value for value in probabilities]
        dlogits[step.action] -= coefficient
        if kl_scale:
            divergence = 0.0
            ratios = []
            for probability, reference_probability in zip(probabilities, reference):
                ratio = math.log(max(probability, 1e-12) / max(reference_probability, 1e-12))
                ratios.append(ratio)
                divergence += probability * ratio
            for index, probability in enumerate(probabilities):
                dlogits[index] += kl_scale * probability * (ratios[index] - divergence)
        if ENTROPY_BONUS:
            entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
            for index, probability in enumerate(probabilities):
                logarithm = math.log(max(probability, 1e-12))
                dlogits[index] += ENTROPY_BONUS * probability * (logarithm + entropy)
        policy.accumulate(step.cache, dlogits, gradients)


def _scale_and_clip(gradients, scale: float) -> None:
    total = 0.0
    for tensor in gradients:
        for row in tensor:
            for index, value in enumerate(row):
                scaled = value * scale
                row[index] = scaled
                total += scaled * scaled
    norm = math.sqrt(total)
    if GRADIENT_CLIP and norm > GRADIENT_CLIP:
        factor = GRADIENT_CLIP / norm
        for tensor in gradients:
            for row in tensor:
                for index, value in enumerate(row):
                    row[index] = value * factor


def train_policy(environment: TrainingEnvironment, policy: Policy, budget: dict, seed: int, algorithm: str | None = None) -> Policy:
    """Spend the rollout budget improving `policy` in place, then return it."""
    solver = make_algorithm(algorithm or ALGORITHM)
    rng = random.Random(seed ^ 0x5EED)
    reference = _ReferencePolicy(policy.clone())
    optimizer = Adam(policy.tensors(), LEARNING_RATE)
    per_update = GROUP_SIZE * BATCH_PROBLEMS
    updates = 0
    while environment.rollouts_remaining >= per_update:
        batch: list[Rollout] = []
        for group, problem in enumerate(environment.sample_problems(BATCH_PROBLEMS, rng)):
            for _ in range(GROUP_SIZE):
                item, emitted = rollout(policy, reference, problem, group, rng)
                reward = environment.score(problem, emitted)
                batch.append(Rollout(item.logprob, reward, item.reference_logprob, group, item.steps))
        weights = solver.coefficients(batch)
        kl_scale = solver.kl_gradient_scale()
        gradients = policy.zero_gradients()
        for item, weight in zip(batch, weights):
            _accumulate_rollout(policy, item, weight, kl_scale, gradients)
        _scale_and_clip(gradients, 1.0 / len(batch))
        optimizer.step(policy.tensors(), gradients)
        solver.observe(batch)
        updates += 1
        if ANCHOR_REFRESH and updates % ANCHOR_REFRESH == 0:
            reference.refresh(policy)
    return policy
