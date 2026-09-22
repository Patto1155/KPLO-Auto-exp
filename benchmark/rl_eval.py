"""Protected RL benchmark: fixed checkpoint, metered training, held-out Pass@1.

The candidate's mutable code trains a policy against the train split only. Decoding at
evaluation time happens here, greedily, so a candidate cannot change how it is scored.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.reasoning_env import (  # noqa: E402
    ANSWER_DIGITS,
    TASK_VERSION,
    Problem,
    TrainingEnvironment,
    context,
    digit_matches,
    heldout_sample,
    solved,
    split_digest,
    train_split,
)

CHECKPOINT_SALT = 0x0C0FFEE
DEFAULT_ROLLOUTS = 20000
DEFAULT_EVAL_PROBLEMS = 2048


def decode(policy, problem: Problem) -> tuple[int, ...]:
    """Fixed greedy decoding; ties resolve to the lowest digit."""
    emitted: list[int] = []
    for _ in range(ANSWER_DIGITS):
        distribution = policy.distribution(context(problem.prompt, tuple(emitted)))
        best = max(range(len(distribution)), key=distribution.__getitem__)
        emitted.append(best)
    return tuple(emitted)


def score_split(policy, problems: list[Problem]) -> tuple[float, float]:
    exact = 0
    digits = 0
    for problem in problems:
        emitted = decode(policy, problem)
        exact += int(solved(problem, emitted))
        digits += digit_matches(problem, emitted)
    count = max(len(problems), 1)
    return exact / count, digits / (count * ANSWER_DIGITS)


def divergence_and_entropy(policy, reference, problems: list[Problem]) -> tuple[float, float]:
    """Mean analytic KL to the starting checkpoint and mean entropy, per decoded step."""
    total_kl = 0.0
    total_entropy = 0.0
    steps = 0
    for problem in problems:
        emitted: list[int] = []
        for _ in range(ANSWER_DIGITS):
            slots = context(problem.prompt, tuple(emitted))
            current = policy.distribution(slots)
            prior = reference.distribution(slots)
            for probability, reference_probability in zip(current, prior):
                if probability > 1e-12:
                    total_kl += probability * math.log(probability / max(reference_probability, 1e-12))
                    total_entropy -= probability * math.log(probability)
            steps += 1
            emitted.append(max(range(len(current)), key=current.__getitem__))
    divisor = max(steps, 1)
    return total_kl / divisor, total_entropy / divisor


def main() -> None:
    seed = int(os.environ.get("NRL_SEED", "101"))
    budget = json.loads(os.environ.get("NRL_BUDGET", "{}"))
    algorithm = os.environ.get("NRL_ALGORITHM") or None
    rollout_budget = int(budget.get("rollouts", DEFAULT_ROLLOUTS))
    eval_problems = int(budget.get("eval_problems", DEFAULT_EVAL_PROBLEMS))

    from model import Policy  # imported after sys.path so the candidate's copy is used
    from rl import train_policy

    initial = Policy.initialise(seed ^ CHECKPOINT_SALT)
    reference = initial.clone()
    environment = TrainingEnvironment(rollout_budget, seed)

    started = time.perf_counter()
    crashed = 0.0
    failure = None
    try:
        policy = train_policy(environment, initial, budget, seed, algorithm)
    except Exception as error:  # a broken candidate scores its untrained checkpoint
        policy = reference.clone()
        crashed = 1.0
        failure = f"{type(error).__name__}: {error}"
    train_seconds = time.perf_counter() - started

    heldout = heldout_sample(eval_problems, seed)
    trainable = train_split()
    train_problems = random.Random(seed ^ 0x7A11).sample(trainable, min(eval_problems, len(trainable)))

    inference_started = time.perf_counter()
    heldout_pass, heldout_digits = score_split(policy, heldout)
    train_pass, train_digits = score_split(policy, train_problems)
    inference_seconds = time.perf_counter() - inference_started
    kl, entropy = divergence_and_entropy(policy, reference, heldout)

    metrics = {
        "heldout_pass_at_1": heldout_pass,
        "heldout_digit_accuracy": heldout_digits,
        "train_pass_at_1": train_pass,
        "train_digit_accuracy": train_digits,
        "generalisation_gap": train_pass - heldout_pass,
        "train_reward_mean": environment.train_reward_mean,
        "train_solve_rate": environment.train_solve_rate,
        "kl_to_reference": kl,
        "entropy": entropy,
        "rollouts": float(environment.rollouts_used),
        "rollout_budget": float(rollout_budget),
        "eval_problems": float(len(heldout)),
        "parameter_count": float(policy.parameter_count),
        "total_generated_tokens": float(environment.rollouts_used * ANSWER_DIGITS),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "crash_rate": crashed,
    }
    payload = {
        "metrics": metrics,
        "task_version": TASK_VERSION,
        "split_hash": split_digest(),
        "algorithm": algorithm or "declared-in-rl.py",
    }
    if failure:
        payload["failure"] = failure
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
