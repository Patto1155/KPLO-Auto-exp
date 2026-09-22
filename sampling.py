"""Agent-editable sampling helpers.

Evaluation decoding is fixed by the protected benchmark; these helpers only affect
how training explores.
"""

from __future__ import annotations

import random


def categorical(probabilities: list[float], rng: random.Random) -> int:
    draw = rng.random()
    total = 0.0
    for index, probability in enumerate(probabilities):
        total += probability
        if draw <= total:
            return index
    return len(probabilities) - 1


def temper(probabilities: list[float], temperature: float) -> list[float]:
    if temperature == 1.0:
        return probabilities
    powered = [value ** (1.0 / temperature) for value in probabilities]
    total = sum(powered)
    return [value / total for value in powered]


def greedy(probabilities: list[float]) -> int:
    return max(range(len(probabilities)), key=probabilities.__getitem__)
