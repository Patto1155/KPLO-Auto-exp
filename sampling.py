"""Agent-editable sampling helpers."""

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
