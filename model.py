"""Mutable model surface.

`QUALITY_BIAS` drives the smoke benchmark. `Policy` is the experimental organism for
the RL reasoning profile: a token-embedding network with an explicit backward pass so
the harness stays dependency-free and deterministic on CPU. Research agents may change
the architecture, the width, the initialisation, or replace this entirely, provided
`Policy.initialise`, `distribution`, `accumulate`, and `tensors` keep their contracts.
"""

from __future__ import annotations

import math
import random

from benchmark.reasoning_env import ACTION_SIZE, CONTEXT_SLOTS, VOCAB_SIZE

QUALITY_BIAS = 0.03

EMBED_DIM = 4
HIDDEN_DIM = 16
INIT_SCALE = 0.6


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _random(rows: int, cols: int, scale: float, rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


class Policy:
    """Embedding -> tanh hidden layer -> digit logits."""

    def __init__(self, tensors: list[list[list[float]]], embed_dim: int, hidden_dim: int) -> None:
        self.embedding, self.w1, self.b1, self.w2, self.b2 = tensors
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    @classmethod
    def initialise(cls, seed: int, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM) -> "Policy":
        """Deterministic starting checkpoint; both sides of a comparison share it."""
        rng = random.Random(seed ^ 0x11117)
        width = CONTEXT_SLOTS * embed_dim
        return cls(
            [
                _random(VOCAB_SIZE, embed_dim, INIT_SCALE, rng),
                _random(hidden_dim, width, INIT_SCALE / math.sqrt(width), rng),
                _zeros(1, hidden_dim),
                _random(ACTION_SIZE, hidden_dim, INIT_SCALE / math.sqrt(hidden_dim), rng),
                _zeros(1, ACTION_SIZE),
            ],
            embed_dim,
            hidden_dim,
        )

    def tensors(self) -> list[list[list[float]]]:
        return [self.embedding, self.w1, self.b1, self.w2, self.b2]

    def zero_gradients(self) -> list[list[list[float]]]:
        return [[[0.0] * len(row) for row in tensor] for tensor in self.tensors()]

    def clone(self) -> "Policy":
        copied = [[list(row) for row in tensor] for tensor in self.tensors()]
        return Policy(copied, self.embed_dim, self.hidden_dim)

    @property
    def parameter_count(self) -> int:
        return sum(len(row) for tensor in self.tensors() for row in tensor)

    def forward(self, context: list[int]) -> tuple[list[float], dict]:
        """Return the digit distribution plus the cache the backward pass needs."""
        embedding = self.embedding
        features: list[float] = []
        for token in context:
            features.extend(embedding[token])
        bias1 = self.b1[0]
        hidden = []
        for index, row in enumerate(self.w1):
            total = bias1[index]
            for weight, value in zip(row, features):
                total += weight * value
            hidden.append(math.tanh(total))
        bias2 = self.b2[0]
        logits = []
        for index, row in enumerate(self.w2):
            total = bias2[index]
            for weight, value in zip(row, hidden):
                total += weight * value
            logits.append(total)
        highest = max(logits)
        exponentials = [math.exp(value - highest) for value in logits]
        total = sum(exponentials)
        probabilities = [value / total for value in exponentials]
        return probabilities, {"features": features, "hidden": hidden, "context": context}

    def distribution(self, context: list[int]) -> list[float]:
        return self.forward(context)[0]

    def accumulate(self, cache: dict, dlogits: list[float], gradients: list[list[list[float]]]) -> None:
        """Add d(loss)/d(parameters) for one decoding step into `gradients`."""
        grad_embedding, grad_w1, grad_b1, grad_w2, grad_b2 = gradients
        features = cache["features"]
        hidden = cache["hidden"]
        grad_hidden = [0.0] * len(hidden)
        row_b2 = grad_b2[0]
        for index, delta in enumerate(dlogits):
            if delta == 0.0:
                continue
            row = grad_w2[index]
            weights = self.w2[index]
            for position, value in enumerate(hidden):
                row[position] += delta * value
                grad_hidden[position] += delta * weights[position]
            row_b2[index] += delta
        row_b1 = grad_b1[0]
        grad_features = [0.0] * len(features)
        for index, value in enumerate(hidden):
            delta = grad_hidden[index] * (1.0 - value * value)
            if delta == 0.0:
                continue
            row = grad_w1[index]
            weights = self.w1[index]
            for position, feature in enumerate(features):
                row[position] += delta * feature
                grad_features[position] += delta * weights[position]
            row_b1[index] += delta
        width = self.embed_dim
        for slot, token in enumerate(cache["context"]):
            target = grad_embedding[token]
            offset = slot * width
            for position in range(width):
                target[position] += grad_features[offset + position]
