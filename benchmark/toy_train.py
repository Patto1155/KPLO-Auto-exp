"""Deterministic CPU benchmark for end-to-end harness validation only."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import QUALITY_BIAS

seed = int(os.environ.get("NRL_SEED", "0"))
budget = json.loads(os.environ.get("NRL_BUDGET", "{}"))
rng = random.Random(seed)
noise = rng.uniform(-0.002, 0.002)
score = 0.5 + QUALITY_BIAS + noise
metrics = {
    "score": score,
    "val_loss": 2.0 - score,
    "train_loss": 2.1 - score,
    "tokens": float(budget.get("rollouts", 100) * 16),
    "rollouts": float(budget.get("rollouts", 100)),
    "peak_gpu_memory_mb": 0.0,
    "reward_mean": score,
    "reward_std": 0.1,
    "kl": max(0.0, abs(QUALITY_BIAS) / 2),
    "entropy": 0.69,
    "crash_rate": 0.0,
}
print(json.dumps({"metrics": metrics}, sort_keys=True))
