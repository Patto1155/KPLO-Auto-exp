from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from benchmark.metrics import aggregate


SUMMARY_PATTERN = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*):\s+(-?[0-9]+(?:\.[0-9]+)?)$", re.M)


def _parse(stdout: str) -> dict[str, float]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("metrics"), dict):
            return {key: float(item) for key, item in value["metrics"].items()}
    parsed = {key: float(value) for key, value in SUMMARY_PATTERN.findall(stdout)}
    if not parsed:
        raise ValueError("benchmark command emitted no parseable metrics")
    return parsed


def evaluate(root: Path, command: list[str], seeds: list[int], budget: dict, extra_env: dict | None = None) -> tuple[dict, list[dict]]:
    runs = []
    for seed in seeds:
        env = os.environ.copy()
        env["NRL_SEED"] = str(seed)
        env["NRL_BUDGET"] = json.dumps(budget, sort_keys=True)
        env.update(extra_env or {})
        started = time.perf_counter()
        process = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
        elapsed = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(f"benchmark failed ({process.returncode}):\n{process.stderr[-2000:]}")
        metrics = _parse(process.stdout)
        metrics.setdefault("wall_seconds", elapsed)
        runs.append({"seed": seed, "metrics": metrics})
    return aggregate([run["metrics"] for run in runs]), runs
