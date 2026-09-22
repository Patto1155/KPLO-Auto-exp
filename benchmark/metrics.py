from __future__ import annotations

from statistics import fmean, pstdev


def aggregate(runs: list[dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(run) for run in runs))
    result: dict[str, float] = {}
    for key in sorted(keys):
        values = [float(run[key]) for run in runs]
        result[key] = fmean(values)
        if len(values) > 1:
            result[f"{key}_std"] = pstdev(values)
    return result


def improvement(baseline: float, candidate: float, direction: str) -> float:
    return candidate - baseline if direction == "max" else baseline - candidate
