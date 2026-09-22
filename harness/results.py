from __future__ import annotations

import json
from pathlib import Path


def append_record(root: Path, record: dict) -> None:
    path = root / "research" / "experiments.jsonl"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def records(root: Path) -> list[dict]:
    path = root / "research" / "experiments.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def next_id(root: Path) -> str:
    return f"exp_{len(records(root)) + 1:04d}"


def write_leaderboard(root: Path) -> None:
    kept = [item for item in records(root) if item["decision"] == "KEEP"]
    kept.sort(key=lambda item: item["delta"]["primary_improvement"], reverse=True)
    rows = []
    for rank, item in enumerate(kept, 1):
        rows.append({
            "rank": rank,
            "experiment": item["experiment_id"],
            "parent": item["parent"],
            "method": item.get("algorithm", "unspecified"),
            "score": item["candidate_metrics"][item["primary_metric"]],
            "delta": item["delta"]["primary_improvement"],
            "compute": item["candidate_metrics"].get("wall_seconds"),
            "confirmed": item.get("confirmed", False),
        })
    (root / "research" / "leaderboard.json").write_text(json.dumps(rows, indent=2) + "\n")
