"""Load and validate protected benchmark templates."""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE_DIR = Path(__file__).with_name("templates")
REQUIRED_FIELDS = {
    "id",
    "name",
    "version",
    "status",
    "environment",
    "objective",
    "adapter",
    "training_budget",
    "evaluation",
    "metrics",
    "fairness",
    "integrity",
}


def template_paths() -> list[Path]:
    return sorted(TEMPLATE_DIR.glob("*.json"))


def load_template(template_id: str) -> dict:
    path = TEMPLATE_DIR / f"{template_id}.json"
    if not path.is_file():
        available = ", ".join(item.stem for item in template_paths())
        raise ValueError(f"unknown template {template_id!r}; available: {available}")
    payload = json.loads(path.read_text())
    validate_template(payload)
    return payload


def validate_template(payload: dict) -> None:
    missing = sorted(REQUIRED_FIELDS - payload.keys())
    if missing:
        raise ValueError("missing required fields: " + ", ".join(missing))
    if payload["status"] not in {"adapter-required", "runnable"}:
        raise ValueError("status must be adapter-required or runnable")
    metric = payload["metrics"]
    if metric.get("direction") not in {"min", "max"}:
        raise ValueError("metrics.direction must be min or max")
    if not metric.get("primary"):
        raise ValueError("metrics.primary is required")
    for stage in ("quick", "confirmation"):
        if stage not in payload["evaluation"]:
            raise ValueError(f"evaluation.{stage} is required")
    if not payload["fairness"] or not payload["integrity"]:
        raise ValueError("fairness and integrity rules may not be empty")


def summaries() -> list[dict]:
    output = []
    for path in template_paths():
        item = load_template(path.stem)
        output.append({
            "id": item["id"],
            "name": item["name"],
            "status": item["status"],
            "primary_metric": item["metrics"]["primary"],
            "adapter": item["adapter"]["protocol"],
        })
    return output
