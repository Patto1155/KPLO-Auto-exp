from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from benchmark.evaluate import evaluate
from benchmark.protocol import PROTOCOL_VERSION
from harness.accept import adopt, preserve_candidate, promote, return_to_parent
from harness.compare import decide
from harness.git_state import changed_files, fingerprint, git, head, revision_tree, validate_candidate_paths
from harness.results import append_record, next_id, records, write_leaderboard


def load_config(root: Path) -> dict:
    return json.loads((root / "lab.json").read_text())


def current_best(root: Path, profile_name: str) -> dict:
    path = root / "research" / "current_best.json"
    if not path.exists():
        return {"experiment_id": "baseline", "git_commit": head(root), "confirmed": True}
    payload = json.loads(path.read_text())
    if "git_commit" in payload:
        if profile_name == "smoke":
            return payload
        return {"experiment_id": f"{profile_name}-baseline", "git_commit": head(root), "confirmed": True}
    return payload.get("profiles", {}).get(
        profile_name,
        {"experiment_id": f"{profile_name}-baseline", "git_commit": head(root), "confirmed": True},
    )


def _cache_path(root: Path, key: str) -> Path:
    path = root / ".research-cache"
    path.mkdir(exist_ok=True)
    return path / f"{key}.json"


def _baseline(root: Path, revision: str, profile: dict, seeds: list[int]):
    key = fingerprint(root, revision, profile, seeds)
    path = _cache_path(root, key)
    if path.exists():
        payload = json.loads(path.read_text())
        return payload["metrics"], payload["runs"], True
    with revision_tree(root, revision) as tree:
        metrics, runs = evaluate(tree, profile["command"], seeds, profile["budget"])
    path.write_text(json.dumps({"metrics": metrics, "runs": runs}, indent=2) + "\n")
    return metrics, runs, False


def _record(root: Path, record: dict) -> None:
    append_record(root, record)
    write_leaderboard(root)
    paths = ["research/experiments.jsonl", "research/leaderboard.json"]
    if (root / "research" / "current_best.json").exists():
        paths.append("research/current_best.json")
    git(root, "add", *paths)
    git(root, "commit", "-m", f"research: record {record['experiment_id']}")


def run_experiment(root: Path, hypothesis: str, description: str, algorithm: str, profile_name: str | None) -> dict:
    experiment_id = next_id(root)
    config = load_config(root)
    profile_name = profile_name or config["default_profile"]
    profile = config["profiles"][profile_name]
    paths = changed_files(root)
    valid, validity = validate_candidate_paths(paths)
    parent = current_best(root, profile_name)
    base = {
        "experiment_id": experiment_id, "parent": parent["experiment_id"],
        "parent_commit": parent["git_commit"], "hypothesis": hypothesis,
        "description": description, "changed_files": paths, "algorithm": algorithm,
        "parameters": {"profile": profile_name, "profile_version": profile.get("version", "1")}, "budget": profile["budget"],
        "seeds": profile["quick_seeds"], "primary_metric": profile["primary_metric"],
        "protocol_version": PROTOCOL_VERSION,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "quick",
        "confirmed": False, "validity": validity,
    }
    if not valid:
        start_head = head(root)
        candidate_commit = None
        if paths:
            candidate_commit = preserve_candidate(root, experiment_id, paths)
            return_to_parent(root, start_head)
        record = {**base, "git_commit": candidate_commit, "baseline_metrics": {}, "candidate_metrics": {}, "delta": {}, "decision": "INVALID", "notes": validity}
        _record(root, record)
        return record
    start_head = head(root)
    candidate_commit = preserve_candidate(root, experiment_id, paths)
    try:
        baseline_metrics, baseline_runs, cache_hit = _baseline(root, parent["git_commit"], profile, profile["quick_seeds"])
        candidate_metrics, candidate_runs = evaluate(root, profile["command"], profile["quick_seeds"], profile["budget"])
        decision, delta, warnings = decide(baseline_metrics, candidate_metrics, profile["primary_metric"], profile["direction"], "quick")
    except Exception as error:
        decision, delta, warnings = "INVALID", {}, [str(error)]
        baseline_metrics, candidate_metrics, baseline_runs, candidate_runs, cache_hit = {}, {}, [], [], False
    if decision != "KEEP":
        return_to_parent(root, start_head)
    record = {**base, "git_commit": candidate_commit, "baseline_metrics": baseline_metrics,
              "candidate_metrics": candidate_metrics, "baseline_runs": baseline_runs,
              "candidate_runs": candidate_runs, "delta": delta, "decision": decision,
              "baseline_cache_hit": cache_hit, "notes": "; ".join(warnings),
              "reproduce": f"python research.py reproduce {experiment_id}"}
    _record(root, record)
    return record


def confirm(root: Path, experiment_id: str) -> dict:
    original = next((item for item in records(root) if item["experiment_id"] == experiment_id), None)
    if not original or original["decision"] != "KEEP":
        raise ValueError("confirmation requires a quick-stage KEEP experiment")
    profile = load_config(root)["profiles"][original["parameters"]["profile"]]
    seeds = profile["confirmation_seeds"]
    baseline_metrics, baseline_runs, cache_hit = _baseline(root, original["parent_commit"], profile, seeds)
    with revision_tree(root, original["git_commit"]) as tree:
        candidate_metrics, candidate_runs = evaluate(tree, profile["command"], seeds, profile["budget"])
    decision, delta, warnings = decide(baseline_metrics, candidate_metrics, profile["primary_metric"], profile["direction"], "confirmation")
    record = {**original, "experiment_id": next_id(root), "confirmation_of": experiment_id,
              "stage": "confirmation", "seeds": seeds, "baseline_metrics": baseline_metrics,
              "candidate_metrics": candidate_metrics, "baseline_runs": baseline_runs,
              "candidate_runs": candidate_runs, "delta": delta, "decision": decision,
              "confirmed": decision == "KEEP", "baseline_cache_hit": cache_hit,
              "timestamp": datetime.now(timezone.utc).isoformat(), "notes": "; ".join(warnings)}
    if decision == "KEEP":
        promote(root, original["parameters"]["profile"], record["experiment_id"], original["git_commit"])
        adopt(root, record["experiment_id"], original["git_commit"], original["changed_files"])
    _record(root, record)
    return record


def audit(root: Path, experiment_id: str) -> dict:
    original = next((item for item in records(root) if item["experiment_id"] == experiment_id), None)
    if not original or original["decision"] != "INCONCLUSIVE" or "anomalous jump" not in original.get("notes", ""):
        raise ValueError("audit requires an anomalous INCONCLUSIVE experiment")
    profile_name = original["parameters"]["profile"]
    profile = load_config(root)["profiles"][profile_name]
    seeds = profile["confirmation_seeds"]
    baseline_metrics, baseline_runs, cache_hit = _baseline(root, original["parent_commit"], profile, seeds)
    with revision_tree(root, original["git_commit"]) as tree:
        candidate_metrics, candidate_runs = evaluate(tree, profile["command"], seeds, profile["budget"])
    metric = profile["primary_metric"]
    direction = profile["direction"]
    per_seed_improvements = []
    for baseline_run, candidate_run in zip(baseline_runs, candidate_runs):
        baseline_value = baseline_run["metrics"][metric]
        candidate_value = candidate_run["metrics"][metric]
        per_seed_improvements.append(candidate_value - baseline_value if direction == "max" else baseline_value - candidate_value)
    aggregate = candidate_metrics[metric] - baseline_metrics[metric] if direction == "max" else baseline_metrics[metric] - candidate_metrics[metric]
    integrity_ok = candidate_metrics.get("illegal_move_rate", 0.0) == 0.0 and candidate_metrics.get("crash_rate", 0.0) == 0.0
    consistent = all(value >= 0.01 for value in per_seed_improvements)
    decision = "KEEP" if aggregate >= 0.01 and consistent and integrity_ok else "REJECT"
    relative = aggregate / max(abs(baseline_metrics[metric]), 1e-12)
    notes = "audit passed: consistent multi-seed gain with zero illegal moves and crashes" if decision == "KEEP" else "audit failed consistency or integrity checks"
    record = {**original, "experiment_id": next_id(root), "confirmation_of": experiment_id,
              "stage": "audit_confirmation", "seeds": seeds,
              "baseline_metrics": baseline_metrics, "candidate_metrics": candidate_metrics,
              "baseline_runs": baseline_runs, "candidate_runs": candidate_runs,
              "delta": {"primary_improvement": aggregate, "relative_improvement": relative,
                        "per_seed_improvements": per_seed_improvements},
              "decision": decision, "confirmed": decision == "KEEP",
              "baseline_cache_hit": cache_hit, "timestamp": datetime.now(timezone.utc).isoformat(),
              "notes": notes}
    if decision == "KEEP":
        promote(root, profile_name, record["experiment_id"], original["git_commit"])
        adopt(root, record["experiment_id"], original["git_commit"], original["changed_files"])
    _record(root, record)
    return record


ALGORITHM_MATRIX = ("reinforce", "grpo", "klpo", "flashreinforce")
# Wider than a confirmation run: separating algorithms needs more seeds than
# separating a candidate from its own parent.
MATRIX_SEEDS = (101, 211, 307, 401, 509, 601, 701, 809)


def algorithm_matrix(
    root: Path,
    profile_name: str,
    algorithms: tuple[str, ...] = ALGORITHM_MATRIX,
    seeds: tuple[int, ...] = MATRIX_SEEDS,
) -> dict:
    """Measure every algorithm on one profile under an identical budget and seed set.

    This establishes reference numbers; it never promotes a champion, because the
    champion is defined by the committed mutable code rather than by an override.
    """
    profile = load_config(root)["profiles"][profile_name]
    metric = profile["primary_metric"]
    results = []
    for name in algorithms:
        metrics, runs = evaluate(
            root, profile["command"], list(seeds), profile["budget"], {"NRL_ALGORITHM": name}
        )
        results.append({"algorithm": name, "metrics": metrics, "runs": runs})
    results.sort(key=lambda row: row["metrics"][metric], reverse=profile["direction"] == "max")
    payload = {
        "profile": profile_name,
        "profile_version": profile.get("version", "1"),
        "primary_metric": metric,
        "direction": profile["direction"],
        "budget": profile["budget"],
        "seeds": list(seeds),
        "protocol_version": PROTOCOL_VERSION,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    (root / "research" / "algorithms.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def reproduce(root: Path, experiment_id: str) -> dict:
    record = next((item for item in records(root) if item["experiment_id"] == experiment_id), None)
    if not record or not record.get("git_commit"):
        raise ValueError("unknown or non-reproducible experiment")
    profile = load_config(root)["profiles"][record["parameters"]["profile"]]
    with revision_tree(root, record["git_commit"]) as tree:
        metrics, runs = evaluate(tree, profile["command"], record["seeds"], record["budget"])
    return {"experiment_id": experiment_id, "recorded": record["candidate_metrics"], "reproduced": metrics, "runs": runs}
