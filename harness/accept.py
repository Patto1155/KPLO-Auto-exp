import json
from pathlib import Path

from harness.git_state import git


def preserve_candidate(root: Path, experiment_id: str, paths: list[str]) -> str:
    git(root, "add", "--", *paths)
    git(root, "commit", "-m", f"experiment({experiment_id}): candidate")
    commit = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", f"refs/nrl/experiments/{experiment_id}", commit)
    return commit


def return_to_parent(root: Path, parent: str) -> None:
    git(root, "reset", "--hard", parent)


def promote(root: Path, profile: str, experiment_id: str, commit: str) -> None:
    path = root / "research" / "current_best.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    if "git_commit" in existing:  # migrate the original single-profile format
        existing = {"profiles": {"smoke": existing}}
    profiles = existing.setdefault("profiles", {})
    profiles[profile] = {"experiment_id": experiment_id, "git_commit": commit, "confirmed": True}
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
