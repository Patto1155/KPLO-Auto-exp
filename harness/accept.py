import json
from pathlib import Path

from harness.git_state import git, is_ancestor


def preserve_candidate(root: Path, experiment_id: str, paths: list[str]) -> str:
    git(root, "add", "--", *paths)
    git(root, "commit", "-m", f"experiment({experiment_id}): candidate")
    commit = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", f"refs/nrl/experiments/{experiment_id}", commit)
    return commit


def return_to_parent(root: Path, parent: str) -> None:
    git(root, "reset", "--hard", parent)


def adopt(root: Path, experiment_id: str, commit: str, paths: list[str]) -> bool:
    """Put a promoted champion's code back on the active branch.

    A candidate that was rolled back before promotion — an anomalous jump that only
    an audit confirms — lives solely under `refs/nrl/experiments/`. Without this the
    branch would keep building on a baseline worse than its own recorded champion.
    """
    if not paths or is_ancestor(root, commit):
        return False
    git(root, "checkout", commit, "--", *paths)
    git(root, "add", "--", *paths)
    if not git(root, "diff", "--cached", "--name-only", "--", *paths):
        return False
    git(root, "commit", "-m", f"champion({experiment_id}): adopt confirmed candidate")
    return True


def promote(root: Path, profile: str, experiment_id: str, commit: str) -> None:
    path = root / "research" / "current_best.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    if "git_commit" in existing:  # migrate the original single-profile format
        existing = {"profiles": {"smoke": existing}}
    profiles = existing.setdefault("profiles", {})
    profiles[profile] = {"experiment_id": experiment_id, "git_commit": commit, "confirmed": True}
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
