from __future__ import annotations

import hashlib
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

from benchmark.protocol import MUTABLE_PATHS, PROTECTED_PATHS


def git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip())
    # Preserve porcelain status' leading columns; only remove record terminators.
    return result.stdout.rstrip("\r\n")


def head(root: Path) -> str:
    return git(root, "rev-parse", "HEAD")


def changed_files(root: Path) -> list[str]:
    output = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return sorted(line[3:] for line in output.splitlines() if line)


def is_under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def validate_candidate_paths(paths: list[str]) -> tuple[bool, str]:
    protected = [path for path in paths if is_under(path, PROTECTED_PATHS)]
    unknown = [path for path in paths if not is_under(path, MUTABLE_PATHS) and not is_under(path, PROTECTED_PATHS)]
    if protected:
        return False, "protected files changed: " + ", ".join(protected)
    if unknown:
        return False, "files outside mutable surface changed: " + ", ".join(unknown)
    if not paths:
        return False, "no candidate modification found"
    return True, "candidate touches mutable files only"


def fingerprint(root: Path, revision: str, profile: dict, seeds: list[int]) -> str:
    payload = "\n".join([
        revision,
        repr(profile),
        repr(seeds),
        git(root, "show", f"{revision}:lab.json"),
        git(root, "show", f"{revision}:benchmark/protocol.py"),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()


def patch(root: Path, paths: list[str]) -> str:
    tracked = git(root, "diff", "--binary", "--", *paths)
    untracked = [path for path in paths if git(root, "ls-files", "--error-unmatch", path, check=False) == ""]
    extra = []
    for path in untracked:
        extra.append(git(root, "diff", "--binary", "--no-index", "/dev/null", path, check=False))
    return tracked + "\n".join(extra)


@contextmanager
def revision_tree(root: Path, revision: str):
    with tempfile.TemporaryDirectory(prefix="nrl-baseline-") as directory:
        archive = Path(directory) / "tree.tar"
        with archive.open("wb") as stream:
            process = subprocess.run(["git", "archive", revision], cwd=root, stdout=stream)
        if process.returncode:
            raise RuntimeError("could not archive baseline revision")
        target = Path(directory) / "tree"
        target.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(target, filter="data")
        yield target
