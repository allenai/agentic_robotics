"""Utility functions that support validating and recording the current runtime environment."""

import hashlib
import subprocess
from collections.abc import Iterable
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROCESS_STARTED_AT = datetime.fromtimestamp(Path("/proc/self").stat().st_ctime).astimezone()
"""Timestamp of when the currently running process was started."""


def _git(repo: Path, *args: str) -> str | None:
    """Run a Git command in the given repository.

    :param repo: Working directory used for the command
    :param args: Arguments following `git`
    :return: Standard output, or None upon failure
    """
    try:
        completed = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return completed.stdout if (completed.returncode == 0) else None


def _repo_state(name: str, repo: Path, out_dir: Path) -> dict[str, Any]:
    """Capture the state of the given repository well enough to reconstruct the running code.

    :param name: Short name used to identify the repository
    :param repo: Path to the repository's root
    :param out_dir: Output directory, where any large diff file is written
    :return: A JSON-serializable description of the repository's state
    """
    diff = _git(repo, "diff", "HEAD")
    status = _git(repo, "status", "--porcelain")
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")

    state = {
        "path": str(repo),
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "status": [line for line in (status or "").splitlines() if line.strip()],
        "diff_file": None,
        "diff_sha256": None,
    }

    if diff and diff.strip():
        diff_path = out_dir / f"{name}.diff"
        diff_path.write_text(diff)
        state["diff_file"] = diff_path.name
        state["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()

    return state


def _package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    """Read the installed version of each of the given packages."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    return versions
