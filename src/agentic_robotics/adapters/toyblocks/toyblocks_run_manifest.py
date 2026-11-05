"""Define utilities to log a manifest of the state of the development environment per run."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_robotics.core.runtime_env_checks import (
    PROCESS_STARTED_AT,
    _package_versions,
    _repo_state,
)

if TYPE_CHECKING:
    from agentic_robotics.adapters.toyblocks.config import ToyBlocksConfig
    from agentic_robotics.adapters.toyblocks.toyblocks_mcp_server import ToyBlocksMCPServer

MANIFEST_SCHEMA_VERSION = 3.1
"""Used to indicate changes to the manifest schema, so that old manifests remain readable."""

MANIFEST_PACKAGES = ("torch", "numpy", "isaaclab", "mcp", "openai")
"""Packages whose version could plausibly affect a run's behavior. Absent ones record as null."""


def _repo_root(package: str, parents_up: int) -> Path | None:
    """Attempt to find where a package imports from, or return None if it's not installed."""
    spec = find_spec(package)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parents[parents_up]


_THIS_FILE = Path(__file__).resolve()
AGENTIC_ROBOTICS_ROOT = _THIS_FILE.parents[4]

MANIFEST_REPOS = {
    "toyblocks": _repo_root("toyblocks", 2),
    "agentic_robotics": AGENTIC_ROBOTICS_ROOT,
    "curobo": _repo_root("curobo", 1),  # cuRobo v2: <repo>/curobo/__init__.py
}
"""Repositories whose state is recorded in the manifest for a run."""

assert all(root.exists() for root in MANIFEST_REPOS.values() if root is not None)


def write_run_manifest(mcp_server: ToyBlocksMCPServer, config: ToyBlocksConfig) -> Path:
    """Record a manifest (`run.json`) of the development environment state that produced a run.

    :param mcp_server: Server defining the MCP tools available for the run
    :param config: Configuration parameters for the run
    :return: Path to the written manifest file
    """
    if config.output_dir is None:
        raise RuntimeError("Cannot record run manifest; config must specify an output path.")

    camera = mcp_server._toyblocks_env.read_camera_model(config.camera_name)

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "process_started_at": PROCESS_STARTED_AT.isoformat(),
        "manifest_written_at": datetime.now().astimezone().isoformat(),
        "command": sys.argv,
        "config": asdict(config),
        "tools": [t.model_dump() for t in mcp_server.mcp._tool_manager.list_tools()],
        "observation": {
            "camera_name": config.camera_name,
            "camera_resolution": asdict(mcp_server.camera_resolution),
            "observation_resolution": asdict(mcp_server.observation_resolution),
            "camera_model": camera.asdict(),
        },
        "repos": {
            name: _repo_state(name, repo, config.output_dir) if repo is not None else None
            for name, repo in MANIFEST_REPOS.items()
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "packages": _package_versions(packages=MANIFEST_PACKAGES),
        },
    }

    manifest_path = config.output_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path
