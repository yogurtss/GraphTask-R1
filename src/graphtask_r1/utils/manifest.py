from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphtask_r1 import __version__
from graphtask_r1.utils.hashing import file_hash, stable_hash
from graphtask_r1.utils.io import write_json


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_manifest(
    output_dir: Path, config: dict[str, Any], artifacts: list[str]
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "lock_hash": file_hash(Path("uv.lock")),
        "config": config,
        "config_hash": stable_hash(config),
        "artifacts": sorted(artifacts),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
