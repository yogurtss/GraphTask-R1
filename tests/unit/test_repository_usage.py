from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cli_runs_from_repository_without_installing_package() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "graphtask_r1.cli", "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "graphtask-r1" in result.stdout


def test_runtime_requirements_do_not_manage_gpu_stack() -> None:
    requirements = {
        line.split("#", maxsplit=1)[0].strip().lower()
        for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
    }

    externally_managed = ("torch", "verl", "sglang", "ray", "flash-attn")
    for package in externally_managed:
        assert not any(line.startswith(package) for line in requirements)


def test_cli_logs_to_stderr_and_keeps_json_on_stdout() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "graphtask_r1.cli",
            "--log-level",
            "INFO",
            "graph",
            "preflight",
            "--snapshot",
            "toy-v1",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["snapshot"] == "toy-v1"
    assert "command_started group=graph action=preflight" in result.stderr
    assert "command_completed group=graph action=preflight" in result.stderr
