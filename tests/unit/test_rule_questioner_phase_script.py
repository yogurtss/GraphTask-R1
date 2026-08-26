from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/run_rule_questioner_selfplay_phases.sh"


def test_rule_questioner_phase_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_rule_questioner_phase_script_runs_all_six_phases(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "commands.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$CAPTURE_ARGS"\n'
    )
    fake_python.chmod(0o755)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n")
    required_files = {
        "BASE_TASKS": tmp_path / "base.parquet",
        "QUESTIONER_SEEDS": tmp_path / "questioner-selfplay.parquet",
        "KQAPRO_RELATION_CATALOG": tmp_path / "relation_catalog.json",
        "GRAPHTASK_KQAPRO_DB": tmp_path / "graph.sqlite",
    }
    for path in required_files.values():
        path.touch()

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE_ARGS": str(capture),
        "INITIAL_ADAPTER": str(adapter),
        **{name: str(path) for name, path in required_files.items()},
    }
    output_dir = tmp_path / "selfplay"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(
                PROJECT_ROOT
                / "configs/training/selfplay_qwen3_4b_rule_questioner_large.yaml"
            ),
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = capture.read_text().splitlines()
    assert len(commands) == 6
    assert "--round-index 1 --phase questioner" in commands[0]
    assert "--round-index 1 --phase solver" in commands[1]
    assert "--round-index 3 --phase solver" in commands[-1]
    assert "all six commands completed successfully" in result.stdout


def test_rule_questioner_phase_script_rejects_missing_inputs(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "INITIAL_ADAPTER": str(tmp_path / "missing-adapter"),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "INITIAL_ADAPTER is not a complete adapter" in result.stderr
