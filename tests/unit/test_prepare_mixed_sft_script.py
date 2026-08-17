from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/prepare_mixed_sft_data.sh"


def test_prepare_mixed_sft_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_prepare_mixed_sft_script_computes_questioner_count_from_ratio(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    train.touch()
    val.touch()
    capture = tmp_path / "commands.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$CAPTURE_ARGS\"\n"
        "if [[ \"$1\" == '-c' ]]; then printf '90\\n'; fi\n"
    )
    fake_python.chmod(0o755)
    work_dir = tmp_path / "work"
    environment = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "TRAIN_TASKS": str(train),
        "VAL_TASKS": str(val),
        "WORK_DIR": str(work_dir),
        "SOLVER_RATIO": "9",
        "QUESTIONER_RATIO": "1",
        "PYTHON_BIN": str(fake_python),
        "CAPTURE_ARGS": str(capture),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = capture.read_text()
    assert "data export-questioner-sft" in commands
    assert "data export-questioner-seeds" in commands
    assert "--count 10" in commands
    assert "--allow-oversample" not in commands
    assert "--solver-weight 9 --questioner-weight 1" in commands
    assert "Solver=90 Questioner=10 Total=100" in result.stdout
    env_file = (work_dir / "sft_data.env").read_text()
    assert "SFT_TRAIN_DATA=" in env_file
    assert "SFT_VAL_DATA=" in env_file
    assert "QUESTIONER_SEEDS=" in env_file


def test_prepare_mixed_sft_script_exact_count_overrides_ratio(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    train.touch()
    val.touch()
    capture = tmp_path / "commands.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$CAPTURE_ARGS\"\n"
        "if [[ \"$1\" == '-c' ]]; then printf '90\\n'; fi\n"
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "TRAIN_TASKS": str(train),
        "VAL_TASKS": str(val),
        "WORK_DIR": str(tmp_path / "work"),
        "QUESTIONER_COUNT_OVERRIDE": "7",
        "PYTHON_BIN": str(fake_python),
        "CAPTURE_ARGS": str(capture),
    }

    subprocess.run(["bash", str(SCRIPT)], cwd=PROJECT_ROOT, env=environment, check=True)

    assert "--count 7" in capture.read_text()
