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


def test_cuda124_sft_uses_json_compatible_dataset() -> None:
    script = (PROJECT_ROOT / "scripts/train_sft.sh").read_text()
    assert "cuda124)" in script
    assert "graphtask_r1/training/verl_sft_dataset.py" in script
    assert "data.custom_cls.name=GraphTaskMultiTurnSFTDataset" in script


def test_ms_swift_sft_reads_existing_parquet_through_runtime_plugin() -> None:
    script = (PROJECT_ROOT / "scripts/train_ms_swift_sft.sh").read_text()
    assert "GRAPHTASK_MS_SWIFT_TRAIN_DATA" in script
    assert "graphtask_r1/training/ms_swift_plugin.py" in script
    assert ': "${MODEL_TYPE:=qwen3}"' in script
    assert '--model_type "$MODEL_TYPE"' in script
    assert "data export-sft" not in script
    assert "data prepare" not in script


def test_ms_swift_grpo_keeps_rollout_and_trainer_gpus_separate() -> None:
    rollout = (PROJECT_ROOT / "scripts/rollout_ms_swift.sh").read_text()
    trainer = (PROJECT_ROOT / "scripts/train_ms_swift_grpo.sh").read_text()
    assert 'ROLLOUT_CUDA_VISIBLE_DEVICES:-0' in rollout
    assert 'TRAIN_CUDA_VISIBLE_DEVICES:-1,2,3' in trainer
    assert '--model_type "$MODEL_TYPE"' in rollout
    assert '--model_type "$MODEL_TYPE"' in trainer
    assert "multi_turn_scheduler graphtask_solver" in trainer
    assert "multi_turn_scheduler graphtask_solver" in rollout


def test_main_readme_links_dedicated_ms_swift_cuda124_guide() -> None:
    main_readme = (PROJECT_ROOT / "README.md").read_text()
    guide = PROJECT_ROOT / "README_MS_SWIFT_CUDA124.md"

    assert "README_MS_SWIFT_CUDA124.md" in main_readme
    assert guide.is_file()
    assert "qwen3_4b_sft_ms_swift_cuda124.yaml" in guide.read_text()


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
