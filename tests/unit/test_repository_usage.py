from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

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

    externally_managed = ("torch", "sglang", "ray", "flash-attn")
    for package in externally_managed:
        assert not any(line.startswith(package) for line in requirements)


def test_only_ms_swift_training_scripts_remain() -> None:
    scripts = {path.name for path in (PROJECT_ROOT / "scripts").glob("train_*.sh")}
    assert scripts == {"train_ms_swift_sft.sh", "train_ms_swift_grpo.sh"}


def test_gpu_profiles_keep_kqapro_and_kilt_graphscript_versions_separate() -> None:
    for path in sorted((PROJECT_ROOT / "configs/experiments").glob("qwen*.yaml")):
        config = yaml.safe_load(path.read_text())
        assert config["training_backend"] == "ms_swift", path
        assert config["interaction_mode"] == "graphscript", path
        expected_version = "0.2" if "kilt" in path.name else "0.3"
        assert config["graphscript_version"] == expected_version, path


def test_ms_swift_sft_reads_existing_parquet_through_runtime_plugin() -> None:
    script = (PROJECT_ROOT / "scripts/train_ms_swift_sft.sh").read_text()
    assert "GRAPHTASK_MS_SWIFT_TRAIN_DATA" in script
    assert "graphtask_r1/training/ms_swift_plugin.py" in script
    assert ': "${MODEL_TYPE:=qwen3}"' in script
    assert '--model_type "$MODEL_TYPE"' in script
    assert 'MAX_LENGTH="${MAX_LENGTH:-32768}"' in script
    assert "MAX_LENGTH > 40960" in script
    assert "data export-sft" not in script
    assert "data prepare" not in script


def test_ms_swift_grpo_keeps_rollout_and_trainer_gpus_separate() -> None:
    rollout = (PROJECT_ROOT / "scripts/rollout_ms_swift.sh").read_text()
    trainer = (PROJECT_ROOT / "scripts/train_ms_swift_grpo.sh").read_text()
    assert "ROLLOUT_CUDA_VISIBLE_DEVICES:-0" in rollout
    assert "TRAIN_CUDA_VISIBLE_DEVICES:-1,2,3" in trainer
    assert '--model_type "$MODEL_TYPE"' in rollout
    assert '--model_type "$MODEL_TYPE"' in trainer
    assert 'INTERACTION_MODE="${INTERACTION_MODE:-graphscript}"' in trainer
    assert "export INTERACTION_MODE" in trainer
    assert "multi_turn_scheduler graphtask_solver" in trainer
    assert "multi_turn_scheduler graphtask_solver" in rollout


def test_main_readme_links_dedicated_ms_swift_cuda124_guide() -> None:
    main_readme = (PROJECT_ROOT / "README.md").read_text()
    guide = PROJECT_ROOT / "docs/MS_SWIFT_CUDA_12_4.md"

    assert "docs/MS_SWIFT_CUDA_12_4.md" in main_readme
    assert guide.is_file()
    assert "qwen3_4b_sft_ms_swift_cuda124.yaml" in guide.read_text()


def test_documented_mainline_runs_selfplay_directly_from_sft() -> None:
    main_readme = (PROJECT_ROOT / "README.md").read_text()
    training_guide = (PROJECT_ROOT / "docs/TRAINING.md").read_text()
    kqapro_guide = (PROJECT_ROOT / "docs/KQAPRO_TRAINING.md").read_text()

    assert "SFT → self-play → val 选模" in main_readme
    assert "Solver-only GRPO 不是前置依赖" in main_readme
    assert (
        "export INITIAL_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last"
        in training_guide
    )
    assert "## 4. 可选：KQAPro Solver-only GRPO warm-up" in training_guide
    assert "## 5. 可选：Solver-only GRPO warm-up" in kqapro_guide
    assert "默认直接用 SFT adapter 初始化 self-play" in kqapro_guide


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
