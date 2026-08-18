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
    assert "--vllm_gpu_memory_utilization" in trainer
    assert "--vllm_max_model_len" in trainer
    assert "--sleep_level" in trainer
    assert "multi_turn_scheduler graphtask_solver" in trainer
    assert "multi_turn_scheduler graphtask_solver" in rollout
    assert "import math_verify" in trainer


def test_ms_swift_grpo_only_passes_server_address_in_server_mode(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_swift = fake_bin / "swift"
    fake_swift.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    fake_swift.chmod(0o755)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    train_data = tmp_path / "train.parquet"
    val_data = tmp_path / "val.parquet"
    train_data.touch()
    val_data.touch()
    script = PROJECT_ROOT / "scripts/train_ms_swift_grpo.sh"

    def launch(mode: str, *, deepspeed: str = "none") -> list[str]:
        capture = tmp_path / f"{mode}-{deepspeed}.args"
        environment = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE_ARGS": str(capture),
            "LORA_ADAPTER_PATH": str(adapter),
            "TRAIN_DATA": str(train_data),
            "VAL_DATA": str(val_data),
            "OUTPUT_DIR": str(tmp_path / f"output-{mode}"),
            "VLLM_MODE": mode,
            "DEEPSPEED": deepspeed,
        }
        subprocess.run(["bash", str(script)], cwd=PROJECT_ROOT, env=environment, check=True)
        return capture.read_text().splitlines()

    colocate = launch("colocate")
    assert "--vllm_gpu_memory_utilization" in colocate
    assert "--vllm_server_host" not in colocate
    assert "--vllm_server_port" not in colocate
    assert "--deepspeed" not in colocate

    zero3 = launch("colocate", deepspeed="zero3")
    assert zero3[zero3.index("--deepspeed") + 1] == "zero3"

    server = launch("server")
    assert server[server.index("--vllm_server_host") + 1] == "127.0.0.1"
    assert server[server.index("--vllm_server_port") + 1] == "8000"
    assert "--vllm_gpu_memory_utilization" not in server

    no_vllm_capture = tmp_path / "no-vllm.args"
    no_vllm_environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE_ARGS": str(no_vllm_capture),
        "LORA_ADAPTER_PATH": str(adapter),
        "TRAIN_DATA": str(train_data),
        "VAL_DATA": str(val_data),
        "OUTPUT_DIR": str(tmp_path / "output-no-vllm"),
        "USE_VLLM": "false",
    }
    subprocess.run(
        ["bash", str(script)], cwd=PROJECT_ROOT, env=no_vllm_environment, check=True
    )
    no_vllm = no_vllm_capture.read_text().splitlines()
    assert no_vllm[no_vllm.index("--use_vllm") + 1] == "false"
    assert "--vllm_mode" not in no_vllm
    assert "--vllm_server_host" not in no_vllm
    assert "--vllm_gpu_memory_utilization" not in no_vllm


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
    assert "## 5. Questioner/Solver self-play" in kqapro_guide
    assert "## 附录 A：可选的 Solver-only GRPO warm-up" in kqapro_guide
    assert kqapro_guide.index("## 5. Questioner/Solver self-play") < kqapro_guide.index(
        "## 附录 A：可选的 Solver-only GRPO warm-up"
    )
    assert "默认直接用 mixed-role SFT adapter 初始化 self-play" in kqapro_guide
    assert "scripts/prepare_mixed_sft_data.sh" in main_readme
    assert "scripts/prepare_mixed_sft_data.sh" in training_guide
    assert "scripts/prepare_mixed_sft_data.sh" in kqapro_guide
    assert "SOLVER_RATIO=1" in kqapro_guide
    assert "QUESTIONER_RATIO=1" in kqapro_guide
    assert "QUESTIONER_COUNT_OVERRIDE=2048" in kqapro_guide


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
