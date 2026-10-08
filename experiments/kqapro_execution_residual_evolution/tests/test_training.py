from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EXPERIMENT_ROOT / "scripts" / "train_dpo.py"


def _training_module():
    spec = importlib.util.spec_from_file_location("ere_train_dpo", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(path: Path) -> None:
    path.write_text(
        """
repo_root: .
seed: 17
paths:
  model: model
  adapter: adapter
  dataset: pairs/train.jsonl
  output_dir: run
runtime:
  swift: swift
  cuda_visible_devices: "3"
  nproc_per_node: 1
  dataset_num_proc: 1
  dataloader_num_workers: 1
model:
  model_type: qwen3
  template: qwen3
  response_prefix: "<think>\\n\\n</think>\\n\\n"
  torch_dtype: bfloat16
  attn_impl: sdpa
  max_length: 4096
training:
  loss_type: sigmoid
  beta: 0.1
  epochs: 1
  micro_batch_size: 1
  gradient_accumulation_steps: 2
  learning_rate: 5.0e-7
  weight_decay: 0.0
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: all-linear
  save_steps: 4
  save_total_limit: 1
  logging_steps: 1
  warmup_ratio: 0.05
  extra_args: [--save_only_model, "true"]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_build_invocation_is_dpo_and_seeded(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    invocation = _training_module().build_invocation(config)

    command = list(invocation.command)
    assert command[:4] == ["swift", "rlhf", "--rlhf_type", "dpo"]
    assert command[command.index("--dataset") + 1] == str(
        (tmp_path / "pairs/train.jsonl").resolve()
    )
    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--data_seed") + 1] == "17"
    assert command[command.index("--response_prefix") + 1] == "<think>\n\n</think>\n\n"
    assert command[-2:] == ["--save_only_model", "true"]
    assert invocation.environment == {
        "CUDA_VISIBLE_DEVICES": "3",
        "NPROC_PER_NODE": "1",
        "PYTHONHASHSEED": "17",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_merged_incumbent_does_not_load_a_second_initial_adapter(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    config.write_text(
        config.read_text(encoding="utf-8").replace("  adapter: adapter\n", ""),
        encoding="utf-8",
    )

    command = list(_training_module().build_invocation(config).command)

    assert "--adapters" not in command
    assert command[command.index("--model") + 1] == str((tmp_path / "model").resolve())


def test_dry_run_emits_replayable_invocation(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["cwd"] == str(tmp_path.resolve())
    assert payload["environment"]["PYTHONHASHSEED"] == "17"
    assert payload["command"][0:2] == ["swift", "rlhf"]


def test_scale444_config_is_an_isolated_one_epoch_scale_diagnostic() -> None:
    module = _training_module()
    mini = module.build_invocation(EXPERIMENT_ROOT / "configs" / "mini.yaml")
    scale = module.build_invocation(EXPERIMENT_ROOT / "configs" / "scale444.yaml")

    def flags(command: tuple[str, ...]) -> dict[str, str]:
        return dict(zip(command[2::2], command[3::2], strict=True))

    mini_flags = flags(mini.command)
    scale_flags = flags(scale.command)
    assert scale_flags.pop("--dataset").endswith("/pairs-scale444/train.jsonl")
    assert scale_flags.pop("--output_dir").endswith("/dpo-scale444")
    assert scale_flags.pop("--save_steps") == "111"
    assert scale_flags.pop("--save_total_limit") == "1"
    mini_flags.pop("--dataset")
    mini_flags.pop("--output_dir")
    mini_flags.pop("--save_steps")
    mini_flags.pop("--save_total_limit")

    assert scale.command[:2] == mini.command[:2]
    assert scale_flags == mini_flags
    assert scale.cwd == mini.cwd
    assert scale.environment == mini.environment
