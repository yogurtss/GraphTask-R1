#!/usr/bin/env python3
"""Launch the isolated ERE preference update from one explicit YAML config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingInvocation:
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "cwd": str(self.cwd),
            "environment": dict(sorted(self.environment.items())),
        }


def _resolve_path(root: Path, value: object) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _flag(command: list[str], name: str, value: object) -> None:
    command.extend((name, str(value).lower() if isinstance(value, bool) else str(value)))


def build_invocation(config_path: Path) -> TrainingInvocation:
    config_path = config_path.resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root_value = raw.get("repo_root", "../../..")
    repo_root = _resolve_path(config_path.parent, root_value)
    paths = raw["paths"]
    runtime = raw["runtime"]
    model = raw["model"]
    training = raw["training"]
    seed = int(raw["seed"])

    swift_value = str(runtime.get("swift", "swift"))
    swift = str(_resolve_path(repo_root, swift_value)) if "/" in swift_value else swift_value
    command = [swift, "rlhf"]
    fixed_args: list[tuple[str, object]] = [
        ("--rlhf_type", "dpo"),
        ("--model", _resolve_path(repo_root, paths["model"])),
        ("--model_type", model["model_type"]),
        ("--template", model["template"]),
        ("--response_prefix", model["response_prefix"]),
        ("--train_type", "lora"),
        ("--dataset", _resolve_path(repo_root, paths["dataset"])),
        ("--torch_dtype", model["torch_dtype"]),
        ("--attn_impl", model["attn_impl"]),
        ("--max_length", model["max_length"]),
        ("--loss_type", training["loss_type"]),
        ("--beta", training["beta"]),
        ("--num_train_epochs", training["epochs"]),
        ("--per_device_train_batch_size", training["micro_batch_size"]),
        ("--gradient_accumulation_steps", training["gradient_accumulation_steps"]),
        ("--learning_rate", training["learning_rate"]),
        ("--weight_decay", training["weight_decay"]),
        ("--lora_rank", training["lora_rank"]),
        ("--lora_alpha", training["lora_alpha"]),
        ("--lora_dropout", training["lora_dropout"]),
        ("--target_modules", training["target_modules"]),
        ("--gradient_checkpointing", True),
        ("--gradient_checkpointing_kwargs", '{"use_reentrant":false}'),
        ("--dataset_num_proc", runtime["dataset_num_proc"]),
        ("--dataloader_num_workers", runtime["dataloader_num_workers"]),
        ("--load_from_cache_file", False),
        ("--split_dataset_ratio", 0),
        ("--eval_strategy", "no"),
        ("--save_strategy", "steps"),
        ("--save_steps", training["save_steps"]),
        ("--save_total_limit", training["save_total_limit"]),
        ("--logging_steps", training["logging_steps"]),
        ("--warmup_ratio", training["warmup_ratio"]),
        ("--report_to", "none"),
        ("--seed", seed),
        ("--data_seed", seed),
        ("--output_dir", _resolve_path(repo_root, paths["output_dir"])),
    ]
    adapter = paths.get("adapter")
    if adapter is not None:
        fixed_args.insert(5, ("--adapters", _resolve_path(repo_root, adapter)))
    for name, value in fixed_args:
        _flag(command, name, value)
    command.extend(str(value) for value in training.get("extra_args", []))

    environment = {
        "CUDA_VISIBLE_DEVICES": str(runtime["cuda_visible_devices"]),
        "NPROC_PER_NODE": str(runtime["nproc_per_node"]),
        "PYTHONHASHSEED": str(seed),
        "TOKENIZERS_PARALLELISM": "false",
    }
    return TrainingInvocation(tuple(command), repo_root, environment)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated KQAPro ERE DPO update")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    invocation = build_invocation(args.config)
    print(json.dumps(invocation.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    environment = os.environ.copy()
    environment.update(invocation.environment)
    subprocess.run(
        invocation.command,
        cwd=invocation.cwd,
        env=environment,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
