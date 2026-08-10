from __future__ import annotations

import argparse
import importlib
import json
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast


def export_command(
    checkpoint: Path, target: Path, *, trust_remote_code: bool = False
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint),
        "--target_dir",
        str(target),
        "--use_cpu_initialization",
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    return command


def patch_lora_alpha(adapter_dir: Path, lora_alpha: int) -> dict[str, Any]:
    """Repair the zero alpha emitted by the verl v0.5 FSDP model merger."""
    if lora_alpha < 1:
        raise ValueError("LoRA alpha must be positive")
    config_path = adapter_dir / "adapter_config.json"
    raw_config = json.loads(config_path.read_text())
    if not isinstance(raw_config, dict):
        raise ValueError(f"invalid LoRA config in {config_path}")
    config = cast(dict[str, Any], raw_config)
    if int(config.get("r", 0)) < 1:
        raise ValueError(f"invalid LoRA rank in {config_path}")
    config["lora_alpha"] = lora_alpha
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    return config


def merge_sft_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    lora_alpha: int = 64,
    trust_remote_code: bool = False,
) -> Path:
    """Export a verl v0.5 FSDP SFT checkpoint and fold its LoRA into the base model."""
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"SFT checkpoint directory does not exist: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".graphtask-sft-merge-", dir=output.parent) as raw:
        work_dir = Path(raw)
        exported_dir = work_dir / "exported"
        merged_dir = work_dir / "merged"
        subprocess.run(
            export_command(
                checkpoint, exported_dir, trust_remote_code=trust_remote_code
            ),
            check=True,
        )
        adapter_dir = exported_dir / "lora_adapter"
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(
                f"verl model merger did not export a LoRA adapter under {adapter_dir}"
            )
        patch_lora_alpha(adapter_dir, lora_alpha)

        torch = importlib.import_module("torch")
        peft = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")
        load_options = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": trust_remote_code,
        }
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            exported_dir, **load_options
        )
        peft_model = peft.PeftModel.from_pretrained(
            base_model, adapter_dir, is_trainable=False
        )
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(
            merged_dir, safe_serialization=True, max_shard_size="5GB"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            exported_dir, trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(merged_dir)
        merged_dir.rename(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a verl v0.5 FSDP SFT LoRA checkpoint into a Hugging Face model"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.lora_alpha < 1:
        raise ValueError("--lora-alpha must be positive")
    if args.dry_run:
        target = args.output.resolve().parent / "<temporary-export>"
        print(
            shlex.join(
                export_command(
                    args.checkpoint.resolve(),
                    target,
                    trust_remote_code=args.trust_remote_code,
                )
            )
        )
        return 0
    output = merge_sft_checkpoint(
        args.checkpoint,
        args.output,
        lora_alpha=args.lora_alpha,
        trust_remote_code=args.trust_remote_code,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
