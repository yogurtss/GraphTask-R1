from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_RUN_DIRECTORY = re.compile(r"v(?P<version>\d+)(?:-(?P<timestamp>\d{8}-\d{6}))?")
_CHECKPOINT_DIRECTORY = re.compile(r"checkpoint-(?P<step>\d+)")


def _checkpoint_order(checkpoint: Path, output_dir: Path) -> tuple[str, int, int, int, str]:
    """Order ms-swift checkpoints by run timestamp, version, step, and mtime."""

    run_timestamp = ""
    run_version = -1
    for parent in checkpoint.parents:
        if parent == output_dir.parent:
            break
        match = _RUN_DIRECTORY.fullmatch(parent.name)
        if match:
            run_timestamp = match.group("timestamp") or ""
            run_version = int(match.group("version"))
            break
        if parent == output_dir:
            break
    step_match = _CHECKPOINT_DIRECTORY.fullmatch(checkpoint.name)
    step = int(step_match.group("step")) if step_match else -1
    state = checkpoint / "trainer_state.json"
    return (
        run_timestamp,
        run_version,
        step,
        state.stat().st_mtime_ns,
        str(checkpoint),
    )


def _has_optimizer_state(checkpoint: Path) -> bool:
    if (checkpoint / "optimizer.pt").is_file():
        return True
    return any(
        path.is_dir()
        for pattern in ("global_step*", "zero_*", "mp_rank_*")
        for path in checkpoint.glob(pattern)
    )


def is_resumable_checkpoint(checkpoint: Path) -> bool:
    """Return whether a checkpoint contains model and full trainer state."""

    if _CHECKPOINT_DIRECTORY.fullmatch(checkpoint.name) is None:
        return False
    required = (
        checkpoint / "adapter_config.json",
        checkpoint / "trainer_state.json",
        checkpoint / "scheduler.pt",
    )
    if not all(path.is_file() for path in required):
        return False
    if not any(
        (checkpoint / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        return False
    if not _has_optimizer_state(checkpoint):
        return False
    try:
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
        global_step = int(trainer_state["global_step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    return global_step >= 0


def latest_resumable_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest complete ms-swift checkpoint below one training output."""

    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        return None
    candidates = {
        state.parent
        for state in output_dir.rglob("trainer_state.json")
        if is_resumable_checkpoint(state.parent)
    }
    if not candidates:
        return None
    return max(candidates, key=lambda path: _checkpoint_order(path, output_dir))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the latest fully resumable ms-swift checkpoint."
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    checkpoint = latest_resumable_checkpoint(args.output_dir)
    if checkpoint is not None:
        print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
