from __future__ import annotations

from pathlib import Path

from graphtask_r1.training.checkpointing import (
    is_resumable_checkpoint,
    latest_resumable_checkpoint,
)
from graphtask_r1.utils import write_json


def _write_checkpoint(path: Path, *, full_state: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    write_json(
        path / "trainer_state.json",
        {"global_step": int(path.name.removeprefix("checkpoint-")), "max_steps": 100},
    )
    if full_state:
        (path / "scheduler.pt").write_bytes(b"scheduler")
        (path / "optimizer.pt").write_bytes(b"optimizer")
    return path


def test_latest_resumable_checkpoint_prefers_latest_timestamp_then_step(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    _write_checkpoint(output / "v9-20260901-235959" / "checkpoint-99")
    _write_checkpoint(output / "v0-20260902-010000" / "checkpoint-10")
    expected = _write_checkpoint(
        output / "v0-20260902-010000" / "checkpoint-20"
    )

    assert latest_resumable_checkpoint(output) == expected.resolve()


def test_latest_resumable_checkpoint_ignores_partial_and_model_only_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    expected = _write_checkpoint(output / "v0-20260902-010000" / "checkpoint-20")
    model_only = _write_checkpoint(
        output / "v1-20260902-020000" / "checkpoint-30", full_state=False
    )
    partial = output / "v2-20260902-030000" / "checkpoint-40"
    partial.mkdir(parents=True)
    (partial / "trainer_state.json").write_text("{")

    assert not is_resumable_checkpoint(model_only)
    assert not is_resumable_checkpoint(partial)
    assert latest_resumable_checkpoint(output) == expected.resolve()
