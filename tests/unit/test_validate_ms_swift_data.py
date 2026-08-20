from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.validate_ms_swift_data import validate_path


def test_validate_ms_swift_sft_accepts_message_rows(tmp_path: Path) -> None:
    path = tmp_path / "accepted.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Answer"},
                    ],
                    "role": "solver",
                    "task_id": "task-1",
                    "interaction_mode": "graphscript",
                }
            ]
        ),
        path,
    )

    assert validate_path(path, kind="sft") == {
        "path": str(path),
        "kind": "sft",
        "rows": 1,
        "message_column": "messages",
    }


def test_validate_ms_swift_sft_rejects_certified_task_parquet_with_path_hint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"record_json": "{}"}]), path)

    with pytest.raises(ValueError, match="preflight SFT accepted Parquet"):
        validate_path(path, kind="sft")


def test_validate_ms_swift_sft_reports_serialized_message_type(tmp_path: Path) -> None:
    path = tmp_path / "serialized.parquet"
    pq.write_table(pa.Table.from_pylist([{"messages": "[]"}]), path)

    with pytest.raises(ValueError, match="messages must be a list, got str"):
        validate_path(path, kind="sft")
