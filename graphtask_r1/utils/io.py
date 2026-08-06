from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    rows = []
    for record in records:
        row: dict[str, Any] = {
            "record_json": json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
        }
        for key in ("task_id", "index", "round", "role", "reason_code", "program_signature"):
            value = record.get(key)
            if value is not None and isinstance(value, str | int | float | bool):
                row[key] = value
        rows.append(row)
    if not rows:
        rows = [{"record_json": "{}"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(value) for value in pq.read_table(path)["record_json"].to_pylist()]
