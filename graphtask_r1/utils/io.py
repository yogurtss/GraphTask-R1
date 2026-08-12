from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

RECORD_INDEX_FIELDS = (
    "task_id",
    "index",
    "round",
    "role",
    "reason_code",
    "program_signature",
)
RECORD_SCHEMA = pa.schema(
    [
        ("record_json", pa.string()),
        ("task_id", pa.string()),
        ("index", pa.int64()),
        ("round", pa.int64()),
        ("role", pa.string()),
        ("reason_code", pa.string()),
        ("program_signature", pa.string()),
    ]
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def iter_json_array(
    path: Path,
    *,
    limit: int | None = None,
    chunk_chars: int = 1024 * 1024,
) -> Iterator[dict[str, Any]]:
    """Incrementally decode a top-level JSON array without loading the source file."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be at least 1")
    decoder = json.JSONDecoder()

    def values() -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as stream:
            buffer = ""
            position = 0
            eof = False

            def refill() -> None:
                nonlocal buffer, position, eof
                buffer = buffer[position:]
                position = 0
                chunk = stream.read(chunk_chars)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            def skip_space() -> None:
                nonlocal position
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer) or eof:
                        return
                    refill()

            refill()
            skip_space()
            if position >= len(buffer) or buffer[position] != "[":
                raise ValueError(f"{path} must contain a top-level JSON array")
            position += 1
            first = True
            while True:
                skip_space()
                if position < len(buffer) and buffer[position] == "]":
                    position += 1
                    break
                if not first:
                    if position >= len(buffer):
                        refill()
                        skip_space()
                    if position >= len(buffer) or buffer[position] != ",":
                        raise ValueError(f"invalid JSON array separator in {path}")
                    position += 1
                    skip_space()
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                        position = end
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        refill()
                if not isinstance(value, dict):
                    raise ValueError(f"{path} array entries must be JSON objects")
                yield value
                first = False
            skip_space()
            if position < len(buffer) or not eof and stream.read(1):
                raise ValueError(f"unexpected trailing content in {path}")

    return islice(values(), limit) if limit is not None else values()


class ParquetRowWriter:
    """Write Python rows in bounded batches using one stable Arrow schema."""

    def __init__(
        self,
        path: Path,
        *,
        schema: pa.Schema | None = None,
        batch_size: int = 256,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self._rows: list[Mapping[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def __enter__(self) -> ParquetRowWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=self.schema)
        if self._writer is None:
            self.schema = table.schema
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)
        self.rows_written += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is None:
            if self.schema is None:
                raise ValueError("cannot write an empty parquet file without an explicit schema")
            self._writer = pq.ParquetWriter(self.path, self.schema)
        self._writer.close()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        elif self._writer is not None:
            self._writer.close()


class RecordWriter:
    """Stream JSON records to the repository's indexed Parquet contract."""

    def __init__(self, path: Path, *, batch_size: int = 128) -> None:
        self._writer = ParquetRowWriter(path, schema=RECORD_SCHEMA, batch_size=batch_size)

    def __enter__(self) -> RecordWriter:
        self._writer.__enter__()
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        row: dict[str, Any] = {
            "record_json": json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
        }
        for key in RECORD_INDEX_FIELDS:
            value = record.get(key)
            if value is not None and isinstance(value, str | int | bool):
                row[key] = value
        self._writer.write(row)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._writer.__exit__(exc_type, exc, traceback)


def write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with RecordWriter(path) as writer:
        for record in records:
            writer.write(record)


def record_count(path: Path) -> int:
    """Return the number of stored records without reading any column data."""
    return int(pq.ParquetFile(path).metadata.num_rows)


def iter_record_json(
    path: Path,
    *,
    limit: int | None = None,
    batch_size: int | None = None,
) -> Iterator[str]:
    """Yield raw JSON records from bounded Parquet batches."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    parquet = pq.ParquetFile(path)
    if batch_size is None:
        rows = max(1, int(parquet.metadata.num_rows))
        column_index = parquet.schema_arrow.get_field_index("record_json")
        uncompressed = sum(
            parquet.metadata.row_group(group).column(column_index).total_uncompressed_size
            for group in range(parquet.metadata.num_row_groups)
        )
        average_bytes = max(1, uncompressed // rows)
        batch_size = max(1, min(256, (16 * 1024 * 1024) // average_bytes))
    yielded = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["record_json"],
    ):
        for value in batch.column(0):
            if limit is not None and yielded >= limit:
                return
            raw = value.as_py()
            if not isinstance(raw, str):
                raise ValueError(f"record_json row {yielded} is not a string")
            yielded += 1
            yield raw


def iter_records(
    path: Path,
    *,
    limit: int | None = None,
    batch_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    for raw in iter_record_json(path, limit=limit, batch_size=batch_size):
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("record_json must contain a JSON object")
        yield value


def read_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_records(path))
