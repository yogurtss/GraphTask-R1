#!/usr/bin/env python3
"""Fail early when ms-swift is pointed at the wrong GraphTask Parquet schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graphtask_r1.training.ms_swift_data import convert_rl_row, convert_sft_row


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("sft", "rl"), required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    return parser.parse_args()


def validate_path(path: Path, *, kind: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    expected = "messages" if kind == "sft" else "prompt"
    columns = parquet.schema_arrow.names
    if expected not in columns:
        raise ValueError(
            f"{kind.upper()} dataset {path} has columns {columns}, but requires {expected!r}; "
            "use the preflight SFT accepted Parquet from sft_data.env"
            if kind == "sft"
            else f"RL dataset {path} has columns {columns}, but requires 'prompt'"
        )
    if parquet.metadata.num_rows < 1:
        raise ValueError(f"{kind.upper()} dataset is empty: {path}")
    row = next(parquet.iter_batches(batch_size=1)).to_pylist()[0]
    try:
        (convert_sft_row if kind == "sft" else convert_rl_row)(row)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {kind.upper()} row in {path}: {exc}") from exc
    return {
        "path": str(path),
        "kind": kind,
        "rows": parquet.metadata.num_rows,
        "message_column": expected,
    }


def main() -> int:
    args = _arguments()
    seen: set[Path] = set()
    results = []
    for path in args.input:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        results.append(validate_path(path, kind=args.kind))
    print(json.dumps({"validated": results}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
