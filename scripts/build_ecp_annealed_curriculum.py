#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.utils import write_json


def _rows_by_uid(table: pa.Table) -> dict[str, dict[str, object]]:
    return {str(row["uid"]): row for row in table.to_pylist()}


def build_annealed_curriculum(
    dense_rows_path: Path,
    strict_rows_path: Path,
    output_dir: Path,
    *,
    dense_size: int,
    seed: int,
) -> dict[str, object]:
    dense_table = pq.read_table(dense_rows_path)
    strict_table = pq.read_table(strict_rows_path)
    dense_by_uid = _rows_by_uid(dense_table)
    strict_by_uid = _rows_by_uid(strict_table)
    if dense_by_uid.keys() != strict_by_uid.keys():
        raise ValueError("dense and strict rows must describe the same tasks")
    if not 0 < dense_size < len(dense_by_uid):
        raise ValueError("dense_size must leave at least one row in each stage")

    uids = sorted(dense_by_uid)
    random.Random(seed).shuffle(uids)
    dense_uids = uids[:dense_size]
    strict_uids = uids[dense_size:]
    dense_rows = [dense_by_uid[uid] for uid in dense_uids]
    strict_rows = [strict_by_uid[uid] for uid in strict_uids]

    dense_dir = output_dir / "stage_1_dense"
    strict_dir = output_dir / "stage_2_strict"
    dense_dir.mkdir(parents=True, exist_ok=True)
    strict_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(dense_rows, schema=dense_table.schema),
        dense_dir / "solver_train.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(strict_rows, schema=strict_table.schema),
        strict_dir / "solver_train.parquet",
    )
    manifest: dict[str, object] = {
        "seed": seed,
        "total_updates": len(uids),
        "dense_reward": "ecp_v5",
        "strict_reward": "ecp_v4",
        "dense_uids": dense_uids,
        "strict_uids": strict_uids,
    }
    write_json(output_dir / "annealing_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build disjoint dense/strict stages for ECP reward annealing."
    )
    parser.add_argument("--dense-rows", type=Path, required=True)
    parser.add_argument("--strict-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    build_annealed_curriculum(
        args.dense_rows,
        args.strict_rows,
        args.output_dir,
        dense_size=args.dense_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
