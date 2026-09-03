#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from graphtask_r1.training.evidence_selfplay_rl import (
    build_promoted_solver_curriculum,
)
from graphtask_r1.utils import ParquetRowWriter, write_json


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with ParquetRowWriter(path, batch_size=128) as writer:
        for row in rows:
            writer.write(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote online Questioner proposals into a round-2 Solver curriculum."
    )
    parser.add_argument("--questioner-rows", type=Path, required=True)
    parser.add_argument("--solver-rows", type=Path, required=True)
    parser.add_argument("--solver-val", type=Path, required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--retain-all-solver-rows", action="store_true")
    args = parser.parse_args()

    questioner_rows = pq.read_table(args.questioner_rows).to_pylist()
    solver_rows = pq.read_table(args.solver_rows).to_pylist()
    validation_rows = pq.read_table(args.solver_val).to_pylist()
    curriculum, manifest = build_promoted_solver_curriculum(
        questioner_rows,
        solver_rows,
        _read_jsonl(args.completions),
        target_size=args.target_size,
        seed=args.seed,
        retain_all_solver_rows=args.retain_all_solver_rows,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "solver_train.parquet", curriculum)
    _write_rows(args.output_dir / "solver_val.parquet", validation_rows)
    write_json(args.output_dir / "promotion_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
