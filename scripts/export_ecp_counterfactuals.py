#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphtask_r1.experiments import (
    EvidenceSelfPlayRecord,
    counterfactual_retrieval_examples,
    generate_counterfactual_proof_pairs,
)
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.utils import read_json, read_records, write_json, write_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export execution-certified counterfactual proof and retriever rows."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    split = read_json(args.split_manifest)
    train_ids = {str(value) for value in split["train_ids"]}
    records = tuple(
        EvidenceSelfPlayRecord.model_validate(value)
        for value in read_records(args.records)
    )
    challenges = tuple(
        record.challenge
        for record in records
        if record.challenge.challenge_id in train_ids
    )
    backend = SQLiteGraphBackend(args.graph_db, snapshot_id="kilt-2019-08-01-v1")
    try:
        pairs = generate_counterfactual_proof_pairs(
            challenges,
            backend=backend,
            seed=args.seed,
            limit=args.limit,
        )
        retrieval_rows = counterfactual_retrieval_examples(pairs, backend=backend)
    finally:
        backend.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_records(
        args.output_dir / "counterfactual_pairs.parquet",
        (value.model_dump(mode="json", by_alias=True) for value in pairs),
    )
    write_records(
        args.output_dir / "retriever_contrastive.parquet",
        (value.model_dump(mode="json") for value in retrieval_rows),
    )
    manifest = {
        "schema_version": "ecp-counterfactual-export-v1",
        "seed": args.seed,
        "uses_sft": False,
        "train_challenges": len(challenges),
        "counterfactual_pairs": len(pairs),
        "retriever_examples": len(retrieval_rows),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
