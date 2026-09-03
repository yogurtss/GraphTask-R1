#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from graphtask_r1.experiments import (
    TransformersInteractiveEvidencePolicy,
    benchmark_examples_from_records,
    evaluate_interactive_evidence,
)
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.graphscript import BackendEvidenceRetriever, CounterfactualEvidenceRetriever
from graphtask_r1.utils import read_json, read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the exact interactive evidence protocol used during RL."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--counterfactual-reranker", type=Path)
    parser.add_argument("--causal-selection", action="store_true")
    args = parser.parse_args()

    split = read_json(args.split_manifest)
    validation_ids = frozenset(str(value) for value in split["validation_ids"])
    examples = benchmark_examples_from_records(
        read_records(args.examples),
        example_ids=validation_ids,
        limit=args.limit,
    )
    if not examples:
        raise ValueError("the selected interactive evaluation set is empty")
    backend = SQLiteGraphBackend(args.graph_db, snapshot_id="kilt-2019-08-01-v1")
    try:
        policy = TransformersInteractiveEvidencePolicy(
            args.model,
            adapter_path=args.adapter,
            max_new_tokens=args.max_new_tokens,
        )
        retriever = (
            CounterfactualEvidenceRetriever.from_path(
                backend, args.counterfactual_reranker
            )
            if args.counterfactual_reranker is not None
            else BackendEvidenceRetriever(backend)
        )
        report = evaluate_interactive_evidence(
            examples,
            retriever=retriever,
            policy=policy,
            max_turns=args.max_turns,
            causal_selection=args.causal_selection,
        )
        write_json(args.output, report.model_dump(mode="json"))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
