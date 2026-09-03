#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from graphtask_r1.experiments import (
    EvidenceABConfig,
    HyperlinkEvidenceFlow,
    SingleRetrievalBaseline,
    TransformersEvidenceAnswerer,
    benchmark_examples_from_records,
    evaluate_evidence_ab,
)
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.graphscript import BackendEvidenceRetriever
from graphtask_r1.utils import read_json, read_records, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired single-retrieval vs Evidence-Flow KILT QA smoke experiment."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument("--expand-k", type=int, default=3)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    example_ids = None
    if args.split_manifest is not None:
        split = read_json(args.split_manifest)
        example_ids = frozenset(str(value) for value in split["validation_ids"])
    examples = benchmark_examples_from_records(
        read_records(args.examples),
        example_ids=example_ids,
        limit=args.limit,
    )
    if not examples:
        raise ValueError("the selected test set is empty")
    config = EvidenceABConfig(
        retrieve_k=args.retrieve_k,
        expand_k=args.expand_k,
        context_k=args.context_k,
    )
    backend = SQLiteGraphBackend(args.graph_db, snapshot_id="kilt-evidence-ab")
    try:
        retriever = BackendEvidenceRetriever(backend)
        answerer = TransformersEvidenceAnswerer(
            args.model,
            adapter_path=args.adapter,
            max_new_tokens=args.max_new_tokens,
        )
        report = evaluate_evidence_ab(
            examples,
            SingleRetrievalBaseline(retriever, answerer, config),
            HyperlinkEvidenceFlow(retriever, answerer, config),
        )
        write_json(args.output, report.model_dump(mode="json"))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
