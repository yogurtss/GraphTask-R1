#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphtask_r1.experiments import (
    CounterfactualRetrievalExample,
    train_counterfactual_reranker,
)
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.utils import read_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ECP pairwise reranker from certified counterfactual rows."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    args = parser.parse_args()

    examples = tuple(
        CounterfactualRetrievalExample.model_validate(value)
        for value in read_records(args.examples)
    )
    backend = SQLiteGraphBackend(args.graph_db, snapshot_id="kilt-2019-08-01-v1")
    try:
        state = train_counterfactual_reranker(
            examples,
            backend=backend,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        write_json(args.output, state.model_dump(mode="json"))
        print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
