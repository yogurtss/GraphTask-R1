#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast

from graphtask_r1.experiments.path_sampling import (
    ExperimentalPathSampler,
    PathSamplingConfig,
    SamplingStrategy,
    load_reference_profile,
    write_experiment,
)
from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.training.relations import load_relation_catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare isolated rule-based path sampling strategies."
    )
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--reference-tasks", type=Path, required=True)
    parser.add_argument("--relation-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--neighbor-limit", type=int, default=200)
    parser.add_argument("--branch-trials", type=int, default=24)
    parser.add_argument("--max-prefix-entities", type=int, default=100)
    parser.add_argument("--max-answers", type=int, default=20)
    parser.add_argument("--bounded-retries", type=int, default=4)
    parser.add_argument("--family-retries", type=int, default=16)
    parser.add_argument(
        "--strategies",
        default="naive_path,bounded_path,family_balanced",
        help="Comma-separated strategies: naive_path,bounded_path,family_balanced",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.attempts < 1:
        raise SystemExit("--attempts must be positive")
    os.environ["GRAPHTASK_KQAPRO_DB"] = str(args.graph_db)
    reference = load_reference_profile(args.reference_tasks)
    relations = load_relation_catalog(args.relation_catalog)
    config = PathSamplingConfig(
        seed=args.seed,
        max_depth=args.max_depth,
        neighbor_limit=args.neighbor_limit,
        branch_trials=args.branch_trials,
        max_prefix_entities=args.max_prefix_entities,
        max_answers=args.max_answers,
        bounded_retries=args.bounded_retries,
        family_retries=args.family_retries,
    )
    sampler = ExperimentalPathSampler(
        backend_from_snapshot("kqapro-v1"),
        seed_entities=reference.seed_entities,
        allowed_relations=frozenset(relation.relation_id for relation in relations),
        config=config,
    )
    valid = {"naive_path", "bounded_path", "family_balanced"}
    raw_strategies = tuple(value.strip() for value in args.strategies.split(",") if value.strip())
    invalid = sorted(set(raw_strategies) - valid)
    if invalid:
        raise SystemExit(f"unknown strategies: {', '.join(invalid)}")
    strategies = tuple(cast(SamplingStrategy, value) for value in raw_strategies)
    experiments = tuple(
        sampler.run(strategy=strategy, attempts=args.attempts) for strategy in strategies
    )
    comparison = write_experiment(
        args.output_dir,
        config=config,
        reference=reference,
        experiments=experiments,
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
