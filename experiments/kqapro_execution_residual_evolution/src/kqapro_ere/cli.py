from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from graphtask_r1.graph.sqlite import SQLiteGraphBackend
from kqapro_ere.compare import compare_predictions
from kqapro_ere.eval_split import freeze_eval_sets
from kqapro_ere.pairs import build_preference_pairs
from kqapro_ere.profile import mine_residual_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent KQAPro execution-residual evolution experiment"
    )
    actions = parser.add_subparsers(dest="action", required=True)

    freeze = actions.add_parser("freeze-eval")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--exclude-root", type=Path, action="append", default=[])
    freeze.add_argument("--dev-count", type=int, default=128)
    freeze.add_argument("--blind-count", type=int, default=128)
    freeze.add_argument("--seed", type=int, default=4242)

    pairs = actions.add_parser("build-pairs")
    pairs.add_argument("--tasks", type=Path, required=True)
    pairs.add_argument("--graph-db", type=Path, required=True)
    pairs.add_argument("--relation-catalog", type=Path, required=True)
    pairs.add_argument("--output-dir", type=Path, required=True)
    pairs.add_argument("--residual-profile", type=Path)
    pairs.add_argument("--count", type=int, default=128)
    pairs.add_argument("--seed", type=int, default=4242)
    pairs.add_argument("--max-edge-visits", type=int, default=200)
    pairs.add_argument("--max-returned-entities", type=int, default=1000)

    residuals = actions.add_parser("mine-residuals")
    residuals.add_argument("--tasks", type=Path, required=True)
    residuals.add_argument("--completions", type=Path, required=True)
    residuals.add_argument("--graph-db", type=Path, required=True)
    residuals.add_argument("--relation-catalog", type=Path, required=True)
    residuals.add_argument("--output-dir", type=Path, required=True)
    residuals.add_argument("--seed", type=int, default=4242)
    residuals.add_argument("--max-edge-visits", type=int, default=200)
    residuals.add_argument("--max-returned-entities", type=int, default=1000)

    compare = actions.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--bootstrap-seed", type=int, default=4242)
    compare.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "freeze-eval":
        result = freeze_eval_sets(
            args.source,
            args.output_dir,
            exclude_roots=tuple(args.exclude_root),
            dev_count=args.dev_count,
            blind_count=args.blind_count,
            seed=args.seed,
        )
    elif args.action == "build-pairs":
        backend = SQLiteGraphBackend(args.graph_db, snapshot_id="kqapro-v1")
        result = build_preference_pairs(
            args.tasks,
            args.relation_catalog,
            args.output_dir,
            backend=backend,
            count=args.count,
            seed=args.seed,
            residual_profile_path=args.residual_profile,
            max_edge_visits=args.max_edge_visits,
            max_returned_entities=args.max_returned_entities,
        )
    elif args.action == "mine-residuals":
        backend = SQLiteGraphBackend(args.graph_db, snapshot_id="kqapro-v1")
        result = mine_residual_profile(
            args.tasks,
            args.completions,
            args.relation_catalog,
            args.output_dir,
            backend=backend,
            seed=args.seed,
            max_edge_visits=args.max_edge_visits,
            max_returned_entities=args.max_returned_entities,
        )
    else:
        result = compare_predictions(
            args.baseline,
            args.candidate,
            output_path=args.output,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0
