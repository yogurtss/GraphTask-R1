from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from kqapro_grr.boundary import build_boundary_pairs
from kqapro_grr.compare import compare_predictions
from kqapro_grr.eval_split import freeze_eval_sets
from kqapro_grr.pipeline import (
    build_replay_pairs,
    collect_fixed_incumbent_rollouts,
    load_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent KQAPro graph-residual replay experiment"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("collect-rollouts", "build-pairs", "build-boundary-pairs"):
        command = actions.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
    freeze = actions.add_parser("freeze-eval")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--exclude-root", type=Path, action="append", default=[])
    freeze.add_argument("--dev-count", type=int, default=128)
    freeze.add_argument("--blind-count", type=int, default=128)
    freeze.add_argument("--seed", type=int, default=4242)
    compare = actions.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
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
    elif args.action == "collect-rollouts":
        config = load_config(args.config)
        result = asyncio.run(collect_fixed_incumbent_rollouts(config))
    elif args.action == "build-boundary-pairs":
        config = load_config(args.config)
        result = build_boundary_pairs(config)
    elif args.action == "compare":
        result = compare_predictions(
            args.baseline,
            args.candidate,
            output_path=args.output,
        )
    else:
        config = load_config(args.config)
        result = build_replay_pairs(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0
