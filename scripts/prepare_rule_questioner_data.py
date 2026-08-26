#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from graphtask_r1.experiments.rule_questioner import (
    build_rule_questioner_mixed_sft,
    export_rule_questioner_rl,
    export_rule_questioner_sft,
    rule_questioner_replacement_count,
)
from graphtask_r1.graph import backend_from_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare isolated program-to-question SFT and self-play data."
    )
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--reference-tasks", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sft-count",
        type=int,
        help=(
            "Questioner SFT rows to export. When omitted with --baseline-mixed-sft, "
            "use its exact Questioner row count."
        ),
    )
    parser.add_argument(
        "--baseline-mixed-sft",
        type=Path,
        help="Optional baseline mixed SFT whose Questioner rows are replaced for controlled A/B.",
    )
    parser.add_argument("--selfplay-count", type=int)
    parser.add_argument("--opponent-url", default="http://127.0.0.1:18080")
    parser.add_argument("--opponent-samples", type=int, default=4)
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    os.environ["GRAPHTASK_KQAPRO_DB"] = str(args.graph_db)
    backend = backend_from_snapshot("kqapro-v1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_count = args.sft_count
    if sft_count is None:
        if args.baseline_mixed_sft is None:
            raise ValueError(
                "set --sft-count when --baseline-mixed-sft is not provided"
            )
        sft_count = rule_questioner_replacement_count(args.baseline_mixed_sft)
    sft = export_rule_questioner_sft(
        args.reference_tasks,
        args.output_dir / "questioner-sft.parquet",
        backend=backend,
        count=sft_count,
        seed=args.seed,
    )
    rl = export_rule_questioner_rl(
        args.candidates,
        args.output_dir / "questioner-selfplay.parquet",
        backend=backend,
        graph_snapshot="kqapro-v1",
        opponent_url=args.opponent_url,
        opponent_samples=args.opponent_samples,
        count=args.selfplay_count,
        round_index=args.round_index,
        seed=args.seed,
    )
    mixed = None
    if args.baseline_mixed_sft is not None:
        mixed = build_rule_questioner_mixed_sft(
            args.baseline_mixed_sft,
            args.output_dir / "questioner-sft.parquet",
            args.output_dir / "mixed-sft.parquet",
            seed=args.seed,
        )
    print(
        json.dumps(
            {"sft": sft, "selfplay": rl, "mixed_sft": mixed},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
