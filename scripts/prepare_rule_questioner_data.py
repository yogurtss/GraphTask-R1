#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from graphtask_r1.experiments.rule_questioner import (
    build_rule_questioner_mixed_sft,
    export_rule_questioner_rl,
    export_rule_questioner_sft,
    rule_questioner_replacement_count,
)
from graphtask_r1.graph import backend_from_snapshot

LOGGER = logging.getLogger("graphtask_r1.prepare_rule_questioner_data")


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    os.environ["GRAPHTASK_KQAPRO_DB"] = str(args.graph_db)
    LOGGER.info("loading_graph database=%s", args.graph_db)
    backend = backend_from_snapshot("kqapro-v1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_count = args.sft_count
    if sft_count is None:
        if args.baseline_mixed_sft is None:
            raise ValueError(
                "set --sft-count when --baseline-mixed-sft is not provided"
            )
        sft_count = rule_questioner_replacement_count(args.baseline_mixed_sft)
    LOGGER.info(
        "exporting_questioner_sft input=%s requested=%d output=%s",
        args.reference_tasks,
        sft_count,
        args.output_dir / "questioner-sft.parquet",
    )
    sft = export_rule_questioner_sft(
        args.reference_tasks,
        args.output_dir / "questioner-sft.parquet",
        backend=backend,
        count=sft_count,
        seed=args.seed,
    )
    LOGGER.info(
        "questioner_sft_exported scanned=%s eligible=%s selected=%s",
        sft["scanned"],
        sft["eligible"],
        sft["selected"],
    )
    LOGGER.info(
        "exporting_questioner_selfplay input=%s requested=%s output=%s",
        args.candidates,
        args.selfplay_count,
        args.output_dir / "questioner-selfplay.parquet",
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
    LOGGER.info(
        "questioner_selfplay_exported scanned=%s selected=%s rejected_uncertified=%s",
        rl["scanned"],
        rl["selected"],
        rl["rejected_uncertified"],
    )
    mixed = None
    if args.baseline_mixed_sft is not None:
        LOGGER.info(
            "building_mixed_sft baseline=%s output=%s",
            args.baseline_mixed_sft,
            args.output_dir / "mixed-sft.parquet",
        )
        mixed = build_rule_questioner_mixed_sft(
            args.baseline_mixed_sft,
            args.output_dir / "questioner-sft.parquet",
            args.output_dir / "mixed-sft.parquet",
            seed=args.seed,
        )
        LOGGER.info(
            "mixed_sft_built solver_rows=%s questioner_rows=%s total=%s",
            mixed["solver_rows"],
            mixed["replacement_questioner_rows"],
            mixed["total"],
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
