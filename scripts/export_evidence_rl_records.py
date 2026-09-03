#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphtask_r1.experiments import EvidenceSelfPlayRecord, EvidenceSelfPlayRound
from graphtask_r1.training.evidence_selfplay_rl import export_evidence_rl_round
from graphtask_r1.utils import read_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an existing frozen Solver audit to no-SFT self-play RL rows."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument(
        "--solver-reward-variant",
        choices=(
            "additive_v1",
            "search_r1_em",
            "ecp_v1",
            "ecp_v2",
            "ecp_v3",
            "ecp_v4",
            "ecp_v5",
        ),
        default="additive_v1",
    )
    args = parser.parse_args()
    records = tuple(
        EvidenceSelfPlayRecord.model_validate(value)
        for value in read_records(args.records)
    )
    result = EvidenceSelfPlayRound(
        seed=args.seed,
        records=records,
        retrieval_residuals=sum(record.residual.kind == "retrieval" for record in records),
        reasoning_residuals=sum(record.residual.kind == "reasoning" for record in records),
        successes=sum(record.residual.kind == "success" for record in records),
    )
    summary = export_evidence_rl_round(
        result,
        args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        split_manifest=args.split_manifest,
        solver_reward_variant=args.solver_reward_variant,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
