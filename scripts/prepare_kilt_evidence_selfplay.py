#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from graphtask_r1.experiments import (
    EvidenceSelfPlayConfig,
    TransformersEvidenceAnswerer,
    export_evidence_selfplay_round,
    generate_evidence_challenges,
    run_evidence_selfplay_round,
)
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.graphscript import BackendEvidenceRetriever
from graphtask_r1.training.evidence_selfplay_rl import export_evidence_rl_round


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one certified KILT Evidence-Flow self-play round."
    )
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument("--expand-k", type=int, default=3)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--export-mode",
        choices=("rl", "sft-diagnostic"),
        default="rl",
        help="Direct RL is the research mainline; SFT remains a diagnostic ablation.",
    )
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EvidenceSelfPlayConfig(
        seed=args.seed,
        candidate_limit=args.candidate_limit,
        retrieve_k=args.retrieve_k,
        expand_k=args.expand_k,
        context_k=args.context_k,
    )
    backend = SQLiteGraphBackend(args.graph_db, snapshot_id="kilt-2019-08-01-v1")
    try:
        challenges = generate_evidence_challenges(backend, config)
        answerer = TransformersEvidenceAnswerer(
            args.model,
            adapter_path=args.adapter,
            max_new_tokens=args.max_new_tokens,
        )
        result = run_evidence_selfplay_round(
            challenges,
            backend=backend,
            retriever=BackendEvidenceRetriever(backend),
            answerer=answerer,
            config=config,
        )
        if args.export_mode == "rl":
            summary = export_evidence_rl_round(
                result,
                args.output_dir,
                seed=args.seed,
                train_ratio=config.train_ratio,
                split_manifest=args.split_manifest,
                frontier_target=config.frontier_target,
                frontier_sigma=config.frontier_sigma,
                solver_reward_variant=args.solver_reward_variant,
            )
        else:
            summary = export_evidence_selfplay_round(
                result,
                args.output_dir,
                config=config,
                split_manifest=args.split_manifest,
            )
        print(summary)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
