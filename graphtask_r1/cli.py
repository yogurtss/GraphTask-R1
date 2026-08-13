from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
from collections.abc import Iterator, Sequence
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from graphtask_r1.archive import TaskArchive
from graphtask_r1.data import (
    audit_records,
    bootstrap_kilt_grpo,
    merge_denylists,
    prepare_benchmark,
    prepare_kilt,
    prepare_kqapro,
    sample_questioner_seeds,
    select_graphscript_tasks,
)
from graphtask_r1.evaluation import (
    KQAProValConfig,
    compare_kqapro_val_metrics,
    evaluate_benchmark,
    evaluate_kqapro_val,
    inspect_kqapro_val,
    visualize_kqapro_val,
)
from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.pipeline import run_mini_pipeline
from graphtask_r1.schema import TaskCertificate, TaskTrainingRecord
from graphtask_r1.training.relations import build_relation_catalog, load_relation_catalog
from graphtask_r1.training.rl_dataset import export_role_dataset
from graphtask_r1.training.scripted import run_scripted_selfplay
from graphtask_r1.training.selfplay import run_self_play
from graphtask_r1.training.sft_dataset import export_sft_dataset
from graphtask_r1.utils import (
    ProgressLogger,
    iter_record_json,
    record_count,
    write_records,
)

LOGGER = logging.getLogger("graphtask_r1.cli")
DEFAULT_DATA_WORKERS = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")


def _load_kqapro_val_config(path: Path) -> KQAProValConfig:
    raw = yaml.safe_load(os.path.expandvars(path.read_text()))
    if not isinstance(raw, dict):
        raise ValueError(f"KQAPro evaluation config must be a mapping: {path}")
    return KQAProValConfig.model_validate(raw)


def _parse_indices(value: str | None) -> frozenset[int] | None:
    if value is None:
        return None
    try:
        indices = frozenset(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--indices must be comma-separated non-negative integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise ValueError("--indices must contain non-negative integers")
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphtask-r1")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="stderr log level (default: INFO)",
    )
    groups = parser.add_subparsers(dest="group", required=True)

    graph = groups.add_parser("graph")
    graph_actions = graph.add_subparsers(dest="action", required=True)
    preflight = graph_actions.add_parser("preflight")
    preflight.add_argument("--snapshot", required=True)
    preflight.add_argument("--limit", type=int, default=5)

    e2e = groups.add_parser("e2e")
    e2e_actions = e2e.add_subparsers(dest="action", required=True)
    mini = e2e_actions.add_parser("mini-pipeline")
    mini.add_argument("--graph", choices=["toy"], default="toy")
    mini.add_argument("--num-programs", type=int, default=100)
    mini.add_argument("--output-dir", type=Path, required=True)
    _add_common(mini)
    scripted = e2e_actions.add_parser("scripted-self-play")
    scripted.add_argument("--rounds", type=int, default=3)
    scripted.add_argument("--candidates-per-round", type=int, default=16)
    scripted.add_argument("--seed", type=int, default=42)
    scripted.add_argument("--resume", action="store_true")
    scripted.add_argument("--output-dir", type=Path, required=True)

    data = groups.add_parser("data")
    data_actions = data.add_subparsers(dest="action", required=True)
    fetch = data_actions.add_parser("fetch")
    fetch.add_argument(
        "--dataset",
        choices=["kqapro", "kilt", "ssp", "freebase", "webqsp", "cwq", "grailqa"],
        required=True,
    )
    fetch.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    prepare = data_actions.add_parser("prepare")
    prepare.add_argument(
        "--dataset",
        choices=[
            "kqapro",
            "kilt",
            "ssp",
            "webqsp",
            "cwq",
            "grailqa",
            "nq",
            "triviaqa",
            "popqa",
            "hotpotqa",
            "2wikimultihopqa",
            "bamboogle",
            "musique",
        ],
        required=True,
    )
    prepare.add_argument("--raw-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--splits", default="train,val")
    prepare.add_argument(
        "--include-datasets",
        help="comma-separated SSP buckets (default: the six CoEvoKG evaluation buckets)",
    )
    prepare.add_argument(
        "--no-text-index",
        action="store_true",
        help="for KILT only: build hyperlink graph without the FTS5 passage index",
    )
    prepare.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="rebuild graph.sqlite even when kb.json and converter metadata match",
    )
    prepare.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_DATA_WORKERS,
        help=f"parallel record workers (default: {DEFAULT_DATA_WORKERS})",
    )
    prepare.add_argument(
        "--max-trace-tool-calls",
        type=_positive_int,
        default=32,
        help="for KQAPro: maximum graph calls plus final answer in a compiled trace",
    )
    prepare.add_argument(
        "--max-trace-query-results",
        type=_positive_int,
        default=1_024,
        help="for KQAPro: maximum entities retained by a compact graph query",
    )
    prepare.add_argument(
        "--max-witness-facts",
        type=_non_negative_int,
        default=0,
        help="for KQAPro: causal witness triples stored inline per task (default: 0)",
    )
    prepare.add_argument(
        "--train-sample-size",
        type=_non_negative_int,
        default=20_000,
        help="for KQAPro: stratified train rows; 0 processes the full train split",
    )
    prepare.add_argument(
        "--trace-mode",
        choices=("none", "canonical"),
        default="none",
        help="for KQAPro: omit expensive canonical traces by default",
    )
    prepare.add_argument(
        "--verification-mode",
        choices=("source", "full"),
        default="source",
        help="for KQAPro: source-program execution certification or full self-play gates",
    )
    _add_common(prepare)

    bootstrap_kilt = data_actions.add_parser("bootstrap-kilt-grpo")
    bootstrap_kilt.add_argument("--output-dir", type=Path, required=True)
    bootstrap_kilt.add_argument("--snapshot", default="kilt-2019-08-01-v1")
    bootstrap_kilt.add_argument("--count", type=_positive_int, default=1_024)
    bootstrap_kilt.add_argument("--pool-limit", type=_positive_int, default=100_000)
    bootstrap_kilt.add_argument("--max-attempts", type=_positive_int)
    bootstrap_kilt.add_argument("--min-degree", type=_positive_int, default=2)
    bootstrap_kilt.add_argument("--max-degree", type=_positive_int, default=100)
    bootstrap_kilt.add_argument("--val-ratio", type=float, default=0.1)
    bootstrap_kilt.add_argument("--families", default="hop1,hop2,type_filter,count")
    bootstrap_kilt.add_argument(
        "--interaction-mode", choices=["tool", "graphscript"], default="graphscript"
    )
    bootstrap_kilt.add_argument("--graphscript-version", choices=["0.1", "0.2"], default="0.2")
    bootstrap_kilt.add_argument("--seed", type=int, default=42)
    bootstrap_kilt.add_argument("--dry-run", action="store_true")

    audit = data_actions.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--kind", choices=["auto", "task", "benchmark"], default="auto")
    audit.add_argument(
        "--deep",
        action="store_true",
        help="also validate bulky witness facts; default audit checks training-critical fields",
    )
    audit.add_argument(
        "--training-view-output",
        type=Path,
        help="write valid task fields needed downstream while streaming the audit",
    )

    export = data_actions.add_parser("export-rl")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--roles", choices=["both", "questioner", "solver"], default="both")
    export.add_argument("--opponent-url")
    export.add_argument("--opponent-samples", type=int, default=8)
    export.add_argument("--interaction-mode", choices=["tool", "graphscript"], default="tool")
    export.add_argument(
        "--graphscript-version", choices=["0.1", "0.2", "0.3"], default="0.1"
    )
    export.add_argument("--relation-catalog", type=Path)
    _add_common(export)

    export_sft = data_actions.add_parser("export-sft")
    export_sft.add_argument("--input", type=Path, required=True)
    export_sft.add_argument("--output", type=Path, required=True)
    export_sft.add_argument("--roles", choices=["both", "questioner", "solver"], default="both")
    export_sft.add_argument("--interaction-mode", choices=["tool", "graphscript"], default="tool")
    export_sft.add_argument(
        "--graphscript-version", choices=["0.1", "0.2", "0.3"], default="0.1"
    )
    export_sft.add_argument("--relation-catalog", type=Path)
    _add_common(export_sft)

    seeds = data_actions.add_parser("sample-seeds")
    seeds.add_argument("--snapshot", default="kqapro-v1")
    seeds.add_argument("--output", type=Path, required=True)
    seeds.add_argument("--count", type=int, default=256)
    seeds.add_argument("--exclude", type=Path)
    seeds.add_argument("--pool-limit", type=int, default=100_000)
    seeds.add_argument("--opponent-url")
    seeds.add_argument("--opponent-samples", type=int, default=8)
    seeds.add_argument("--seed", type=int, default=42)
    seeds.add_argument("--interaction-mode", choices=["tool", "graphscript"], default="tool")
    seeds.add_argument(
        "--graphscript-version", choices=["0.1", "0.2", "0.3"], default="0.3"
    )
    seeds.add_argument("--relation-catalog", type=Path)

    select_interaction = data_actions.add_parser("select-interaction-tasks")
    select_interaction.add_argument("--input", type=Path, required=True)
    select_interaction.add_argument("--output", type=Path, required=True)
    select_interaction.add_argument("--limit", type=int)
    select_interaction.add_argument("--snapshot")
    select_interaction.add_argument("--max-follow-limit", type=int, default=100)
    select_interaction.add_argument("--max-edge-visits", type=int, default=200)
    select_interaction.add_argument("--max-returned-entities", type=int, default=1_000)

    catalog = data_actions.add_parser("build-relation-catalog")
    catalog.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="one or more task files; relations are unioned into one shared catalog",
    )
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--snapshot")
    catalog.add_argument("--limit", type=int)
    catalog.add_argument(
        "--scope",
        choices=("graph", "tasks"),
        default="graph",
        help="catalog scope; graph is stable across train/val samples (default: graph)",
    )
    export_archive = data_actions.add_parser("export-archive")
    export_archive.add_argument("--archive", type=Path, required=True)
    export_archive.add_argument("--output", type=Path, required=True)
    denylist = data_actions.add_parser("merge-denylists")
    denylist.add_argument("--inputs", type=Path, nargs="+", required=True)
    denylist.add_argument("--output", type=Path, required=True)

    train = groups.add_parser("train")
    train_actions = train.add_subparsers(dest="action", required=True)
    for name in ("sft", "solver-grpo"):
        stage = train_actions.add_parser(name)
        stage.add_argument("--config", type=Path, required=True)
        stage.add_argument("--dry-run", action="store_true")
    selfplay = train_actions.add_parser("self-play")
    selfplay.add_argument("--config", type=Path, required=True)
    selfplay.add_argument("--output-dir", type=Path, required=True)
    selfplay.add_argument("--resume", action="store_true")
    selfplay.add_argument("--dry-run", action="store_true")

    evaluate = groups.add_parser("evaluate")
    evaluate_actions = evaluate.add_subparsers(dest="action", required=True)
    benchmark = evaluate_actions.add_parser("benchmark")
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--solver-url", default="http://127.0.0.1:18080")
    benchmark.add_argument("--snapshot", default="freebase-v1")
    benchmark.add_argument("--samples", type=int, default=1)
    benchmark.add_argument("--concurrency", type=int, default=16)
    benchmark.add_argument(
        "--graphscript-version", choices=["0.1", "0.2", "0.3"], default="0.2"
    )

    kqapro_val = evaluate_actions.add_parser("kqapro-val")
    kqapro_val.add_argument("--config", type=Path, required=True)
    kqapro_val.add_argument("--model-stage", choices=("base", "sft", "grpo"), required=True)
    kqapro_val.add_argument("--input", type=Path, help="override input_path from config")
    kqapro_val.add_argument(
        "--output-dir", type=Path, help="default: outputs/evaluation/kqapro-<model-stage>"
    )
    kqapro_val.add_argument("--limit", type=_positive_int)

    kqapro_compare = evaluate_actions.add_parser("kqapro-compare")
    kqapro_compare.add_argument(
        "--metrics",
        type=Path,
        nargs=3,
        required=True,
        help="metrics.json files from separate base, SFT, and GRPO runs",
    )
    kqapro_compare.add_argument(
        "--output", type=Path, default=Path("outputs/evaluation/kqapro-comparison.json")
    )

    visualize = groups.add_parser("visualize")
    visualize_actions = visualize.add_subparsers(dest="action", required=True)
    kqapro_visualize = visualize_actions.add_parser("kqapro")
    kqapro_visualize.add_argument("--config", type=Path, required=True)
    kqapro_visualize.add_argument(
        "--model-stage", choices=("base", "sft", "grpo"), required=True
    )
    kqapro_visualize.add_argument("--input", type=Path, help="override input_path from config")
    kqapro_visualize.add_argument(
        "--output-dir", type=Path, help="default: outputs/visualization/kqapro-<model-stage>"
    )
    kqapro_visualize.add_argument(
        "--indices", help="zero-based comma-separated dataset rows, for example 0,12,41"
    )
    kqapro_visualize.add_argument(
        "--limit", type=_positive_int, default=3, help="sample count when --indices is omitted"
    )
    kqapro_visualize.add_argument(
        "--inspect-only", action="store_true", help="print selected dataset rows without inference"
    )
    return parser


def _load_tasks(path: Path, limit: int | None) -> list[TaskCertificate]:
    total = min(record_count(path), limit) if limit is not None else record_count(path)
    progress = ProgressLogger("data.load_tasks", total=total)
    progress.start(path=str(path), loading="streaming", include_witness=True)
    tasks: list[TaskCertificate] = []
    for index, raw in enumerate(iter_record_json(path, limit=limit)):
        tasks.append(TaskCertificate.model_validate_json(raw))
        progress.update(index + 1)
    progress.finish(len(tasks), path=str(path))
    return tasks


def _iter_training_tasks(
    paths: Path | Sequence[Path], limit: int | None
) -> tuple[Iterator[TaskTrainingRecord], int]:
    source_paths = (paths,) if isinstance(paths, Path) else tuple(paths)
    if not source_paths:
        raise ValueError("at least one task input is required")
    available = sum(record_count(path) for path in source_paths)
    total = min(available, limit) if limit is not None else available

    def generate() -> Iterator[TaskTrainingRecord]:
        progress = ProgressLogger("data.load_training_tasks", total=total)
        progress.start(
            paths=[str(path) for path in source_paths],
            loading="streaming",
            include_witness=False,
        )
        completed = 0
        for path in source_paths:
            remaining = None if limit is None else limit - completed
            if remaining is not None and remaining <= 0:
                break
            for raw in iter_record_json(path, limit=remaining):
                completed += 1
                yield TaskTrainingRecord.model_validate_json(raw)
                progress.update(completed, path=str(path))
        progress.finish(completed, paths=[str(path) for path in source_paths])

    return generate(), total


def _launch_stage(stage: str, config_path: Path, *, dry_run: bool) -> dict[str, Any]:
    config = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    backend = str(config.get("training_backend", "ms_swift"))
    if backend != "ms_swift":
        raise ValueError(f"unsupported training backend: {backend}; only ms_swift is supported")
    scripts = {
        "sft": "scripts/train_ms_swift_sft.sh",
        "solver-grpo": "scripts/train_ms_swift_grpo.sh",
    }
    script = scripts[stage]
    env_keys = {
        "model_path": "MODEL_PATH",
        "model_type": "MODEL_TYPE",
        "train_data": "TRAIN_DATA",
        "val_data": "VAL_DATA",
        "output_dir": "OUTPUT_DIR",
        "num_gpus": "NUM_GPUS",
        "micro_batch_size": "MICRO_BATCH_SIZE",
        "eval_batch_size": "EVAL_BATCH_SIZE",
        "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
        "steps_per_generation": "STEPS_PER_GENERATION",
        "rollout_n": "ROLLOUT_N",
        "experiment_name": "EXPERIMENT_NAME",
        "lora_adapter_path": "LORA_ADAPTER_PATH",
        "interaction_mode": "INTERACTION_MODE",
        "graphscript_version": "GRAPHSCRIPT_VERSION",
    }
    selected_env: dict[str, str] = {}
    positive_integer_fields = {
        "num_gpus",
        "micro_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "steps_per_generation",
        "rollout_n",
    }
    for source, target in env_keys.items():
        if target in os.environ:
            selected_env[target] = os.environ[target]
        elif source in config:
            selected_env[target] = str(config[source])
        if source in positive_integer_fields and target in selected_env:
            try:
                parsed = int(selected_env[target])
            except ValueError as exc:
                raise ValueError(f"{target} must be a positive integer") from exc
            if parsed < 1 or str(parsed) != selected_env[target]:
                raise ValueError(f"{target} must be a positive integer")
    if stage == "solver-grpo":
        num_gpus = int(selected_env.get("NUM_GPUS", "1"))
        micro_batch = int(selected_env.get("MICRO_BATCH_SIZE", "1"))
        eval_batch = int(selected_env.get("EVAL_BATCH_SIZE", "1"))
        accumulation = int(selected_env.get("GRADIENT_ACCUMULATION_STEPS", "4"))
        generation_steps = int(selected_env.get("STEPS_PER_GENERATION", str(accumulation)))
        generations = int(selected_env.get("ROLLOUT_N", "4"))
        generation_batch = num_gpus * micro_batch * generation_steps
        evaluation_batch = num_gpus * eval_batch
        if generation_steps % accumulation:
            raise ValueError(
                "STEPS_PER_GENERATION must be an integer multiple of "
                "GRADIENT_ACCUMULATION_STEPS"
            )
        if generation_batch % generations:
            raise ValueError(
                "NUM_GPUS * MICRO_BATCH_SIZE * STEPS_PER_GENERATION must be "
                "divisible by ROLLOUT_N"
            )
        if evaluation_batch % generations:
            raise ValueError(
                "NUM_GPUS * EVAL_BATCH_SIZE must be divisible by ROLLOUT_N"
            )
    result = {
        "stage": stage,
        "training_backend": backend,
        "command": ["bash", script],
        "environment": selected_env,
    }
    if not dry_run:
        subprocess.run(["bash", script], env={**os.environ, **selected_env}, check=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    command_started = perf_counter()
    LOGGER.info("command_started group=%s action=%s", args.group, args.action)
    result: object
    if args.group == "graph":
        started = perf_counter()
        backend = backend_from_snapshot(args.snapshot)
        entities = backend.all_entities(limit=args.limit)
        neighbors = (
            backend.neighbors(entities[:1], direction="both", limit=args.limit) if entities else []
        )
        result = {
            "snapshot": args.snapshot,
            "entities": list(entities),
            "neighbor_count": len(neighbors),
            "latency_ms": (perf_counter() - started) * 1000,
            "passed": bool(entities),
        }
    elif args.group == "e2e":
        if args.action == "mini-pipeline":
            count = args.limit if args.limit is not None else args.num_programs
            result = run_mini_pipeline(
                args.output_dir, num_programs=count, seed=args.seed, dry_run=args.dry_run
            )
        else:
            result = run_scripted_selfplay(
                args.output_dir,
                rounds=args.rounds,
                candidates_per_round=args.candidates_per_round,
                seed=args.seed,
                resume=args.resume,
            )
    elif args.group == "data" and args.action == "fetch":
        env = {**os.environ, "DATA_ROOT": str(args.raw_dir)}
        subprocess.run(["bash", "scripts/download_data.sh", args.dataset], env=env, check=True)
        result = {"dataset": args.dataset, "raw_dir": str(args.raw_dir)}
    elif args.group == "data" and args.action == "prepare":
        if args.dry_run:
            result = vars(args)
        elif args.dataset == "kqapro":
            result = prepare_kqapro(
                args.raw_dir,
                args.output_dir,
                splits=tuple(value.strip() for value in args.splits.split(",")),
                limit=args.limit,
                seed=args.seed,
                workers=args.workers,
                rebuild_graph=args.rebuild_graph,
                max_trace_tool_calls=args.max_trace_tool_calls,
                max_trace_query_results=args.max_trace_query_results,
                max_witness_facts=args.max_witness_facts,
                train_sample_size=(
                    args.train_sample_size if args.train_sample_size > 0 else None
                ),
                trace_mode=args.trace_mode,
                verification_mode=args.verification_mode,
            )
        elif args.dataset == "kilt":
            source_path = (
                args.raw_dir
                if args.raw_dir.is_file()
                else args.raw_dir / "kilt_knowledgesource.json"
            )
            result = prepare_kilt(
                source_path,
                args.output_dir,
                limit=args.limit,
                with_text_index=not args.no_text_index,
                rebuild_graph=args.rebuild_graph,
            )
        else:
            result = prepare_benchmark(
                args.dataset,
                args.raw_dir,
                args.output_dir,
                workers=args.workers,
                include_datasets=tuple(
                    value.strip()
                    for value in (args.include_datasets or "").split(",")
                    if value.strip()
                )
                or None,
                limit=args.limit,
            )
    elif args.group == "data" and args.action == "audit":
        result = audit_records(
            args.input,
            kind=args.kind,
            deep=args.deep,
            training_view_output=args.training_view_output,
        )
    elif args.group == "data" and args.action == "bootstrap-kilt-grpo":
        if args.dry_run:
            result = vars(args)
        else:
            result = bootstrap_kilt_grpo(
                args.output_dir,
                snapshot=args.snapshot,
                count=args.count,
                seed=args.seed,
                pool_limit=args.pool_limit,
                max_attempts=args.max_attempts,
                min_degree=args.min_degree,
                max_degree=args.max_degree,
                val_ratio=args.val_ratio,
                families=tuple(
                    value.strip() for value in args.families.split(",") if value.strip()
                ),
                interaction_mode=args.interaction_mode,
                graphscript_version=args.graphscript_version,
            )
    elif args.group == "data" and args.action == "export-rl":
        if args.dry_run:
            total = (
                min(record_count(args.input), args.limit)
                if args.limit is not None
                else record_count(args.input)
            )
            result = {"would_export": total, "roles": args.roles}
        else:
            rl_tasks, total = _iter_training_tasks(args.input, args.limit)
            rows = export_role_dataset(
                rl_tasks,
                args.output,
                total=total,
                include_questioner=args.roles in {"both", "questioner"},
                include_solver=args.roles in {"both", "solver"},
                opponent_url=args.opponent_url,
                opponent_samples=args.opponent_samples,
                interaction_mode=args.interaction_mode,
                graphscript_version=args.graphscript_version,
                relation_catalog=load_relation_catalog(args.relation_catalog),
            )
            result = {"rows": rows, "output": str(args.output)}
    elif args.group == "data" and args.action == "export-sft":
        if args.dry_run:
            total = (
                min(record_count(args.input), args.limit)
                if args.limit is not None
                else record_count(args.input)
            )
            result = {"would_export": total, "roles": args.roles}
        else:
            sft_tasks, total = _iter_training_tasks(args.input, args.limit)
            rows = export_sft_dataset(
                sft_tasks,
                args.output,
                total=total,
                include_questioner=args.roles in {"both", "questioner"},
                include_solver=args.roles in {"both", "solver"},
                seed=args.seed,
                interaction_mode=args.interaction_mode,
                graphscript_version=args.graphscript_version,
                relation_catalog=load_relation_catalog(args.relation_catalog),
            )
            result = {"rows": rows, "output": str(args.output)}
    elif args.group == "data" and args.action == "sample-seeds":
        result = sample_questioner_seeds(
            args.snapshot,
            args.output,
            count=args.count,
            seed=args.seed,
            exclude_path=args.exclude,
            pool_limit=args.pool_limit,
            opponent_url=args.opponent_url,
            opponent_samples=args.opponent_samples,
            interaction_mode=args.interaction_mode,
            graphscript_version=args.graphscript_version,
            relation_catalog=load_relation_catalog(args.relation_catalog),
        )
    elif args.group == "data" and args.action == "select-interaction-tasks":
        selected_tasks = _load_tasks(args.input, args.limit)
        if not selected_tasks:
            raise ValueError("cannot select interaction tasks from an empty task set")
        snapshot = args.snapshot or selected_tasks[0].graph_snapshot
        mismatched = sorted(
            {
                task.graph_snapshot
                for task in selected_tasks
                if task.graph_snapshot != snapshot
            }
        )
        if mismatched:
            raise ValueError(
                "interaction task input contains snapshots other than "
                f"{snapshot}: {', '.join(mismatched)}"
            )
        result = select_graphscript_tasks(
            selected_tasks,
            args.output,
            backend=backend_from_snapshot(snapshot),
            max_follow_limit=args.max_follow_limit,
            max_edge_visits=args.max_edge_visits,
            max_returned_entities=args.max_returned_entities,
        )
    elif args.group == "data" and args.action == "build-relation-catalog":
        catalog_tasks, total = _iter_training_tasks(args.input, args.limit)
        first = next(catalog_tasks, None)
        if first is None:
            raise ValueError("cannot build a relation catalog from an empty task set")
        snapshot = args.snapshot or first.graph_snapshot

        def matching_tasks() -> Iterator[TaskTrainingRecord]:
            for task in chain((first,), catalog_tasks):
                if task.graph_snapshot != snapshot:
                    raise ValueError(
                        "relation catalog input contains snapshot "
                        f"{task.graph_snapshot!r}; expected {snapshot!r}"
                    )
                yield task

        relations = build_relation_catalog(
            matching_tasks(),
            backend_from_snapshot(snapshot),
            args.output,
            total=total,
            include_graph_schema=args.scope == "graph",
        )
        result = {
            "relations": len(relations),
            "output": str(args.output),
            "snapshot": snapshot,
            "scope": args.scope,
            "inputs": [str(path) for path in args.input],
        }
    elif args.group == "data" and args.action == "export-archive":
        if not args.archive.exists():
            raise FileNotFoundError(args.archive)
        with TaskArchive(args.archive) as archive:
            archived_tasks = archive.all()
        write_records(
            args.output, (task.model_dump(mode="json") for task in archived_tasks)
        )
        result = {"tasks": len(archived_tasks), "output": str(args.output)}
    elif args.group == "data" and args.action == "merge-denylists":
        result = merge_denylists(args.inputs, args.output)
    elif args.group == "train" and args.action in {"sft", "solver-grpo"}:
        result = _launch_stage(args.action, args.config, dry_run=args.dry_run)
    elif args.group == "train":
        result = run_self_play(
            args.config, args.output_dir, resume=args.resume, dry_run=args.dry_run
        )
    elif args.group == "evaluate" and args.action == "benchmark":
        result = asyncio.run(
            evaluate_benchmark(
                args.input,
                args.output_dir,
                solver_url=args.solver_url,
                graph_snapshot=args.snapshot,
                samples=args.samples,
                concurrency=args.concurrency,
                graphscript_version=args.graphscript_version,
            )
        )
    elif args.group == "evaluate" and args.action == "kqapro-compare":
        result = compare_kqapro_val_metrics(args.metrics, output_path=args.output)
    elif args.group == "evaluate":
        val_config = _load_kqapro_val_config(args.config)
        result = asyncio.run(
            evaluate_kqapro_val(
                args.input or val_config.input_path,
                args.output_dir or Path(f"outputs/evaluation/kqapro-{args.model_stage}"),
                val_config,
                model_stage=args.model_stage,
                limit=args.limit,
            )
        )
    else:
        val_config = _load_kqapro_val_config(args.config)
        selected_indices = _parse_indices(args.indices)
        input_path = args.input or val_config.input_path
        if args.inspect_only:
            result = {
                "dataset_preview": inspect_kqapro_val(
                    input_path,
                    val_config,
                    limit=args.limit,
                    selected_indices=selected_indices,
                ),
                "models_called": False,
            }
        else:
            result = asyncio.run(
                visualize_kqapro_val(
                    input_path,
                    args.output_dir
                    or Path(f"outputs/visualization/kqapro-{args.model_stage}"),
                    val_config,
                    model_stage=args.model_stage,
                    limit=args.limit,
                    selected_indices=selected_indices,
                )
            )
    LOGGER.info(
        "command_completed group=%s action=%s elapsed_s=%.1f",
        args.group,
        args.action,
        perf_counter() - command_started,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
