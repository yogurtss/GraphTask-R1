from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from graphtask_r1.archive import TaskArchive
from graphtask_r1.data import (
    audit_records,
    merge_denylists,
    prepare_benchmark,
    prepare_kqapro,
    sample_questioner_seeds,
    select_graphscript_tasks,
)
from graphtask_r1.evaluation import evaluate_benchmark
from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.pipeline import run_mini_pipeline
from graphtask_r1.schema import TaskCertificate
from graphtask_r1.training.relations import build_relation_catalog, load_relation_catalog
from graphtask_r1.training.scripted import run_scripted_selfplay
from graphtask_r1.training.selfplay import run_self_play
from graphtask_r1.training.sft_dataset import export_sft_dataset
from graphtask_r1.training.verl_dataset import export_role_dataset
from graphtask_r1.utils import ProgressLogger, read_records, write_records

LOGGER = logging.getLogger("graphtask_r1.cli")
DEFAULT_DATA_WORKERS = 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")


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
        choices=["kqapro", "freebase", "webqsp", "cwq", "grailqa"],
        required=True,
    )
    fetch.add_argument("--raw-dir", type=Path, default=Path("data/raw"))

    prepare = data_actions.add_parser("prepare")
    prepare.add_argument("--dataset", choices=["kqapro", "webqsp", "cwq", "grailqa"], required=True)
    prepare.add_argument("--raw-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--splits", default="train,val")
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
    _add_common(prepare)

    audit = data_actions.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--kind", choices=["auto", "task", "benchmark"], default="auto")

    export = data_actions.add_parser("export-verl")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--roles", choices=["both", "questioner", "solver"], default="both")
    export.add_argument("--opponent-url")
    export.add_argument("--opponent-samples", type=int, default=8)
    export.add_argument("--interaction-mode", choices=["tool", "graphscript"], default="tool")
    export.add_argument("--relation-catalog", type=Path)
    _add_common(export)

    export_sft = data_actions.add_parser("export-sft")
    export_sft.add_argument("--input", type=Path, required=True)
    export_sft.add_argument("--output", type=Path, required=True)
    export_sft.add_argument("--roles", choices=["both", "questioner", "solver"], default="both")
    export_sft.add_argument(
        "--interaction-mode", choices=["tool", "graphscript"], default="tool"
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
    catalog.add_argument("--input", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--snapshot")
    catalog.add_argument("--limit", type=int)
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
    return parser


def _load_tasks(path: Path, limit: int | None) -> list[TaskCertificate]:
    values = read_records(path)
    if limit is not None:
        values = values[:limit]
    progress = ProgressLogger("data.load_tasks", total=len(values))
    progress.start(path=str(path))
    tasks: list[TaskCertificate] = []
    for index, value in enumerate(values):
        tasks.append(TaskCertificate.model_validate(value))
        progress.update(index + 1)
    progress.finish(len(tasks), path=str(path))
    return tasks


def _launch_stage(stage: str, config_path: Path, *, dry_run: bool) -> dict[str, Any]:
    config = yaml.safe_load(os.path.expandvars(config_path.read_text()))
    script = "scripts/train_sft.sh" if stage == "sft" else "scripts/train_verl.sh"
    env_keys = {
        "model_path": "MODEL_PATH",
        "train_data": "TRAIN_DATA",
        "val_data": "VAL_DATA",
        "output_dir": "OUTPUT_DIR",
        "num_gpus": "NUM_GPUS",
        "experiment_name": "EXPERIMENT_NAME",
        "lora_adapter_path": "LORA_ADAPTER_PATH",
    }
    selected_env = {
        target: str(config[source]) for source, target in env_keys.items() if source in config
    }
    result = {"stage": stage, "command": ["bash", script], "environment": selected_env}
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
            )
        else:
            result = prepare_benchmark(
                args.dataset, args.raw_dir, args.output_dir, workers=args.workers
            )
    elif args.group == "data" and args.action == "audit":
        result = audit_records(args.input, kind=args.kind)
    elif args.group == "data" and args.action == "export-verl":
        tasks = _load_tasks(args.input, args.limit)
        if args.dry_run:
            result = {"would_export": len(tasks), "roles": args.roles}
        else:
            rows = export_role_dataset(
                tasks,
                args.output,
                include_questioner=args.roles in {"both", "questioner"},
                include_solver=args.roles in {"both", "solver"},
                opponent_url=args.opponent_url,
                opponent_samples=args.opponent_samples,
                interaction_mode=args.interaction_mode,
                relation_catalog=load_relation_catalog(args.relation_catalog),
            )
            result = {"rows": rows, "output": str(args.output)}
    elif args.group == "data" and args.action == "export-sft":
        tasks = _load_tasks(args.input, args.limit)
        if args.dry_run:
            result = {"would_export": len(tasks), "roles": args.roles}
        else:
            rows = export_sft_dataset(
                tasks,
                args.output,
                include_questioner=args.roles in {"both", "questioner"},
                include_solver=args.roles in {"both", "solver"},
                seed=args.seed,
                interaction_mode=args.interaction_mode,
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
            relation_catalog=load_relation_catalog(args.relation_catalog),
        )
    elif args.group == "data" and args.action == "select-interaction-tasks":
        tasks = _load_tasks(args.input, args.limit)
        if not tasks:
            raise ValueError("cannot select interaction tasks from an empty task set")
        snapshot = args.snapshot or tasks[0].graph_snapshot
        mismatched = sorted(
            {task.graph_snapshot for task in tasks if task.graph_snapshot != snapshot}
        )
        if mismatched:
            raise ValueError(
                "interaction task input contains snapshots other than "
                f"{snapshot}: {', '.join(mismatched)}"
            )
        result = select_graphscript_tasks(
            tasks,
            args.output,
            backend=backend_from_snapshot(snapshot),
            max_follow_limit=args.max_follow_limit,
            max_edge_visits=args.max_edge_visits,
            max_returned_entities=args.max_returned_entities,
        )
    elif args.group == "data" and args.action == "build-relation-catalog":
        tasks = _load_tasks(args.input, args.limit)
        if not tasks:
            raise ValueError("cannot build a relation catalog from an empty task set")
        snapshot = args.snapshot or tasks[0].graph_snapshot
        mismatched = sorted(
            {task.graph_snapshot for task in tasks if task.graph_snapshot != snapshot}
        )
        if mismatched:
            raise ValueError(
                "relation catalog input contains snapshots other than "
                f"{snapshot}: {', '.join(mismatched)}"
            )
        relations = build_relation_catalog(tasks, backend_from_snapshot(snapshot), args.output)
        result = {"relations": len(relations), "output": str(args.output), "snapshot": snapshot}
    elif args.group == "data" and args.action == "export-archive":
        if not args.archive.exists():
            raise FileNotFoundError(args.archive)
        with TaskArchive(args.archive) as archive:
            tasks = archive.all()
        write_records(args.output, (task.model_dump(mode="json") for task in tasks))
        result = {"tasks": len(tasks), "output": str(args.output)}
    elif args.group == "data" and args.action == "merge-denylists":
        result = merge_denylists(args.inputs, args.output)
    elif args.group == "train" and args.action in {"sft", "solver-grpo"}:
        result = _launch_stage(args.action, args.config, dry_run=args.dry_run)
    elif args.group == "train":
        result = run_self_play(
            args.config, args.output_dir, resume=args.resume, dry_run=args.dry_run
        )
    else:
        result = asyncio.run(
            evaluate_benchmark(
                args.input,
                args.output_dir,
                solver_url=args.solver_url,
                graph_snapshot=args.snapshot,
                samples=args.samples,
                concurrency=args.concurrency,
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
