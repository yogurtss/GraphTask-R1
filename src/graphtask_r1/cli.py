from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from graphtask_r1.data import (
    audit_records,
    merge_denylists,
    prepare_benchmark,
    prepare_kqapro,
    sample_questioner_seeds,
)
from graphtask_r1.evaluation import evaluate_benchmark
from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.pipeline import run_mini_pipeline
from graphtask_r1.schema import TaskCertificate
from graphtask_r1.training.scripted import run_scripted_selfplay
from graphtask_r1.training.selfplay import run_self_play
from graphtask_r1.training.sft_dataset import export_sft_dataset
from graphtask_r1.training.verl_dataset import export_role_dataset
from graphtask_r1.utils import read_records


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphtask-r1")
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
    _add_common(export)

    export_sft = data_actions.add_parser("export-sft")
    export_sft.add_argument("--input", type=Path, required=True)
    export_sft.add_argument("--output", type=Path, required=True)
    export_sft.add_argument("--roles", choices=["both", "questioner", "solver"], default="both")
    _add_common(export_sft)

    seeds = data_actions.add_parser("sample-seeds")
    seeds.add_argument("--snapshot", default="freebase-v1")
    seeds.add_argument("--output", type=Path, required=True)
    seeds.add_argument("--count", type=int, default=1024)
    seeds.add_argument("--exclude", type=Path)
    seeds.add_argument("--pool-limit", type=int, default=100_000)
    seeds.add_argument("--opponent-url")
    seeds.add_argument("--opponent-samples", type=int, default=8)
    seeds.add_argument("--seed", type=int, default=42)
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
    return [TaskCertificate.model_validate(value) for value in values]


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
            )
        else:
            result = prepare_benchmark(args.dataset, args.raw_dir, args.output_dir)
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
        )
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
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
