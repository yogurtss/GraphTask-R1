from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from graphtask_r1.graph import toy_graph
from graphtask_r1.pipeline import run_mini_pipeline
from graphtask_r1.schema import TaskCertificate
from graphtask_r1.training.selfplay import run_orchestration_smoke
from graphtask_r1.training.verl_dataset import export_role_dataset
from graphtask_r1.utils import read_records


def _bool(value: str) -> bool:
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphtask-r1")
    groups = parser.add_subparsers(dest="group", required=True)

    graph = groups.add_parser("graph")
    graph_actions = graph.add_subparsers(dest="action", required=True)
    smoke = graph_actions.add_parser("smoke-test")
    smoke.add_argument("--config", default="configs/graph/toy.yaml")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--limit", type=int, default=100)
    smoke.add_argument("--dry-run", action="store_true")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--output-dir", type=Path, default=Path("outputs/graph-smoke"))

    e2e = groups.add_parser("e2e")
    e2e_actions = e2e.add_subparsers(dest="action", required=True)
    mini = e2e_actions.add_parser("mini-pipeline")
    mini.add_argument("--graph", choices=["toy"], default="toy")
    mini.add_argument("--num-programs", type=int, default=100)
    mini.add_argument("--limit", type=int)
    mini.add_argument("--seed", type=int, default=42)
    mini.add_argument("--dry-run", action="store_true")
    mini.add_argument("--resume", action="store_true")
    mini.add_argument("--output-dir", type=Path, required=True)

    data = groups.add_parser("data")
    data_actions = data.add_subparsers(dest="action", required=True)
    export = data_actions.add_parser("export-verl")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--roles", choices=["both", "questioner", "solver"], default="both")
    export.add_argument("--limit", type=int)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--dry-run", action="store_true")
    export.add_argument("--resume", action="store_true")
    export.add_argument("--output-dir", type=Path)

    train = groups.add_parser("train")
    train_actions = train.add_subparsers(dest="action", required=True)
    selfplay = train_actions.add_parser("mini-self-play")
    selfplay.add_argument("--graph", choices=["toy"], default="toy")
    selfplay.add_argument("--model", required=True)
    selfplay.add_argument("--shared-policy", type=_bool, default=True)
    selfplay.add_argument("--rounds", type=int, default=3)
    selfplay.add_argument("--questioner-groups", type=int, default=16)
    selfplay.add_argument("--solver-episodes", type=int, default=64)
    selfplay.add_argument("--limit", type=int)
    selfplay.add_argument("--seed", type=int, default=42)
    selfplay.add_argument("--dry-run", action="store_true")
    selfplay.add_argument("--resume", action="store_true")
    selfplay.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: object
    if args.group == "graph":
        backend = toy_graph()
        result = {
            "triples": len(backend.triples),
            "entities": len({t.subject for t in backend.triples}),
        }
    elif args.group == "e2e":
        count = args.limit if args.limit is not None else args.num_programs
        result = run_mini_pipeline(
            args.output_dir, num_programs=count, seed=args.seed, dry_run=args.dry_run
        )
    elif args.group == "data":
        records = read_records(args.input)
        if args.limit is not None:
            records = records[: args.limit]
        tasks = [TaskCertificate.model_validate(record) for record in records]
        if args.dry_run:
            result = {"would_export": len(tasks), "roles": args.roles}
        else:
            rows = export_role_dataset(
                tasks,
                args.output,
                include_questioner=args.roles in {"both", "questioner"},
                include_solver=args.roles in {"both", "solver"},
            )
            result = {"rows": rows, "output": str(args.output)}
    else:
        if not args.shared_policy:
            raise SystemExit("the first-version protocol requires --shared-policy true")
        if args.dry_run:
            result = vars(args)
        else:
            result = run_orchestration_smoke(
                args.output_dir,
                rounds=args.rounds,
                questioner_groups=args.questioner_groups,
                solver_episodes=args.solver_episodes,
                seed=args.seed,
                model=args.model,
                resume=args.resume,
            )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
