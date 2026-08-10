from __future__ import annotations

import os
import random
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphtask_r1.archive import TaskArchive
from graphtask_r1.schema import TaskCertificate
from graphtask_r1.training.prompts import role_prompt
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.training.verl_dataset import export_role_dataset, tool_kwargs
from graphtask_r1.utils import file_hash, read_json, read_records, write_json

VERL_COMMIT = "bec9ef74768dd201881cd4e54cd0385e87caae27"


class SelfPlayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    initial_adapter: str
    base_tasks: Path
    val_data: Path
    questioner_seeds: Path
    graph_snapshot: str = "kqapro-v1"
    rounds: int = Field(default=3, gt=0)
    solver_episodes: int = Field(default=256, gt=0)
    opponent_samples: int = Field(default=8, gt=0)
    base_ratio: float = 0.35
    archive_ratio: float = 0.35
    new_ratio: float = 0.30
    seed: int = 42
    actor_gpus: str = "0,1"
    opponent_gpus: str = "2,3"
    sglang_port: int = 30000
    opponent_port: int = 18080
    train_script: Path = Path("scripts/train_verl.sh")
    sglang_start_timeout_s: int = 300
    interaction_mode: Literal["tool", "graphscript"] = "tool"
    relation_catalog: Path | None = None
    relation_catalogs: dict[str, Path] = Field(default_factory=dict)
    max_follow_limit: int = Field(default=100, gt=0, le=1_000)
    max_edge_visits: int = Field(default=200, gt=0)
    max_returned_entities: int = Field(default=1_000, gt=0)
    program_profile: Literal["full", "graphscript_v0_1"] = "full"

    @model_validator(mode="after")
    def validate_interaction_contract(self) -> SelfPlayConfig:
        if self.program_profile == "graphscript_v0_1" and self.relation_catalog is None:
            raise ValueError("comparison profile requires relation_catalog")
        if self.interaction_mode == "graphscript" and self.program_profile != "graphscript_v0_1":
            raise ValueError("graphscript mode requires program_profile=graphscript_v0_1")
        return self


def load_selfplay_config(path: Path) -> SelfPlayConfig:
    raw = os.path.expandvars(path.read_text())
    return SelfPlayConfig.model_validate(yaml.safe_load(raw))


def _sample(values: list[TaskCertificate], count: int, rng: random.Random) -> list[TaskCertificate]:
    if not values or count <= 0:
        return []
    if count <= len(values):
        return rng.sample(values, count)
    return [rng.choice(values) for _ in range(count)]


def _round_tasks(
    config: SelfPlayConfig, archive_path: Path, *, round_index: int
) -> list[TaskCertificate]:
    base = [TaskCertificate.model_validate(value) for value in read_records(config.base_tasks)]
    with TaskArchive(archive_path) as archive:
        archived = archive.all()
    new = [task for task in archived if task.generation.get("round") == round_index - 1]
    old = [task for task in archived if task not in new]
    rng = random.Random(config.seed + round_index)
    base_count = round(config.solver_episodes * config.base_ratio)
    archive_count = round(config.solver_episodes * config.archive_ratio)
    new_count = max(0, config.solver_episodes - base_count - archive_count)
    if not new:
        base_count += new_count
        new_count = 0
    if not old:
        base_count += archive_count
        archive_count = 0
    return [
        *_sample(base, base_count, rng),
        *_sample(old, archive_count, rng),
        *_sample(new, new_count, rng),
    ]


def _assemble_dataset(
    config: SelfPlayConfig,
    archive_path: Path,
    output_path: Path,
    *,
    round_index: int,
    opponent_url: str,
) -> dict[str, int]:
    relation_catalog = load_relation_catalog(config.relation_catalog)
    relation_catalogs = {
        snapshot: load_relation_catalog(path) for snapshot, path in config.relation_catalogs.items()
    }
    solver_tasks = _round_tasks(config, archive_path, round_index=round_index)
    solver_path = output_path.with_name("solver.parquet")
    export_role_dataset(
        solver_tasks,
        solver_path,
        include_questioner=False,
        include_solver=True,
        interaction_mode=config.interaction_mode,
        relation_catalog=relation_catalog,
        relation_catalogs=relation_catalogs,
        max_follow_limit=config.max_follow_limit,
        max_edge_visits=config.max_edge_visits,
        max_returned_entities=config.max_returned_entities,
        program_profile=config.program_profile,
    )
    seed_table = pq.read_table(config.questioner_seeds)
    seed_rows = seed_table.to_pylist()
    for row in seed_rows:
        extra = dict(row["extra_info"])
        topic_ids = [str(value) for value in extra.get("topic_entity_ids", [])]
        extra.update(
            {
                "opponent_url": opponent_url,
                "opponent_samples": config.opponent_samples,
                "round": round_index,
                "interaction_mode": config.interaction_mode,
                "allowed_relations": [value.relation_id for value in relation_catalog],
                "max_follow_limit": config.max_follow_limit,
                "max_edge_visits": config.max_edge_visits,
                "max_returned_entities": config.max_returned_entities,
                "program_profile": config.program_profile,
            }
        )
        row["extra_info"] = extra
        legacy_tool_mode = (
            config.interaction_mode == "tool"
            and config.program_profile == "full"
            and not relation_catalog
        )
        if legacy_tool_mode:
            continue
        row["prompt"] = role_prompt(
            "questioner",
            "Explore from these seed entities and construct one certified task: "
            + ", ".join(topic_ids),
            interaction_mode=config.interaction_mode,
            relation_catalog=relation_catalog,
        )
        if config.interaction_mode == "tool":
            row["agent_name"] = "tool_agent"
            questioner_tools = tool_kwargs(
                ("graph_search", "inspect_entity", "execute_program"),
                extra,
                role="questioner",
            )
            row["tools_kwargs"] = questioner_tools
            extra["need_tools_kwargs"] = True
            extra["tools_kwargs"] = questioner_tools
            row["extra_info"] = extra
        else:
            row.pop("agent_name", None)
            row.pop("tools_kwargs", None)
    questioner_table = pa.Table.from_pylist(seed_rows)
    solver_table = pq.read_table(solver_path)
    combined = pa.concat_tables([questioner_table, solver_table], promote_options="default")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, output_path)
    return {"questioner": len(seed_rows), "solver": len(solver_tasks), "total": len(combined)}


def _wait_for(url: str, *, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"service did not become healthy: {url}") from last_error


def _adapter_from_checkpoint(round_dir: Path) -> Path:
    candidates = sorted(
        round_dir.rglob("adapter_model.safetensors"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"verl did not emit a LoRA adapter under {round_dir}")
    return candidates[-1].parent


def _stop(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _commands(
    config: SelfPlayConfig,
    *,
    adapter: Path,
    archive_path: Path,
    mixed_data: Path,
    round_dir: Path,
) -> dict[str, list[str]]:
    model_name = "frozen-opponent"
    sglang = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        config.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.sglang_port),
        "--tp-size",
        "1",
        "--dp-size",
        str(len(config.opponent_gpus.split(","))),
        "--enable-lora",
        "--lora-paths",
        f"{model_name}={adapter}",
    ]
    opponent = [
        "python",
        "-m",
        "graphtask_r1.training.opponent",
        "--model-url",
        f"http://127.0.0.1:{config.sglang_port}",
        "--model",
        model_name,
        "--archive",
        str(archive_path),
        "--port",
        str(config.opponent_port),
        "--interaction-mode",
        config.interaction_mode,
        "--max-follow-limit",
        str(config.max_follow_limit),
    ]
    if config.interaction_mode == "graphscript" or config.program_profile == "graphscript_v0_1":
        opponent.extend(["--max-edge-visits", str(config.max_edge_visits)])
    if config.relation_catalog is not None:
        opponent.extend(["--relation-catalog", str(config.relation_catalog)])
    train = ["bash", str(config.train_script)]
    return {"sglang": sglang, "opponent": opponent, "train": train}


def run_self_play(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    config = load_selfplay_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "archive.sqlite"
    manifest_path = output_dir / "manifest.json"
    completed = 0
    adapter = Path(config.initial_adapter)
    if resume and manifest_path.exists():
        state = read_json(manifest_path)
        completed = int(state["last_completed_round"])
        adapter = Path(state["adapter"])
        if state["config_hash"] != file_hash(config_path):
            raise ValueError("cannot resume: self-play config changed")
    plans: list[dict[str, Any]] = []
    for round_index in range(completed + 1, config.rounds + 1):
        round_dir = output_dir / f"round_{round_index:03d}"
        mixed_data = round_dir / "mixed.parquet"
        opponent_url = f"http://127.0.0.1:{config.opponent_port}"
        counts = (
            {"questioner": 0, "solver": 0, "total": 0}
            if dry_run
            else _assemble_dataset(
                config,
                archive_path,
                mixed_data,
                round_index=round_index,
                opponent_url=opponent_url,
            )
        )
        commands = _commands(
            config,
            adapter=adapter,
            archive_path=archive_path,
            mixed_data=mixed_data,
            round_dir=round_dir,
        )
        plan = {
            "round": round_index,
            "adapter_in": str(adapter),
            "dataset": str(mixed_data),
            "counts": counts,
            "commands": commands,
            "actor_gpus": config.actor_gpus,
            "opponent_gpus": config.opponent_gpus,
            "interaction_mode": config.interaction_mode,
            "relation_catalog": str(config.relation_catalog)
            if config.relation_catalog is not None
            else None,
            "program_profile": config.program_profile,
        }
        plans.append(plan)
        if dry_run:
            continue
        round_dir.mkdir(parents=True, exist_ok=True)
        write_json(round_dir / "plan.json", plan)
        logs = round_dir / "logs"
        logs.mkdir(exist_ok=True)
        opponent_env = {**os.environ, "CUDA_VISIBLE_DEVICES": config.opponent_gpus}
        with (logs / "sglang.log").open("w") as sglang_log:
            sglang_process = subprocess.Popen(
                commands["sglang"], env=opponent_env, stdout=sglang_log, stderr=subprocess.STDOUT
            )
        try:
            _wait_for(
                f"http://127.0.0.1:{config.sglang_port}/health",
                timeout_s=config.sglang_start_timeout_s,
            )
            with (logs / "opponent.log").open("w") as opponent_log:
                opponent_process = subprocess.Popen(
                    commands["opponent"],
                    env=opponent_env,
                    stdout=opponent_log,
                    stderr=subprocess.STDOUT,
                )
            try:
                _wait_for(opponent_url + "/health", timeout_s=60)
                train_env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": config.actor_gpus,
                    "MODEL_PATH": config.model_path,
                    "LORA_ADAPTER_PATH": str(adapter),
                    "TRAIN_DATA": str(mixed_data.resolve()),
                    "VAL_DATA": str(config.val_data.resolve()),
                    "NUM_GPUS": str(len(config.actor_gpus.split(","))),
                    "OUTPUT_DIR": str(round_dir.resolve()),
                    "EXPERIMENT_NAME": f"graphtask-selfplay-r{round_index:03d}",
                    "SAVE_FREQ": "1",
                }
                with (logs / "verl.log").open("w") as train_log:
                    subprocess.run(
                        commands["train"],
                        env=train_env,
                        stdout=train_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            finally:
                _stop(opponent_process)
        finally:
            _stop(sglang_process)
        adapter = _adapter_from_checkpoint(round_dir)
        write_json(
            round_dir / "manifest.json",
            {
                "round": round_index,
                "adapter": str(adapter),
                "dataset_hash": file_hash(mixed_data),
                "completed": True,
            },
        )
        write_json(
            manifest_path,
            {
                "last_completed_round": round_index,
                "adapter": str(adapter),
                "config_hash": file_hash(config_path),
                "verl_commit": VERL_COMMIT,
            },
        )
    return {
        "dry_run": dry_run,
        "rounds_requested": config.rounds,
        "rounds_completed": completed if dry_run else config.rounds,
        "plans": plans,
        "verl_commit": VERL_COMMIT,
    }
