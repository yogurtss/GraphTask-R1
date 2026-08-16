from __future__ import annotations

import os
import random
import re
import subprocess
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphtask_r1.archive import TaskArchive
from graphtask_r1.schema import TaskCertificate, TaskTrainingRecord
from graphtask_r1.training.prompts import GraphScriptVersion, role_prompt
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.training.rl_dataset import export_role_dataset
from graphtask_r1.utils import file_hash, read_json, read_records, write_json

MS_SWIFT_VERSION = "3.6.4"
SelfPlayTask = TaskCertificate | TaskTrainingRecord
TaskT = TypeVar("TaskT")


def _gpu_ids(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


class SelfPlayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_type: str = "qwen3"
    initial_adapter: str
    base_tasks: Path
    val_data: Path
    questioner_seeds: Path
    graph_snapshot: str = "kqapro-v1"
    rounds: int = Field(default=3, gt=0)
    questioner_episodes: int = Field(default=256, gt=0, le=4_096)
    solver_episodes: int = Field(default=256, gt=0)
    opponent_samples: int = Field(default=4, gt=0, le=64)
    base_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    archive_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    new_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    seed: int = 42
    actor_gpus: str = "0,1,2"
    opponent_gpus: str = "3"
    allow_gpu_overlap: bool = False
    use_vllm: bool = True
    opponent_backend: Literal["sglang", "transformers"] = "sglang"
    opponent_device: str = "cuda:0"
    sglang_port: int = 30000
    opponent_port: int = 18080
    train_script: Path = Path("scripts/train_ms_swift_grpo.sh")
    sglang_start_timeout_s: int = 300
    interaction_mode: Literal["tool", "graphscript"] = "graphscript"
    graphscript_version: GraphScriptVersion = "0.3"
    relation_catalog: Path | None = None
    relation_catalogs: dict[str, Path] = Field(default_factory=dict)
    max_follow_limit: int = Field(default=100, gt=0, le=1_000)
    max_edge_visits: int = Field(default=200, gt=0)
    max_returned_entities: int = Field(default=1_000, gt=0)
    max_completion_tokens: int = Field(default=4_096, gt=0, le=40_960)
    vllm_max_model_len: int = Field(default=16_384, gt=0, le=40_960)
    vllm_gpu_memory_utilization: float = Field(default=0.6, gt=0.0, lt=1.0)
    vllm_sleep_level: Literal[0, 1, 2] = 1
    micro_batch_size: int = Field(default=4, gt=0)
    eval_batch_size: int = Field(default=8, gt=0)
    gradient_accumulation_steps: int = Field(default=2, gt=0)
    steps_per_generation: int = Field(default=4, gt=0)
    rollout_n: int = Field(default=4, gt=0)
    program_profile: Literal[
        "full", "graphscript_v0_1", "graphscript_v0_2", "graphscript_v0_3"
    ] = "graphscript_v0_3"

    @model_validator(mode="after")
    def validate_interaction_contract(self) -> SelfPlayConfig:
        if self.program_profile == "graphscript_v0_1" and self.relation_catalog is None:
            raise ValueError("comparison profile requires relation_catalog")
        expected_profile = (
            "graphscript_v" + self.graphscript_version.replace(".", "_")
            if self.interaction_mode == "graphscript"
            else "full"
        )
        if self.program_profile != expected_profile:
            raise ValueError(
                f"{self.interaction_mode} mode requires "
                f"program_profile={expected_profile}"
            )
        if abs(self.base_ratio + self.archive_ratio + self.new_ratio - 1.0) > 1e-9:
            raise ValueError("base_ratio, archive_ratio, and new_ratio must sum to 1")
        if self.opponent_backend == "transformers" and self.opponent_samples != 1:
            raise ValueError("the deterministic Transformers opponent requires opponent_samples=1")
        if self.vllm_max_model_len <= self.max_completion_tokens:
            raise ValueError(
                "vllm_max_model_len must exceed max_completion_tokens to leave room "
                "for the prompt"
            )
        actor_ids = _gpu_ids(self.actor_gpus)
        opponent_ids = _gpu_ids(self.opponent_gpus)
        if not actor_ids or not opponent_ids:
            raise ValueError("actor_gpus and opponent_gpus must each select at least one GPU")
        if len(set(actor_ids)) != len(actor_ids) or len(set(opponent_ids)) != len(
            opponent_ids
        ):
            raise ValueError("actor_gpus and opponent_gpus cannot contain duplicate GPU IDs")
        overlap = sorted(set(actor_ids) & set(opponent_ids))
        if overlap and not self.allow_gpu_overlap:
            raise ValueError(
                "actor_gpus and opponent_gpus must be disjoint; overlap: "
                + ", ".join(overlap)
            )
        if overlap and (self.use_vllm or self.opponent_backend != "transformers"):
            raise ValueError(
                "overlapping GPUs require use_vllm=false and "
                "opponent_backend=transformers"
            )
        actor_count = len(actor_ids)
        generation_batch = actor_count * self.micro_batch_size * self.steps_per_generation
        evaluation_batch = actor_count * self.eval_batch_size
        if self.steps_per_generation % self.gradient_accumulation_steps:
            raise ValueError(
                "steps_per_generation must be an integer multiple of "
                "gradient_accumulation_steps"
            )
        if generation_batch % self.rollout_n:
            raise ValueError("self-play generation batch must be divisible by rollout_n")
        if evaluation_batch % self.rollout_n:
            raise ValueError("self-play evaluation batch must be divisible by rollout_n")
        return self


def load_selfplay_config(path: Path) -> SelfPlayConfig:
    raw = os.path.expandvars(path.read_text())
    return SelfPlayConfig.model_validate(yaml.safe_load(raw))


def _sample(values: Sequence[TaskT], count: int, rng: random.Random) -> list[TaskT]:
    if not values or count <= 0:
        return []
    if count <= len(values):
        return rng.sample(values, count)
    return [rng.choice(values) for _ in range(count)]


def _round_tasks(
    config: SelfPlayConfig, archive_path: Path, *, round_index: int
) -> list[SelfPlayTask]:
    # ``base_tasks`` normally points at the compact training view emitted by
    # ``data audit --training-view-output``. Archive entries remain full certificates.
    base = [
        TaskTrainingRecord.model_validate(value) for value in read_records(config.base_tasks)
    ]
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
    selected: list[SelfPlayTask] = []
    selected.extend(_sample(base, base_count, rng))
    selected.extend(_sample(old, archive_count, rng))
    selected.extend(_sample(new, new_count, rng))
    return selected


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
        graphscript_version=config.graphscript_version,
        relation_catalog=relation_catalog,
        relation_catalogs=relation_catalogs,
        max_follow_limit=config.max_follow_limit,
        max_edge_visits=config.max_edge_visits,
        max_returned_entities=config.max_returned_entities,
        program_profile=config.program_profile,
    )
    seed_table = pq.read_table(config.questioner_seeds)
    seed_rows = seed_table.to_pylist()
    questioner_rng = random.Random(config.seed + 10_000 + round_index)
    seed_rows = questioner_rng.sample(
        seed_rows, k=min(config.questioner_episodes, len(seed_rows))
    )
    for row in seed_rows:
        extra = dict(row["extra_info"])
        topic_ids = [str(value) for value in extra.get("topic_entity_ids", [])]
        extra.update(
            {
                "opponent_url": opponent_url,
                "opponent_samples": config.opponent_samples,
                "round": round_index,
                "interaction_mode": config.interaction_mode,
                "graphscript_version": config.graphscript_version,
                "allowed_relations": [value.relation_id for value in relation_catalog],
                "max_follow_limit": config.max_follow_limit,
                "max_edge_visits": config.max_edge_visits,
                "max_returned_entities": config.max_returned_entities,
                "program_profile": config.program_profile,
            }
        )
        row["extra_info"] = extra
        row["prompt"] = role_prompt(
            "questioner",
            "Explore from these seed entities and construct one certified task: "
            + ", ".join(topic_ids),
            interaction_mode=config.interaction_mode,
            relation_catalog=relation_catalog,
            graphscript_version=config.graphscript_version,
        )
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


def _checkpoint_step(path: Path) -> int:
    for parent in path.parents:
        match = re.fullmatch(r"checkpoint-(\d+)", parent.name)
        if match:
            return int(match.group(1))
        if parent.name.startswith("round_"):
            break
    return -1


def _adapter_from_checkpoint(
    round_dir: Path, *, known_adapters: frozenset[Path] = frozenset()
) -> Path:
    candidates = [
        path
        for path in round_dir.rglob("adapter_model.safetensors")
        if path.resolve() not in known_adapters
    ]
    if not candidates:
        raise RuntimeError(f"ms-swift did not emit a new LoRA adapter under {round_dir}")
    return max(candidates, key=lambda path: (_checkpoint_step(path), str(path))).parent


def _validate_merged_model(model_dir: Path) -> None:
    missing = [
        name
        for name in ("config.json", "tokenizer_config.json")
        if not (model_dir / name).is_file()
    ]
    has_weights = any(model_dir.glob("model*.safetensors")) or (
        model_dir / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        missing.append("model*.safetensors")
    if missing:
        raise RuntimeError(
            f"merged opponent model is incomplete under {model_dir}; missing: "
            + ", ".join(missing)
        )


def _opponent_merge_spec(config: SelfPlayConfig, adapter: Path) -> dict[str, str]:
    adapter_config = adapter / "adapter_config.json"
    adapter_weights = adapter / "adapter_model.safetensors"
    missing = [
        str(path)
        for path in (adapter_config, adapter_weights)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError("cannot merge incomplete LoRA adapter; missing: " + ", ".join(missing))
    adapter_config_hash = file_hash(adapter_config)
    adapter_weights_hash = file_hash(adapter_weights)
    if adapter_config_hash is None or adapter_weights_hash is None:
        raise RuntimeError(f"cannot hash LoRA adapter under {adapter}")
    return {
        "base_model": config.model_path,
        "adapter": str(adapter.resolve()),
        "adapter_config_sha256": adapter_config_hash,
        "adapter_weights_sha256": adapter_weights_hash,
    }


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
    merged_model = (round_dir / "opponent_merged").resolve()
    merge = [
        "swift",
        "export",
        "--model",
        config.model_path,
        "--model_type",
        config.model_type,
        "--adapters",
        str(adapter.resolve()),
        "--train_type",
        "lora",
        "--torch_dtype",
        "bfloat16",
        "--load_args",
        "false",
        "--merge_lora",
        "true",
        "--output_dir",
        str(merged_model),
    ]
    sglang = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(merged_model),
        "--host",
        "127.0.0.1",
        "--port",
        str(config.sglang_port),
        "--tp-size",
        "1",
        "--dp-size",
        str(len(_gpu_ids(config.opponent_gpus))),
    ]
    opponent = [
        "python",
        "-m",
        "graphtask_r1.training.opponent",
        "--model-url",
        f"http://127.0.0.1:{config.sglang_port}",
        "--model",
        str(merged_model),
        "--archive",
        str(archive_path),
        "--port",
        str(config.opponent_port),
        "--interaction-mode",
        config.interaction_mode,
        "--graphscript-version",
        config.graphscript_version,
        "--max-follow-limit",
        str(config.max_follow_limit),
        "--max-completion-tokens",
        str(config.max_completion_tokens),
    ]
    if config.opponent_backend == "transformers":
        sglang = []
        model_url_index = opponent.index("--model-url")
        del opponent[model_url_index : model_url_index + 2]
        opponent.extend(
            [
                "--local-model",
                "--device",
                config.opponent_device,
            ]
        )
    if config.interaction_mode == "graphscript" or config.program_profile == "graphscript_v0_1":
        opponent.extend(["--max-edge-visits", str(config.max_edge_visits)])
    if config.relation_catalog is not None:
        opponent.extend(["--relation-catalog", str(config.relation_catalog)])
    train = ["bash", str(config.train_script)]
    return {"merge": merge, "sglang": sglang, "opponent": opponent, "train": train}


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
        train_overrides = {
            "CUDA_VISIBLE_DEVICES": config.actor_gpus,
            "TRAIN_CUDA_VISIBLE_DEVICES": config.actor_gpus,
            "MODEL_PATH": config.model_path,
            "MODEL_TYPE": config.model_type,
            "LORA_ADAPTER_PATH": str(adapter),
            "TRAIN_DATA": str(mixed_data.resolve()),
            "VAL_DATA": str(config.val_data.resolve()),
            "NUM_GPUS": str(len(_gpu_ids(config.actor_gpus))),
            "MICRO_BATCH_SIZE": str(config.micro_batch_size),
            "EVAL_BATCH_SIZE": str(config.eval_batch_size),
            "GRADIENT_ACCUMULATION_STEPS": str(config.gradient_accumulation_steps),
            "STEPS_PER_GENERATION": str(config.steps_per_generation),
            "ROLLOUT_N": str(config.rollout_n),
            "MAX_COMPLETION_LENGTH": str(config.max_completion_tokens),
            "VLLM_MAX_MODEL_LEN": str(config.vllm_max_model_len),
            "VLLM_GPU_MEMORY_UTILIZATION": str(config.vllm_gpu_memory_utilization),
            "VLLM_SLEEP_LEVEL": str(config.vllm_sleep_level),
            "USE_VLLM": str(config.use_vllm).lower(),
            "OUTPUT_DIR": str(round_dir.resolve()),
            "EXPERIMENT_NAME": f"graphtask-selfplay-r{round_index:03d}",
            "INTERACTION_MODE": config.interaction_mode,
            "GRAPHSCRIPT_VERSION": config.graphscript_version,
            "VLLM_MODE": "colocate",
            "SAVE_STEPS": "1",
        }
        plan = {
            "round": round_index,
            "adapter_in": str(adapter),
            "dataset": str(mixed_data),
            "counts": counts,
            "commands": commands,
            "train_environment": train_overrides,
            "actor_gpus": config.actor_gpus,
            "opponent_gpus": config.opponent_gpus,
            "allow_gpu_overlap": config.allow_gpu_overlap,
            "use_vllm": config.use_vllm,
            "opponent_backend": config.opponent_backend,
            "merged_opponent_model": str((round_dir / "opponent_merged").resolve()),
            "interaction_mode": config.interaction_mode,
            "graphscript_version": config.graphscript_version,
            "relation_catalog": str(config.relation_catalog)
            if config.relation_catalog is not None
            else None,
            "program_profile": config.program_profile,
            "max_completion_tokens": config.max_completion_tokens,
            "rollout_budget": {
                "questioner_prompts": (
                    counts["questioner"] if not dry_run else config.questioner_episodes
                ),
                "solver_prompts": counts["solver"] if not dry_run else config.solver_episodes,
                "actor_completions_upper_bound": (
                    (
                        counts["questioner"] + counts["solver"]
                        if not dry_run
                        else config.questioner_episodes + config.solver_episodes
                    )
                    * config.rollout_n
                ),
                "opponent_completions_upper_bound": (
                    (counts["questioner"] if not dry_run else config.questioner_episodes)
                    * config.rollout_n
                    * config.opponent_samples
                ),
            },
        }
        plans.append(plan)
        if dry_run:
            continue
        round_dir.mkdir(parents=True, exist_ok=True)
        known_adapters = frozenset(
            path.resolve() for path in round_dir.rglob("adapter_model.safetensors")
        )
        write_json(round_dir / "plan.json", plan)
        logs = round_dir / "logs"
        logs.mkdir(exist_ok=True)
        opponent_env = {**os.environ, "CUDA_VISIBLE_DEVICES": config.opponent_gpus}
        merged_model = (round_dir / "opponent_merged").resolve()
        merge_manifest_path = round_dir / "opponent_merge.json"
        merge_spec = _opponent_merge_spec(config, adapter)
        if merged_model.exists():
            if not merge_manifest_path.is_file() or read_json(merge_manifest_path) != merge_spec:
                raise RuntimeError(
                    f"refusing to reuse unverified merged opponent model at {merged_model}; "
                    "move or remove that directory before retrying"
                )
            _validate_merged_model(merged_model)
        else:
            merge_log_path = logs / "merge.log"
            with merge_log_path.open("w") as merge_log:
                try:
                    subprocess.run(
                        commands["merge"],
                        env=opponent_env,
                        stdout=merge_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise RuntimeError(
                        f"failed to merge frozen opponent model; see {merge_log_path}"
                    ) from exc
            _validate_merged_model(merged_model)
            write_json(merge_manifest_path, merge_spec)
        sglang_process: subprocess.Popen[Any] | None = None
        if commands["sglang"]:
            with (logs / "sglang.log").open("w") as sglang_log:
                sglang_process = subprocess.Popen(
                    commands["sglang"],
                    env=opponent_env,
                    stdout=sglang_log,
                    stderr=subprocess.STDOUT,
                )
        try:
            if sglang_process is not None:
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
                train_env = {**os.environ, **train_overrides}
                train_log_path = logs / "ms_swift.log"
                with train_log_path.open("w") as train_log:
                    try:
                        subprocess.run(
                            commands["train"],
                            env=train_env,
                            stdout=train_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    except (OSError, subprocess.CalledProcessError) as exc:
                        raise RuntimeError(
                            f"self-play GRPO training failed; see {train_log_path}"
                        ) from exc
            finally:
                _stop(opponent_process)
        finally:
            if sglang_process is not None:
                _stop(sglang_process)
        adapter = _adapter_from_checkpoint(round_dir, known_adapters=known_adapters)
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
                "ms_swift_version": MS_SWIFT_VERSION,
            },
        )
    return {
        "dry_run": dry_run,
        "rounds_requested": config.rounds,
        "rounds_completed": completed if dry_run else config.rounds,
        "plans": plans,
        "ms_swift_version": MS_SWIFT_VERSION,
    }
