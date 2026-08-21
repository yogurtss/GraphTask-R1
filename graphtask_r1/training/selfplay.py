from __future__ import annotations

import logging
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphtask_r1.archive import TaskArchive, promote_staged_tasks
from graphtask_r1.dsl import program_cost
from graphtask_r1.schema import TaskCertificate, TaskTrainingRecord
from graphtask_r1.training.prompts import GraphScriptVersion, role_prompt
from graphtask_r1.training.questioner_context import render_questioner_seed_payload
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.training.rl_dataset import export_role_dataset
from graphtask_r1.training.selfplay_metrics import (
    find_trainer_log,
    summarize_selfplay_round,
    write_selfplay_report,
)
from graphtask_r1.utils import file_hash, read_json, read_records, write_json

MS_SWIFT_VERSION = "3.10.3"
LOGGER = logging.getLogger(__name__)
SelfPlayTask = TaskCertificate | TaskTrainingRecord
TaskT = TypeVar("TaskT")
DeepSpeedStage = Literal[
    "none",
    "zero0",
    "zero1",
    "zero2",
    "zero3",
    "zero2_offload",
    "zero3_offload",
]
RLAlgorithm = Literal["grpo", "reinforce_plus_plus"]
SelfPlayVariant = Literal["legacy", "frontier_v2", "curriculum_v3"]
CurriculumPhase = Literal["production", "grounding", "frontier"]
UpdatePhase = Literal["questioner", "solver"]


def _gpu_ids(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


class SelfPlayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_type: str = "qwen3"
    response_prefix: str | None = None
    initial_adapter: str
    base_tasks: Path
    val_data: Path
    questioner_seeds: Path
    graph_snapshot: str = "kqapro-v1"
    selfplay_variant: SelfPlayVariant = "legacy"
    rounds: int = Field(default=3, gt=0)
    questioner_episodes: int = Field(default=256, gt=0, le=4_096)
    solver_episodes: int = Field(default=256, gt=0)
    questioner_reward_weight: float = Field(default=1.0, gt=0.0)
    solver_reward_weight: float = Field(default=1.0, gt=0.0)
    opponent_samples: int = Field(default=4, gt=0, le=64)
    frontier_target_start: float = Field(default=0.5, ge=0.0, le=1.0)
    frontier_target_end: float = Field(default=0.5, ge=0.0, le=1.0)
    frontier_sigma: float = Field(default=0.2, gt=0.0)
    archive_min_pass_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    archive_max_pass_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    archive_min_novelty: float = Field(default=0.25, ge=0.0, le=1.0)
    curriculum_production_rounds: int = Field(default=1, ge=0)
    curriculum_grounding_rounds: int = Field(default=1, ge=0)
    curriculum_solver_fraction_start: float = Field(default=0.4, gt=0.0, le=1.0)
    curriculum_replay_ratio: float = Field(default=0.3, ge=0.0, lt=1.0)
    curriculum_question_alignment_min: float = Field(default=0.35, ge=0.0, le=1.0)
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
    deepspeed: DeepSpeedStage = "none"
    rl_algorithm: RLAlgorithm = "grpo"
    micro_batch_size: int = Field(default=4, gt=0)
    eval_batch_size: int = Field(default=8, gt=0)
    validation_samples: int | None = Field(default=256, gt=0)
    gradient_accumulation_steps: int = Field(default=2, gt=0)
    steps_per_generation: int = Field(default=4, gt=0)
    rollout_n: int = Field(default=4, gt=0)
    save_steps: int = Field(default=20, gt=0)
    save_total_limit: int = Field(default=2, gt=0)
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
        if self.archive_min_pass_rate > self.archive_max_pass_rate:
            raise ValueError("archive_min_pass_rate cannot exceed archive_max_pass_rate")
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


def _frontier_target(config: SelfPlayConfig, round_index: int) -> float:
    if config.rounds == 1:
        return config.frontier_target_end
    progress = (round_index - 1) / (config.rounds - 1)
    return config.frontier_target_start + progress * (
        config.frontier_target_end - config.frontier_target_start
    )


def _curriculum_phase(config: SelfPlayConfig, round_index: int) -> CurriculumPhase:
    if round_index <= config.curriculum_production_rounds:
        return "production"
    if round_index <= config.curriculum_production_rounds + config.curriculum_grounding_rounds:
        return "grounding"
    return "frontier"


def _curriculum_progress(config: SelfPlayConfig, round_index: int) -> float:
    if config.rounds == 1:
        return 1.0
    return (round_index - 1) / (config.rounds - 1)


def _task_difficulty(task: SelfPlayTask) -> tuple[float, int, int, str]:
    return (
        program_cost(task.program),
        len(task.topic_entities),
        len(task.gold_answers.answers),
        task.task_id,
    )


def _curriculum_sample(
    values: Sequence[SelfPlayTask],
    count: int,
    rng: random.Random,
    *,
    visible_fraction: float,
    replay_ratio: float,
) -> list[SelfPlayTask]:
    """Draw mostly from the current structural frontier while replaying easier tasks."""
    if not values or count <= 0:
        return []
    ranked = sorted(values, key=_task_difficulty)
    visible_count = max(1, math.ceil(len(ranked) * visible_fraction))
    visible = ranked[:visible_count]
    split = max(1, len(visible) // 2)
    replay_pool = visible[:split]
    frontier_pool = visible[split:] or visible
    replay_count = round(count * replay_ratio)
    frontier_count = count - replay_count
    selected = _sample(frontier_pool, frontier_count, rng)
    selected.extend(_sample(replay_pool, replay_count, rng))
    rng.shuffle(selected)
    return selected


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
    new_round = round_index if config.selfplay_variant == "curriculum_v3" else round_index - 1
    new = [task for task in archived if task.generation.get("round") == new_round]
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
    if config.selfplay_variant == "curriculum_v3":
        progress = _curriculum_progress(config, round_index)
        visible_fraction = config.curriculum_solver_fraction_start + progress * (
            1.0 - config.curriculum_solver_fraction_start
        )

        def choose(values: Sequence[SelfPlayTask], count: int) -> list[SelfPlayTask]:
            return _curriculum_sample(
                values,
                count,
                rng,
                visible_fraction=visible_fraction,
                replay_ratio=config.curriculum_replay_ratio,
            )

    else:

        def choose(values: Sequence[SelfPlayTask], count: int) -> list[SelfPlayTask]:
            return _sample(values, count, rng)

    selected: list[SelfPlayTask] = []
    selected.extend(choose(base, base_count))
    selected.extend(choose(old, archive_count))
    selected.extend(choose(new, new_count))
    return selected


def _write_solver_dataset(
    config: SelfPlayConfig,
    archive_path: Path,
    output_path: Path,
    *,
    round_index: int,
) -> int:
    relation_catalog = load_relation_catalog(config.relation_catalog)
    relation_catalogs = {
        snapshot: load_relation_catalog(path) for snapshot, path in config.relation_catalogs.items()
    }
    solver_tasks = _round_tasks(config, archive_path, round_index=round_index)
    export_role_dataset(
        solver_tasks,
        output_path,
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
        questioner_weight=config.questioner_reward_weight,
        solver_weight=config.solver_reward_weight,
        solver_reward_variant=(
            "curriculum_v3" if config.selfplay_variant == "curriculum_v3" else "legacy"
        ),
        curriculum_phase=(
            _curriculum_phase(config, round_index)
            if config.selfplay_variant == "curriculum_v3"
            else None
        ),
    )
    return len(solver_tasks)


def _assemble_dataset(
    config: SelfPlayConfig,
    archive_path: Path,
    output_path: Path,
    *,
    round_index: int,
    opponent_url: str,
) -> dict[str, int]:
    relation_catalog = load_relation_catalog(config.relation_catalog)
    solver_path = output_path.with_name("solver.parquet")
    solver_count = _write_solver_dataset(
        config,
        archive_path,
        solver_path,
        round_index=round_index,
    )
    seed_table = pq.read_table(config.questioner_seeds)
    seed_rows = seed_table.to_pylist()
    questioner_rng = random.Random(config.seed + 10_000 + round_index)
    seed_rows = (
        _sample(seed_rows, config.questioner_episodes, questioner_rng)
        if config.selfplay_variant in {"frontier_v2", "curriculum_v3"}
        else questioner_rng.sample(
            seed_rows, k=min(config.questioner_episodes, len(seed_rows))
        )
    )
    for row in seed_rows:
        extra = dict(row["extra_info"])
        topic_ids = [str(value) for value in extra.get("topic_entity_ids", [])]
        raw_seed_context = extra.get("seed_context")
        prompt_relation_catalog = relation_catalog
        observed_relations: set[str] = set()
        if (
            config.selfplay_variant == "curriculum_v3"
            and isinstance(raw_seed_context, list)
        ):
            observed_relations = {
                str(relation_id)
                for context in raw_seed_context
                if isinstance(context, dict)
                for field in ("outgoing_relation_ids", "incoming_relation_ids")
                for relation_id in context.get(field, [])
            }
            local_catalog = tuple(
                relation
                for relation in relation_catalog
                if relation.relation_id in observed_relations
            )
            if local_catalog:
                prompt_relation_catalog = local_catalog
        extra.update(
            {
                "opponent_url": opponent_url,
                "opponent_samples": config.opponent_samples,
                "round": round_index,
                "interaction_mode": config.interaction_mode,
                "graphscript_version": config.graphscript_version,
                "allowed_relations": [
                    value.relation_id for value in relation_catalog
                ],
                "max_follow_limit": config.max_follow_limit,
                "max_edge_visits": config.max_edge_visits,
                "max_returned_entities": config.max_returned_entities,
                "program_profile": config.program_profile,
                "role_weight": config.questioner_reward_weight,
                "questioner_reward_variant": config.selfplay_variant,
                "frontier_target": _frontier_target(config, round_index),
                "frontier_sigma": config.frontier_sigma,
            }
        )
        if config.selfplay_variant in {"frontier_v2", "curriculum_v3"}:
            extra["opponent_seed"] = config.seed
        if config.selfplay_variant == "curriculum_v3":
            extra.update(
                {
                    "curriculum_phase": _curriculum_phase(config, round_index),
                    "question_alignment_min": config.curriculum_question_alignment_min,
                    "use_proposed_question": True,
                    "observed_relation_ids": sorted(observed_relations),
                }
            )
        row["extra_info"] = extra
        payload = (
            render_questioner_seed_payload(raw_seed_context)
            if isinstance(raw_seed_context, list) and raw_seed_context
            else "Explore from these seed entities and construct one certified task: "
            + ", ".join(topic_ids)
        )
        row["prompt"] = role_prompt(
            "questioner",
            payload,
            interaction_mode=config.interaction_mode,
            relation_catalog=prompt_relation_catalog,
            graphscript_version=config.graphscript_version,
            questioner_contract=(
                "question_program" if config.selfplay_variant == "curriculum_v3" else "program"
            ),
        )
        row.pop("agent_name", None)
        row.pop("tools_kwargs", None)
    questioner_table = pa.Table.from_pylist(seed_rows)
    questioner_path = output_path.with_name("questioner.parquet")
    pq.write_table(questioner_table, questioner_path)
    solver_table = pq.read_table(solver_path)
    combined = pa.concat_tables([questioner_table, solver_table], promote_options="default")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, output_path)
    return {"questioner": len(seed_rows), "solver": solver_count, "total": len(combined)}


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


def _completed_phase_adapter(phase_dir: Path) -> Path | None:
    """Return a certified final adapter from a completed or legacy phase run."""

    phase_manifest = phase_dir / "phase_manifest.json"
    if phase_manifest.is_file():
        state = read_json(phase_manifest)
        adapter = Path(str(state.get("adapter", "")))
        if state.get("completed") is True and all(
            (adapter / name).is_file()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        ):
            return adapter

    candidates = sorted(
        phase_dir.rglob("adapter_model.safetensors"),
        key=lambda path: (_checkpoint_step(path), str(path)),
        reverse=True,
    )
    for weights in candidates:
        adapter = weights.parent
        trainer_state_path = adapter / "trainer_state.json"
        if not (adapter / "adapter_config.json").is_file() or not trainer_state_path.is_file():
            continue
        trainer_state = read_json(trainer_state_path)
        global_step = int(trainer_state.get("global_step", -1))
        max_steps = int(trainer_state.get("max_steps", 0))
        if max_steps > 0 and global_step >= max_steps:
            return adapter
    return None


def _write_phase_manifest(
    phase_dir: Path,
    *,
    phase: str,
    adapter: Path,
    train_data: Path,
    trainer_log: Path | None,
) -> None:
    write_json(
        phase_dir / "phase_manifest.json",
        {
            "phase": phase,
            "adapter": str(adapter.resolve()),
            "adapter_config_hash": file_hash(adapter / "adapter_config.json"),
            "adapter_weights_hash": file_hash(adapter / "adapter_model.safetensors"),
            "train_data": str(train_data.resolve()),
            "train_data_hash": file_hash(train_data),
            "trainer_log": str(trainer_log.resolve()) if trainer_log is not None else None,
            "completed": True,
        },
    )


def _manifest_adapter(state: dict[str, Any], key: str) -> Path | None:
    raw_adapter = state.get(key)
    if not raw_adapter:
        return None
    adapter = Path(str(raw_adapter))
    if all(
        (adapter / name).is_file()
        for name in ("adapter_config.json", "adapter_model.safetensors")
    ):
        return adapter
    return None


def _discover_curriculum_progress(
    output_dir: Path, *, rounds: int
) -> dict[str, Any]:
    """Discover the last durable curriculum phase directly from output artifacts."""

    completed = 0
    questioner_adapter: Path | None = None
    solver_adapter: Path | None = None
    next_phase: str | None = "questioner"
    for round_index in range(1, rounds + 1):
        round_dir = output_dir / f"round_{round_index:03d}"
        round_manifest_path = round_dir / "manifest.json"
        round_state = (
            read_json(round_manifest_path) if round_manifest_path.is_file() else {}
        )
        questioner = _completed_phase_adapter(round_dir / "questioner_update")
        solver = _completed_phase_adapter(round_dir / "solver_update")
        if round_state.get("completed") is True:
            questioner = questioner or _manifest_adapter(round_state, "questioner_adapter")
            solver = solver or _manifest_adapter(round_state, "solver_adapter")
        if solver is not None and questioner is None:
            raise RuntimeError(
                f"round {round_index} has a completed Solver but no completed Questioner"
            )
        if questioner is None:
            next_phase = "questioner"
            break
        if solver is None:
            next_phase = "solver"
            break
        completed = round_index
        questioner_adapter = questioner
        solver_adapter = solver
        next_phase = "questioner" if round_index < rounds else None
    return {
        "last_completed_round": completed,
        "questioner_adapter": (
            str(questioner_adapter) if questioner_adapter is not None else None
        ),
        "solver_adapter": str(solver_adapter) if solver_adapter is not None else None,
        "next_round": completed + 1 if completed < rounds else None,
        "next_phase": next_phase,
    }


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


def _signal_service(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        return
    if process.poll() is None:  # pragma: no cover - Windows fallback
        process.send_signal(sig)


def _stop(process: subprocess.Popen[Any]) -> None:
    """Stop a long-lived service and all workers it spawned."""

    _signal_service(process, signal.SIGTERM)
    try:
        if process.poll() is None:
            process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _signal_service(process, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=10)


def _run_with_tee(command: Sequence[str], *, env: dict[str, str], log_path: Path) -> None:
    """Stream a child process to the terminal while retaining the complete byte log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        list(command),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
        raise RuntimeError("failed to capture training output")
    with log_path.open("wb") as log_stream:
        while True:
            chunk = os.read(process.stdout.fileno(), 64 * 1024)
            if not chunk:
                break
            log_stream.write(chunk)
            log_stream.flush()
            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            sys.stdout.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))


def _next_metrics_attempt(logs_dir: Path) -> Path:
    indices = []
    for path in logs_dir.glob("metrics_attempt_*"):
        match = re.fullmatch(r"metrics_attempt_(\d+)", path.name)
        if match:
            indices.append(int(match.group(1)))
    return logs_dir / f"metrics_attempt_{max(indices, default=0) + 1:03d}"


def _archive_size(path: Path) -> int:
    with TaskArchive(path) as archive:
        return len(archive.all())


def _prepare_validation_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    max_samples: int | None,
    seed: int,
) -> dict[str, Any]:
    """Create one replayable validation subset shared by every self-play round."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source_path.resolve()
    parquet = pq.ParquetFile(source)
    total_rows = parquet.metadata.num_rows
    selected_indices: list[int] | None = None
    output = source
    if max_samples is not None and total_rows > max_samples:
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(range(total_rows), max_samples))
        table = pq.read_table(source)
        sampled = table.take(pa.array(selected_indices, type=pa.int64()))
        output = (output_dir / "validation.parquet").resolve()
        pq.write_table(sampled, output)
    summary = {
        "source": str(source),
        "source_sha256": file_hash(source),
        "output": str(output),
        "total_rows": total_rows,
        "selected_rows": total_rows if selected_indices is None else len(selected_indices),
        "max_samples": max_samples,
        "seed": seed,
        "selected_indices": selected_indices,
    }
    write_json(output_dir / "validation_sample.json", summary)
    return summary


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
    if config.selfplay_variant in {"frontier_v2", "curriculum_v3"}:
        opponent.extend(
            [
                "--candidate-archive",
                str((round_dir / "candidate_archive.sqlite").resolve()),
                "--cache-evaluations",
            ]
        )
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
    one_round: bool = False,
    target_round: int | None = None,
    target_phase: UpdatePhase | None = None,
) -> dict[str, Any]:
    config = load_selfplay_config(config_path)
    if (target_round is None) != (target_phase is None):
        raise ValueError("--round-index and --phase must be provided together")
    if target_round is not None:
        if one_round:
            raise ValueError("--one-round cannot be combined with --round-index/--phase")
        if config.selfplay_variant != "curriculum_v3":
            raise ValueError("exact phase execution requires selfplay_variant=curriculum_v3")
        if target_round > config.rounds:
            raise ValueError(
                f"--round-index {target_round} exceeds configured rounds={config.rounds}"
            )
        resume = True
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seed = config.seed + 20_000
    validation = (
        {
            "source": str(config.val_data.resolve()),
            "output": str(
                (
                    output_dir / "validation.parquet"
                    if config.validation_samples is not None
                    else config.val_data
                ).resolve()
            ),
            "total_rows": None,
            "selected_rows": config.validation_samples,
            "max_samples": config.validation_samples,
            "seed": validation_seed,
            "selected_indices": None,
        }
        if dry_run
        else _prepare_validation_dataset(
            config.val_data,
            output_dir,
            max_samples=config.validation_samples,
            seed=validation_seed,
        )
    )
    validation_path = Path(str(validation["output"]))
    archive_path = output_dir / "archive.sqlite"
    manifest_path = output_dir / "manifest.json"
    completed = 0
    adapter = Path(config.initial_adapter)
    questioner_adapter = adapter
    solver_adapter = adapter
    manifest_completed = 0
    if resume and manifest_path.exists():
        state = read_json(manifest_path)
        manifest_completed = int(state["last_completed_round"])
        completed = manifest_completed
        adapter = Path(state["adapter"])
        questioner_adapter = Path(state.get("questioner_adapter", state["adapter"]))
        solver_adapter = Path(state.get("solver_adapter", state["adapter"]))
        if state["config_hash"] != file_hash(config_path):
            raise ValueError("cannot resume: self-play config changed")
    resume_progress: dict[str, Any] | None = None
    if resume and config.selfplay_variant == "curriculum_v3":
        resume_progress = _discover_curriculum_progress(output_dir, rounds=config.rounds)
        discovered = int(resume_progress["last_completed_round"])
        if discovered < manifest_completed:
            raise RuntimeError(
                "cannot resume: top-level manifest is ahead of complete phase artifacts "
                f"({manifest_completed} > {discovered})"
            )
        completed = discovered
        discovered_questioner = resume_progress["questioner_adapter"]
        discovered_solver = resume_progress["solver_adapter"]
        if discovered_questioner is not None:
            questioner_adapter = Path(str(discovered_questioner))
        if discovered_solver is not None:
            solver_adapter = Path(str(discovered_solver))
            adapter = solver_adapter
        LOGGER.info(
            "selfplay_resume_discovered completed_rounds=%d next_round=%s next_phase=%s",
            completed,
            resume_progress["next_round"],
            resume_progress["next_phase"],
        )
        if not dry_run and completed > manifest_completed:
            write_json(
                manifest_path,
                {
                    "last_completed_round": completed,
                    "adapter": str(adapter),
                    "questioner_adapter": str(questioner_adapter),
                    "solver_adapter": str(solver_adapter),
                    "config_hash": file_hash(config_path),
                    "ms_swift_version": MS_SWIFT_VERSION,
                    "selfplay_variant": config.selfplay_variant,
                    "recovered_from_phase_artifacts": True,
                },
            )
    if target_round is not None and target_phase is not None:
        next_round = resume_progress["next_round"] if resume_progress is not None else 1
        requested_dir = output_dir / f"round_{target_round:03d}"
        requested_questioner = _completed_phase_adapter(
            requested_dir / "questioner_update"
        )
        requested_adapter = _completed_phase_adapter(
            requested_dir / f"{target_phase}_update"
        )
        if requested_adapter is not None or target_round <= completed:
            LOGGER.info(
                "selfplay_phase_already_completed round=%d phase=%s adapter=%s",
                target_round,
                target_phase,
                requested_adapter,
            )
            return {
                "dry_run": dry_run,
                "rounds_requested": config.rounds,
                "rounds_completed": completed,
                "rounds_planned": 0,
                "rounds_remaining": config.rounds - completed,
                "one_round": False,
                "target_round": target_round,
                "target_phase": target_phase,
                "phase_skipped": True,
                "resume_progress": resume_progress,
                "plans": [],
                "ms_swift_version": MS_SWIFT_VERSION,
                "report_artifacts": None,
            }
        if target_round != next_round:
            raise RuntimeError(
                f"cannot run round {target_round} {target_phase}: next unfinished round is "
                f"{next_round}"
            )
        if target_phase == "solver" and requested_questioner is None:
            raise RuntimeError(
                f"cannot run round {target_round} solver before its Questioner completes"
            )
        if target_phase == "solver":
            # The scanner intentionally returns adapters only for fully completed
            # rounds. An exact Solver invocation must consume this round's durable
            # Questioner adapter instead of the previous round's Questioner.
            assert requested_questioner is not None
            questioner_adapter = requested_questioner
        first_round = target_round
        final_round = target_round
    else:
        first_round = completed + 1
        final_round = min(config.rounds, completed + 1) if one_round else config.rounds
    plans: list[dict[str, Any]] = []
    report_artifacts: dict[str, str] | None = None
    completed_after = completed
    for round_index in range(first_round, final_round + 1):
        round_dir = output_dir / f"round_{round_index:03d}"
        logs = round_dir / "logs"
        reward_metrics_dir = _next_metrics_attempt(logs)
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
        archive_size_before = 0 if dry_run else _archive_size(archive_path)
        commands = _commands(
            config,
            adapter=(solver_adapter if config.selfplay_variant == "curriculum_v3" else adapter),
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
            "VAL_DATA": str(validation_path),
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
            "DEEPSPEED": config.deepspeed,
            "RL_ALGORITHM": config.rl_algorithm,
            "USE_VLLM": str(config.use_vllm).lower(),
            "OUTPUT_DIR": str(round_dir.resolve()),
            "EXPERIMENT_NAME": f"graphtask-selfplay-r{round_index:03d}",
            "INTERACTION_MODE": config.interaction_mode,
            "GRAPHSCRIPT_VERSION": config.graphscript_version,
            "VLLM_MODE": "colocate",
            "SAVE_STEPS": str(config.save_steps),
            "SAVE_TOTAL_LIMIT": str(config.save_total_limit),
            "GRAPHTASK_REWARD_METRICS_DIR": str(reward_metrics_dir.resolve()),
            "SEED": str(config.seed),
            "PYTHONUNBUFFERED": "1",
        }
        if config.selfplay_variant == "curriculum_v3":
            train_overrides["MULTI_TURN_SCHEDULER"] = "graphtask_curriculum_solver"
        if config.response_prefix is not None:
            train_overrides["RESPONSE_PREFIX"] = config.response_prefix
        plan = {
            "round": round_index,
            "one_round": one_round,
            "target_phase": target_phase,
            "phase_resume": {
                "questioner": _completed_phase_adapter(round_dir / "questioner_update")
                is not None,
                "solver": _completed_phase_adapter(round_dir / "solver_update") is not None,
            },
            "selfplay_variant": config.selfplay_variant,
            "adapter_in": str(adapter),
            "questioner_adapter_in": (
                str(questioner_adapter)
                if config.selfplay_variant == "curriculum_v3"
                else None
            ),
            "solver_adapter_in": (
                str(solver_adapter) if config.selfplay_variant == "curriculum_v3" else None
            ),
            "dataset": str(mixed_data),
            "counts": counts,
            "validation": validation,
            "archive_size_before": archive_size_before,
            "commands": commands,
            "train_environment": train_overrides,
            "actor_gpus": config.actor_gpus,
            "opponent_gpus": config.opponent_gpus,
            "allow_gpu_overlap": config.allow_gpu_overlap,
            "use_vllm": config.use_vllm,
            "deepspeed": config.deepspeed,
            "rl_algorithm": config.rl_algorithm,
            "opponent_backend": config.opponent_backend,
            "merged_opponent_model": str((round_dir / "opponent_merged").resolve()),
            "interaction_mode": config.interaction_mode,
            "graphscript_version": config.graphscript_version,
            "relation_catalog": str(config.relation_catalog)
            if config.relation_catalog is not None
            else None,
            "program_profile": config.program_profile,
            "update_order": (
                ["questioner", "archive", "solver"]
                if config.selfplay_variant == "curriculum_v3"
                else (
                    ["solver", "questioner"]
                    if config.selfplay_variant == "frontier_v2"
                    else ["mixed"]
                )
            ),
            "max_completion_tokens": config.max_completion_tokens,
            "reward_metrics_dir": str(reward_metrics_dir),
            "questioner_reward": {
                "variant": config.selfplay_variant,
                "frontier_target": _frontier_target(config, round_index),
                "frontier_sigma": config.frontier_sigma,
                "curriculum_phase": (
                    _curriculum_phase(config, round_index)
                    if config.selfplay_variant == "curriculum_v3"
                    else None
                ),
            },
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
        reward_metrics_dir.mkdir(parents=True, exist_ok=False)
        LOGGER.info(
            "selfplay_round_started round=%d/%d questioner_rows=%d solver_rows=%d logs=%s",
            round_index,
            config.rounds,
            counts["questioner"],
            counts["solver"],
            logs,
        )
        known_adapters = frozenset(
            path.resolve() for path in round_dir.rglob("adapter_model.safetensors")
        )
        write_json(round_dir / "plan.json", plan)
        logs.mkdir(exist_ok=True)
        opponent_env = {**os.environ, "CUDA_VISIBLE_DEVICES": config.opponent_gpus}
        merged_model = (round_dir / "opponent_merged").resolve()
        merge_manifest_path = round_dir / "opponent_merge.json"
        opponent_adapter = (
            solver_adapter if config.selfplay_variant == "curriculum_v3" else adapter
        )
        merge_spec = _opponent_merge_spec(config, opponent_adapter)
        if merged_model.exists():
            LOGGER.info("selfplay_merge_reused round=%d model=%s", round_index, merged_model)
            if not merge_manifest_path.is_file() or read_json(merge_manifest_path) != merge_spec:
                raise RuntimeError(
                    f"refusing to reuse unverified merged opponent model at {merged_model}; "
                    "move or remove that directory before retrying"
                )
            _validate_merged_model(merged_model)
        else:
            merge_log_path = logs / "merge.log"
            LOGGER.info("selfplay_merge_started round=%d log=%s", round_index, merge_log_path)
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
            LOGGER.info("selfplay_merge_completed round=%d", round_index)
        sglang_process: subprocess.Popen[Any] | None = None
        phase_trainer_logs: dict[str, Path] = {}
        if target_phase == "solver":
            questioner_log = find_trainer_log(round_dir, questioner_adapter)
            if questioner_log is not None:
                phase_trainer_logs["questioner"] = questioner_log
        admission_summary: dict[str, Any] | None = None
        if commands["sglang"]:
            LOGGER.info("selfplay_sglang_started round=%d log=%s", round_index, logs / "sglang.log")
            with (logs / "sglang.log").open("w") as sglang_log:
                sglang_process = subprocess.Popen(
                    commands["sglang"],
                    env=opponent_env,
                    stdout=sglang_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
        try:
            if sglang_process is not None:
                _wait_for(
                    f"http://127.0.0.1:{config.sglang_port}/health",
                    timeout_s=config.sglang_start_timeout_s,
                )
            with (logs / "opponent.log").open("w") as opponent_log:
                LOGGER.info(
                    "selfplay_opponent_started round=%d backend=%s log=%s",
                    round_index,
                    config.opponent_backend,
                    logs / "opponent.log",
                )
                opponent_process = subprocess.Popen(
                    commands["opponent"],
                    env=opponent_env,
                    stdout=opponent_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
            try:
                _wait_for(opponent_url + "/health", timeout_s=60)
                train_env = {**os.environ, **train_overrides}
                train_log_path = logs / "ms_swift.log"
                LOGGER.info(
                    "selfplay_training_started round=%d log=%s terminal_output=true",
                    round_index,
                    train_log_path,
                )
                try:
                    if config.selfplay_variant == "legacy":
                        _run_with_tee(commands["train"], env=train_env, log_path=train_log_path)
                    elif config.selfplay_variant == "frontier_v2":
                        phase_adapter = adapter
                        for phase, phase_data in (
                            ("solver", round_dir / "solver.parquet"),
                            ("questioner", round_dir / "questioner.parquet"),
                        ):
                            phase_dir = round_dir / f"{phase}_update"
                            phase_known_adapters = frozenset(
                                path.resolve()
                                for path in phase_dir.rglob("adapter_model.safetensors")
                            )
                            phase_log_path = logs / f"ms_swift_{phase}.log"
                            phase_env = {
                                **train_env,
                                "LORA_ADAPTER_PATH": str(phase_adapter),
                                "TRAIN_DATA": str(phase_data.resolve()),
                                "OUTPUT_DIR": str(phase_dir.resolve()),
                                "EXPERIMENT_NAME": (
                                    f"graphtask-selfplay-frontier-v2-r{round_index:03d}-{phase}"
                                ),
                            }
                            LOGGER.info(
                                "selfplay_phase_training_started round=%d phase=%s data=%s log=%s",
                                round_index,
                                phase,
                                phase_data,
                                phase_log_path,
                            )
                            _run_with_tee(
                                commands["train"], env=phase_env, log_path=phase_log_path
                            )
                            phase_adapter = _adapter_from_checkpoint(
                                phase_dir, known_adapters=phase_known_adapters
                            )
                            phase_log = find_trainer_log(phase_dir, phase_adapter)
                            if phase_log is not None:
                                phase_trainer_logs[phase] = phase_log
                            LOGGER.info(
                                "selfplay_phase_training_completed round=%d phase=%s adapter=%s",
                                round_index,
                                phase,
                                phase_adapter,
                            )
                        adapter = phase_adapter
                    else:
                        curriculum_phase = _curriculum_phase(config, round_index)
                        phase_updates: tuple[tuple[UpdatePhase, Path], ...] = (
                            ("questioner", round_dir / "questioner.parquet"),
                            ("solver", round_dir / "solver.parquet"),
                        )
                        if target_phase is not None:
                            phase_updates = tuple(
                                update for update in phase_updates if update[0] == target_phase
                            )
                        for phase, phase_data in phase_updates:
                            if phase == "solver":
                                relaxed = curriculum_phase != "frontier"
                                admission_summary = promote_staged_tasks(
                                    round_dir / "candidate_archive.sqlite",
                                    archive_path,
                                    min_pass_rate=(
                                        0.0 if relaxed else config.archive_min_pass_rate
                                    ),
                                    max_pass_rate=(
                                        1.0 if relaxed else config.archive_max_pass_rate
                                    ),
                                    min_novelty=(
                                        0.0 if relaxed else config.archive_min_novelty
                                    ),
                                )
                                admission_summary["curriculum_phase"] = curriculum_phase
                                write_json(logs / "archive_admission.json", admission_summary)
                                counts["solver"] = _write_solver_dataset(
                                    config,
                                    archive_path,
                                    phase_data,
                                    round_index=round_index,
                                )
                                counts["total"] = counts["questioner"] + counts["solver"]

                            phase_dir = round_dir / f"{phase}_update"
                            recovered_adapter = _completed_phase_adapter(phase_dir)
                            if recovered_adapter is not None:
                                if phase == "questioner":
                                    questioner_adapter = recovered_adapter
                                else:
                                    solver_adapter = recovered_adapter
                                phase_log = find_trainer_log(phase_dir, recovered_adapter)
                                if phase_log is not None:
                                    phase_trainer_logs[phase] = phase_log
                                _write_phase_manifest(
                                    phase_dir,
                                    phase=phase,
                                    adapter=recovered_adapter,
                                    train_data=phase_data,
                                    trainer_log=phase_log,
                                )
                                LOGGER.info(
                                    "selfplay_phase_training_reused round=%d phase=%s adapter=%s",
                                    round_index,
                                    phase,
                                    recovered_adapter,
                                )
                                continue
                            phase_known_adapters = frozenset(
                                path.resolve()
                                for path in phase_dir.rglob("adapter_model.safetensors")
                            )
                            phase_log_path = logs / f"ms_swift_{phase}.log"
                            phase_adapter = (
                                questioner_adapter if phase == "questioner" else solver_adapter
                            )
                            phase_env = {
                                **train_env,
                                "LORA_ADAPTER_PATH": str(phase_adapter),
                                "TRAIN_DATA": str(phase_data.resolve()),
                                "OUTPUT_DIR": str(phase_dir.resolve()),
                                "EXPERIMENT_NAME": (
                                    f"graphtask-selfplay-curriculum-v3-"
                                    f"r{round_index:03d}-{phase}"
                                ),
                                "SEED": str(
                                    config.seed
                                    + round_index * 1_000
                                    + (1 if phase == "questioner" else 2)
                                ),
                            }
                            LOGGER.info(
                                "selfplay_phase_training_started round=%d phase=%s data=%s log=%s",
                                round_index,
                                phase,
                                phase_data,
                                phase_log_path,
                            )
                            _run_with_tee(
                                commands["train"], env=phase_env, log_path=phase_log_path
                            )
                            updated_adapter = _adapter_from_checkpoint(
                                phase_dir, known_adapters=phase_known_adapters
                            )
                            if phase == "questioner":
                                questioner_adapter = updated_adapter
                            else:
                                solver_adapter = updated_adapter
                            phase_log = find_trainer_log(phase_dir, updated_adapter)
                            if phase_log is not None:
                                phase_trainer_logs[phase] = phase_log
                            _write_phase_manifest(
                                phase_dir,
                                phase=phase,
                                adapter=updated_adapter,
                                train_data=phase_data,
                                trainer_log=phase_log,
                            )
                            LOGGER.info(
                                "selfplay_phase_training_completed round=%d phase=%s adapter=%s",
                                round_index,
                                phase,
                                updated_adapter,
                            )
                        adapter = solver_adapter
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise RuntimeError(
                        f"self-play GRPO training failed; see {train_log_path}"
                    ) from exc
                LOGGER.info("selfplay_training_completed round=%d", round_index)
            finally:
                _stop(opponent_process)
        finally:
            if sglang_process is not None:
                _stop(sglang_process)
        if target_phase == "questioner":
            LOGGER.info(
                "selfplay_exact_phase_completed round=%d phase=questioner",
                round_index,
            )
            continue
        if config.selfplay_variant == "legacy":
            adapter = _adapter_from_checkpoint(round_dir, known_adapters=known_adapters)
        if config.selfplay_variant == "frontier_v2":
            admission_summary = promote_staged_tasks(
                round_dir / "candidate_archive.sqlite",
                archive_path,
                min_pass_rate=config.archive_min_pass_rate,
                max_pass_rate=config.archive_max_pass_rate,
                min_novelty=config.archive_min_novelty,
            )
            write_json(logs / "archive_admission.json", admission_summary)
        archive_size_after = _archive_size(archive_path)
        trainer_log = (
            find_trainer_log(round_dir, adapter)
            if config.selfplay_variant == "legacy"
            else None
        )
        round_metrics = summarize_selfplay_round(
            round_index,
            counts,
            trainer_log=trainer_log,
            reward_metrics_dir=reward_metrics_dir,
            archive_size_before=archive_size_before,
            archive_size_after=archive_size_after,
            trainer_logs=phase_trainer_logs or None,
        )
        if admission_summary is not None:
            round_metrics["archive_admission"] = admission_summary
        metrics_summary_path = logs / "metrics_summary.json"
        write_json(metrics_summary_path, round_metrics)
        report_artifacts = write_selfplay_report(output_dir)
        write_json(
            round_dir / "manifest.json",
            {
                "round": round_index,
                "adapter": str(adapter),
                "questioner_adapter": (
                    str(questioner_adapter)
                    if config.selfplay_variant == "curriculum_v3"
                    else None
                ),
                "solver_adapter": (
                    str(solver_adapter)
                    if config.selfplay_variant == "curriculum_v3"
                    else None
                ),
                "dataset_hash": file_hash(mixed_data),
                "questioner_dataset_hash": file_hash(round_dir / "questioner.parquet"),
                "solver_dataset_hash": file_hash(round_dir / "solver.parquet"),
                "metrics_summary": str(metrics_summary_path),
                "reward_metrics_dir": str(reward_metrics_dir),
                "selfplay_variant": config.selfplay_variant,
                "trainer_logs": {
                    phase: str(path) for phase, path in sorted(phase_trainer_logs.items())
                },
                "archive_admission": (
                    str(logs / "archive_admission.json")
                    if admission_summary is not None
                    else None
                ),
                "completed": True,
            },
        )
        write_json(
            manifest_path,
            {
                "last_completed_round": round_index,
                "adapter": str(adapter),
                "questioner_adapter": (
                    str(questioner_adapter)
                    if config.selfplay_variant == "curriculum_v3"
                    else None
                ),
                "solver_adapter": (
                    str(solver_adapter)
                    if config.selfplay_variant == "curriculum_v3"
                    else None
                ),
                "config_hash": file_hash(config_path),
                "ms_swift_version": MS_SWIFT_VERSION,
                "selfplay_variant": config.selfplay_variant,
            },
        )
        LOGGER.info(
            "selfplay_round_completed round=%d adapter=%s curves=%s",
            round_index,
            adapter,
            report_artifacts["plot"],
        )
        completed_after = round_index
    if not dry_run and report_artifacts is None:
        report_artifacts = write_selfplay_report(output_dir)
    return {
        "dry_run": dry_run,
        "rounds_requested": config.rounds,
        "rounds_completed": completed if dry_run else completed_after,
        "rounds_planned": len(plans),
        "rounds_remaining": config.rounds - (completed if dry_run else completed_after),
        "one_round": one_round,
        "target_round": target_round,
        "target_phase": target_phase,
        "phase_skipped": False,
        "resume_progress": resume_progress,
        "plans": plans,
        "ms_swift_version": MS_SWIFT_VERSION,
        "report_artifacts": report_artifacts,
    }
