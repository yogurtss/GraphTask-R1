from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.utils import write_json

MS_SWIFT_VERSION = "3.10.3"


class EvidenceRLTrainConfig(BaseModel):
    """One alternating direct-RL update on a frozen self-play audit."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    model_type: str = "qwen3"
    initial_adapter: Path | None = None
    questioner_initial_adapter: Path | None = None
    existing_solver_adapter: Path | None = None
    policy_topology: Literal["shared", "role_separated"] = "shared"
    train_solver: bool = True
    train_questioner: bool = True
    graph_db: Path
    counterfactual_reranker: Path | None = None
    round_data: Path
    output_dir: Path
    train_script: Path = Path("scripts/train_ms_swift_grpo.sh")
    actor_gpus: str = "0"
    use_vllm: bool = False
    seed: int = 42
    epochs: int = Field(default=1, gt=0)
    micro_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    steps_per_generation: int = Field(default=1, gt=0)
    solver_rollouts: int = Field(default=4, gt=1)
    questioner_rollouts: int = Field(default=4, gt=1)
    solver_max_completion_length: int = Field(default=1_024, gt=0)
    questioner_max_completion_length: int = Field(default=128, gt=0)
    response_prefix: str | None = None
    solver_learning_rate: float = Field(default=2e-6, gt=0.0)
    questioner_learning_rate: float = Field(default=2e-6, gt=0.0)
    solver_algorithm: Literal["grpo"] = "grpo"
    questioner_algorithm: Literal["reinforce_plus_plus"] = "reinforce_plus_plus"


def load_evidence_rl_train_config(path: Path) -> EvidenceRLTrainConfig:
    raw = yaml.safe_load(os.path.expandvars(path.read_text()))
    return EvidenceRLTrainConfig.model_validate(raw)


def evidence_selfplay_plan(config: EvidenceRLTrainConfig) -> dict[str, object]:
    common = {
        "MODEL_PATH": config.model_path,
        "MODEL_TYPE": config.model_type,
        "TRAIN_CUDA_VISIBLE_DEVICES": config.actor_gpus,
        "NUM_GPUS": str(len([value for value in config.actor_gpus.split(",") if value])),
        "USE_VLLM": str(config.use_vllm).lower(),
        "VLLM_MODE": "colocate",
        "GRAPHTASK_KILT_DB": str(config.graph_db.resolve()),
        "MICRO_BATCH_SIZE": str(config.micro_batch_size),
        "GRADIENT_ACCUMULATION_STEPS": str(config.gradient_accumulation_steps),
        "STEPS_PER_GENERATION": str(config.steps_per_generation),
        "EPOCHS": str(config.epochs),
        "SEED": str(config.seed),
        "AUTO_RESUME": "false",
        "EVAL_STRATEGY": "no",
    }
    if config.counterfactual_reranker is not None:
        common["EVIDENCE_RERANKER_PATH"] = str(
            config.counterfactual_reranker.resolve()
        )
    all_phases = [
        {
            "role": "solver",
            "algorithm": config.solver_algorithm,
            "dataset": str((config.round_data / "solver_train.parquet").resolve()),
            "output_dir": str((config.output_dir / "solver_update").resolve()),
            "rollouts": config.solver_rollouts,
            "max_completion_length": config.solver_max_completion_length,
            "learning_rate": config.solver_learning_rate,
            "interaction_mode": "tool",
            "multi_turn_scheduler": "graphtask_curriculum_solver",
            "max_turns": 4,
        },
        {
            "role": "questioner",
            "algorithm": config.questioner_algorithm,
            "dataset": str((config.round_data / "questioner_train.parquet").resolve()),
            "output_dir": str((config.output_dir / "questioner_update").resolve()),
            "rollouts": config.questioner_rollouts,
            "max_completion_length": config.questioner_max_completion_length,
            "learning_rate": config.questioner_learning_rate,
            "interaction_mode": "graphscript",
        },
    ]
    enabled_roles = {
        role
        for role, enabled in (
            ("solver", config.train_solver),
            ("questioner", config.train_questioner),
        )
        if enabled
    }
    phases = [phase for phase in all_phases if phase["role"] in enabled_roles]
    phase_order = [str(phase["role"]) for phase in phases]
    return {
        "framework": "ms-swift",
        "ms_swift_version": MS_SWIFT_VERSION,
        "uses_sft": False,
        "policy_topology": config.policy_topology,
        "phase_order": phase_order,
        "shared_policy_update_order": (
            phase_order if config.policy_topology == "shared" else []
        ),
        "common_environment": common,
        "phases": phases,
    }


def run_evidence_selfplay_update(
    config: EvidenceRLTrainConfig,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    plan = evidence_selfplay_plan(config)
    if dry_run:
        return plan
    train_script = config.train_script.resolve()
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    if not config.graph_db.is_file():
        raise FileNotFoundError(config.graph_db)
    phases = plan["phases"]
    assert isinstance(phases, list)
    common = plan["common_environment"]
    assert isinstance(common, dict)
    completed: list[dict[str, str]] = []
    role_adapters: dict[str, Path] = {}
    adapter: Path | None
    for raw_phase in phases:
        assert isinstance(raw_phase, dict)
        role = str(raw_phase["role"])
        if role == "solver" and config.existing_solver_adapter is not None:
            adapter = config.existing_solver_adapter.resolve()
            role_adapters[role] = adapter
            completed.append(
                {
                    "role": role,
                    "algorithm": str(raw_phase["algorithm"]),
                    "adapter": str(adapter),
                    "status": "reused",
                }
            )
            continue
        dataset = Path(str(raw_phase["dataset"]))
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        phase_dir = Path(str(raw_phase["output_dir"]))
        environment = {
            **os.environ,
            **{str(key): str(value) for key, value in common.items()},
            "TRAIN_DATA": str(dataset),
            "OUTPUT_DIR": str(phase_dir),
            "RL_ALGORITHM": str(raw_phase["algorithm"]),
            "ROLLOUT_N": str(raw_phase["rollouts"]),
            "MAX_COMPLETION_LENGTH": str(raw_phase["max_completion_length"]),
            "LR": str(raw_phase["learning_rate"]),
            "INTERACTION_MODE": str(raw_phase["interaction_mode"]),
        }
        if raw_phase.get("multi_turn_scheduler") is not None:
            environment["MULTI_TURN_SCHEDULER"] = str(raw_phase["multi_turn_scheduler"])
            environment["MAX_TURNS"] = str(raw_phase["max_turns"])
        if config.response_prefix is not None:
            environment["RESPONSE_PREFIX"] = config.response_prefix
        environment.pop("MS_SWIFT_SFT_ADAPTER", None)
        if role == "solver":
            adapter = config.initial_adapter.resolve() if config.initial_adapter else None
        elif config.policy_topology == "shared":
            adapter = role_adapters.get("solver")
        else:
            adapter = (
                config.questioner_initial_adapter.resolve()
                if config.questioner_initial_adapter
                else None
            )
        if adapter is None:
            environment.pop("LORA_ADAPTER_PATH", None)
        else:
            environment["LORA_ADAPTER_PATH"] = str(adapter)
        subprocess.run(
            ["bash", str(train_script)],
            cwd=train_script.parent.parent,
            env=environment,
            check=True,
        )
        adapter = latest_lora_adapter(phase_dir)
        role_adapters[role] = adapter
        completed.append(
            {
                "role": role,
                "algorithm": str(raw_phase["algorithm"]),
                "adapter": str(adapter),
                "status": "trained",
            }
        )
    final_adapter = role_adapters[str(phases[-1]["role"])]
    if config.policy_topology == "role_separated" and "solver" in role_adapters:
        final_adapter = role_adapters["solver"]
    result = {
        **plan,
        "completed_phases": completed,
        "role_adapters": {
            role: str(adapter_path) for role, adapter_path in role_adapters.items()
        },
        "final_adapter": str(final_adapter),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "direct_rl_update.json", result)
    return result


def latest_lora_adapter(output_dir: Path) -> Path:
    candidates = [
        path.parent
        for pattern in ("adapter_model.safetensors", "adapter_model.bin")
        for path in output_dir.rglob(pattern)
    ]
    if not candidates:
        raise RuntimeError(f"ms-swift emitted no LoRA adapter below {output_dir}")
    return max(candidates, key=_adapter_order)


def _adapter_order(path: Path) -> tuple[int, int, str]:
    step = -1
    for parent in (path, *path.parents):
        match = re.fullmatch(r"checkpoint-(\d+)", parent.name)
        if match:
            step = int(match.group(1))
            break
    weights = next(
        candidate
        for candidate in (path / "adapter_model.safetensors", path / "adapter_model.bin")
        if candidate.is_file()
    )
    return step, weights.stat().st_mtime_ns, str(path)
