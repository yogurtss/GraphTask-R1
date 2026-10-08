from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = ROOT / "scripts" / "train_dpo.py"
SPEC = importlib.util.spec_from_file_location("grr_train_dpo", TRAINING_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _flags(command: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(command[2::2], command[3::2], strict=True))


def test_mini_training_is_isolated_and_matches_ere_optimizer_contract() -> None:
    invocation = MODULE.build_invocation(ROOT / "configs" / "train_mini.yaml")
    flags = _flags(invocation.command)
    assert invocation.command[:2] == (
        str((ROOT.parents[1] / ".conda/ms-swift-debug/bin/swift").resolve()),
        "rlhf",
    )
    assert flags["--rlhf_type"] == "dpo"
    assert flags["--dataset"].endswith(
        "/outputs/experiments/kqapro-graph-residual-replay/mini/pairs/train.jsonl"
    )
    assert flags["--output_dir"].endswith(
        "/outputs/experiments/kqapro-graph-residual-replay/dpo-mini"
    )
    assert flags["--num_train_epochs"] == "1"
    assert flags["--gradient_accumulation_steps"] == "4"
    assert flags["--learning_rate"] == "5e-07"
    assert flags["--beta"] == "0.1"
    assert flags["--lora_rank"] == "32"
    assert invocation.environment["PYTHONHASHSEED"] == "4242"

    ere = yaml.safe_load(
        (
            ROOT.parent / "kqapro_execution_residual_evolution/configs/mini.yaml"
        ).read_text()
    )
    grr = yaml.safe_load((ROOT / "configs" / "train_mini.yaml").read_text())
    assert grr["seed"] == ere["seed"]
    assert grr["model"] == ere["model"]
    assert grr["runtime"] == ere["runtime"]
    assert grr["training"] == ere["training"]


def test_boundary_training_changes_only_the_dataset_and_loss_scope() -> None:
    invocation = MODULE.build_invocation(ROOT / "configs" / "train_boundary.yaml")
    flags = _flags(invocation.command)

    assert flags["--dataset"].endswith(
        "/outputs/experiments/kqapro-graph-residual-replay/mini/boundary-pairs/train.jsonl"
    )
    assert flags["--output_dir"].endswith(
        "/outputs/experiments/kqapro-graph-residual-replay/dpo-boundary"
    )
    assert flags["--external_plugins"].endswith("ms_swift_boundary_plugin.py")
    assert flags["--template"] == "qwen3_execution_boundary"
    assert flags["--loss_scale"] == "execution_boundary_asymmetric"
    assert flags["--num_train_epochs"] == "2"
    assert flags["--gradient_accumulation_steps"] == "4"
    assert flags["--learning_rate"] == "5e-07"
    assert flags["--beta"] == "0.1"


def test_balanced_boundary_control_has_the_same_optimizer_contract() -> None:
    treatment = _flags(
        MODULE.build_invocation(ROOT / "configs" / "train_boundary_balanced.yaml").command
    )
    control = _flags(
        MODULE.build_invocation(
            ROOT / "configs" / "train_boundary_balanced_control.yaml"
        ).command
    )

    for flag in (
        "--model",
        "--num_train_epochs",
        "--gradient_accumulation_steps",
        "--learning_rate",
        "--beta",
        "--lora_rank",
        "--seed",
        "--data_seed",
    ):
        assert treatment[flag] == control[flag]
    assert treatment["--dataset"].endswith("/hard-boundary-pairs/train.jsonl")
    assert control["--dataset"].endswith(
        "/hard-boundary-pairs/train_control.jsonl"
    )
    assert treatment["--template"] == "qwen3_execution_boundary"
    assert treatment["--loss_scale"] == "execution_boundary_asymmetric"
    assert "--loss_scale" not in control


def test_candidate_set_retry_preserves_the_balanced_optimizer_contract() -> None:
    treatment = _flags(
        MODULE.build_invocation(
            ROOT / "configs" / "train_boundary_candidate_set.yaml"
        ).command
    )
    control = _flags(
        MODULE.build_invocation(
            ROOT / "configs" / "train_boundary_candidate_set_control.yaml"
        ).command
    )

    for flag in (
        "--model",
        "--num_train_epochs",
        "--gradient_accumulation_steps",
        "--learning_rate",
        "--beta",
        "--lora_rank",
        "--seed",
        "--data_seed",
    ):
        assert treatment[flag] == control[flag]
    assert treatment["--dataset"].endswith(
        "/boundary-v2-candidate-set/hard-boundary-pairs/train.jsonl"
    )
    assert control["--dataset"].endswith(
        "/boundary-v2-candidate-set/hard-boundary-pairs/train_control.jsonl"
    )
    assert treatment["--output_dir"].endswith(
        "/boundary-v2-candidate-set/dpo-boundary"
    )
    assert control["--output_dir"].endswith(
        "/boundary-v2-candidate-set/dpo-boundary-control"
    )
    assert treatment["--num_train_epochs"] == "2"
    assert treatment["--gradient_accumulation_steps"] == "4"
    assert treatment["--template"] == "qwen3_execution_boundary"
    assert treatment["--loss_scale"] == "execution_boundary_asymmetric"
    assert "--loss_scale" not in control

    retry = yaml.safe_load(
        (ROOT / "configs" / "boundary_candidate_set.yaml").read_text()
    )
    balanced = yaml.safe_load((ROOT / "configs" / "boundary_balanced.yaml").read_text())
    assert retry["seed"] == balanced["seed"]
    assert retry["model"] == balanced["model"]
    assert retry["rollout"] == balanced["rollout"]
    assert retry["pairs"] == balanced["pairs"]
    assert retry["execution"] == balanced["execution"]
    assert retry["paths"]["output_dir"].endswith("boundary-v2-candidate-set")
