from pathlib import Path

import pytest

from graphtask_r1.training.evidence_selfplay_runner import (
    evidence_selfplay_plan,
    load_evidence_rl_train_config,
)


@pytest.mark.parametrize("model_size", ["qwen3_4b", "qwen3_8b"])
def test_scaleup_baseline_and_candidate_are_compute_matched(
    monkeypatch: pytest.MonkeyPatch,
    model_size: str,
) -> None:
    root = Path("experiments/kilt_ecp_scaleup/configs") / model_size
    monkeypatch.setenv("EVIDENCE_MODEL_PATH", f"models/{model_size}")
    monkeypatch.setenv("GRAPHTASK_KILT_DB", "data/kilt.sqlite")
    monkeypatch.setenv("EVIDENCE_RERANKER_PATH", "artifacts/reranker.json")
    monkeypatch.setenv("EVIDENCE_RL_DATA", "artifacts/rl-data")
    monkeypatch.setenv("EVIDENCE_RL_OUTPUT", "artifacts/output")
    monkeypatch.setenv("EVIDENCE_SEED", "113")

    baseline = load_evidence_rl_train_config(root / "search_r1.yaml")
    candidate = load_evidence_rl_train_config(root / "ecp_v4.yaml")

    assert baseline.seed == candidate.seed == 113
    assert baseline.model_path == candidate.model_path
    assert baseline.counterfactual_reranker is None
    assert candidate.counterfactual_reranker == Path("artifacts/reranker.json")
    paired = (
        "actor_gpus",
        "epochs",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "steps_per_generation",
        "solver_rollouts",
        "solver_max_completion_length",
        "solver_learning_rate",
    )
    for field in paired:
        assert getattr(baseline, field) == getattr(candidate, field)
    assert evidence_selfplay_plan(baseline)["ms_swift_version"] == "3.10.3"
    assert evidence_selfplay_plan(candidate)["uses_sft"] is False
