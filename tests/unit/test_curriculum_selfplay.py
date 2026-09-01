from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from graphtask_r1.archive import TaskArchive
from graphtask_r1.experiments.rule_questioner import (
    RULE_QUESTIONER_VARIANT,
    export_rule_questioner_rl,
)
from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import (
    AnswerSet,
    Entity,
    EntityInfo,
    Hop,
    Program,
    TaskProposal,
    TaskTrainingRecord,
    VerificationSummary,
)
from graphtask_r1.training.selfplay import (
    SelfPlayConfig,
    _archive_growth_gate,
    _archive_round_size,
    _assemble_dataset,
    _completed_phase_adapter,
    _curriculum_max_seed_entities,
    _curriculum_phase,
    _curriculum_sample,
    _discover_curriculum_progress,
    _round_tasks,
    _solver_episode_count,
    _solver_signal_gate,
    _write_phase_manifest,
    load_selfplay_config,
    run_self_play,
)
from graphtask_r1.utils import file_hash, read_json, write_json, write_records


def _task(index: int, hops: int) -> TaskTrainingRecord:
    program: Program = Entity(entity_id=f"seed-{index}")
    for hop in range(hops):
        program = Hop(input=program, relation=f"r-{hop}")
    return TaskTrainingRecord(
        task_id=f"task-{index}",
        question=f"Question {index}?",
        topic_entities=(EntityInfo(entity_id=f"seed-{index}", label=f"Seed {index}"),),
        program=program,
        gold_answers=AnswerSet.entities([f"answer-{index}"]),
        verification=VerificationSummary(executable=True),
    )


def _config(**updates: object) -> SelfPlayConfig:
    values: dict[str, object] = {
        "initial_adapter": "adapter",
        "base_tasks": "tasks.parquet",
        "val_data": "val.parquet",
        "questioner_seeds": "seeds.parquet",
        "selfplay_variant": "curriculum_v3",
        "rounds": 3,
    }
    values.update(updates)
    return SelfPlayConfig.model_validate(values)


def test_curriculum_stages_advance_from_production_to_frontier() -> None:
    config = _config()

    assert [_curriculum_phase(config, round_index) for round_index in (1, 2, 3)] == [
        "production",
        "grounding",
        "frontier",
    ]


def test_questioner_root_curriculum_expands_only_after_grounding() -> None:
    config = _config(
        curriculum_max_seed_entities_start=1,
        curriculum_max_seed_entities_end=2,
    )

    assert [_curriculum_max_seed_entities(config, value) for value in (1, 2, 3)] == [
        1,
        1,
        2,
    ]


def test_rule_questioner_selfplay_preserves_fixed_program_prompt(tmp_path: Path) -> None:
    base_tasks = tmp_path / "base.parquet"
    seeds = tmp_path / "rule-questioner.parquet"
    candidates = tmp_path / "candidates.jsonl"
    relation_catalog = tmp_path / "relations.json"
    output = tmp_path / "round" / "mixed.parquet"
    archive = tmp_path / "archive.sqlite"
    program = Hop(input=Entity(entity_id="alice"), relation="works_at")
    task = TaskTrainingRecord(
        task_id="task-0",
        question="Where does Alice work?",
        topic_entities=(EntityInfo(entity_id="alice", label="Alice"),),
        program=program,
        gold_answers=AnswerSet.entities(["acme"]),
        verification=VerificationSummary(executable=True),
    )
    write_records(base_tasks, [task.model_dump(mode="json")])
    write_json(
        relation_catalog,
        [
            toy_graph().relation_info(relation_id).model_dump(mode="json")
            for relation_id in ("works_at", "located_in")
        ],
    )
    candidates.write_text(
        json.dumps(
            {
                "strict_certified": True,
                "topic_entities": ["alice"],
                "program": program.model_dump(mode="json"),
            }
        )
        + "\n"
    )
    export_rule_questioner_rl(
        candidates,
        seeds,
        backend=toy_graph(),
        graph_snapshot="toy-v1",
        opponent_url="http://old-opponent",
        opponent_samples=1,
    )
    original = pq.read_table(seeds).to_pylist()[0]
    config = _config(
        base_tasks=base_tasks,
        questioner_seeds=seeds,
        questioner_reward_variant=RULE_QUESTIONER_VARIANT,
        questioner_episodes=1,
        solver_episodes=1,
        base_ratio=1.0,
        archive_ratio=0.0,
        new_ratio=0.0,
        relation_catalog=relation_catalog,
        curriculum_max_seed_entities_start=1,
        curriculum_max_seed_entities_end=1,
    )

    counts = _assemble_dataset(
        config,
        archive,
        output,
        round_index=1,
        opponent_url="http://new-opponent",
    )

    assembled = pq.read_table(output.with_name("questioner.parquet")).to_pylist()[0]
    info = dict(assembled["extra_info"])
    assert counts == {"questioner": 1, "solver": 1, "total": 2}
    assert assembled["prompt"] == original["prompt"]
    assert "certified_graphscript" in assembled["prompt"][1]["content"]
    assert info["questioner_reward_variant"] == RULE_QUESTIONER_VARIANT
    assert info["fixed_program_json"] == dict(original["extra_info"])[
        "fixed_program_json"
    ]
    assert info["opponent_url"] == "http://new-opponent"
    assert info["allowed_relations"] == ["works_at"]
    assert info["opponent_request_timeout_s"] == 300.0


def test_rule_questioner_variant_rejects_legacy_selfplay() -> None:
    with pytest.raises(
        ValueError,
        match="rule_program_question_v1 requires selfplay_variant=curriculum_v3",
    ):
        _config(
            selfplay_variant="legacy",
            questioner_reward_variant=RULE_QUESTIONER_VARIANT,
        )


def test_opponent_outer_timeout_must_exceed_model_timeout() -> None:
    with pytest.raises(ValueError, match="must exceed opponent_model_request_timeout_s"):
        _config(
            opponent_request_timeout_s=120,
            opponent_model_request_timeout_s=120,
        )


def test_solver_syntax_stage_can_use_a_bounded_maintenance_budget() -> None:
    config = _config(
        solver_episodes=64,
        curriculum_production_solver_episodes=16,
    )

    assert [_solver_episode_count(config, value) for value in (1, 2, 3)] == [16, 64, 64]


def test_archive_round_size_makes_closed_loop_gate_resume_safe(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.sqlite"
    backend = toy_graph()
    first = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        ),
        backend,
        graph_snapshot="toy-v1",
        round_index=1,
    )
    second = certify_proposal(
        TaskProposal(
            topic_entities=("bob",),
            program=Hop(input=Entity(entity_id="bob"), relation="friend"),
        ),
        backend,
        graph_snapshot="toy-v1",
        round_index=2,
    )
    with TaskArchive(archive_path) as archive:
        assert archive.add(first)
        assert archive.add(second)

    assert _archive_round_size(archive_path, 1) == 1
    assert _archive_round_size(archive_path, 2) == 1
    assert _archive_round_size(archive_path, 3) == 0


def test_archive_growth_gate_allows_nonzero_shortfall_but_blocks_empty_round() -> None:
    shortfall = _archive_growth_gate(available=72, configured_minimum=128)
    empty = _archive_growth_gate(available=0, configured_minimum=128)

    assert shortfall == {
        "configured_minimum": 128,
        "available": 72,
        "shortfall": 56,
        "empty_policy": "block",
        "status": "shortfall_allowed",
        "passed": True,
    }
    assert empty["status"] == "empty_blocked"
    assert empty["passed"] is False


def test_archive_growth_gate_can_backfill_an_empty_round_when_configured() -> None:
    empty = _archive_growth_gate(
        available=0,
        configured_minimum=128,
        empty_policy="base_backfill",
    )

    assert empty == {
        "configured_minimum": 128,
        "available": 0,
        "shortfall": 128,
        "empty_policy": "base_backfill",
        "status": "empty_backfill",
        "passed": True,
    }


def test_solver_signal_gate_requires_enough_executable_rollouts() -> None:
    observed_round = {
        "roles": {
            "solver": {
                "samples": 67,
                "means": {"milestone_execution_progress": 7 / 67},
            }
        }
    }
    weak = _solver_signal_gate(
        observed_round,
        minimum_execution_rate=0.2,
    )
    empirically_ready = _solver_signal_gate(
        observed_round,
        minimum_execution_rate=0.1,
    )
    ready = _solver_signal_gate(
        {
            "roles": {
                "solver": {
                    "samples": 64,
                    "means": {"milestone_execution_progress": 0.25},
                }
            }
        },
        minimum_execution_rate=0.2,
    )

    assert weak["status"] == "insufficient"
    assert weak["effective_executions"] == pytest.approx(7.0)
    assert weak["passed"] is False
    assert empirically_ready["status"] == "satisfied"
    assert empirically_ready["passed"] is True
    assert ready["status"] == "satisfied"
    assert ready["passed"] is True


def test_solver_curriculum_expands_the_visible_structural_band() -> None:
    tasks = [_task(index, hops=index + 1) for index in range(10)]

    early = _curriculum_sample(
        tasks,
        40,
        random.Random(7),
        visible_fraction=0.4,
        replay_ratio=0.25,
    )
    mature = _curriculum_sample(
        tasks,
        40,
        random.Random(7),
        visible_fraction=1.0,
        replay_ratio=0.25,
    )

    assert {task.task_id for task in early} <= {f"task-{index}" for index in range(4)}
    assert any(task.task_id == "task-9" for task in mature)
    assert len(early) == len(mature) == 40


def test_curriculum_config_and_dry_run_are_isolated(tmp_path: Path) -> None:
    config_path = tmp_path / "curriculum.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"initial_adapter: {tmp_path / 'adapter'}",
                f"base_tasks: {tmp_path / 'tasks.parquet'}",
                f"val_data: {tmp_path / 'val.parquet'}",
                f"questioner_seeds: {tmp_path / 'seeds.parquet'}",
                "selfplay_variant: curriculum_v3",
                "rounds: 3",
                "curriculum_production_rounds: 1",
                "curriculum_grounding_rounds: 1",
            ]
        )
        + "\n"
    )

    result = run_self_play(
        config_path,
        tmp_path / "run",
        resume=False,
        dry_run=True,
    )

    assert [plan["questioner_reward"]["curriculum_phase"] for plan in result["plans"]] == [
        "production",
        "grounding",
        "frontier",
    ]
    assert all(
        plan["update_order"] == ["questioner", "archive", "solver"]
        for plan in result["plans"]
    )
    assert all(plan["questioner_adapter_in"] for plan in result["plans"])
    assert all(plan["solver_adapter_in"] for plan in result["plans"])
    assert "--candidate-archive" in result["plans"][0]["commands"]["opponent"]


def test_one_round_resume_plans_only_the_next_unfinished_round(tmp_path: Path) -> None:
    config_path = tmp_path / "curriculum.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"initial_adapter: {tmp_path / 'initial-adapter'}",
                f"base_tasks: {tmp_path / 'tasks.parquet'}",
                f"val_data: {tmp_path / 'val.parquet'}",
                f"questioner_seeds: {tmp_path / 'seeds.parquet'}",
                "selfplay_variant: curriculum_v3",
                "rounds: 3",
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    questioner_1 = _write_completed_phase(output_dir / "round_001", "questioner")
    solver_1 = _write_completed_phase(output_dir / "round_001", "solver")
    write_json(
        output_dir / "manifest.json",
        {
            "last_completed_round": 1,
            "adapter": str(solver_1),
            "questioner_adapter": str(questioner_1),
            "solver_adapter": str(solver_1),
            "config_hash": file_hash(config_path),
        },
    )

    result = run_self_play(
        config_path,
        output_dir,
        resume=True,
        dry_run=True,
        one_round=True,
    )

    assert [plan["round"] for plan in result["plans"]] == [2]
    assert result["rounds_completed"] == 1
    assert result["rounds_planned"] == 1
    assert result["rounds_remaining"] == 2
    assert result["one_round"] is True


def test_completed_phase_adapter_recovers_legacy_finished_checkpoint(tmp_path: Path) -> None:
    phase_dir = tmp_path / "questioner_update"
    adapter = phase_dir / "v0" / "checkpoint-4"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    write_json(adapter / "trainer_state.json", {"global_step": 4, "max_steps": 4})

    assert _completed_phase_adapter(phase_dir) == adapter


def test_incomplete_phase_checkpoint_is_not_reused(tmp_path: Path) -> None:
    phase_dir = tmp_path / "questioner_update"
    adapter = phase_dir / "v0" / "checkpoint-2"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    write_json(adapter / "trainer_state.json", {"global_step": 2, "max_steps": 4})

    assert _completed_phase_adapter(phase_dir) is None


def test_completed_phase_adapter_uses_highest_complete_checkpoint(tmp_path: Path) -> None:
    phase_dir = tmp_path / "questioner_update"
    checkpoint_2 = phase_dir / "v0" / "checkpoint-2"
    checkpoint_4 = phase_dir / "v0" / "checkpoint-4"
    for checkpoint, step in ((checkpoint_2, 2), (checkpoint_4, 4)):
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "adapter_model.safetensors").write_bytes(str(step).encode())
        write_json(
            checkpoint / "trainer_state.json",
            {"global_step": step, "max_steps": step},
        )

    assert _completed_phase_adapter(phase_dir) == checkpoint_4


def test_completed_phase_adapter_uses_latest_run_version_before_checkpoint(
    tmp_path: Path,
) -> None:
    phase_dir = tmp_path / "questioner_update"
    checkpoint_v0 = phase_dir / "v0" / "checkpoint-8"
    checkpoint_v1 = phase_dir / "v1" / "checkpoint-4"
    for checkpoint, step in ((checkpoint_v0, 8), (checkpoint_v1, 4)):
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}")
        (checkpoint / "adapter_model.safetensors").write_bytes(str(step).encode())
        write_json(
            checkpoint / "trainer_state.json",
            {"global_step": step, "max_steps": step},
        )

    assert _completed_phase_adapter(phase_dir) == checkpoint_v1


def test_newer_complete_checkpoint_supersedes_stale_phase_manifest(
    tmp_path: Path,
) -> None:
    phase_dir = tmp_path / "questioner_update"
    checkpoint_2 = phase_dir / "v0" / "checkpoint-8"
    checkpoint_2.mkdir(parents=True)
    (checkpoint_2 / "adapter_config.json").write_text("{}")
    (checkpoint_2 / "adapter_model.safetensors").write_bytes(b"two")
    write_json(
        phase_dir / "phase_manifest.json",
        {"completed": True, "adapter": str(checkpoint_2)},
    )
    checkpoint_4 = phase_dir / "v1" / "checkpoint-4"
    checkpoint_4.mkdir(parents=True)
    (checkpoint_4 / "adapter_config.json").write_text("{}")
    (checkpoint_4 / "adapter_model.safetensors").write_bytes(b"four")
    write_json(
        checkpoint_4 / "trainer_state.json",
        {"global_step": 4, "max_steps": 4},
    )

    assert _completed_phase_adapter(phase_dir) == checkpoint_4


def test_phase_manifest_records_replayable_artifacts(tmp_path: Path) -> None:
    phase_dir = tmp_path / "solver_update"
    adapter = phase_dir / "v0" / "checkpoint-4"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    train_data = tmp_path / "solver.parquet"
    train_data.write_bytes(b"dataset")
    trainer_log = phase_dir / "v0" / "logging.jsonl"
    trainer_log.write_text("{}\n")

    _write_phase_manifest(
        phase_dir,
        phase="solver",
        adapter=adapter,
        train_data=train_data,
        trainer_log=trainer_log,
    )

    state = read_json(phase_dir / "phase_manifest.json")
    assert state["completed"] is True
    assert state["phase"] == "solver"
    assert state["adapter"] == str(adapter.resolve())
    assert state["adapter_weights_hash"] == file_hash(adapter / "adapter_model.safetensors")
    assert state["train_data_hash"] == file_hash(train_data)


def _write_completed_phase(round_dir: Path, phase: str, step: int = 4) -> Path:
    adapter = round_dir / f"{phase}_update" / "v0" / f"checkpoint-{step}"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(phase.encode())
    write_json(adapter / "trainer_state.json", {"global_step": step, "max_steps": 4})
    return adapter


def test_resume_discovers_completed_round_without_round_manifest(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_001"
    questioner = _write_completed_phase(round_dir, "questioner")
    solver = _write_completed_phase(round_dir, "solver")

    progress = _discover_curriculum_progress(tmp_path, rounds=3)

    assert progress == {
        "last_completed_round": 1,
        "questioner_adapter": str(questioner),
        "solver_adapter": str(solver),
        "next_round": 2,
        "next_phase": "questioner",
    }


def test_resume_discovers_questioner_only_phase(tmp_path: Path) -> None:
    round_1 = tmp_path / "round_001"
    _write_completed_phase(round_1, "questioner")
    _write_completed_phase(round_1, "solver")
    questioner_2 = _write_completed_phase(tmp_path / "round_002", "questioner")

    progress = _discover_curriculum_progress(tmp_path, rounds=3)

    assert progress["last_completed_round"] == 1
    assert progress["questioner_adapter"] != questioner_2
    assert progress["next_round"] == 2
    assert progress["next_phase"] == "solver"


def test_one_round_resume_uses_filesystem_progress_ahead_of_manifest(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "curriculum.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"initial_adapter: {tmp_path / 'initial-adapter'}",
                f"base_tasks: {tmp_path / 'tasks.parquet'}",
                f"val_data: {tmp_path / 'val.parquet'}",
                f"questioner_seeds: {tmp_path / 'seeds.parquet'}",
                "selfplay_variant: curriculum_v3",
                "rounds: 3",
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "run"
    _write_completed_phase(output_dir / "round_001", "questioner")
    solver_1 = _write_completed_phase(output_dir / "round_001", "solver")
    questioner_2 = _write_completed_phase(output_dir / "round_002", "questioner")

    result = run_self_play(
        config_path,
        output_dir,
        resume=True,
        dry_run=True,
        one_round=True,
    )

    assert [plan["round"] for plan in result["plans"]] == [2]
    assert result["plans"][0]["solver_adapter_in"] == str(solver_1)
    assert result["plans"][0]["phase_resume"] == {
        "questioner": True,
        "solver": False,
    }
    assert result["resume_progress"]["next_phase"] == "solver"
    assert questioner_2.exists()


def test_exact_solver_phase_requires_completed_questioner(tmp_path: Path) -> None:
    config_path = tmp_path / "curriculum.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"initial_adapter: {tmp_path / 'initial-adapter'}",
                f"base_tasks: {tmp_path / 'tasks.parquet'}",
                f"val_data: {tmp_path / 'val.parquet'}",
                f"questioner_seeds: {tmp_path / 'seeds.parquet'}",
                "selfplay_variant: curriculum_v3",
                "rounds: 3",
            ]
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="before its Questioner completes"):
        run_self_play(
            config_path,
            tmp_path / "run",
            resume=False,
            dry_run=True,
            target_round=1,
            target_phase="solver",
        )


def test_exact_completed_phase_is_an_idempotent_noop(tmp_path: Path) -> None:
    config_path = tmp_path / "curriculum.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"initial_adapter: {tmp_path / 'initial-adapter'}",
                f"base_tasks: {tmp_path / 'tasks.parquet'}",
                f"val_data: {tmp_path / 'val.parquet'}",
                f"questioner_seeds: {tmp_path / 'seeds.parquet'}",
                "selfplay_variant: curriculum_v3",
                "rounds: 3",
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "run"
    _write_completed_phase(output_dir / "round_001", "questioner")

    result = run_self_play(
        config_path,
        output_dir,
        resume=False,
        dry_run=True,
        target_round=1,
        target_phase="questioner",
    )

    assert result["phase_skipped"] is True
    assert result["rounds_planned"] == 0


def test_exact_solver_uses_same_round_questioner_adapter(tmp_path: Path) -> None:
    config_path = tmp_path / "curriculum.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"initial_adapter: {tmp_path / 'initial-adapter'}",
                f"base_tasks: {tmp_path / 'tasks.parquet'}",
                f"val_data: {tmp_path / 'val.parquet'}",
                f"questioner_seeds: {tmp_path / 'seeds.parquet'}",
                "selfplay_variant: curriculum_v3",
                "rounds: 3",
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "run"
    _write_completed_phase(output_dir / "round_001", "questioner")
    solver_1 = _write_completed_phase(output_dir / "round_001", "solver")
    questioner_2 = _write_completed_phase(output_dir / "round_002", "questioner")

    result = run_self_play(
        config_path,
        output_dir,
        resume=False,
        dry_run=True,
        target_round=2,
        target_phase="solver",
    )

    assert result["plans"][0]["questioner_adapter_in"] == str(questioner_2)
    assert result["plans"][0]["solver_adapter_in"] == str(solver_1)


def test_curriculum_consumes_same_round_generated_tasks(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    write_records(base_path, [_task(0, 1).model_dump(mode="json")])
    generated = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=Hop(
                input=Hop(input=Entity(entity_id="alice"), relation="works_at"),
                relation="located_in",
            ),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
        round_index=2,
    )
    archive_path = tmp_path / "archive.sqlite"
    with TaskArchive(archive_path) as archive:
        assert archive.add(generated)
    config = _config(
        base_tasks=base_path,
        solver_episodes=4,
        base_ratio=0.5,
        archive_ratio=0.0,
        new_ratio=0.5,
    )

    selected = _round_tasks(config, archive_path, round_index=2)

    assert len(selected) == 4
    assert sum(task.task_id == generated.task_id for task in selected) == 1
    assert len({task.task_id for task in selected}) == 2


def test_repository_curriculum_config_is_opt_in() -> None:
    root = Path(__file__).parents[2]
    legacy = load_selfplay_config(root / "configs/training/selfplay.yaml")
    curriculum = load_selfplay_config(
        root / "configs/training/selfplay_curriculum_v3.yaml"
    )
    smoke = load_selfplay_config(
        root / "configs/training/selfplay_qwen3_0_6b_curriculum_v3_smoke.yaml"
    )
    targeted_v2 = load_selfplay_config(
        root / "configs/training/selfplay_qwen3_0_6b_kqapro_targeted_v2.yaml"
    )
    rule_questioner_4b = load_selfplay_config(
        root / "configs/training/selfplay_qwen3_4b_rule_questioner_large.yaml"
    )

    assert legacy.selfplay_variant == "legacy"
    assert curriculum.selfplay_variant == "curriculum_v3"
    assert curriculum.model_path == "Qwen/Qwen3-4B-Instruct-2507"
    assert curriculum.questioner_episodes == 1024
    assert curriculum.solver_episodes == 2048
    assert curriculum.opponent_samples == 4
    assert curriculum.curriculum_production_rounds == 0
    assert curriculum.curriculum_grounding_rounds == 0
    assert curriculum.frontier_target_start == 0.5
    assert curriculum.frontier_target_end == 0.5
    assert curriculum.curriculum_replay_ratio == 0.3
    assert curriculum.curriculum_min_archive_growth == 128
    assert curriculum.curriculum_archive_empty_policy == "base_backfill"
    assert curriculum.curriculum_min_solver_execution_rate == 0.1
    assert curriculum.archive_min_target_alignment == 0.65
    assert curriculum.learning_rate == 7e-7
    assert curriculum.kl_beta == 0.002
    assert curriculum.val_data is None
    assert curriculum.validation_samples is None
    assert curriculum.enable_grpo_validation is False
    assert curriculum.response_prefix == "<think>\\n\\n</think>\\n\\n"
    assert smoke.response_prefix == "<think>\\n\\n</think>\\n\\n"
    assert targeted_v2.questioner_episodes == 192
    assert targeted_v2.solver_episodes == 256
    assert targeted_v2.enable_grpo_validation is True
    assert rule_questioner_4b.model_path == "Qwen/Qwen3-4B-Instruct-2507"
    assert rule_questioner_4b.questioner_episodes == 1024
    assert rule_questioner_4b.solver_episodes == 2048
    assert rule_questioner_4b.opponent_samples == 4
    assert rule_questioner_4b.curriculum_min_archive_growth == 128
    assert rule_questioner_4b.curriculum_archive_empty_policy == "base_backfill"
    assert rule_questioner_4b.curriculum_min_solver_execution_rate == 0.1
    assert rule_questioner_4b.questioner_reward_variant == RULE_QUESTIONER_VARIANT
    assert rule_questioner_4b.opponent_backend == "sglang"
    assert rule_questioner_4b.opponent_max_concurrency == 8
    assert rule_questioner_4b.opponent_model_request_timeout_s == 240.0
    assert rule_questioner_4b.opponent_request_timeout_s == 300.0
    assert rule_questioner_4b.deepspeed == "zero2"
    assert rule_questioner_4b.val_data is None
    assert rule_questioner_4b.validation_samples is None
    assert rule_questioner_4b.enable_grpo_validation is False


def test_selfplay_validation_requires_an_explicit_val_dataset() -> None:
    with pytest.raises(
        ValueError, match="val_data is required when enable_grpo_validation=true"
    ):
        SelfPlayConfig.model_validate(
            {
                "initial_adapter": "adapter",
                "base_tasks": "tasks.parquet",
                "questioner_seeds": "seeds.parquet",
                "enable_grpo_validation": True,
            }
        )


def test_questioner_episode_budget_is_not_capped_at_the_old_4096_scale() -> None:
    assert _config(questioner_episodes=8_192).questioner_episodes == 8_192
