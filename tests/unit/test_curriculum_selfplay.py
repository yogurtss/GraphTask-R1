from __future__ import annotations

import random
from pathlib import Path

from graphtask_r1.archive import TaskArchive
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
    _curriculum_phase,
    _curriculum_sample,
    _round_tasks,
    load_selfplay_config,
    run_self_play,
)
from graphtask_r1.utils import write_records


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
    assert sum(task.task_id == generated.task_id for task in selected) == 2


def test_repository_curriculum_config_is_opt_in() -> None:
    root = Path(__file__).parents[2]
    legacy = load_selfplay_config(root / "configs/training/selfplay.yaml")
    curriculum = load_selfplay_config(
        root / "configs/training/selfplay_curriculum_v3.yaml"
    )
    smoke = load_selfplay_config(
        root / "configs/training/selfplay_qwen3_0_6b_curriculum_v3_smoke.yaml"
    )

    assert legacy.selfplay_variant == "legacy"
    assert curriculum.selfplay_variant == "curriculum_v3"
    assert curriculum.frontier_target_start == 0.8
    assert curriculum.frontier_target_end == 0.5
    assert curriculum.curriculum_replay_ratio == 0.3
    assert smoke.response_prefix == "<think>\\n\\n</think>\\n\\n"
