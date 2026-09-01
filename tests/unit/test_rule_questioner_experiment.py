import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from graphtask_r1.archive import TaskArchive
from graphtask_r1.experiments import rule_questioner as rule_questioner_module
from graphtask_r1.experiments.rule_questioner import (
    RULE_QUESTIONER_VARIANT,
    build_rule_questioner_mixed_sft,
    compute_rule_questioner_score,
    export_rule_questioner_rl,
    export_rule_questioner_sft,
    promote_rule_questioner_candidates,
    rule_questioner_messages,
    rule_questioner_replacement_count,
)
from graphtask_r1.generation import certify_proposal, verbalize
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import (
    AnswerSet,
    Count,
    Entity,
    EntityInfo,
    Hop,
    TaskProposal,
    VerificationSummary,
)
from graphtask_r1.training.ms_swift_data import convert_rl_row, convert_sft_row
from graphtask_r1.utils import write_records


def _program() -> Hop:
    return Hop(input=Entity(entity_id="alice"), relation="works_at", direction="out")


def _write_task(path: Path) -> None:
    graph = toy_graph()
    program = _program()
    write_records(
        path,
        [
            {
                "task_id": "task-1",
                "split": "train",
                "graph_snapshot": "toy-v1",
                "question": verbalize(program, graph),
                "topic_entities": [
                    EntityInfo(
                        entity_id="alice", label="Alice", type_ids=("person",)
                    ).model_dump(mode="json")
                ],
                "program": program.model_dump(mode="json"),
                "gold_answers": AnswerSet.entities(["acme"]).model_dump(mode="json"),
                "verification": VerificationSummary(
                    executable=True,
                    semantic_equivalent=True,
                ).model_dump(mode="json"),
            }
        ],
    )


def test_rule_questioner_sft_changes_direction_to_program_to_question(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.parquet"
    output = tmp_path / "questioner-sft.parquet"
    _write_task(tasks)
    metrics = export_rule_questioner_sft(
        tasks,
        output,
        backend=toy_graph(),
        count=1,
        seed=7,
    )
    row = pq.read_table(output).to_pylist()[0]
    converted = convert_sft_row(row)
    messages = converted["messages"]
    assert metrics["selected"] == 1
    assert isinstance(messages, list)
    assert "certified_graphscript" in messages[1]["content"]
    assert json.loads(messages[-1]["content"]) == {
        "question": verbalize(_program(), toy_graph())
    }


def test_rule_questioner_rl_keeps_program_hidden_from_completion(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "questioner-rl.parquet"
    program = _program()
    candidate = {
        "strict_certified": True,
        "topic_entities": ["alice"],
        "program": program.model_dump(mode="json"),
    }
    candidates.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    export_rule_questioner_rl(
        candidates,
        output,
        backend=toy_graph(),
        graph_snapshot="toy-v1",
        opponent_url="",
        opponent_samples=1,
    )
    row = pq.read_table(output).to_pylist()[0]
    converted = convert_rl_row(row)
    info = converted["extra_info"]
    assert info["questioner_reward_variant"] == RULE_QUESTIONER_VARIANT
    assert "fixed_program_json" in info
    assert "certified_graphscript" in converted["messages"][1]["content"]


def test_rule_questioner_reward_only_scores_generated_question() -> None:
    program = _program()
    info = {
        "questioner_reward_variant": RULE_QUESTIONER_VARIANT,
        "graph_snapshot": "toy-v1",
        "topic_entity_ids": ["alice"],
        "fixed_program_json": program.model_dump_json(),
        "question_alignment_min": 0.4,
        "role_weight": 1.0,
    }
    valid = asyncio.run(
        compute_rule_questioner_score(
            json.dumps({"question": verbalize(program, toy_graph())}),
            info,
        )
    )
    invalid = asyncio.run(compute_rule_questioner_score("not-json", info))
    assert valid["json_valid"] == 1.0
    assert valid["question_program_alignment"] == 1.0
    assert valid["score"] > invalid["score"]


def test_rule_questioner_uses_bounded_catalog_and_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_opponent(*args: object, **kwargs: object) -> dict[str, float]:
        del args
        captured.update(kwargs)
        return {
            "program_parse_rate": 1.0,
            "execution_rate_given_parse": 1.0,
            "semantic_success_given_execution": 0.5,
            "novelty_textual": 1.0,
        }

    monkeypatch.setattr(rule_questioner_module, "request_opponent", fake_opponent)
    program = _program()
    question = verbalize(program, toy_graph())
    info = {
        "graph_snapshot": "toy-v1",
        "topic_entity_ids": ["alice"],
        "fixed_program_json": program.model_dump_json(),
        "question_alignment_min": 0.4,
        "opponent_url": "http://unused",
        "opponent_samples": 4,
        "opponent_request_timeout_s": 321.0,
        "allowed_relations": ["works_at"],
    }

    asyncio.run(
        compute_rule_questioner_score(json.dumps({"question": question}), info)
    )

    assert captured["allowed_relations"] == ("works_at",)
    assert captured["restrict_relation_catalog"] is True
    assert captured["timeout_s"] == 321.0


def test_rule_questioner_reward_extracts_json_after_serving_prefix() -> None:
    program = _program()
    info = {
        "graph_snapshot": "toy-v1",
        "fixed_program_json": program.model_dump_json(),
        "question_alignment_min": 0.4,
    }
    question = verbalize(program, toy_graph())

    result = asyncio.run(
        compute_rule_questioner_score(
            f'</tool_call>diagnostic {json.dumps({"question": question})} trailing',
            info,
        )
    )

    assert result["json_valid"] == 1.0
    assert result["certified"] == 1.0
    assert result.get("reject_non_json") is None


def test_rule_questioner_prompt_never_contains_gold_answer() -> None:
    messages = rule_questioner_messages(
        _program(),
        topic_entities=("alice",),
        backend=toy_graph(),
    )
    assert "acme" not in json.dumps(messages).casefold()


def test_mixed_sft_replaces_only_questioner_rows(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.parquet"
    questioner = tmp_path / "questioner.parquet"
    baseline = tmp_path / "baseline.parquet"
    output = tmp_path / "mixed.parquet"
    _write_task(tasks)
    export_rule_questioner_sft(
        tasks,
        questioner,
        backend=toy_graph(),
        count=1,
        seed=7,
    )
    replacement = pq.read_table(questioner)
    role_index = replacement.schema.get_field_index("role")
    task_id_index = replacement.schema.get_field_index("task_id")
    solver = replacement.set_column(role_index, "role", pa.array(["solver"])).set_column(
        task_id_index,
        "task_id",
        pa.array(["solver-kept"]),
    )
    baseline_questioner = replacement.set_column(
        task_id_index,
        "task_id",
        pa.array(["old-questioner"]),
    )
    pq.write_table(pa.concat_tables([solver, baseline_questioner]), baseline)

    metrics = build_rule_questioner_mixed_sft(
        baseline,
        questioner,
        output,
        seed=42,
    )

    rows = pq.read_table(output).to_pylist()
    assert metrics["solver_rows"] == 1
    assert rule_questioner_replacement_count(baseline) == 1
    assert {row["task_id"] for row in rows} == {"solver-kept", "task-1"}
    assert next(row for row in rows if row["role"] == "solver") == solver.to_pylist()[0]


def test_rule_archive_frontier_works_with_one_deterministic_sample(
    tmp_path: Path,
) -> None:
    graph = toy_graph()
    programs = (
        _program(),
        Count(input=_program()),
        Hop(input=Entity(entity_id="alice"), relation="friend", direction="out"),
    )
    certificates = tuple(
        certify_proposal(
            TaskProposal(
                topic_entities=("alice",),
                program=program,
                paraphrase=verbalize(program, graph),
            ),
            graph,
            graph_snapshot="toy-v1",
        )
        for program in programs
    )
    variants = (
        (
            certificates[0].model_copy(update={"task_id": "hard"}),
            {"program_parse_rate": 0.0, "program_execution_rate": 0.0, "mean_f1": 0.0},
        ),
        (
            certificates[1].model_copy(update={"task_id": "frontier"}),
            {"program_parse_rate": 1.0, "program_execution_rate": 0.0, "mean_f1": 0.0},
        ),
        (
            certificates[2].model_copy(update={"task_id": "easy"}),
            {"program_parse_rate": 1.0, "program_execution_rate": 1.0, "mean_f1": 1.0},
        ),
    )
    staged = tmp_path / "staged.sqlite"
    archive = tmp_path / "archive.sqlite"
    with TaskArchive(staged) as store:
        for task, stats in variants:
            assert store.add(task.model_copy(update={"solver_stats": stats}))

    summary = promote_rule_questioner_candidates(
        staged,
        archive,
        min_difficulty=0.25,
        max_difficulty=0.75,
    )

    assert summary["accepted"] == 1
    assert summary["reason_counts"] == {"TOO_EASY": 1, "TOO_HARD": 1}
    with TaskArchive(archive) as store:
        assert [task.task_id for task in store.all()] == ["frontier"]
