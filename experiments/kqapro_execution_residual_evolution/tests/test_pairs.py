from __future__ import annotations

import json
from pathlib import Path

from kqapro_ere.pairs import build_preference_pairs

from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import (
    AnswerSet,
    Entity,
    EntityInfo,
    Hop,
    TaskTrainingRecord,
    VerificationSummary,
)
from graphtask_r1.utils import write_json, write_records


def _tasks() -> list[TaskTrainingRecord]:
    return [
        TaskTrainingRecord(
            task_id="toy-friend",
            question="Who is Bob's friend?",
            topic_entities=(EntityInfo(entity_id="bob", label="Bob"),),
            program=Hop(input=Entity(entity_id="bob"), relation="friend"),
            gold_answers=AnswerSet.entities(["cara"]),
            verification=VerificationSummary(executable=True),
        ),
        TaskTrainingRecord(
            task_id="toy-work",
            question="Where does Alice work?",
            topic_entities=(EntityInfo(entity_id="alice", label="Alice"),),
            program=Hop(input=Entity(entity_id="alice"), relation="works_at"),
            gold_answers=AnswerSet.entities(["acme"]),
            verification=VerificationSummary(executable=True),
        ),
    ]


def _input_paths(tmp_path: Path) -> tuple[Path, Path]:
    backend = toy_graph()
    tasks_path = tmp_path / "tasks.parquet"
    catalog_path = tmp_path / "relations.json"
    if not tasks_path.exists():
        write_records(
            tasks_path,
            (task.model_dump(mode="json") for task in _tasks()),
        )
        write_json(
            catalog_path,
            [relation.model_dump(mode="json") for relation in backend.all_relation_infos()],
        )
    return tasks_path, catalog_path


def _build(tmp_path: Path, output_name: str) -> dict[str, object]:
    backend = toy_graph()
    tasks_path, catalog_path = _input_paths(tmp_path)
    return build_preference_pairs(
        tasks_path,
        catalog_path,
        tmp_path / output_name,
        backend=backend,
        count=2,
        seed=17,
        residual_profile_path=None,
        max_edge_visits=100,
        max_returned_entities=100,
    )


def test_pair_builder_writes_deterministic_standard_dpo_jsonl(tmp_path: Path) -> None:
    first = _build(tmp_path, "first")
    replay = _build(tmp_path, "replay")

    assert first["selected_pairs"] == 2
    assert first["selected_task_ids"] == replay["selected_task_ids"]
    assert first["train_sha256"] == replay["train_sha256"]
    assert first["certificates_sha256"] == replay["certificates_sha256"]
    assert (
        first["mutation_attempt_rejections_sha256"] == replay["mutation_attempt_rejections_sha256"]
    )
    assert first["residual_profile_sha256"] is None
    assert first["generation_mode"] == "dual_executed_default_single_slot_ablation"
    assert first["residual_guided"] is False
    assert first["residual_effective_signal_count"] == 0

    train_path = Path(str(first["train"]))
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row["messages"], list)
        assert [message["role"] for message in row["messages"]] == [
            "system",
            "user",
            "assistant",
        ]
        assert row["messages"][-1]["content"] != row["rejected_response"]
        assert isinstance(row["rejected_response"], str)
        assert isinstance(row["task_id"], str)
        assert isinstance(row["ere_edit_slot"], str)

    certificate_path = Path(str(first["certificates"]))
    certificates = [json.loads(line) for line in certificate_path.read_text().splitlines()]
    assert all(certificate["single_slot"] for certificate in certificates)
    assert all(certificate["both_executable"] for certificate in certificates)
    assert all(certificate["denotation_changed"] for certificate in certificates)


def test_pair_manifest_marks_effective_profile_and_writes_attempt_ledger(
    tmp_path: Path,
) -> None:
    backend = toy_graph()
    tasks_path, catalog_path = _input_paths(tmp_path)
    profile_path = tmp_path / "residual_profile.json"
    write_json(
        profile_path,
        {
            "slot_priority": ["direction"],
            "observed_values": {"direction": ["sideways"]},
            "effective_signal_count": 1,
        },
    )

    manifest = build_preference_pairs(
        tasks_path,
        catalog_path,
        tmp_path / "guided",
        backend=backend,
        count=1,
        seed=17,
        residual_profile_path=profile_path,
        max_edge_visits=100,
        max_returned_entities=100,
    )

    assert manifest["selected_pairs"] == 1
    assert manifest["generation_mode"] == "execution_residual_guided_single_slot"
    assert manifest["residual_guided"] is True
    assert manifest["residual_effective_signal_count"] == 1
    attempt_rejection_count = manifest["attempt_rejection_count"]
    assert isinstance(attempt_rejection_count, int)
    assert attempt_rejection_count >= 1
    assert manifest["attempt_reason_counts"]
    assert manifest["attempt_stage_counts"]
    ledger_path = Path(str(manifest["mutation_attempt_rejections"]))
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert ledger
    assert all(
        set(record)
        == {
            "task_id",
            "slot",
            "operation_index",
            "value_sha256",
            "stage",
            "reason_code",
        }
        for record in ledger
    )
    assert "sideways" not in ledger_path.read_text()


def test_zero_signal_profile_uses_explicit_default_ablation_mode(tmp_path: Path) -> None:
    backend = toy_graph()
    tasks_path, catalog_path = _input_paths(tmp_path)
    profile_path = tmp_path / "zero_signal_profile.json"
    write_json(
        profile_path,
        {
            "slot_priority": ["query"],
            "observed_values": {"query": ["ignored-with-zero-signal"]},
            "effective_signal_count": 0,
        },
    )

    manifest = build_preference_pairs(
        tasks_path,
        catalog_path,
        tmp_path / "ablation",
        backend=backend,
        count=1,
        seed=17,
        residual_profile_path=profile_path,
        max_edge_visits=100,
        max_returned_entities=100,
    )

    assert manifest["selected_pairs"] == 1
    assert manifest["residual_profile_sha256"] is not None
    assert manifest["generation_mode"] == "dual_executed_default_single_slot_ablation"
    assert manifest["residual_guided"] is False
    assert manifest["residual_effective_signal_count"] == 0
