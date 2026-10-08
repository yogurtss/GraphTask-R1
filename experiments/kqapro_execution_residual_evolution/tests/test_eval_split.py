from __future__ import annotations

from pathlib import Path

import pytest
from kqapro_ere.eval_split import freeze_eval_sets

from graphtask_r1.schema import (
    AnswerSet,
    Entity,
    EntityInfo,
    TaskCertificate,
    VerificationSummary,
)
from graphtask_r1.utils import read_records, stable_hash, write_records


def _certificate(index: int) -> TaskCertificate:
    return TaskCertificate(
        task_id=f"task-{index}",
        source="toy",
        split="val",
        question=f"Which entity is item {index}?",
        topic_entities=(EntityInfo(entity_id="alice", label="Alice"),),
        program=Entity(entity_id="alice"),
        sparql="SELECT ?x WHERE {}",
        gold_answers=AnswerSet.entities(["alice"]),
        witness_facts=(),
        program_signature=f"entity-{index}",
        program_cost=float(1 + index % 7),
        verification=VerificationSummary(executable=True),
    )


def _ids(section: object) -> list[str]:
    assert isinstance(section, dict)
    raw = section["task_ids"]
    assert isinstance(raw, list)
    return [str(value) for value in raw]


def test_freeze_sets_exclude_history_and_replay_without_overlap(tmp_path: Path) -> None:
    source = tmp_path / "val.parquet"
    history = tmp_path / "history" / "run"
    predictions = history / "predictions.parquet"
    write_records(
        source,
        (_certificate(index).model_dump(mode="json") for index in range(12)),
    )
    write_records(predictions, [{"task_id": "task-0"}])

    first = freeze_eval_sets(
        source,
        tmp_path / "freeze-a",
        exclude_roots=(tmp_path / "history",),
        dev_count=3,
        blind_count=3,
        seed=4242,
    )
    replay = freeze_eval_sets(
        source,
        tmp_path / "freeze-b",
        exclude_roots=(tmp_path / "history",),
        dev_count=3,
        blind_count=3,
        seed=4242,
    )

    dev_ids = _ids(first["dev"])
    blind_ids = _ids(first["blind"])
    assert len(dev_ids) == 3
    assert len(blind_ids) == 3
    assert set(dev_ids).isdisjoint(blind_ids)
    assert "task-0" not in {*dev_ids, *blind_ids}
    assert first["excluded_task_count"] == 1
    assert first["excluded_task_ids_sha256"] == stable_hash(["task-0"])
    assert dev_ids == _ids(replay["dev"])
    assert blind_ids == _ids(replay["blind"])
    assert first["dev_sha256"] == replay["dev_sha256"]
    assert first["blind_sha256"] == replay["blind_sha256"]

    dev_path = Path(str(first["dev"]["path"]))  # type: ignore[index]
    blind_path = Path(str(first["blind"]["path"]))  # type: ignore[index]
    assert [row["task_id"] for row in read_records(dev_path)] == dev_ids
    assert [row["task_id"] for row in read_records(blind_path)] == blind_ids


def test_freeze_sets_refuse_to_overwrite_existing_manifest(tmp_path: Path) -> None:
    source = tmp_path / "val.parquet"
    output_dir = tmp_path / "freeze"
    write_records(
        source,
        (_certificate(index).model_dump(mode="json") for index in range(8)),
    )
    freeze_eval_sets(
        source,
        output_dir,
        exclude_roots=(),
        dev_count=2,
        blind_count=2,
        seed=4242,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite frozen"):
        freeze_eval_sets(
            source,
            output_dir,
            exclude_roots=(),
            dev_count=2,
            blind_count=2,
            seed=4242,
        )


def test_freeze_sets_refuse_a_short_blind_partition(tmp_path: Path) -> None:
    source = tmp_path / "val.parquet"
    write_records(
        source,
        (_certificate(index).model_dump(mode="json") for index in range(4)),
    )

    with pytest.raises(ValueError, match="requested 5 unseen tasks"):
        freeze_eval_sets(
            source,
            tmp_path / "freeze",
            exclude_roots=(),
            dev_count=3,
            blind_count=2,
            seed=4242,
        )
