from __future__ import annotations

from pathlib import Path

from kqapro_grr.eval_split import freeze_eval_sets

from graphtask_r1.schema import (
    AnswerSet,
    Entity,
    EntityInfo,
    TaskCertificate,
    VerificationSummary,
)
from graphtask_r1.utils import read_records, write_json, write_records


def _task(index: int) -> TaskCertificate:
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


def test_frozen_split_excludes_prior_predictions(tmp_path: Path) -> None:
    source = tmp_path / "val.parquet"
    history = tmp_path / "history"
    write_records(
        source,
        (_task(index).model_dump(mode="json") for index in range(12)),
    )
    write_records(history / "run" / "predictions.parquet", [{"task_id": "task-0"}])
    write_json(
        history / "reserved" / "eval_manifest.json",
        {"blind": {"task_ids": ["task-1"]}},
    )
    manifest = freeze_eval_sets(
        source,
        tmp_path / "split",
        exclude_roots=(history,),
        dev_count=3,
        blind_count=3,
        seed=4242,
    )
    dev = read_records(Path(str(manifest["dev"]["path"])))  # type: ignore[index]
    blind = read_records(Path(str(manifest["blind"]["path"])))  # type: ignore[index]
    ids = {str(row["task_id"]) for row in [*dev, *blind]}
    assert len(dev) == 3
    assert len(blind) == 3
    assert "task-0" not in ids
    assert "task-1" not in ids
    assert set(row["task_id"] for row in dev).isdisjoint(
        row["task_id"] for row in blind
    )
