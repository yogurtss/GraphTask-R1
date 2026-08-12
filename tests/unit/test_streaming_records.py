from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from graphtask_r1.data import audit_records
from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import Entity, Hop, TaskProposal
from graphtask_r1.utils import (
    iter_json_array,
    iter_records,
    read_records,
    record_count,
    write_records,
)


def test_iter_json_array_handles_chunk_boundaries_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text(json.dumps([{"text": "甲乙丙"}, {"text": "second"}]))

    assert list(iter_json_array(path, chunk_chars=3)) == [
        {"text": "甲乙丙"},
        {"text": "second"},
    ]
    assert list(iter_json_array(path, limit=1, chunk_chars=2)) == [{"text": "甲乙丙"}]


def test_iter_records_applies_limit_before_decoding_later_rows(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    pq.write_table(
        pa.table({"record_json": [json.dumps({"index": 0}), "not-json"]}),
        path,
    )

    assert record_count(path) == 2
    assert list(iter_records(path, limit=1, batch_size=1)) == [{"index": 0}]


def test_fast_task_audit_skips_witness_payload_but_deep_audit_checks_it(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="graphtask_r1.progress")
    task = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
    ).model_dump(mode="json")
    task["witness_facts"] = ["intentionally-invalid-witness"]
    path = tmp_path / "tasks.parquet"
    write_records(path, [task])

    fast = audit_records(path, kind="task")
    deep = audit_records(path, kind="task", deep=True)

    assert fast["passed"] is True
    assert fast["deep"] is False
    assert deep["passed"] is False
    assert deep["deep"] is True
    assert any('phase="started" completed=0 total=1' in record.message for record in caplog.records)


def test_task_audit_can_write_a_compact_training_view(tmp_path: Path) -> None:
    task = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        ),
        toy_graph(),
        graph_snapshot="toy-v1",
    ).model_dump(mode="json")
    task["witness_facts"] = [
        {
            "subject": "large",
            "relation": "payload",
            "object": "removed",
        }
    ]
    source = tmp_path / "tasks.parquet"
    output = tmp_path / "training_tasks.parquet"
    write_records(source, [task])

    result = audit_records(
        source,
        kind="task",
        training_view_output=output,
    )

    compact = read_records(output)
    assert result["passed"] is True
    assert result["training_view_output"] == str(output)
    assert len(compact) == 1
    assert compact[0]["task_id"] == task["task_id"]
    assert "witness_facts" not in compact[0]
