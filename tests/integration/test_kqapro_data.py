import json
import logging
import sqlite3
from pathlib import Path

import pytest

from graphtask_r1.data.kqapro import KoPLConversionError, KoPLMapper, prepare_kqapro
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.schema import (
    Entity,
    FilterLiteral,
    FilterType,
    LiteralValue,
    Union,
    parse_program,
    program_to_dict,
)
from graphtask_r1.utils import read_records


def _write_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir()
    kb = {
        "concepts": {
            "c_person": {"name": "person", "instanceOf": []},
        },
        "entities": {
            "e_alice": {
                "name": "Alice",
                "instanceOf": ["c_person"],
                "attributes": [],
                "relations": [
                    {
                        "predicate": "friend",
                        "direction": "forward",
                        "object": "e_bob",
                        "qualifiers": {},
                    }
                ],
            },
            "e_bob": {
                "name": "Bob",
                "instanceOf": ["c_person"],
                "attributes": [
                    {
                        "key": "age",
                        "value": {"type": "quantity", "value": 30, "unit": "year"},
                        "qualifiers": {},
                    }
                ],
                "relations": [],
            },
        },
    }
    row = {
        "id": "q1",
        "question": "Who is Alice's friend?",
        "program": [
            {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
            {"function": "Relate", "dependencies": [0], "inputs": ["friend", "forward"]},
            {"function": "What", "dependencies": [1], "inputs": []},
        ],
        "answer": "Bob",
    }
    (raw_dir / "kb.json").write_text(json.dumps(kb))
    (raw_dir / "train.json").write_text(json.dumps([row]))
    (raw_dir / "val.json").write_text(json.dumps([row]))
    (raw_dir / "test.json").write_text(json.dumps([{"question": "test"}]))


def test_kqapro_prepare_executes_and_replays(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="graphtask_r1.progress")
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"
    metrics = prepare_kqapro(raw_dir, output_dir, splits=("train",), seed=7)
    assert metrics["accepted"] == 1
    tasks = read_records(output_dir / "train" / "tasks.parquet")
    assert tasks[0]["gold_answers"]["answers"][0]["value"] == "e_bob"
    assert (
        read_records(output_dir / "train" / "traces.parquet")[0]["final_answers"]
        == tasks[0]["gold_answers"]
    )
    messages = [record.message for record in caplog.records]
    assert any('operation="data.prepare.kqapro.build_graph"' in value for value in messages)
    assert any('operation="data.prepare.kqapro.split.train"' in value for value in messages)
    assert any('phase="completed"' in value and 'accepted=1' in value for value in messages)


def test_kqapro_parallel_output_matches_serial_output(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"

    prepare_kqapro(raw_dir, serial_dir, splits=("train",), seed=7, workers=1)
    prepare_kqapro(raw_dir, parallel_dir, splits=("train",), seed=7, workers=3)

    for name in ("tasks.parquet", "traces.parquet", "rejections.parquet"):
        assert read_records(serial_dir / "train" / name) == read_records(
            parallel_dir / "train" / name
        )


def test_mapper_rejects_non_core_kopl(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"
    prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    backend = SQLiteGraphBackend(output_dir / "graph.sqlite")
    mapper = KoPLMapper(backend)
    try:
        mapper.convert([{"function": "VerifyStr", "dependencies": [], "inputs": ["x"]}])
    except KoPLConversionError as exc:
        assert exc.reason_code == "UNSUPPORTED_KOPL_OPERATOR"
    else:
        raise AssertionError("unsupported KoPL must be rejected")


def test_sqlite_backend_batches_queries_below_runtime_variable_limit(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"
    prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    backend = SQLiteGraphBackend(output_dir / "graph.sqlite")
    backend.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 12)
    candidate_ids = ["e_alice", "e_bob", *(f"missing_{index}" for index in range(30))]
    program = Union(inputs=tuple(Entity(entity_id=value) for value in candidate_ids))
    try:
        edges = backend.neighbors(
            candidate_ids,
            direction="both",
            relation_ids=["friend", *(f"missing_relation_{index}" for index in range(20))],
            limit=100,
        )
        assert [edge.sort_key() for edge in edges] == [("e_alice", "friend", "e_bob")]
        assert backend.execute_program(FilterType(input=program, type_id="c_person")).values() == (
            "e_alice",
            "e_bob",
        )
        assert backend.execute_program(
            FilterLiteral(
                input=program,
                relation="age",
                comparator="ge",
                value=LiteralValue(value=18, datatype="quantity", unit="year"),
            )
        ).values() == ("e_bob",)
    finally:
        backend.close()


def test_union_and_typed_literal_round_trip() -> None:
    program = Union(
        inputs=(
            parse_program({"op": "entity", "entity_id": "a"}),
            FilterLiteral(
                input=parse_program({"op": "entity", "entity_id": "b"}),
                relation="age",
                comparator="ge",
                value=LiteralValue(value=18, datatype="quantity", unit="year"),
            ),
        )
    )
    assert parse_program(program_to_dict(program)) == program
