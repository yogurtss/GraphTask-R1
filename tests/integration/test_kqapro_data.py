import json
import logging
import sqlite3
from pathlib import Path

import pytest

from graphtask_r1.data.kqapro import KoPLConversionError, KoPLMapper, prepare_kqapro
from graphtask_r1.generation import compile_trace
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.schema import (
    AllEntities,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
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
    assert tasks[0]["witness_complete"] is False
    assert tasks[0]["witness_facts"] == []
    assert tasks[0]["generation"]["max_witness_facts"] == 0
    assert tasks[0]["generation"]["witness_omitted"] is True
    assert (
        read_records(output_dir / "train" / "traces.parquet")[0]["final_answers"]
        == tasks[0]["gold_answers"]
    )
    messages = [record.message for record in caplog.records]
    assert any('operation="data.prepare.kqapro.build_graph"' in value for value in messages)
    assert any('operation="data.prepare.kqapro.split.train"' in value for value in messages)
    assert any('phase="completed"' in value and "accepted=1" in value for value in messages)


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


def test_kqapro_reuses_matching_graph_snapshot(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"

    first = prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    second = prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    rebuilt = prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0, rebuild_graph=True)

    assert first["build"]["reused"] is False
    assert second["build"]["reused"] is True
    assert rebuilt["build"]["reused"] is False


def test_sqlite_shortcut_primitives_avoid_full_answer_materialization(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"
    prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    backend = SQLiteGraphBackend(output_dir / "graph.sqlite")
    try:
        program = Hop(input=Entity(entity_id="e_alice"), relation="friend")
        assert backend.execute_entity_ids(program) == frozenset({"e_bob"})
        assert ("friend", "out") in backend.relation_hops(("e_alice",))
        assert ("friend", "in") in backend.relation_hops(("e_bob",))
        assert {relation.relation_id for relation in backend.all_relation_infos()} == {
            "age",
            "friend",
        }
    finally:
        backend.close()


def test_kqapro_bounds_and_marks_large_causal_witness(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    kb = json.loads((raw_dir / "kb.json").read_text())
    for index in range(5):
        entity_id = f"e_friend_{index}"
        kb["entities"][entity_id] = {
            "name": f"Friend {index}",
            "instanceOf": ["c_person"],
            "attributes": [],
            "relations": [],
        }
        kb["entities"]["e_alice"]["relations"].append(
            {
                "predicate": "colleague",
                "direction": "forward",
                "object": entity_id,
                "qualifiers": {},
            }
        )
    (raw_dir / "kb.json").write_text(json.dumps(kb))
    row = {
        "id": "q-count",
        "question": "How many colleagues does Alice have?",
        "program": [
            {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
            {
                "function": "Relate",
                "dependencies": [0],
                "inputs": ["colleague", "forward"],
            },
            {"function": "Count", "dependencies": [1], "inputs": []},
        ],
        "answer": "5",
    }
    (raw_dir / "train.json").write_text(json.dumps([row]))

    output_dir = tmp_path / "processed"
    metrics = prepare_kqapro(
        raw_dir,
        output_dir,
        splits=("train",),
        max_witness_facts=2,
    )
    task = read_records(output_dir / "train" / "tasks.parquet")[0]

    assert metrics["accepted"] == 1
    assert metrics["max_witness_facts"] == 2
    assert len(task["witness_facts"]) == 2
    assert task["witness_complete"] is False
    assert task["generation"]["witness_truncated"] is True


def test_kopl_mapper_supports_query_and_selection_operators(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"
    prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    backend = SQLiteGraphBackend(output_dir / "graph.sqlite")
    mapper = KoPLMapper(backend)
    try:
        query_attribute = mapper.convert(
            [
                {"function": "Find", "dependencies": [], "inputs": ["Bob"]},
                {"function": "QueryAttr", "dependencies": [0], "inputs": ["age"]},
            ]
        )
        query_relation = mapper.convert(
            [
                {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
                {"function": "Find", "dependencies": [], "inputs": ["Bob"]},
                {"function": "QueryRelation", "dependencies": [0, 1], "inputs": []},
            ]
        )
        select_between = mapper.convert(
            [
                {"function": "Find", "dependencies": [], "inputs": ["Alice"]},
                {"function": "Find", "dependencies": [], "inputs": ["Bob"]},
                {
                    "function": "SelectBetween",
                    "dependencies": [0, 1],
                    "inputs": ["age", "greater"],
                },
            ]
        )
    finally:
        backend.close()

    assert query_attribute.op == "query_attribute"
    assert query_relation.op == "query_relation"
    assert select_between.op == "select_between"


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
        infos = backend.entity_infos(candidate_ids)
        assert infos[0].label == "Alice"
        assert infos[1].label == "Bob"
    finally:
        backend.close()


def test_sqlite_task_cache_reuses_program_and_entity_queries(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"
    prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    backend = SQLiteGraphBackend(output_dir / "graph.sqlite")
    statements: list[str] = []
    backend.connection.set_trace_callback(statements.append)
    program = parse_program(
        {
            "op": "hop",
            "input": {"op": "entity", "entity_id": "e_alice"},
            "relation": "friend",
        }
    )
    try:
        with backend.query_cache():
            expected = backend.execute_program(program)
            backend.entity_info("e_bob")
            first_query_count = len(statements)
            assert backend.execute_program(program) == expected
            backend.entity_info("e_bob")
            assert len(statements) == first_query_count
        backend.execute_program(program)
        assert len(statements) > first_query_count
    finally:
        backend.close()


def test_sqlite_pushes_find_all_filters_into_single_queries(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"
    prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    backend = SQLiteGraphBackend(output_dir / "graph.sqlite")
    statements: list[str] = []
    backend.connection.set_trace_callback(statements.append)
    all_entities = AllEntities(max_results=1_000_000)
    try:
        assert backend.execute_program(
            FilterType(input=all_entities, type_id="c_person")
        ).values() == ("e_alice", "e_bob")
        assert backend.execute_program(
            FilterLiteral(
                input=all_entities,
                relation="age",
                comparator="ge",
                value=LiteralValue(value=18, datatype="quantity", unit="year"),
            )
        ).values() == ("e_bob",)
        assert len(statements) == 2
    finally:
        backend.close()


def test_trace_filters_candidates_in_one_compact_query(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fixture(raw_dir)
    output_dir = tmp_path / "processed"
    prepare_kqapro(raw_dir, output_dir, splits=("train",), limit=0)
    backend = SQLiteGraphBackend(output_dir / "graph.sqlite")
    statements: list[str] = []
    backend.connection.set_trace_callback(statements.append)
    program = FilterType(input=AllEntities(max_results=100), type_id="c_person")
    try:
        with backend.query_cache():
            trace = compile_trace("bulk-prefetch", "Which people?", program, backend, seed=7)
        assert trace.final_answers.values() == ("e_alice", "e_bob")
        assert [call.name for call in trace.calls] == ["search", "final_answer"]
        assert trace.calls[0].arguments["query"]["root"]["kind"] == "all_entities"
        assert trace.observations[1].total_entities == 2
        assert not any(call.name == "inspect_entity" for call in trace.calls)
        assert len(statements) == 2
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
