from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphtask_r1.envs.graph_query import CompactGraphQuery, execute_compact_query
from graphtask_r1.generation import TraceCompilationError, compile_trace
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import AllEntities, Count, FilterType


def test_compact_query_executes_multiple_steps_without_intermediate_ids() -> None:
    result = execute_compact_query(
        toy_graph(),
        {
            "root": {"kind": "entities", "entity_ids": ["alice"]},
            "steps": [
                {"op": "hop", "relation": "works_at", "direction": "out"},
                {"op": "hop", "relation": "located_in", "direction": "out"},
                {"op": "filter_type", "type_ids": ["city"]},
            ],
            "limit": 10,
        },
    )

    assert [entity.entity_id for entity in result.entities] == ["paris"]
    assert result.total_entities == 1
    assert result.truncated is False


def test_compact_query_returns_exact_count_and_reports_truncation() -> None:
    graph = toy_graph()
    base = {
        "root": {"kind": "all_entities"},
        "steps": [{"op": "filter_type", "type_ids": ["person"]}],
    }

    truncated = execute_compact_query(graph, {**base, "limit": 2})
    counted = execute_compact_query(graph, {**base, "return_count": True})

    assert len(truncated.entities) == 2
    assert truncated.total_entities == 3
    assert truncated.truncated is True
    assert counted.count == 3
    assert counted.entities == ()


def test_global_query_without_filter_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CompactGraphQuery.model_validate({"root": {"kind": "all_entities"}, "steps": []})


def test_trace_compiler_uses_one_bounded_query_for_global_filter_count() -> None:
    program = Count(input=FilterType(input=AllEntities(max_results=100), type_id="person"))

    trace = compile_trace("people", "How many people?", program, toy_graph(), seed=7)

    assert trace.final_answers.values() == (3,)
    assert [call.name for call in trace.calls] == ["search", "final_answer"]
    assert trace.calls[0].arguments["query"]["return_count"] is True
    assert trace.observations[1].count == 3


def test_trace_compiler_rejects_oversized_entity_observation() -> None:
    program = FilterType(input=AllEntities(max_results=100), type_id="person")

    with pytest.raises(TraceCompilationError) as raised:
        compile_trace(
            "people",
            "Who are the people?",
            program,
            toy_graph(),
            seed=7,
            max_query_results=2,
        )

    assert raised.value.reason_code == "TRACE_ENTITY_BUDGET_EXCEEDED"
