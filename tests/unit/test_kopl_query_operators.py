from __future__ import annotations

import pytest

from graphtask_r1.dsl import compile_sparql
from graphtask_r1.dsl.interventions import necessity_scores
from graphtask_r1.generation import compile_trace
from graphtask_r1.graph import InMemoryGraphBackend, toy_graph
from graphtask_r1.graph.sqlite import _compare
from graphtask_r1.schema import (
    AllEntities,
    Entity,
    EntityInfo,
    FilterType,
    Hop,
    Program,
    QueryAttribute,
    QueryRelation,
    SelectAmong,
    SelectBetween,
    Triple,
    parse_program,
    program_to_dict,
)
from graphtask_r1.verification import verify_task


@pytest.mark.parametrize(
    ("program", "values"),
    [
        (QueryAttribute(input=Entity(entity_id="alice"), attribute="age"), ("34",)),
        (
            QueryRelation(
                subject=Entity(entity_id="alice"),
                object=Entity(entity_id="acme"),
            ),
            ("works_at",),
        ),
        (
            SelectBetween(
                left=Entity(entity_id="alice"),
                right=Entity(entity_id="bob"),
                attribute="age",
                mode="max",
            ),
            ("alice",),
        ),
        (
            SelectAmong(
                input=FilterType(
                    input=AllEntities(max_results=100),
                    type_id="person",
                ),
                attribute="age",
                mode="min",
            ),
            ("bob",),
        ),
    ],
)
def test_query_operators_execute_round_trip_and_compile_trace(
    program: Program, values: tuple[str, ...]
) -> None:
    typed = parse_program(program_to_dict(program))
    graph = toy_graph()

    assert graph.execute_program(typed).values() == values
    assert graph.execute_sparql(compile_sparql(typed)).values() == values
    trace = compile_trace("new-op", "Test query", typed, graph, seed=7)
    assert trace.final_answers.values() == values
    assert trace.calls[-1].name == "final_answer"
    assert len(trace.calls) <= 3


def test_literal_query_trace_preserves_literal_answer_kind() -> None:
    program = QueryAttribute(input=Entity(entity_id="alice"), attribute="age")

    trace = compile_trace("attribute", "How old is Alice?", program, toy_graph(), seed=7)

    assert trace.final_answers.answers[0].kind == "literal"
    assert trace.observations[1].answer_kind == "literal"
    assert trace.observations[1].values == ("34",)


def test_kqapro_time_and_quantity_comparison_semantics() -> None:
    assert _compare("1906/2/5", "date", "eq", 1906, "year", None, None)
    assert not _compare("1829/1/1", "date", "gt", 1829, "year", None, None)
    assert _compare("1830/1/1", "date", "gt", 1829, "year", None, None)
    assert not _compare(
        "100",
        "quantity",
        "gt",
        50,
        "quantity",
        "square kilometre",
        "square mile",
    )


def test_invalid_selection_counterfactual_is_non_equivalent() -> None:
    graph = InMemoryGraphBackend(
        [
            Triple(subject="seed", relation="link", object="candidate"),
            Triple(subject="candidate", relation="score", object="10"),
        ],
        [
            EntityInfo(entity_id="seed", label="Seed"),
            EntityInfo(entity_id="candidate", label="Candidate"),
        ],
    )
    program = SelectAmong(
        input=Hop(input=Entity(entity_id="seed"), relation="link"),
        attribute="score",
        mode="min",
    )

    mean, minimum, components = necessity_scores(program, graph)

    assert mean > 0
    assert minimum > 0
    assert components


def test_select_between_allows_explicit_answer_options() -> None:
    program = SelectBetween(
        left=Entity(entity_id="alice"),
        right=Entity(entity_id="bob"),
        attribute="age",
        mode="max",
    )

    result = verify_task("Who is older, Alice or Bob?", program, toy_graph())

    assert result.passed
    assert result.answer_leak is False
