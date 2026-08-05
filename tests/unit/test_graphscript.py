import asyncio
import json

import pytest

from graphtask_r1.generation import compile_trace
from graphtask_r1.graph import toy_graph
from graphtask_r1.graphscript import (
    GraphScriptError,
    execute_graphscript,
    graphscript_to_program,
    parse_graphscript,
    program_to_graphscript,
)
from graphtask_r1.schema import AnswerSet, Entity, Hop
from graphtask_r1.training.verl_reward import compute_score


def _script(first: str = "works_at", second: str = "located_in") -> str:
    return json.dumps(
        {
            "version": "0.1",
            "ops": [
                {"op": "start", "entity": "$seed", "out": "h0"},
                {
                    "op": "follow",
                    "in": "h0",
                    "relation": first,
                    "direction": "out",
                    "limit": 100,
                    "out": "h1",
                },
                {
                    "op": "follow",
                    "in": "h1",
                    "relation": second,
                    "direction": "out",
                    "limit": 100,
                    "out": "h2",
                },
                {"op": "require_unique", "in": "h2"},
                {"op": "emit", "in": "h2"},
            ],
        }
    )


def test_graphscript_executes_and_compiles_to_certified_program() -> None:
    graph = toy_graph()
    script = parse_graphscript(_script())
    execution = execute_graphscript(
        script,
        graph,
        seed_entity="alice",
        allowed_relations=frozenset({"works_at", "located_in"}),
        max_edge_visits=10,
        trace_id="test",
    )
    assert execution.answers.values() == ("paris",)
    assert execution.usage.edge_visits == 2
    assert execution.usage.graph_calls == 2
    assert graph.execute_program(execution.program) == execution.answers
    assert graphscript_to_program(script, seed_entity="alice") == execution.program
    assert program_to_graphscript(execution.program).model_dump(by_alias=True) == script.model_dump(
        by_alias=True
    )
    tool_trace = compile_trace(
        "equivalence", "Where does Alice work?", execution.program, graph, seed=42
    )
    assert tool_trace.final_answers == execution.answers
    assert sum(len(observation.triples) for observation in tool_trace.observations) == 2


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (_script() + " trailing", "EXTRA_TEXT"),
        ("not-json", "NON_JSON"),
        (_script().replace('"version": "0.1"', '"version": "9"'), "UNSUPPORTED_VERSION"),
    ],
)
def test_graphscript_parser_preserves_structured_reasons(text: str, reason: str) -> None:
    with pytest.raises(GraphScriptError) as raised:
        parse_graphscript(text)
    assert raised.value.reason_code == reason


def test_graphscript_enforces_relation_uniqueness_and_budget() -> None:
    graph = toy_graph()
    with pytest.raises(GraphScriptError, match="RELATION_NOT_ALLOWED"):
        execute_graphscript(
            parse_graphscript(_script()),
            graph,
            seed_entity="alice",
            allowed_relations=frozenset({"works_at"}),
            max_edge_visits=10,
        )
    with pytest.raises(GraphScriptError, match="BUDGET_EXCEEDED"):
        execute_graphscript(
            parse_graphscript(_script()),
            graph,
            seed_entity="alice",
            allowed_relations=frozenset({"works_at", "located_in"}),
            max_edge_visits=1,
        )
    non_unique = _script("works_at", "works_at").replace(
        '"direction": "out", "limit": 100, "out": "h2"',
        '"direction": "in", "limit": 100, "out": "h2"',
    )
    with pytest.raises(GraphScriptError, match="NON_UNIQUE_RESULT"):
        execute_graphscript(
            parse_graphscript(non_unique),
            graph,
            seed_entity="alice",
            allowed_relations=frozenset({"works_at"}),
            max_edge_visits=10,
        )


def test_graphscript_solver_reward_executes_program() -> None:
    score = asyncio.run(
        compute_score(
            "graphtask/solver",
            _script(),
            AnswerSet.entities(["paris"]).model_dump_json(),
            {
                "interaction_mode": "graphscript",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
                "allowed_relations": ["works_at", "located_in"],
                "max_edge_visits": 10,
            },
        )
    )
    assert score["score"] == 1.0
    assert score["edge_visits"] == 2.0


def test_program_converter_rejects_non_chain_program() -> None:
    with pytest.raises(GraphScriptError, match="INVALID_SHAPE"):
        program_to_graphscript(Hop(input=Entity(entity_id="alice"), relation="friend"))
