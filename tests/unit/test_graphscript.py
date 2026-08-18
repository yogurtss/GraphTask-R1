import asyncio
import json

import pytest

from graphtask_r1.generation import compile_trace
from graphtask_r1.graph import InMemoryGraphBackend, toy_graph
from graphtask_r1.graphscript import (
    GraphScriptError,
    execute_graphscript,
    graphscript_to_program,
    parse_graphscript,
    program_to_graphscript,
)
from graphtask_r1.schema import (
    AllEntities,
    AnswerSet,
    Entity,
    EntityInfo,
    FilterType,
    Hop,
    QueryAttribute,
    SelectBetween,
    Triple,
    parse_program,
)
from graphtask_r1.training import ms_swift_reward as reward_module
from graphtask_r1.training.ms_swift_reward import compute_score


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
    assert [call.name for call in tool_trace.calls] == ["search", "final_answer"]
    assert len(tool_trace.calls[0].arguments["query"]["steps"]) == 2
    assert tool_trace.observations[1].entities[0].entity_id == "paris"


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


def test_questioner_reward_records_frozen_solver_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_opponent(*args: object, **kwargs: object) -> dict[str, float]:
        del args, kwargs
        return {
            "pass_rate": 0.5,
            "novelty_structural": 1.0,
            "novelty_textual": 0.8,
        }

    monkeypatch.setattr(reward_module, "request_opponent", fake_opponent)

    score = asyncio.run(
        compute_score(
            "graphtask/questioner",
            _script(),
            "{}",
            {
                "interaction_mode": "graphscript",
                "graphscript_version": "0.1",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
                "allowed_relations": ["works_at", "located_in"],
                "max_edge_visits": 10,
                "opponent_url": "http://unused",
                "opponent_samples": 2,
                "round": 1,
                "source_stratum": (
                    "roots=1-1|terminal=hop|nodes=1-3|"
                    "ops=entity,hop|answers=entity"
                ),
            },
        )
    )

    assert score["opponent_success_rate"] == 0.5
    assert score["reward_stage"] == 6.0
    assert score["raw_score"] > 0.2
    assert score["target_alignment"] == 1.0
    assert score["target_terminal_match"] == 1.0


@pytest.mark.parametrize(
    ("solution", "allowed_relations", "raw_score", "stage", "reason"),
    [
        ("not-json", ["works_at", "located_in"], -1.0, 0.0, "reject_non_json"),
        (
            _script() + " trailing",
            ["works_at", "located_in"],
            -0.9,
            1.0,
            "reject_extra_text",
        ),
        (
            _script().replace('"version": "0.1"', '"version": "9"'),
            ["works_at", "located_in"],
            -0.75,
            2.0,
            "reject_unsupported_version",
        ),
        (_script(), ["works_at"], -0.6, 3.0, "reject_relation_not_allowed"),
        (
            _script("friend", "located_in"),
            ["friend", "located_in"],
            -0.4,
            4.0,
            "reject_empty_result",
        ),
    ],
)
def test_questioner_reward_distinguishes_pre_certification_progress(
    solution: str,
    allowed_relations: list[str],
    raw_score: float,
    stage: float,
    reason: str,
) -> None:
    score = asyncio.run(
        compute_score(
            "graphtask/questioner",
            solution,
            "{}",
            {
                "interaction_mode": "graphscript",
                "graphscript_version": "0.1",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
                "allowed_relations": allowed_relations,
                "max_edge_visits": 10,
                "role_weight": 0.35,
            },
        )
    )

    assert score["raw_score"] == raw_score
    assert score["score"] == pytest.approx(raw_score * 0.35)
    assert score["reward_stage"] == stage
    assert score[reason] == 1.0


def test_program_converter_rejects_non_chain_program() -> None:
    with pytest.raises(GraphScriptError, match="INVALID_SHAPE"):
        program_to_graphscript(Hop(input=Entity(entity_id="alice"), relation="friend"))


def test_graphscript_v02_resolves_question_entity_and_queries_literal() -> None:
    graph = toy_graph()
    script = parse_graphscript(
        {
            "version": "0.2",
            "ops": [
                {
                    "op": "resolve_entity",
                    "query": "Alice",
                    "match": "exact",
                    "limit": 1,
                    "out": "h0",
                },
                {
                    "op": "query_attribute",
                    "in": "h0",
                    "attribute": "age",
                    "out": "h1",
                },
                {"op": "emit", "in": "h1"},
            ],
        }
    )

    execution = execute_graphscript(
        script,
        graph,
        allowed_relations=frozenset({"age"}),
        max_edge_visits=10,
    )

    assert execution.answers == AnswerSet.literals(["34"])
    assert execution.program == QueryAttribute(input=Entity(entity_id="alice"), attribute="age")
    assert execution.usage.graph_calls == 1
    assert execution.usage.edge_visits == 1


def test_graphscript_v02_round_trips_new_program_operator() -> None:
    graph = toy_graph()
    program = SelectBetween(
        left=Entity(entity_id="alice"),
        right=Entity(entity_id="bob"),
        attribute="age",
        mode="max",
    )

    script = program_to_graphscript(program, version="0.2")
    execution = execute_graphscript(
        parse_graphscript(script.model_dump(mode="json", by_alias=True)),
        graph,
        allowed_relations=frozenset({"age"}),
        max_edge_visits=10,
    )

    assert script.version == "0.2"
    assert execution.program == program
    assert execution.answers == AnswerSet.entities(["alice"])


def test_graphscript_kqapro_and_kilt_profiles_are_separate() -> None:
    qualifier_program = parse_program(
        {
            "op": "filter_qualifier",
            "input": {
                "op": "hop",
                "input": {"op": "entity", "entity_id": "alice"},
                "relation": "friend",
            },
            "qualifier": "since",
            "value": {"value": 2020, "datatype": "year"},
        }
    )

    assert program_to_graphscript(qualifier_program, version="0.3").version == "0.3"
    with pytest.raises(GraphScriptError, match="OP_NOT_IN_PROFILE"):
        program_to_graphscript(qualifier_program, version="0.2")
    with pytest.raises(GraphScriptError, match="OP_NOT_IN_PROFILE"):
        parse_graphscript(
            {
                "version": "0.3",
                "ops": [
                    {"op": "search_passage", "query": "Alice", "out": "h0"},
                    {"op": "emit", "in": "h0"},
                ],
            }
        )


def test_graphscript_v02_supports_bounded_global_filter_program() -> None:
    graph = toy_graph()
    program = FilterType(input=AllEntities(max_results=100), type_id="city")
    script = parse_graphscript(program_to_graphscript(program, version="0.2").model_dump())

    execution = execute_graphscript(
        script,
        graph,
        allowed_relations=frozenset(),
        max_edge_visits=10,
        capture_steps=True,
    )

    assert execution.program == program
    assert execution.answers == graph.execute_program(program)
    all_entities = execution.steps[0]
    filtered = execution.steps[1]
    assert all_entities.output is not None
    assert all_entities.output.kind == "program"
    assert all_entities.output.state == "deferred"
    assert all_entities.output.limit == 100
    assert filtered.input_handles["h0"].state == "deferred"
    assert filtered.output is not None
    assert filtered.output.state == "materialized"
    assert filtered.output.total_count == 2


def test_execution_trace_bounds_large_sets_and_keeps_downstream_entity() -> None:
    neighbors = tuple(f"candidate-{index:02d}" for index in range(12))
    target = neighbors[-1]
    graph = InMemoryGraphBackend(
        triples=[
            Triple(subject="root", relation="candidate", object=entity_id)
            for entity_id in neighbors
        ],
        entities=[
            EntityInfo(entity_id="root", label="Root"),
            *[
                EntityInfo(
                    entity_id=entity_id,
                    label=f"Candidate {index}",
                    type_ids=("target",) if entity_id == target else ("distractor",),
                )
                for index, entity_id in enumerate(neighbors)
            ],
        ],
    )
    script = parse_graphscript(
        {
            "version": "0.3",
            "ops": [
                {
                    "op": "resolve_entity",
                    "query": "root",
                    "match": "id",
                    "limit": 1,
                    "out": "h0",
                },
                {
                    "op": "follow",
                    "in": "h0",
                    "relation": "candidate",
                    "direction": "out",
                    "limit": 20,
                    "out": "h1",
                },
                {"op": "filter_type", "in": "h1", "type_id": "target", "out": "h2"},
                {"op": "emit", "in": "h2"},
            ],
        }
    )

    execution = execute_graphscript(
        script,
        graph,
        allowed_relations=frozenset({"candidate"}),
        max_edge_visits=20,
        capture_steps=True,
        trace_preview_limit=3,
    )

    follow = execution.steps[1]
    filtered = execution.steps[2]
    assert follow.output is not None
    assert follow.output.total_count == 12
    assert follow.output.truncated is True
    assert len(follow.output.values) == 3
    assert target in follow.output.values
    assert target in follow.retrieved_entities
    assert any(edge.object == target for edge in follow.new_evidence)
    assert filtered.selected_entities == (target,)
    assert len(filtered.discarded_entities) == 3


def test_graphscript_v02_solver_reward_does_not_require_topic_seed() -> None:
    solution = json.dumps(
        {
            "version": "0.2",
            "ops": [
                {
                    "op": "resolve_entity",
                    "query": "Alice",
                    "match": "exact",
                    "limit": 1,
                    "out": "h0",
                },
                {
                    "op": "follow",
                    "in": "h0",
                    "relation": "works_at",
                    "direction": "out",
                    "limit": 10,
                    "out": "h1",
                },
                {"op": "emit", "in": "h1"},
            ],
        }
    )
    score = asyncio.run(
        compute_score(
            "graphtask/solver",
            solution,
            AnswerSet.entities(["acme"]).model_dump_json(),
            {
                "interaction_mode": "graphscript",
                "graphscript_version": "0.2",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": [],
                "allowed_relations": ["works_at"],
                "max_edge_visits": 10,
            },
        )
    )
    assert score["score"] == 1.0
    assert score["edge_visits"] == 1.0


def test_graphscript_v03_solver_reward_uses_kqapro_profile() -> None:
    program = Hop(input=Entity(entity_id="alice"), relation="works_at")
    script = program_to_graphscript(program, version="0.3")

    score = asyncio.run(
        compute_score(
            "graphtask/solver",
            script.model_dump_json(by_alias=True),
            AnswerSet.entities(["acme"]).model_dump_json(),
            {
                "interaction_mode": "graphscript",
                "graphscript_version": "0.3",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": [],
                "allowed_relations": ["works_at"],
                "max_edge_visits": 10,
            },
        )
    )

    assert score["score"] == 1.0
    assert score["edge_visits"] == 1.0


def test_tool_comparison_questioner_cannot_change_episode_seed() -> None:
    solution = (
        '<task>{"topic_entities":["bob"],"program":'
        '{"op":"hop","input":{"op":"hop","input":{"op":"entity",'
        '"entity_id":"bob"},"relation":"works_at","direction":"out"},'
        '"relation":"located_in","direction":"out"}}</task>'
    )
    score = asyncio.run(
        compute_score(
            "graphtask/questioner",
            solution,
            "{}",
            {
                "interaction_mode": "tool",
                "program_profile": "graphscript_v0_1",
                "graph_snapshot": "toy-v1",
                "topic_entity_ids": ["alice"],
                "allowed_relations": ["works_at", "located_in"],
                "max_edge_visits": 10,
            },
        )
    )
    assert score["score"] == -0.6
    assert score["reward_stage"] == 3.0
    assert score["reject_seed_mismatch"] == 1.0
