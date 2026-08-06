from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from graphtask_r1.graph.base import GraphBackend
from graphtask_r1.graphscript.schema import (
    BudgetUsage,
    EmitOp,
    FollowOp,
    GraphScript,
    GraphScriptError,
    RequireUniqueOp,
    StartOp,
)
from graphtask_r1.schema import AnswerSet, Entity, Hop, Program, Triple


class GraphScriptExecution(BaseModel):
    model_config = ConfigDict(frozen=True)

    program: Program
    answers: AnswerSet
    support: tuple[Triple, ...]
    relation_path: tuple[str, ...]
    usage: BudgetUsage


def graphscript_to_program(script: GraphScript, *, seed_entity: str) -> Program:
    handles: dict[str, Program] = {}
    emitted: Program | None = None
    for op in script.ops:
        if isinstance(op, StartOp):
            handles[op.out] = Entity(entity_id=seed_entity)
        elif isinstance(op, FollowOp):
            handles[op.out] = Hop(
                input=handles[op.input_handle],
                relation=op.relation,
                direction=op.direction,
            )
        elif isinstance(op, EmitOp):
            emitted = handles[op.input_handle]
    if emitted is None:
        raise GraphScriptError("MISSING_EMIT", "script does not emit a handle")
    return emitted


def program_to_graphscript(program: Program, *, follow_limit: int = 100) -> GraphScript:
    if not isinstance(program, Hop) or not isinstance(program.input, Hop):
        raise GraphScriptError("INVALID_SHAPE", "program must contain exactly two chained hops")
    first = program.input
    if not isinstance(first.input, Entity):
        raise GraphScriptError("INVALID_SHAPE", "program must start from one entity")
    return GraphScript.model_validate(
        {
            "version": "0.1",
            "ops": [
                {"op": "start", "entity": "$seed", "out": "h0"},
                {
                    "op": "follow",
                    "in": "h0",
                    "relation": first.relation,
                    "direction": first.direction,
                    "limit": follow_limit,
                    "out": "h1",
                },
                {
                    "op": "follow",
                    "in": "h1",
                    "relation": program.relation,
                    "direction": program.direction,
                    "limit": follow_limit,
                    "out": "h2",
                },
                {"op": "require_unique", "in": "h2"},
                {"op": "emit", "in": "h2"},
            ],
        }
    )


def execute_graphscript(
    script: GraphScript,
    backend: GraphBackend,
    *,
    seed_entity: str,
    allowed_relations: frozenset[str],
    max_edge_visits: int,
    max_returned_entities: int = 1_000,
    trace_id: str | None = None,
) -> GraphScriptExecution:
    handles: dict[str, set[str]] = {}
    support: set[Triple] = set()
    relation_path: list[str] = []
    edge_visits = 0
    returned_entities = 0
    graph_calls = 0
    emitted: set[str] | None = None
    for index, op in enumerate(script.ops):
        if isinstance(op, StartOp):
            handles[op.out] = {seed_entity}
        elif isinstance(op, FollowOp):
            if op.relation not in allowed_relations:
                raise GraphScriptError(
                    "RELATION_NOT_ALLOWED", f"relation is not in the episode catalog: {op.relation}"
                )
            remaining = max_edge_visits - edge_visits
            if remaining <= 0:
                raise GraphScriptError("BUDGET_EXCEEDED", "edge-visit budget exhausted")
            triples = backend.neighbors(
                sorted(handles[op.input_handle]),
                direction=op.direction,
                relation_ids=[op.relation],
                limit=min(op.limit, remaining + 1),
                trace_id=f"{trace_id or 'graphscript'}:{index}",
            )
            graph_calls += 1
            edge_visits += len(triples)
            if edge_visits > max_edge_visits:
                raise GraphScriptError("BUDGET_EXCEEDED", "edge-visit budget exceeded")
            values = {
                triple.object if op.direction == "out" else triple.subject for triple in triples
            }
            returned_entities += len(values)
            if returned_entities > max_returned_entities:
                raise GraphScriptError("BUDGET_EXCEEDED", "returned-entity budget exceeded")
            handles[op.out] = values
            support.update(triples)
            relation_path.append(op.relation)
        elif isinstance(op, RequireUniqueOp):
            values = handles[op.input_handle]
            if not values:
                raise GraphScriptError("EMPTY_RESULT", "final handle is empty")
            if len(values) != 1:
                raise GraphScriptError(
                    "NON_UNIQUE_RESULT", f"final handle contains {len(values)} entities"
                )
        elif isinstance(op, EmitOp):
            emitted = handles[op.input_handle]
    if emitted is None:
        raise GraphScriptError("MISSING_EMIT", "script did not emit an answer")
    program = graphscript_to_program(script, seed_entity=seed_entity)
    return GraphScriptExecution(
        program=program,
        answers=AnswerSet.entities(emitted),
        support=tuple(sorted(support, key=Triple.sort_key)),
        relation_path=tuple(relation_path),
        usage=BudgetUsage(
            edge_visits=edge_visits,
            operators=len(script.ops),
            returned_entities=returned_entities,
            graph_calls=graph_calls,
        ),
    )
