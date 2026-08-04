from __future__ import annotations

from typing import TYPE_CHECKING

from graphtask_r1.schema import (
    AllEntities,
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    GraphSlice,
    Hop,
    Intersect,
    Program,
    Triple,
    Union,
)

if TYPE_CHECKING:
    from graphtask_r1.graph.base import GraphBackend


def materialize_program(
    backend: GraphBackend,
    program: Program,
    *,
    snapshot_id: str,
    max_nodes: int,
    max_edges: int,
) -> GraphSlice:
    """Materialize the facts touched by a bounded core-DSL execution."""
    facts: set[Triple] = set()
    nodes: set[str] = set()
    truncated = False

    def visit(node: Program) -> None:
        nonlocal truncated
        if isinstance(node, Entity):
            nodes.add(node.entity_id)
            return
        if isinstance(node, AllEntities):
            values = backend.all_entities(limit=min(node.max_results, max_nodes + 1))
            nodes.update(values[:max_nodes])
            truncated |= len(values) > max_nodes
            return
        if isinstance(node, Hop):
            visit(node.input)
            inputs = backend.execute_program(node.input).entity_ids()
            nodes.update(inputs)
            remaining = max_edges - len(facts)
            edges = backend.neighbors(
                inputs,
                direction=node.direction,
                relation_ids=[node.relation],
                limit=max(0, remaining + 1),
            )
            if len(edges) > remaining:
                truncated = True
                edges = edges[:remaining]
            facts.update(edges)
            outputs = (
                (edge.object for edge in edges)
                if node.direction == "out"
                else (edge.subject for edge in edges)
            )
            nodes.update(outputs)
            return
        if isinstance(node, Intersect | Union):
            for branch in node.inputs:
                visit(branch)
            nodes.update(backend.execute_program(node).entity_ids())
            return
        if isinstance(node, FilterType):
            visit(node.input)
            nodes.update(backend.execute_program(node.input).entity_ids())
            return
        if isinstance(node, FilterLiteral):
            visit(node.input)
            inputs = backend.execute_program(node.input).entity_ids()
            nodes.update(inputs)
            remaining = max_edges - len(facts)
            edges = backend.neighbors(
                inputs,
                direction="out",
                relation_ids=[node.relation],
                limit=max(0, remaining + 1),
            )
            if len(edges) > remaining:
                truncated = True
                edges = edges[:remaining]
            facts.update(edges)
            return
        if isinstance(node, Count):
            visit(node.input)
            return
        raise TypeError(type(node).__name__)

    remote_answers = backend.execute_program(program)
    visit(program)
    nodes.update(remote_answers.entity_ids())
    remaining = max_edges - len(facts)
    if remaining > 0 and nodes:
        neighborhood = backend.neighbors(sorted(nodes), direction="both", limit=remaining + 1)
        if len(neighborhood) > remaining:
            truncated = True
            neighborhood = neighborhood[:remaining]
        facts.update(neighborhood)
        nodes.update(
            value
            for fact in neighborhood
            for value in (fact.subject, fact.object)
            if value.startswith(("m.", "g.", "http://", "https://", "urn:"))
        )
    if len(nodes) > max_nodes:
        truncated = True
    selected_nodes = sorted(nodes)[:max_nodes]
    selected_facts = tuple(sorted(facts, key=Triple.sort_key))[:max_edges]
    relations = sorted({fact.relation for fact in selected_facts})
    return GraphSlice(
        snapshot_id=snapshot_id,
        triples=selected_facts,
        entities=tuple(backend.entity_info(entity_id) for entity_id in selected_nodes),
        relations=tuple(backend.relation_info(relation_id) for relation_id in relations),
        complete=not truncated,
        truncated=truncated,
        remote_answers=remote_answers,
        detail="bounded program materialization",
    )
