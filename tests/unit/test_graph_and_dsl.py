import pytest

from graphtask_r1.dsl import canonical_signature, canonicalize, compile_sparql, program_cost
from graphtask_r1.graph import GraphOverlay, toy_graph
from graphtask_r1.schema import Entity, Hop, Intersect, Triple


def test_execution_and_compiled_execution_agree() -> None:
    graph = toy_graph()
    program = Hop(
        input=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        relation="located_in",
    )
    assert graph.execute_program(program).values() == ("paris",)
    assert graph.execute_sparql(compile_sparql(program)) == graph.execute_program(program)


def test_neighbors_are_sorted_and_overlay_is_immutable() -> None:
    graph = toy_graph()
    removed = Triple(subject="alice", relation="works_at", object="acme")
    overlay = graph.with_overlay(GraphOverlay(removed=(removed,)))
    assert removed in graph.triples
    assert removed not in overlay.triples
    assert graph.neighbors(["alice"], direction="both") == sorted(
        graph.neighbors(["alice"], direction="both"), key=Triple.sort_key
    )


def test_canonical_intersection_is_order_invariant_and_idempotent() -> None:
    left = Hop(input=Entity(entity_id="alice"), relation="works_at")
    right = Hop(input=Entity(entity_id="bob"), relation="works_at")
    one = canonicalize(Intersect(inputs=(left, right)))
    two = canonicalize(Intersect(inputs=(right, left)))
    assert canonical_signature(one) == canonical_signature(two)
    assert canonicalize(one) == one
    assert program_cost(one) == pytest.approx(3.5)


def test_sparql_rejects_unsafe_iri() -> None:
    with pytest.raises(ValueError, match="unsafe IRI"):
        compile_sparql(Hop(input=Entity(entity_id="alice"), relation="bad>relation"))
