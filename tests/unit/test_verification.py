from graphtask_r1.dsl import necessity_scores
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import Count, Entity, Hop, Intersect
from graphtask_r1.verification import answer_leak, bounded_shortcut_search, verify_task


def test_shortcut_detects_direct_relation() -> None:
    graph = toy_graph()
    program = Hop(
        input=Hop(input=Entity(entity_id="alice"), relation="friend"),
        relation="friend",
    )
    result = bounded_shortcut_search(program, graph)
    assert result.found is True
    assert result.program is not None


def test_no_shortcut_for_genuine_two_hop_path() -> None:
    graph = toy_graph()
    program = Hop(
        input=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        relation="located_in",
    )
    assert bounded_shortcut_search(program, graph).found is False


def test_redundant_intersection_has_zero_minimum_necessity() -> None:
    graph = toy_graph()
    program = Intersect(
        inputs=(
            Hop(input=Entity(entity_id="alice"), relation="works_at"),
            Hop(input=Entity(entity_id="bob"), relation="works_at"),
        )
    )
    _, minimum, _ = necessity_scores(program, graph)
    assert minimum == 0.0


def test_answer_leak_and_reason_codes() -> None:
    graph = toy_graph()
    program = Hop(input=Entity(entity_id="alice"), relation="friend")
    answers = graph.execute_program(program)
    assert answer_leak("Is Bob the answer?", answers, graph)
    result = verify_task("Is Bob the answer?", program, graph)
    assert not result.passed
    assert "ANSWER_LEAK" in result.rejection_reasons


def test_shortcut_budget_exhaustion_is_unknown() -> None:
    graph = toy_graph()
    program = Hop(
        input=Hop(input=Entity(entity_id="alice"), relation="works_at"),
        relation="located_in",
    )
    result = bounded_shortcut_search(program, graph, max_candidates=0)
    assert result.found is None


def test_non_entity_answer_cannot_match_entity_shortcut_candidates() -> None:
    graph = toy_graph()
    program = Count(input=Hop(input=Entity(entity_id="alice"), relation="friend"))

    result = bounded_shortcut_search(program, graph, max_candidates=0)

    assert result.found is False
    assert result.explored == 0
    assert result.reason == "non_entity_answer"
