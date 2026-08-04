from graphtask_r1.generation import ProgramSampler
from graphtask_r1.graph import toy_graph


def test_same_seed_is_exactly_reproducible() -> None:
    graph = toy_graph()
    assert ProgramSampler(graph, seed=7).signatures(100) == ProgramSampler(
        graph, seed=7
    ).signatures(100)


def test_many_random_legal_programs_execute_without_crash() -> None:
    graph = toy_graph()
    records = ProgramSampler(graph, seed=11).sample(1000)
    for record in records:
        if record.program is not None:
            graph.execute_program(record.program)
