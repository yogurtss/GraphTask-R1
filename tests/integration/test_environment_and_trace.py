import json

from graphtask_r1.envs import SolverEnv
from graphtask_r1.generation import compile_trace
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import Entity, EpisodeInput, Hop, ToolCall


def test_environment_snapshot_restore_and_trace_replay() -> None:
    graph = toy_graph()
    program = Hop(input=Entity(entity_id="alice"), relation="friend")
    answers = graph.execute_program(program)
    episode = EpisodeInput(
        task_id="t1",
        question="Who is Alice's friend?",
        topic_entity_ids=("alice",),
        gold_answers=answers,
    )
    env = SolverEnv(graph)
    env.reset(episode, seed=42)
    env.step(
        ToolCall(
            name="search",
            arguments={"entity_ids": ["alice"], "direction": "out", "relation_ids": ["friend"]},
            trace_id="trace-1",
        )
    )
    snapshot = env.snapshot()
    json.dumps(snapshot)
    restored = SolverEnv(graph)
    restored.restore(snapshot)
    result = restored.step(
        ToolCall(name="final_answer", arguments={"answers": ["bob"]}, trace_id="trace-2")
    )
    assert result.done
    assert restored.final_answers == answers
    trace = compile_trace("t1", episode.question, program, graph, seed=42)
    assert trace.final_answers == answers
