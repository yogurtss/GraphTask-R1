from graphtask_r1.graph.base import GraphBackend
from graphtask_r1.schema import AnswerSet, Program


def execute(program: Program, backend: GraphBackend) -> AnswerSet:
    return backend.execute_program(program)
