import math

from graphtask_r1.rewards import frontier_reward, normalize_advantages
from graphtask_r1.training.parsing import parse_questioner_output, parse_solver_output


def test_frontier_peaks_at_target() -> None:
    assert frontier_reward(0.5) == 1.0
    assert frontier_reward(0.5) > frontier_reward(0.0)


def test_normalization_handles_zero_variance() -> None:
    assert normalize_advantages([1.0, 1.0]) == [0.0, 0.0]
    assert all(math.isfinite(value) for value in normalize_advantages([0.0, 1.0]))


def test_role_output_parsers() -> None:
    output = (
        '<task>{"question":"Who?","topic_entities":["alice"],'
        '"program":{"op":"entity","entity_id":"alice"}}</task>'
    )
    question, topics, program = parse_questioner_output(output)
    assert question == "Who?"
    assert topics == ("alice",)
    assert program.op == "entity"
    assert parse_solver_output('<answer>["alice"]</answer>').values() == ("alice",)
