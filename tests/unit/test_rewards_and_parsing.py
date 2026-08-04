import math
from pathlib import Path

from graphtask_r1.archive import TaskArchive
from graphtask_r1.pipeline import run_mini_pipeline
from graphtask_r1.rewards import frontier_reward, normalize_advantages
from graphtask_r1.schema import TaskCertificate
from graphtask_r1.training.parsing import parse_questioner_output, parse_solver_output
from graphtask_r1.utils import read_records


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


def test_archive_novelty_is_data_dependent(tmp_path: Path) -> None:
    run_mini_pipeline(tmp_path / "data", num_programs=20, seed=42)
    task = TaskCertificate.model_validate(read_records(tmp_path / "data" / "tasks.parquet")[0])
    with TaskArchive(tmp_path / "archive.sqlite") as archive:
        assert archive.novelty(task.program_signature, task.question) == (1.0, 1.0)
        assert archive.add(task)
        structural, textual = archive.novelty(task.program_signature, task.question)
        assert structural == 0.0
        assert textual == 0.0
