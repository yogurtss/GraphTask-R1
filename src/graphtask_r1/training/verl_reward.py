from __future__ import annotations

import json
from typing import Any

from graphtask_r1.dsl import program_cost
from graphtask_r1.evaluation import answer_metrics
from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.rewards import challenger_reward
from graphtask_r1.schema import AnswerSet
from graphtask_r1.training.parsing import parse_questioner_output, parse_solver_output
from graphtask_r1.verification import verify_task


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """verl custom reward entrypoint; returns total score plus auditable components."""
    info = extra_info or {}
    backend = backend_from_snapshot(str(info.get("graph_snapshot", "toy-v1")))
    if data_source == "graphtask/questioner":
        try:
            question, _, program = parse_questioner_output(solution_str)
            result = verify_task(question, program, backend)
            reward = challenger_reward(
                result,
                pass_rate=float(info.get("opponent_pass_rate", 0.5)),
                cost=program_cost(program),
            )
            return {"score": reward.total, **reward.components}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {"score": -1.0, "format": -1.0}
    if data_source == "graphtask/solver":
        gold = AnswerSet.model_validate_json(ground_truth)
        try:
            count = bool(gold.answers and gold.answers[0].kind == "count")
            predicted = parse_solver_output(solution_str, count=count)
            metrics = answer_metrics(predicted, gold)
            return {"score": metrics["f1"], **metrics}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"score": 0.0, "format": 0.0}
    raise ValueError(f"unsupported data source: {data_source}")
