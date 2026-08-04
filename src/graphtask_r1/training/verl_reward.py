from __future__ import annotations

import json
from typing import Any

from graphtask_r1.dsl import program_cost
from graphtask_r1.evaluation import answer_metrics
from graphtask_r1.generation import validate_proposal, verbalize
from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.rewards import challenger_reward
from graphtask_r1.schema import AnswerSet
from graphtask_r1.training.opponent import request_opponent
from graphtask_r1.training.parsing import parse_solver_output, parse_task_proposal
from graphtask_r1.verification import verify_task


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """verl custom reward entrypoint; returns total score plus auditable components."""
    info = extra_info or {}
    backend = backend_from_snapshot(str(info.get("graph_snapshot", "toy-v1")))
    role_weight = float(info.get("role_weight", 1.0))
    if data_source == "graphtask/questioner":
        try:
            proposal = parse_task_proposal(solution_str)
            validate_proposal(proposal)
            question = verbalize(proposal.program, backend)
            result = verify_task(question, proposal.program, backend)
            if result.passed:
                opponent_url = str(info.get("opponent_url") or "")
                if not opponent_url:
                    raise RuntimeError("questioner reward requires extra_info.opponent_url")
                evaluation = await request_opponent(
                    opponent_url,
                    proposal=proposal,
                    graph_snapshot=str(info.get("graph_snapshot", "toy-v1")),
                    samples=int(info.get("opponent_samples", 8)),
                    round_index=int(info["round"]) if info.get("round") is not None else None,
                )
                result = result.model_copy(
                    update={
                        "novelty_structural": float(evaluation["novelty_structural"]),
                        "novelty_textual": float(evaluation["novelty_textual"]),
                    }
                )
                pass_rate = float(evaluation["pass_rate"])
            else:
                pass_rate = 0.0
            reward = challenger_reward(
                result,
                pass_rate=pass_rate,
                cost=program_cost(proposal.program),
            )
            return {"score": reward.total * role_weight, **reward.components}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {"score": -1.0 * role_weight, "format": -1.0}
    if data_source == "graphtask/solver":
        gold = AnswerSet.model_validate_json(ground_truth)
        try:
            count = bool(gold.answers and gold.answers[0].kind == "count")
            predicted = parse_solver_output(solution_str, count=count)
            metrics = answer_metrics(predicted, gold)
            return {"score": metrics["f1"] * role_weight, **metrics}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"score": 0.0, "format": 0.0}
    raise ValueError(f"unsupported data source: {data_source}")
