from __future__ import annotations

import json
from typing import Any, cast

from graphtask_r1.dsl import program_cost
from graphtask_r1.evaluation import answer_metrics
from graphtask_r1.generation import validate_proposal, verbalize
from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.graphscript import (
    GraphScriptError,
    execute_graphscript,
    parse_graphscript,
    program_to_graphscript,
)
from graphtask_r1.rewards import challenger_reward
from graphtask_r1.schema import AnswerSet, TaskProposal
from graphtask_r1.training.opponent import request_opponent
from graphtask_r1.training.parsing import parse_solver_output, parse_task_proposal
from graphtask_r1.training.prompts import InteractionMode
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
    raw_mode = str(info.get("interaction_mode", "tool"))
    if raw_mode not in {"tool", "graphscript"}:
        raise ValueError(f"unsupported interaction mode: {raw_mode}")
    interaction_mode = cast(InteractionMode, raw_mode)
    if data_source == "graphtask/questioner":
        try:
            graph_usage: dict[str, float] = {}
            if interaction_mode == "graphscript":
                topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
                if len(topic_ids) != 1:
                    raise GraphScriptError(
                        "INVALID_SEED", "GraphScript v0.1 requires exactly one topic entity"
                    )
                script = parse_graphscript(
                    solution_str, max_follow_limit=int(info.get("max_follow_limit", 100))
                )
                execution = execute_graphscript(
                    script,
                    backend,
                    seed_entity=topic_ids[0],
                    allowed_relations=frozenset(
                        str(value) for value in info.get("allowed_relations", [])
                    ),
                    max_edge_visits=int(info.get("max_edge_visits", 200)),
                    max_returned_entities=int(info.get("max_returned_entities", 1_000)),
                    trace_id=str(info.get("task_id", "questioner")),
                )
                if backend.execute_program(execution.program) != execution.answers:
                    raise GraphScriptError(
                        "BOUNDED_UNBOUNDED_MISMATCH",
                        "bounded GraphScript result differs from certified Program execution",
                    )
                proposal = TaskProposal(topic_entities=topic_ids, program=execution.program)
                graph_usage = {
                    "edge_visits": float(execution.usage.edge_visits),
                    "graph_calls": float(execution.usage.graph_calls),
                    "program_operators": float(execution.usage.operators),
                }
            else:
                proposal = parse_task_proposal(solution_str)
                if info.get("program_profile") == "graphscript_v0_1":
                    script = program_to_graphscript(
                        proposal.program,
                        follow_limit=int(info.get("max_follow_limit", 100)),
                    )
                    topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
                    if len(topic_ids) != 1:
                        raise GraphScriptError(
                            "INVALID_SEED", "comparison profile requires exactly one topic entity"
                        )
                    execution = execute_graphscript(
                        script,
                        backend,
                        seed_entity=topic_ids[0],
                        allowed_relations=frozenset(
                            str(value) for value in info.get("allowed_relations", [])
                        ),
                        max_edge_visits=int(info.get("max_edge_visits", 200)),
                        max_returned_entities=int(info.get("max_returned_entities", 1_000)),
                        trace_id=str(info.get("task_id", "questioner")),
                    )
                    graph_usage = {
                        "edge_visits": float(execution.usage.edge_visits),
                        "graph_calls": float(execution.usage.graph_calls),
                        "program_operators": float(execution.usage.operators),
                    }
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
                    interaction_mode=interaction_mode,
                    allowed_relations=tuple(
                        str(value) for value in info.get("allowed_relations", [])
                    ),
                    max_follow_limit=int(info.get("max_follow_limit", 100)),
                    max_edge_visits=int(info.get("max_edge_visits", 200))
                    if interaction_mode == "graphscript"
                    or info.get("program_profile") == "graphscript_v0_1"
                    else None,
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
            return {
                "score": reward.total * role_weight,
                **reward.components,
                **graph_usage,
            }
        except GraphScriptError as exc:
            return {
                "score": -1.0 * role_weight,
                "format": -1.0,
                f"reject_{exc.reason_code.lower()}": 1.0,
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {"score": -1.0 * role_weight, "format": -1.0}
    if data_source == "graphtask/solver":
        gold = AnswerSet.model_validate_json(ground_truth)
        try:
            graph_usage = {}
            if interaction_mode == "graphscript":
                topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
                if len(topic_ids) != 1:
                    raise GraphScriptError(
                        "INVALID_SEED", "GraphScript v0.1 requires exactly one topic entity"
                    )
                script = parse_graphscript(
                    solution_str, max_follow_limit=int(info.get("max_follow_limit", 100))
                )
                execution = execute_graphscript(
                    script,
                    backend,
                    seed_entity=topic_ids[0],
                    allowed_relations=frozenset(
                        str(value) for value in info.get("allowed_relations", [])
                    ),
                    max_edge_visits=int(info.get("max_edge_visits", 200)),
                    max_returned_entities=int(info.get("max_returned_entities", 1_000)),
                    trace_id=str(info.get("task_id", "solver")),
                )
                predicted = execution.answers
                graph_usage = {
                    "edge_visits": float(execution.usage.edge_visits),
                    "graph_calls": float(execution.usage.graph_calls),
                    "program_operators": float(execution.usage.operators),
                }
            else:
                count = bool(gold.answers and gold.answers[0].kind == "count")
                predicted = parse_solver_output(solution_str, count=count)
            metrics = answer_metrics(predicted, gold)
            return {"score": metrics["f1"] * role_weight, **metrics, **graph_usage}
        except GraphScriptError as exc:
            return {
                "score": 0.0,
                "format": 0.0,
                f"reject_{exc.reason_code.lower()}": 1.0,
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"score": 0.0, "format": 0.0}
    raise ValueError(f"unsupported data source: {data_source}")
