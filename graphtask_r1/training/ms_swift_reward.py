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
from graphtask_r1.rewards import (
    challenger_reward,
    frontier_gated_challenger_reward,
    questioner_rejection_reward,
    solver_outcome_reward,
    solver_rejection_reward,
)
from graphtask_r1.schema import AnswerSet, TaskProposal
from graphtask_r1.training.opponent import request_opponent
from graphtask_r1.training.parsing import parse_solver_output, parse_task_proposal
from graphtask_r1.training.prompts import GraphScriptVersion, InteractionMode
from graphtask_r1.training.questioner_sampling import target_structure_alignment
from graphtask_r1.verification import verify_task


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """ms-swift reward entrypoint returning the score and auditable components."""
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
            proposal_answers: AnswerSet | None = None
            if interaction_mode == "graphscript":
                topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
                raw_version = str(info.get("graphscript_version", "0.1"))
                if raw_version not in {"0.1", "0.2", "0.3"}:
                    raise GraphScriptError(
                        "UNSUPPORTED_VERSION", f"unsupported version: {raw_version}"
                    )
                graphscript_version = cast(GraphScriptVersion, raw_version)
                if graphscript_version == "0.1" and len(topic_ids) != 1:
                    raise GraphScriptError(
                        "INVALID_SEED", "GraphScript v0.1 requires exactly one topic entity"
                    )
                script = parse_graphscript(
                    solution_str, max_follow_limit=int(info.get("max_follow_limit", 100))
                )
                if script.version != graphscript_version:
                    raise GraphScriptError(
                        "VERSION_MISMATCH",
                        f"expected {graphscript_version}, got {script.version}",
                    )
                execution = execute_graphscript(
                    script,
                    backend,
                    seed_entity=topic_ids[0] if len(topic_ids) == 1 else None,
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
                proposal_answers = execution.answers
                graph_usage = {
                    "edge_visits": float(execution.usage.edge_visits),
                    "graph_calls": float(execution.usage.graph_calls),
                    "program_operators": float(execution.usage.operators),
                    "passage_searches": float(execution.usage.passage_searches),
                }
            else:
                proposal = parse_task_proposal(solution_str)
                if info.get("program_profile") == "graphscript_v0_1":
                    episode_topics = tuple(
                        sorted(str(value) for value in info.get("topic_entity_ids", []))
                    )
                    if tuple(sorted(proposal.topic_entities)) != episode_topics:
                        raise GraphScriptError(
                            "SEED_MISMATCH",
                            "comparison proposal must be rooted at the episode topic entity",
                        )
                    script = program_to_graphscript(
                        proposal.program,
                        follow_limit=int(info.get("max_follow_limit", 100)),
                    )
                    if len(episode_topics) != 1:
                        raise GraphScriptError(
                            "INVALID_SEED", "comparison profile requires exactly one topic entity"
                        )
                    execution = execute_graphscript(
                        script,
                        backend,
                        seed_entity=episode_topics[0],
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
                            "bounded comparison result differs from certified Program execution",
                        )
                    graph_usage = {
                        "edge_visits": float(execution.usage.edge_visits),
                        "graph_calls": float(execution.usage.graph_calls),
                        "program_operators": float(execution.usage.operators),
                    }
            try:
                validate_proposal(proposal)
            except ValueError as exc:
                raise GraphScriptError("PROPOSAL_ROOT_MISMATCH", str(exc)) from exc
            if proposal_answers is None:
                proposal_answers = backend.execute_program(proposal.program)
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
                    graphscript_version=cast(
                        GraphScriptVersion, str(info.get("graphscript_version", "0.1"))
                    ),
                    allowed_relations=tuple(
                        str(value) for value in info.get("allowed_relations", [])
                    ),
                    max_follow_limit=int(info.get("max_follow_limit", 100)),
                    max_edge_visits=int(info.get("max_edge_visits", 200))
                    if interaction_mode == "graphscript"
                    or info.get("program_profile") == "graphscript_v0_1"
                    else None,
                    seed=(
                        int(info["opponent_seed"])
                        if info.get("opponent_seed") is not None
                        else None
                    ),
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
            alignment_components: dict[str, float] = {}
            source_stratum = str(info.get("source_stratum") or "")
            if source_stratum:
                alignment_components = target_structure_alignment(
                    source_stratum,
                    program=proposal.program,
                    root_count=len(proposal.topic_entities),
                    answers=proposal_answers,
                )
            reward_variant = str(info.get("questioner_reward_variant", "legacy"))
            if reward_variant == "frontier_v2":
                reward = frontier_gated_challenger_reward(
                    result,
                    pass_rate=pass_rate,
                    samples=int(info.get("opponent_samples", 8)),
                    cost=program_cost(proposal.program),
                    target_alignment=alignment_components.get("target_alignment"),
                    frontier_target=float(info.get("frontier_target", 0.5)),
                    frontier_sigma=float(info.get("frontier_sigma", 0.2)),
                )
            elif reward_variant == "legacy":
                reward = challenger_reward(
                    result,
                    pass_rate=pass_rate,
                    cost=program_cost(proposal.program),
                    target_alignment=alignment_components.get("target_alignment"),
                )
            else:
                raise ValueError(f"unsupported questioner reward variant: {reward_variant}")
            return {
                "score": reward.total * role_weight,
                "raw_score": reward.total,
                "opponent_success_rate": pass_rate,
                **alignment_components,
                **reward.components,
                **{
                    f"reject_{reason.lower()}": 1.0
                    for reason in result.rejection_reasons
                },
                **graph_usage,
            }
        except GraphScriptError as exc:
            reward = questioner_rejection_reward(exc.reason_code)
            return {
                "score": reward.total * role_weight,
                "raw_score": reward.total,
                f"reject_{exc.reason_code.lower()}": 1.0,
                **reward.components,
            }
        except json.JSONDecodeError:
            reward = questioner_rejection_reward("NON_JSON")
            return {
                "score": reward.total * role_weight,
                "raw_score": reward.total,
                "reject_non_json": 1.0,
                **reward.components,
            }
        except (KeyError, TypeError, ValueError):
            reward = questioner_rejection_reward("INVALID_OUTPUT")
            return {
                "score": reward.total * role_weight,
                "raw_score": reward.total,
                "reject_invalid_output": 1.0,
                **reward.components,
            }
    if data_source == "graphtask/solver":
        gold = AnswerSet.model_validate_json(ground_truth)
        try:
            graph_usage = {}
            if interaction_mode == "graphscript":
                topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
                raw_version = str(info.get("graphscript_version", "0.1"))
                if raw_version not in {"0.1", "0.2", "0.3"}:
                    raise GraphScriptError(
                        "UNSUPPORTED_VERSION", f"unsupported version: {raw_version}"
                    )
                graphscript_version = cast(GraphScriptVersion, raw_version)
                if graphscript_version == "0.1" and len(topic_ids) != 1:
                    raise GraphScriptError(
                        "INVALID_SEED", "GraphScript v0.1 requires exactly one topic entity"
                    )
                script = parse_graphscript(
                    solution_str, max_follow_limit=int(info.get("max_follow_limit", 100))
                )
                if script.version != graphscript_version:
                    raise GraphScriptError(
                        "VERSION_MISMATCH",
                        f"expected {graphscript_version}, got {script.version}",
                    )
                execution = execute_graphscript(
                    script,
                    backend,
                    seed_entity=topic_ids[0] if len(topic_ids) == 1 else None,
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
                    "passage_searches": float(execution.usage.passage_searches),
                }
            else:
                count = bool(gold.answers and gold.answers[0].kind == "count")
                predicted = parse_solver_output(solution_str, count=count)
            metrics = answer_metrics(predicted, gold)
            reward = solver_outcome_reward(
                f1=float(metrics["f1"]), exact_match=float(metrics["exact_match"])
            )
            return {
                "score": reward.total * role_weight,
                "raw_score": reward.total,
                **metrics,
                **reward.components,
                **graph_usage,
            }
        except GraphScriptError as exc:
            reward = solver_rejection_reward(exc.reason_code)
            return {
                "score": reward.total * role_weight,
                "raw_score": reward.total,
                f"reject_{exc.reason_code.lower()}": 1.0,
                **reward.components,
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            reward = solver_rejection_reward("INVALID_OUTPUT")
            return {
                "score": reward.total * role_weight,
                "raw_score": reward.total,
                "reject_invalid_output": 1.0,
                **reward.components,
            }
    raise ValueError(f"unsupported data source: {data_source}")
