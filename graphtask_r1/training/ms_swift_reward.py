from __future__ import annotations

import json
import re
from typing import Any, cast

from graphtask_r1.dsl import program_cost
from graphtask_r1.evaluation import answer_metrics
from graphtask_r1.generation import validate_proposal, verbalize
from graphtask_r1.graph import backend_from_snapshot
from graphtask_r1.graphscript import (
    GraphScriptError,
    execute_graphscript,
    graphscript_operators,
    parse_graphscript,
    program_to_graphscript,
)
from graphtask_r1.graphscript.schema import OP_ADAPTER
from graphtask_r1.rewards import (
    challenger_reward,
    frontier_gated_challenger_reward,
    questioner_rejection_reward,
    solver_outcome_reward,
    solver_rejection_reward,
)
from graphtask_r1.rewards.curriculum import (
    OpponentSignals,
    QuestionerCurriculumStage,
    QuestionerMilestones,
    QuestionerRewardConfig,
    SolverCurriculumStage,
    SolverMilestones,
    questioner_curriculum_reward,
    solver_curriculum_reward,
)
from graphtask_r1.schema import AnswerSet, TaskProposal
from graphtask_r1.training.opponent import request_opponent
from graphtask_r1.training.parsing import (
    decode_questioner_graphscript_output,
    decode_task_proposal_output,
    parse_questioner_graphscript_output,
    parse_solver_output,
    parse_task_proposal,
)
from graphtask_r1.training.prompts import GraphScriptVersion, InteractionMode
from graphtask_r1.training.questioner_sampling import target_structure_alignment
from graphtask_r1.verification import verify_task

_GROUNDING_ALLOWED_REJECTIONS = frozenset(
    {"SHORTCUT_FOUND", "SHORTCUT_UNKNOWN", "REDUNDANT_CONDITION"}
)
_QUESTION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "are",
        "by",
        "does",
        "from",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "which",
        "who",
    }
)


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _question_alignment(generated: str, canonical: str) -> tuple[float, float, float]:
    generated_tokens = set(re.findall(r"[\w]+", generated.casefold()))
    canonical_tokens = set(re.findall(r"[\w]+", canonical.casefold()))
    overlap = len(generated_tokens.intersection(canonical_tokens))
    precision = _fraction(overlap, len(generated_tokens))
    recall = _fraction(overlap, len(canonical_tokens))
    token_f1 = (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    anchors = canonical_tokens - _QUESTION_STOPWORDS
    anchor_overlap = _fraction(len(anchors.intersection(generated_tokens)), len(anchors))
    return 0.5 * token_f1 + 0.5 * anchor_overlap, token_f1, anchor_overlap


def _truncated_graphscript_shape(
    text: str, *, expected_version: GraphScriptVersion
) -> dict[str, float]:
    """Recover bounded construction credit from an incomplete JSON prefix.

    Generation-length truncation is common early in training.  Treating every
    incomplete object as identical recreates the zero-variance reward collapse the
    curriculum is meant to avoid, so this scorer only inspects explicit JSON keys and
    operator names.  The 0.4 cap keeps any invalid JSON below a parsed object.
    """

    scale = 0.4
    version_present = bool(
        re.search(
            rf'"version"\s*:\s*"{re.escape(expected_version)}"',
            text,
        )
    )
    ops_present = bool(re.search(r'"ops"\s*:\s*\[', text))
    program_present = bool(re.search(r'"program"\s*:\s*\{', text))
    op_names = re.findall(r'"op"\s*:\s*"([^"\\]*)"', text)
    op_slots = len(re.findall(r'"op"\s*:', text))
    allowed = frozenset(graphscript_operators(expected_version))
    valid_flags = [name in allowed for name in op_names]
    denominator = max(op_slots, len(op_names))
    operation_fraction = _fraction(sum(valid_flags), denominator)
    valid_prefix = 0
    for valid in valid_flags:
        if not valid:
            break
        valid_prefix += 1
    return {
        "code_present": float(version_present or ops_present or program_present),
        "schema_fraction": scale
        * (float(version_present) + float(ops_present) + operation_fraction)
        / 3.0,
        "valid_operation_fraction": scale * operation_fraction,
        "valid_prefix_fraction": scale * _fraction(valid_prefix, denominator),
    }


def _truncated_question_present(text: str) -> float:
    return float(
        bool(re.search(r'"question"\s*:\s*"(?:[^"\\]|\\.)+"', text))
    )


def _questioner_program_shape(
    raw_program: object, *, expected_version: GraphScriptVersion
) -> dict[str, float]:
    if not isinstance(raw_program, dict):
        return {
            "code_present": 0.0,
            "schema_fraction": 0.0,
            "valid_operation_fraction": 0.0,
            "valid_prefix_fraction": 0.0,
        }
    raw_ops = raw_program.get("ops")
    ops = raw_ops if isinstance(raw_ops, list) else []
    allowed = frozenset(graphscript_operators(expected_version))
    valid_flags: list[bool] = []
    for op in ops:
        if not isinstance(op, dict) or not isinstance(op.get("op"), str) or op["op"] not in allowed:
            valid_flags.append(False)
            continue
        try:
            OP_ADAPTER.validate_python(op)
        except ValueError:
            valid_flags.append(False)
        else:
            valid_flags.append(True)
    valid_operations = sum(valid_flags)
    valid_prefix = 0
    for valid in valid_flags:
        if not valid:
            break
        valid_prefix += 1
    operation_fraction = _fraction(valid_operations, len(ops))
    return {
        "code_present": 1.0,
        "schema_fraction": (
            float(raw_program.get("version") == expected_version)
            + float(isinstance(raw_ops, list) and bool(ops))
            + operation_fraction
        )
        / 3.0,
        "valid_operation_fraction": operation_fraction,
        "valid_prefix_fraction": _fraction(valid_prefix, len(ops)),
    }


def _relation_valid_fraction(raw_program: object, allowed_relations: frozenset[str]) -> float:
    if not isinstance(raw_program, dict) or not isinstance(raw_program.get("ops"), list):
        return 0.0
    relation_values: list[str] = []
    for raw_op in raw_program["ops"]:
        if not isinstance(raw_op, dict):
            continue
        for name in ("relation", "attribute", "qualifier"):
            value = raw_op.get(name)
            if isinstance(value, str):
                relation_values.append(value)
    if not relation_values:
        return 1.0
    return _fraction(
        sum(value in allowed_relations for value in relation_values), len(relation_values)
    )


def _seed_coverage(raw_program: object, topic_ids: tuple[str, ...]) -> float:
    if not topic_ids or not isinstance(raw_program, dict):
        return 0.0
    if raw_program.get("version") == "0.1":
        return float(len(topic_ids) == 1)
    raw_ops = raw_program.get("ops")
    if not isinstance(raw_ops, list):
        return 0.0
    roots = {
        str(op.get("query"))
        for op in raw_ops
        if isinstance(op, dict)
        and op.get("op") == "resolve_entity"
        and op.get("match") == "id"
    }
    return _fraction(len(roots.intersection(topic_ids)), len(set(topic_ids)))


def _questioner_milestones(values: dict[str, float]) -> QuestionerMilestones:
    return QuestionerMilestones(
        question_present=values.get("question_present", 0.0),
        code_present=values.get("code_present", 0.0),
        json_valid=values.get("json_valid", 0.0),
        schema_fraction=values.get("schema_fraction", 0.0),
        valid_operation_fraction=values.get("valid_operation_fraction", 0.0),
        valid_prefix_fraction=values.get("valid_prefix_fraction", 0.0),
        seed_coverage=values.get("seed_coverage", 0.0),
        relation_valid_fraction=values.get("relation_valid_fraction", 0.0),
        handle_valid_fraction=values.get("handle_valid_fraction", 0.0),
        type_valid_fraction=values.get("type_valid_fraction", 0.0),
        executable_prefix_fraction=values.get("executable_prefix_fraction", 0.0),
        program_executable=values.get("program_executable", 0.0),
        answer_nonempty=values.get("answer_nonempty", 0.0),
        cardinality_valid=values.get("cardinality_valid", 0.0),
        question_program_alignment=values.get("question_program_alignment", 0.0),
        no_answer_leak=values.get("no_answer_leak", 0.0),
        certified=values.get("certified", 0.0),
    )


def _curriculum_questioner_result(
    values: dict[str, float],
    *,
    stage: QuestionerCurriculumStage,
    role_weight: float,
    frontier_target: float,
    frontier_sigma: float,
    opponent: OpponentSignals | None = None,
    rejection_reasons: tuple[str, ...] = (),
    extra_components: dict[str, float] | None = None,
) -> dict[str, float]:
    reward = questioner_curriculum_reward(
        _questioner_milestones(values),
        stage=stage,
        opponent=opponent,
        config=QuestionerRewardConfig(
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
        ),
    )
    return {
        "score": reward.total * role_weight,
        "raw_score": reward.total,
        "executable": values.get("program_executable", 0.0),
        "answer_nonempty": values.get("answer_nonempty", 0.0),
        "cardinality_valid": values.get("cardinality_valid", 0.0),
        "certified": values.get("certified", 0.0),
        **reward.components,
        **{
            f"reject_{reason.lower()}": 1.0
            for reason in rejection_reasons
        },
        **(extra_components or {}),
    }


async def _compute_curriculum_questioner_score(
    solution_str: str,
    info: dict[str, Any],
    *,
    role_weight: float,
) -> dict[str, float]:
    raw_stage = str(info.get("curriculum_phase", "production"))
    if raw_stage not in {"production", "grounding", "frontier"}:
        raise ValueError(f"unsupported Questioner curriculum phase: {raw_stage}")
    stage = cast(QuestionerCurriculumStage, raw_stage)
    raw_version = str(info.get("graphscript_version", "0.3"))
    if raw_version not in {"0.1", "0.2", "0.3"}:
        raise ValueError(f"unsupported GraphScript version: {raw_version}")
    graphscript_version = cast(GraphScriptVersion, raw_version)
    topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
    allowed_relations = frozenset(str(value) for value in info.get("allowed_relations", []))
    frontier_target = float(info.get("frontier_target", 0.5))
    frontier_sigma = float(info.get("frontier_sigma", 0.2))
    values: dict[str, float] = {}
    rejection_reasons: tuple[str, ...] = ()
    extra_components: dict[str, float] = {}

    try:
        decoded = decode_questioner_graphscript_output(solution_str)
    except json.JSONDecodeError:
        values["question_present"] = _truncated_question_present(solution_str)
        values.update(
            _truncated_graphscript_shape(
                solution_str,
                expected_version=graphscript_version,
            )
        )
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=("NON_JSON",),
        )
    values["json_valid"] = 1.0
    values["question_present"] = float(decoded.question is not None)
    values.update(
        _questioner_program_shape(decoded.program, expected_version=graphscript_version)
    )
    values["seed_coverage"] = _seed_coverage(decoded.program, topic_ids)
    values["relation_valid_fraction"] = _relation_valid_fraction(
        decoded.program, allowed_relations
    )
    try:
        question, script = parse_questioner_graphscript_output(
            solution_str,
            max_follow_limit=int(info.get("max_follow_limit", 100)),
        )
        if script.version != graphscript_version:
            raise GraphScriptError(
                "VERSION_MISMATCH",
                f"expected {graphscript_version}, got {script.version}",
            )
    except GraphScriptError as exc:
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=(exc.reason_code,),
        )
    values.update(
        {
            "schema_fraction": 1.0,
            "valid_operation_fraction": 1.0,
            "valid_prefix_fraction": 1.0,
            "handle_valid_fraction": 1.0,
            "type_valid_fraction": 1.0,
        }
    )
    backend = backend_from_snapshot(str(info.get("graph_snapshot", "toy-v1")))
    try:
        execution = execute_graphscript(
            script,
            backend,
            seed_entity=topic_ids[0] if len(topic_ids) == 1 else None,
            allowed_relations=allowed_relations,
            max_edge_visits=int(info.get("max_edge_visits", 200)),
            max_returned_entities=int(info.get("max_returned_entities", 1_000)),
            trace_id=str(info.get("task_id", "questioner")),
        )
        if backend.execute_program(execution.program) != execution.answers:
            raise GraphScriptError(
                "BOUNDED_UNBOUNDED_MISMATCH",
                "bounded GraphScript result differs from certified Program execution",
            )
        proposal = TaskProposal(
            topic_entities=topic_ids,
            program=execution.program,
            paraphrase=question,
        )
        validate_proposal(proposal)
    except GraphScriptError as exc:
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=(exc.reason_code,),
        )
    except ValueError:
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=("PROPOSAL_ROOT_MISMATCH",),
        )
    values.update(
        {
            "seed_coverage": 1.0,
            "executable_prefix_fraction": 1.0,
            "program_executable": 1.0,
            "answer_nonempty": float(bool(execution.answers.answers)),
        }
    )
    extra_components.update(
        {
            "edge_visits": float(execution.usage.edge_visits),
            "graph_calls": float(execution.usage.graph_calls),
            "program_operators": float(execution.usage.operators),
            "passage_searches": float(execution.usage.passage_searches),
        }
    )
    if question is None:
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=("MISSING_QUESTION",),
            extra_components=extra_components,
        )

    alignment, token_f1, anchor_overlap = _question_alignment(
        question, verbalize(execution.program, backend)
    )
    values["question_program_alignment"] = alignment
    extra_components.update(
        {
            "question_alignment_token_f1": token_f1,
            "question_alignment_anchor_overlap": anchor_overlap,
        }
    )
    result = verify_task(question, execution.program, backend)
    values.update(
        {
            "answer_nonempty": float(result.answer_nonempty),
            "cardinality_valid": float(result.cardinality_valid),
            "no_answer_leak": float(not result.answer_leak),
            "certified": float(result.passed),
        }
    )
    rejection_reasons = result.rejection_reasons
    eligible_for_opponent = (
        stage != "production"
        and result.executable
        and result.answer_nonempty
        and result.cardinality_valid
        and not result.answer_leak
        and alignment >= float(info.get("question_alignment_min", 0.35))
        and (
            result.passed
            or (
                stage == "grounding"
                and set(result.rejection_reasons).issubset(_GROUNDING_ALLOWED_REJECTIONS)
            )
        )
    )
    opponent: OpponentSignals | None = None
    if eligible_for_opponent:
        opponent_url = str(info.get("opponent_url") or "")
        if not opponent_url:
            raise RuntimeError("Questioner grounding/frontier reward requires opponent_url")
        evaluation = await request_opponent(
            opponent_url,
            proposal=proposal,
            graph_snapshot=str(info.get("graph_snapshot", "toy-v1")),
            samples=int(info.get("opponent_samples", 8)),
            round_index=int(info["round"]) if info.get("round") is not None else None,
            interaction_mode="graphscript",
            graphscript_version=graphscript_version,
            allowed_relations=tuple(allowed_relations),
            max_follow_limit=int(info.get("max_follow_limit", 100)),
            max_edge_visits=int(info.get("max_edge_visits", 200)),
            seed=int(info["opponent_seed"]) if info.get("opponent_seed") is not None else None,
            generated_question=question,
            allowed_rejection_reasons=(
                _GROUNDING_ALLOWED_REJECTIONS if stage == "grounding" else frozenset()
            ),
            recover_invalid_tool_calls=True,
        )
        opponent = OpponentSignals(
            parse_rate=float(evaluation["program_parse_rate"]),
            execution_rate_given_parse=float(evaluation["execution_rate_given_parse"]),
            semantic_success_given_execution=float(
                evaluation["semantic_success_given_execution"]
            ),
        )
        extra_components.update(
            {
                "opponent_success_rate": float(evaluation["pass_rate"]),
                "novelty_structural": float(evaluation["novelty_structural"]),
                "novelty_textual": float(evaluation["novelty_textual"]),
            }
        )
    return _curriculum_questioner_result(
        values,
        stage=stage,
        role_weight=role_weight,
        frontier_target=frontier_target,
        frontier_sigma=frontier_sigma,
        opponent=opponent,
        rejection_reasons=rejection_reasons,
        extra_components=extra_components,
    )


async def _compute_curriculum_tool_questioner_score(
    solution_str: str,
    info: dict[str, Any],
    *,
    role_weight: float,
) -> dict[str, float]:
    raw_stage = str(info.get("curriculum_phase", "production"))
    if raw_stage not in {"production", "grounding", "frontier"}:
        raise ValueError(f"unsupported Questioner curriculum phase: {raw_stage}")
    stage = cast(QuestionerCurriculumStage, raw_stage)
    raw_version = str(info.get("graphscript_version", "0.3"))
    if raw_version not in {"0.1", "0.2", "0.3"}:
        raise ValueError(f"unsupported GraphScript version: {raw_version}")
    graphscript_version = cast(GraphScriptVersion, raw_version)
    frontier_target = float(info.get("frontier_target", 0.5))
    frontier_sigma = float(info.get("frontier_sigma", 0.2))
    values: dict[str, float] = {}
    extra_components: dict[str, float] = {}

    try:
        raw_payload = decode_task_proposal_output(solution_str)
    except json.JSONDecodeError:
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=("NON_JSON",),
        )
    except ValueError:
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=("INVALID_OUTPUT",),
        )
    raw_question = raw_payload.get("question", raw_payload.get("paraphrase"))
    raw_program = raw_payload.get("program")
    values.update(
        {
            "json_valid": 1.0,
            "question_present": float(
                isinstance(raw_question, str) and bool(raw_question.strip())
            ),
            "code_present": float(isinstance(raw_program, dict)),
            "schema_fraction": (
                float(isinstance(raw_program, dict))
                + float(isinstance(raw_payload.get("topic_entities"), list))
            )
            / 2.0,
            "valid_operation_fraction": float(
                isinstance(raw_program, dict) and isinstance(raw_program.get("op"), str)
            ),
            "valid_prefix_fraction": float(isinstance(raw_program, dict)),
        }
    )
    try:
        proposal = parse_task_proposal(solution_str)
        validate_proposal(proposal)
    except (KeyError, TypeError, ValueError):
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=("INVALID_OUTPUT",),
        )
    values.update(
        {
            "schema_fraction": 1.0,
            "valid_operation_fraction": 1.0,
            "valid_prefix_fraction": 1.0,
            "seed_coverage": 1.0,
            "relation_valid_fraction": 1.0,
            "handle_valid_fraction": 1.0,
            "type_valid_fraction": 1.0,
        }
    )
    backend = backend_from_snapshot(str(info.get("graph_snapshot", "toy-v1")))
    try:
        answers = backend.execute_program(proposal.program)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=("EXECUTION_ERROR",),
        )
    values.update(
        {
            "executable_prefix_fraction": 1.0,
            "program_executable": 1.0,
            "answer_nonempty": float(bool(answers.answers)),
        }
    )
    extra_components["program_cost"] = program_cost(proposal.program)
    question = proposal.paraphrase.strip() if proposal.paraphrase is not None else None
    if not question:
        return _curriculum_questioner_result(
            values,
            stage=stage,
            role_weight=role_weight,
            frontier_target=frontier_target,
            frontier_sigma=frontier_sigma,
            rejection_reasons=("MISSING_QUESTION",),
            extra_components=extra_components,
        )

    alignment, token_f1, anchor_overlap = _question_alignment(
        question, verbalize(proposal.program, backend)
    )
    values["question_program_alignment"] = alignment
    extra_components.update(
        {
            "question_alignment_token_f1": token_f1,
            "question_alignment_anchor_overlap": anchor_overlap,
        }
    )
    result = verify_task(question, proposal.program, backend)
    values.update(
        {
            "answer_nonempty": float(result.answer_nonempty),
            "cardinality_valid": float(result.cardinality_valid),
            "no_answer_leak": float(not result.answer_leak),
            "certified": float(result.passed),
        }
    )
    eligible_for_opponent = (
        stage != "production"
        and result.executable
        and result.answer_nonempty
        and result.cardinality_valid
        and not result.answer_leak
        and alignment >= float(info.get("question_alignment_min", 0.35))
        and (
            result.passed
            or (
                stage == "grounding"
                and set(result.rejection_reasons).issubset(_GROUNDING_ALLOWED_REJECTIONS)
            )
        )
    )
    opponent: OpponentSignals | None = None
    if eligible_for_opponent:
        opponent_url = str(info.get("opponent_url") or "")
        if not opponent_url:
            raise RuntimeError("Questioner grounding/frontier reward requires opponent_url")
        allowed_relations = tuple(str(value) for value in info.get("allowed_relations", []))
        evaluation = await request_opponent(
            opponent_url,
            proposal=proposal,
            graph_snapshot=str(info.get("graph_snapshot", "toy-v1")),
            samples=int(info.get("opponent_samples", 8)),
            round_index=int(info["round"]) if info.get("round") is not None else None,
            interaction_mode="tool",
            graphscript_version=graphscript_version,
            allowed_relations=allowed_relations,
            max_follow_limit=int(info.get("max_follow_limit", 100)),
            max_edge_visits=int(info.get("max_edge_visits", 200)),
            seed=int(info["opponent_seed"]) if info.get("opponent_seed") is not None else None,
            generated_question=question,
            allowed_rejection_reasons=(
                _GROUNDING_ALLOWED_REJECTIONS if stage == "grounding" else frozenset()
            ),
            recover_invalid_tool_calls=True,
        )
        opponent = OpponentSignals(
            parse_rate=float(evaluation["program_parse_rate"]),
            execution_rate_given_parse=float(evaluation["execution_rate_given_parse"]),
            semantic_success_given_execution=float(
                evaluation["semantic_success_given_execution"]
            ),
        )
        extra_components.update(
            {
                "opponent_success_rate": float(evaluation["pass_rate"]),
                "novelty_structural": float(evaluation["novelty_structural"]),
                "novelty_textual": float(evaluation["novelty_textual"]),
            }
        )
    return _curriculum_questioner_result(
        values,
        stage=stage,
        role_weight=role_weight,
        frontier_target=frontier_target,
        frontier_sigma=frontier_sigma,
        opponent=opponent,
        rejection_reasons=result.rejection_reasons,
        extra_components=extra_components,
    )


def _solver_curriculum_stage(info: dict[str, Any]) -> SolverCurriculumStage:
    raw_stage = str(info.get("curriculum_phase", "production"))
    mapped = {
        "production": "syntax",
        "grounding": "tool",
        "frontier": "solve",
        "syntax": "syntax",
        "tool": "tool",
        "solve": "solve",
    }.get(raw_stage)
    if mapped is None:
        raise ValueError(f"unsupported Solver curriculum phase: {raw_stage}")
    return cast(SolverCurriculumStage, mapped)


def _solver_milestones(values: dict[str, float]) -> SolverMilestones:
    return SolverMilestones(
        output_present=values.get("output_present", 0.0),
        json_valid=values.get("json_valid", 0.0),
        schema_fraction=values.get("schema_fraction", 0.0),
        valid_operation_fraction=values.get("valid_operation_fraction", 0.0),
        valid_prefix_fraction=values.get("valid_prefix_fraction", 0.0),
        tool_call_attempted=values.get("tool_call_attempted", 0.0),
        valid_tool_call_fraction=values.get("valid_tool_call_fraction", 0.0),
        successful_tool_call_fraction=values.get("successful_tool_call_fraction", 0.0),
        evidence_progress=values.get("evidence_progress", 0.0),
        execution_progress=values.get("execution_progress", 0.0),
        budget_compliance=values.get("budget_compliance", 0.0),
        answer_present=values.get("answer_present", 0.0),
        answer_parse_valid=values.get("answer_parse_valid", 0.0),
        answer_f1=values.get("answer_f1", 0.0),
        exact_match=values.get("exact_match", 0.0),
    )


def _curriculum_solver_result(
    values: dict[str, float],
    *,
    stage: SolverCurriculumStage,
    role_weight: float,
    rejection_reason: str | None = None,
    extra_components: dict[str, float] | None = None,
) -> dict[str, float]:
    reward = solver_curriculum_reward(_solver_milestones(values), stage=stage)
    rejection = (
        {f"reject_{rejection_reason.lower()}": 1.0}
        if rejection_reason is not None
        else {}
    )
    return {
        "score": reward.total * role_weight,
        "raw_score": reward.total,
        "f1": values.get("answer_f1", 0.0),
        "exact_match": values.get("exact_match", 0.0),
        **reward.components,
        **rejection,
        **(extra_components or {}),
    }


def _compute_curriculum_graphscript_solver_score(
    solution_str: str,
    gold: AnswerSet,
    info: dict[str, Any],
    *,
    role_weight: float,
) -> dict[str, float]:
    stage = _solver_curriculum_stage(info)
    raw_version = str(info.get("graphscript_version", "0.3"))
    if raw_version not in {"0.1", "0.2", "0.3"}:
        raise ValueError(f"unsupported GraphScript version: {raw_version}")
    graphscript_version = cast(GraphScriptVersion, raw_version)
    values = {"output_present": float(bool(solution_str.strip()))}
    extra_components: dict[str, float] = {}
    try:
        raw_program: object = json.loads(solution_str)
    except json.JSONDecodeError:
        shape = _truncated_graphscript_shape(
            solution_str,
            expected_version=graphscript_version,
        )
        values.update(
            {
                "schema_fraction": shape["schema_fraction"],
                "valid_operation_fraction": shape["valid_operation_fraction"],
                "valid_prefix_fraction": shape["valid_prefix_fraction"],
            }
        )
        return _curriculum_solver_result(
            values,
            stage=stage,
            role_weight=role_weight,
            rejection_reason="NON_JSON",
        )
    values["json_valid"] = 1.0
    shape = _questioner_program_shape(raw_program, expected_version=graphscript_version)
    values.update(
        {
            "schema_fraction": shape["schema_fraction"],
            "valid_operation_fraction": shape["valid_operation_fraction"],
            "valid_prefix_fraction": shape["valid_prefix_fraction"],
            "tool_call_attempted": float(shape["valid_operation_fraction"] > 0.0),
            "valid_tool_call_fraction": shape["valid_operation_fraction"],
        }
    )
    try:
        script = parse_graphscript(
            solution_str,
            max_follow_limit=int(info.get("max_follow_limit", 100)),
        )
        if script.version != graphscript_version:
            raise GraphScriptError(
                "VERSION_MISMATCH",
                f"expected {graphscript_version}, got {script.version}",
            )
    except GraphScriptError as exc:
        return _curriculum_solver_result(
            values,
            stage=stage,
            role_weight=role_weight,
            rejection_reason=exc.reason_code,
        )
    values.update(
        {
            "schema_fraction": 1.0,
            "valid_operation_fraction": 1.0,
            "valid_prefix_fraction": 1.0,
            "tool_call_attempted": 1.0,
            "valid_tool_call_fraction": 1.0,
        }
    )
    backend = backend_from_snapshot(str(info.get("graph_snapshot", "toy-v1")))
    topic_ids = tuple(str(value) for value in info.get("topic_entity_ids", []))
    try:
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
    except GraphScriptError as exc:
        values["budget_compliance"] = float(exc.reason_code != "BUDGET_EXCEEDED")
        return _curriculum_solver_result(
            values,
            stage=stage,
            role_weight=role_weight,
            rejection_reason=exc.reason_code,
        )
    metrics = answer_metrics(execution.answers, gold)
    values.update(
        {
            "successful_tool_call_fraction": 1.0,
            "evidence_progress": float(bool(execution.support)),
            "execution_progress": 1.0,
            "budget_compliance": 1.0,
            "answer_present": float(bool(execution.answers.answers)),
            "answer_parse_valid": 1.0,
            "answer_f1": float(metrics["f1"]),
            "exact_match": float(metrics["exact_match"]),
        }
    )
    extra_components.update(
        {
            "edge_visits": float(execution.usage.edge_visits),
            "graph_calls": float(execution.usage.graph_calls),
            "program_operators": float(execution.usage.operators),
            "passage_searches": float(execution.usage.passage_searches),
        }
    )
    return _curriculum_solver_result(
        values,
        stage=stage,
        role_weight=role_weight,
        extra_components=extra_components,
    )


def _compute_curriculum_tool_solver_score(
    solution_str: str,
    gold: AnswerSet,
    info: dict[str, Any],
    *,
    role_weight: float,
) -> dict[str, float]:
    stage = _solver_curriculum_stage(info)
    rollout = info.get("solver_rollout", {})
    process = rollout if isinstance(rollout, dict) else {}
    calls = max(0, int(process.get("calls", 0)))
    valid_calls = max(0, min(calls, int(process.get("valid_calls", 0))))
    edge_visits = max(0, int(process.get("edge_visits", 0)))
    new_visible = max(0, int(process.get("new_visible_entities", 0)))
    values = {
        "output_present": float(bool(solution_str.strip())),
        "tool_call_attempted": float(calls > 0),
        "valid_tool_call_fraction": _fraction(valid_calls, calls),
        "successful_tool_call_fraction": _fraction(valid_calls, calls),
        "evidence_progress": min(1.0, new_visible / 3.0),
        "execution_progress": _fraction(valid_calls, calls),
        "budget_compliance": float(
            calls <= int(info.get("max_tool_calls", 8))
            and edge_visits <= int(info.get("max_edge_visits", 200))
        ),
    }
    extra_components = {
        "tool_calls": float(calls),
        "valid_tool_calls": float(valid_calls),
        "invalid_tool_calls": float(max(0, calls - valid_calls)),
        "edge_visits": float(edge_visits),
        "new_visible_entities": float(new_visible),
    }
    answer_kind = gold.answers[0].kind if gold.answers else "entity"
    try:
        predicted = parse_solver_output(solution_str, answer_kind=answer_kind)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _curriculum_solver_result(
            values,
            stage=stage,
            role_weight=role_weight,
            rejection_reason="INVALID_OUTPUT",
            extra_components=extra_components,
        )
    metrics = answer_metrics(predicted, gold)
    values.update(
        {
            "json_valid": 1.0,
            "schema_fraction": 1.0,
            "valid_operation_fraction": 1.0,
            "valid_prefix_fraction": 1.0,
            "answer_present": float(bool(predicted.answers)),
            "answer_parse_valid": 1.0,
            "answer_f1": float(metrics["f1"]),
            "exact_match": float(metrics["exact_match"]),
        }
    )
    return _curriculum_solver_result(
        values,
        stage=stage,
        role_weight=role_weight,
        extra_components=extra_components,
    )


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
        if str(info.get("questioner_reward_variant", "legacy")) == "curriculum_v3":
            return await (
                _compute_curriculum_questioner_score(
                    solution_str,
                    info,
                    role_weight=role_weight,
                )
                if interaction_mode == "graphscript"
                else _compute_curriculum_tool_questioner_score(
                    solution_str,
                    info,
                    role_weight=role_weight,
                )
            )
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
        if str(info.get("solver_reward_variant", "legacy")) == "curriculum_v3":
            return (
                _compute_curriculum_graphscript_solver_score(
                    solution_str,
                    gold,
                    info,
                    role_weight=role_weight,
                )
                if interaction_mode == "graphscript"
                else _compute_curriculum_tool_solver_score(
                    solution_str,
                    gold,
                    info,
                    role_weight=role_weight,
                )
            )
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
