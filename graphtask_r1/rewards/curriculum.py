from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from graphtask_r1.rewards.frontier import frontier_reward
from graphtask_r1.schema import RewardBreakdown

QuestionerCurriculumStage = Literal["production", "grounding", "frontier"]
SolverCurriculumStage = Literal["syntax", "tool", "solve"]


def _require_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _weighted_score(values: dict[str, float], weights: dict[str, float]) -> float:
    return sum(values[name] * weight for name, weight in weights.items())


@dataclass(frozen=True)
class QuestionerMilestones:
    """Dense, independently measurable progress for one Questioner completion."""

    question_present: float = 0.0
    code_present: float = 0.0
    json_valid: float = 0.0
    schema_fraction: float = 0.0
    valid_operation_fraction: float = 0.0
    valid_prefix_fraction: float = 0.0
    seed_coverage: float = 0.0
    relation_valid_fraction: float = 0.0
    handle_valid_fraction: float = 0.0
    type_valid_fraction: float = 0.0
    executable_prefix_fraction: float = 0.0
    program_executable: float = 0.0
    answer_nonempty: float = 0.0
    cardinality_valid: float = 0.0
    question_program_alignment: float = 0.0
    no_answer_leak: float = 0.0
    certified: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_unit_interval(name, value)


@dataclass(frozen=True)
class OpponentSignals:
    """Solver signals that keep interface readiness separate from task semantics.

    ``semantic_success_given_execution`` must be measured only over executable
    opponent rollouts, preferably with mean answer F1.  An unconditional exact-match
    rate does not satisfy this contract because it treats parse and execution failures
    as evidence that the Questioner produced an overly difficult task.
    """

    parse_rate: float
    execution_rate_given_parse: float
    semantic_success_given_execution: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_unit_interval(name, value)

    @property
    def interface_readiness(self) -> float:
        return self.parse_rate * self.execution_rate_given_parse


@dataclass(frozen=True)
class QuestionerRewardConfig:
    production_weight_in_grounding: float = 0.4
    production_weight_in_frontier_base: float = 0.3
    frontier_weight: float = 0.35
    frontier_target: float = 0.5
    frontier_sigma: float = 0.2

    def __post_init__(self) -> None:
        _require_unit_interval(
            "production_weight_in_grounding", self.production_weight_in_grounding
        )
        _require_unit_interval(
            "production_weight_in_frontier_base",
            self.production_weight_in_frontier_base,
        )
        _require_unit_interval("frontier_weight", self.frontier_weight)
        _require_unit_interval("frontier_target", self.frontier_target)
        if self.frontier_sigma <= 0.0:
            raise ValueError("frontier_sigma must be positive")


_QUESTIONER_PRODUCTION_WEIGHTS = {
    "question_present": 0.15,
    "code_present": 0.15,
    "json_valid": 0.15,
    "schema_fraction": 0.20,
    "valid_operation_fraction": 0.20,
    "valid_prefix_fraction": 0.15,
}

_QUESTIONER_GROUNDING_WEIGHTS = {
    "seed_coverage": 0.12,
    "relation_valid_fraction": 0.08,
    "handle_valid_fraction": 0.08,
    "type_valid_fraction": 0.08,
    "executable_prefix_fraction": 0.16,
    "program_executable": 0.14,
    "answer_nonempty": 0.09,
    "cardinality_valid": 0.05,
    "question_program_alignment": 0.10,
    "no_answer_leak": 0.05,
    "certified": 0.05,
}


def questioner_curriculum_reward(
    milestones: QuestionerMilestones,
    *,
    stage: QuestionerCurriculumStage,
    opponent: OpponentSignals | None = None,
    config: QuestionerRewardConfig | None = None,
) -> RewardBreakdown:
    """Score Questioner progress without making certification a learning-signal gate."""

    policy = config or QuestionerRewardConfig()
    values = asdict(milestones)
    production = _weighted_score(values, _QUESTIONER_PRODUCTION_WEIGHTS)
    grounding = _weighted_score(values, _QUESTIONER_GROUNDING_WEIGHTS)
    grounding_production_weight = policy.production_weight_in_grounding
    grounding_progress_weight = 1.0 - grounding_production_weight
    grounding_total = (
        grounding_production_weight * production + grounding_progress_weight * grounding
    )

    frontier_production_weight = policy.production_weight_in_frontier_base
    frontier_grounding_weight = 1.0 - frontier_production_weight
    frontier_base = (
        frontier_production_weight * production + frontier_grounding_weight * grounding
    )
    parse_rate = opponent.parse_rate if opponent is not None else 0.0
    execution_rate = opponent.execution_rate_given_parse if opponent is not None else 0.0
    semantic_success = (
        opponent.semantic_success_given_execution if opponent is not None else 0.0
    )
    interface_readiness = opponent.interface_readiness if opponent is not None else 0.0
    raw_frontier = (
        frontier_reward(
            semantic_success,
            target=policy.frontier_target,
            sigma=policy.frontier_sigma,
        )
        if opponent is not None
        else 0.0
    )
    effective_frontier_weight = policy.frontier_weight * interface_readiness
    frontier_base_weight = 1.0 - effective_frontier_weight
    frontier_total = (
        frontier_base_weight * frontier_base + effective_frontier_weight * raw_frontier
    )

    if stage == "production":
        total = production
        production_contribution = production
        grounding_contribution = 0.0
        frontier_contribution = 0.0
    elif stage == "grounding":
        total = grounding_total
        production_contribution = grounding_production_weight * production
        grounding_contribution = grounding_progress_weight * grounding
        frontier_contribution = 0.0
    else:
        total = frontier_total
        production_contribution = (
            frontier_base_weight * frontier_production_weight * production
        )
        grounding_contribution = (
            frontier_base_weight * frontier_grounding_weight * grounding
        )
        frontier_contribution = effective_frontier_weight * raw_frontier

    components = {
        **{f"milestone_{name}": value for name, value in values.items()},
        "stage_production": float(stage == "production"),
        "stage_grounding": float(stage == "grounding"),
        "stage_frontier": float(stage == "frontier"),
        "production_score": production,
        "grounding_score": grounding,
        "grounding_total": grounding_total,
        "frontier_base_score": frontier_base,
        "opponent_parse_rate": parse_rate,
        "opponent_execution_rate_given_parse": execution_rate,
        "opponent_interface_readiness": interface_readiness,
        "opponent_semantic_success_given_execution": semantic_success,
        "frontier_raw": raw_frontier,
        "frontier_weight_configured": policy.frontier_weight,
        "frontier_weight_effective": effective_frontier_weight,
        "production_contribution": production_contribution,
        "grounding_contribution": grounding_contribution,
        "frontier_contribution": frontier_contribution,
        "curriculum_total": total,
    }
    return RewardBreakdown(
        total=total,
        components=components,
        metadata={
            "stage": stage,
            "frontier_semantic_signal": "conditional_on_successful_execution",
        },
    )


@dataclass(frozen=True)
class SolverMilestones:
    """Dense syntax, tool-use, and answer progress for one Solver completion."""

    output_present: float = 0.0
    json_valid: float = 0.0
    schema_fraction: float = 0.0
    valid_operation_fraction: float = 0.0
    valid_prefix_fraction: float = 0.0
    tool_call_attempted: float = 0.0
    valid_tool_call_fraction: float = 0.0
    successful_tool_call_fraction: float = 0.0
    evidence_progress: float = 0.0
    execution_progress: float = 0.0
    budget_compliance: float = 0.0
    answer_present: float = 0.0
    answer_parse_valid: float = 0.0
    answer_f1: float = 0.0
    exact_match: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_unit_interval(name, value)


@dataclass(frozen=True)
class SolverRewardConfig:
    syntax_weight_in_tool: float = 0.4
    syntax_weight_in_solve: float = 0.2
    tool_weight_in_solve: float = 0.3

    def __post_init__(self) -> None:
        _require_unit_interval("syntax_weight_in_tool", self.syntax_weight_in_tool)
        _require_unit_interval("syntax_weight_in_solve", self.syntax_weight_in_solve)
        _require_unit_interval("tool_weight_in_solve", self.tool_weight_in_solve)
        if self.syntax_weight_in_solve + self.tool_weight_in_solve > 1.0:
            raise ValueError("syntax_weight_in_solve + tool_weight_in_solve cannot exceed 1")


_SOLVER_SYNTAX_WEIGHTS = {
    "output_present": 0.10,
    "json_valid": 0.25,
    "schema_fraction": 0.25,
    "valid_operation_fraction": 0.20,
    "valid_prefix_fraction": 0.20,
}

_SOLVER_TOOL_WEIGHTS = {
    "tool_call_attempted": 0.10,
    "valid_tool_call_fraction": 0.25,
    "successful_tool_call_fraction": 0.25,
    "evidence_progress": 0.15,
    "execution_progress": 0.15,
    "budget_compliance": 0.10,
}

_SOLVER_SOLVE_WEIGHTS = {
    "answer_present": 0.10,
    "answer_parse_valid": 0.15,
    "answer_f1": 0.55,
    "exact_match": 0.20,
}


def solver_curriculum_reward(
    milestones: SolverMilestones,
    *,
    stage: SolverCurriculumStage,
    config: SolverRewardConfig | None = None,
) -> RewardBreakdown:
    """Score Solver progress so valid tool use learns before exact answers are common."""

    policy = config or SolverRewardConfig()
    values = asdict(milestones)
    syntax = _weighted_score(values, _SOLVER_SYNTAX_WEIGHTS)
    tool = _weighted_score(values, _SOLVER_TOOL_WEIGHTS)
    solve = _weighted_score(values, _SOLVER_SOLVE_WEIGHTS)

    tool_syntax_weight = policy.syntax_weight_in_tool
    tool_progress_weight = 1.0 - tool_syntax_weight
    tool_total = tool_syntax_weight * syntax + tool_progress_weight * tool
    solve_syntax_weight = policy.syntax_weight_in_solve
    solve_tool_weight = policy.tool_weight_in_solve
    solve_answer_weight = 1.0 - solve_syntax_weight - solve_tool_weight
    solve_total = (
        solve_syntax_weight * syntax + solve_tool_weight * tool + solve_answer_weight * solve
    )

    if stage == "syntax":
        total = syntax
        syntax_contribution = syntax
        tool_contribution = 0.0
        solve_contribution = 0.0
    elif stage == "tool":
        total = tool_total
        syntax_contribution = tool_syntax_weight * syntax
        tool_contribution = tool_progress_weight * tool
        solve_contribution = 0.0
    else:
        total = solve_total
        syntax_contribution = solve_syntax_weight * syntax
        tool_contribution = solve_tool_weight * tool
        solve_contribution = solve_answer_weight * solve

    components = {
        **{f"milestone_{name}": value for name, value in values.items()},
        "stage_syntax": float(stage == "syntax"),
        "stage_tool": float(stage == "tool"),
        "stage_solve": float(stage == "solve"),
        "syntax_score": syntax,
        "tool_score": tool,
        "solve_score": solve,
        "tool_total": tool_total,
        "solve_total": solve_total,
        "syntax_contribution": syntax_contribution,
        "tool_contribution": tool_contribution,
        "solve_contribution": solve_contribution,
        "curriculum_total": total,
    }
    return RewardBreakdown(
        total=total,
        components=components,
        metadata={"stage": stage},
    )
