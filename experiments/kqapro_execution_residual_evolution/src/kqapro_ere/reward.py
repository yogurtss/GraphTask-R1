from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from graphtask_r1.graph.base import GraphBackend
from kqapro_ere.residual import (
    ExecutionBudget,
    ResidualAnalysis,
    ResidualError,
    analyze_execution_residual,
)


class ERERewardConfig(BaseModel):
    """Independent diagnostic reward; it is not registered in the existing GRPO path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    terminal_bonus_scale: float = Field(default=0.25, ge=0.0)
    rejection_score: float = 0.0


class ERERewardComponents(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alignment_potential: float = Field(ge=0.0, le=1.0)
    terminal_answer_f1: float = Field(ge=0.0, le=1.0)
    terminal_exact_match: float = Field(ge=0.0, le=1.0)
    terminal_bonus: float = Field(ge=0.0)


class RewardRejection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    stage: str
    detail: str


class ERERewardResult(BaseModel):
    """JSON-serializable potential, terminal diagnostics, and structured rejection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    total: float
    components: ERERewardComponents
    rejection: RewardRejection | None = None
    analysis: ResidualAnalysis | None = None


def reward_from_analysis(
    analysis: ResidualAnalysis,
    *,
    config: ERERewardConfig,
) -> ERERewardResult:
    """Apply one denotation-alignment potential plus a bounded terminal-answer bonus."""

    terminal_bonus = (
        config.terminal_bonus_scale
        * (analysis.terminal_answer_f1 + analysis.terminal_exact_match)
        / 2.0
    )
    components = ERERewardComponents(
        alignment_potential=analysis.alignment.score,
        terminal_answer_f1=analysis.terminal_answer_f1,
        terminal_exact_match=analysis.terminal_exact_match,
        terminal_bonus=terminal_bonus,
    )
    return ERERewardResult(
        accepted=True,
        total=components.alignment_potential + components.terminal_bonus,
        components=components,
        analysis=analysis,
    )


def compute_ere_reward(
    predicted_graphscript: str,
    reference_program_json: str,
    *,
    backend: GraphBackend,
    config: ERERewardConfig,
    allowed_relations: frozenset[str],
    seed_entity: str | None,
    seed: int,
    trace_id: str,
) -> ERERewardResult:
    """Strictly parse and replay a candidate and certified Program under one budget."""

    try:
        analysis = analyze_execution_residual(
            predicted_graphscript,
            reference_program_json,
            backend=backend,
            budget=config.budget,
            allowed_relations=allowed_relations,
            seed_entity=seed_entity,
            seed=seed,
            trace_id=trace_id,
        )
    except ResidualError as exc:
        return ERERewardResult(
            accepted=False,
            total=config.rejection_score,
            components=ERERewardComponents(
                alignment_potential=0.0,
                terminal_answer_f1=0.0,
                terminal_exact_match=0.0,
                terminal_bonus=0.0,
            ),
            rejection=RewardRejection(
                reason_code=exc.reason_code,
                stage=exc.stage,
                detail=exc.detail,
            ),
        )
    return reward_from_analysis(analysis, config=config)
