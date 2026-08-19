from __future__ import annotations

import math

from graphtask_r1.rewards.frontier import frontier_reward
from graphtask_r1.schema import RewardBreakdown, VerifierResult

QUESTIONER_STAGE_NON_JSON = 0
QUESTIONER_STAGE_EXACT_OUTPUT = 1
QUESTIONER_STAGE_SCHEMA = 2
QUESTIONER_STAGE_STRUCTURE = 3
QUESTIONER_STAGE_EXECUTION = 4
QUESTIONER_STAGE_VERIFICATION = 5
QUESTIONER_STAGE_CERTIFIED = 6

_SCHEMA_REJECTIONS = frozenset(
    {
        "EXTRA_FIELD",
        "INVALID_DIRECTION",
        "INVALID_OUTPUT",
        "INVALID_SCHEMA",
        "UNKNOWN_OP",
        "UNSUPPORTED_VERSION",
        "VERSION_MISMATCH",
    }
)
_STRUCTURE_REJECTIONS = frozenset(
    {
        "DUPLICATE_HANDLE",
        "INVALID_HANDLE",
        "INVALID_SEED",
        "INVALID_SHAPE",
        "LIMIT_EXCEEDED",
        "MISSING_EMIT",
        "MISSING_SEED",
        "OP_NOT_IN_PROFILE",
        "PROPOSAL_ROOT_MISMATCH",
        "RELATION_NOT_ALLOWED",
        "SEED_MISMATCH",
        "TYPE_MISMATCH",
        "UNSUPPORTED_PROGRAM",
    }
)
_EXECUTION_REJECTIONS = frozenset(
    {
        "BOUNDED_UNBOUNDED_MISMATCH",
        "BUDGET_EXCEEDED",
        "EMPTY_RESULT",
        "ENTITY_NOT_FOUND",
        "ENTITY_RESOLUTION_UNAVAILABLE",
        "EXECUTION_ERROR",
        "MISSING_BACKEND",
        "NON_UNIQUE_RESULT",
        "PASSAGE_SEARCH_ERROR",
        "PROGRAM_TOO_LARGE",
    }
)


def questioner_rejection_reward(reason_code: str) -> RewardBreakdown:
    """Return a deterministic, strictly ordered pre-certification reward."""
    reason = reason_code.upper()
    if reason == "NON_JSON":
        stage, total = QUESTIONER_STAGE_NON_JSON, -1.0
    elif reason == "EXTRA_TEXT":
        stage, total = QUESTIONER_STAGE_EXACT_OUTPUT, -0.9
    elif reason in _SCHEMA_REJECTIONS:
        stage, total = QUESTIONER_STAGE_SCHEMA, -0.75
    elif reason in _STRUCTURE_REJECTIONS:
        stage, total = QUESTIONER_STAGE_STRUCTURE, -0.6
    elif reason in _EXECUTION_REJECTIONS:
        stage, total = QUESTIONER_STAGE_EXECUTION, -0.4
    else:
        # Unknown rejection codes must not accidentally outrank a known later stage.
        stage, total = QUESTIONER_STAGE_SCHEMA, -0.75
    return RewardBreakdown(
        total=total,
        components={
            "reward_stage": float(stage),
            "json_valid": float(stage >= QUESTIONER_STAGE_EXACT_OUTPUT),
            "exact_output": float(stage >= QUESTIONER_STAGE_SCHEMA),
            "schema_valid": float(stage >= QUESTIONER_STAGE_STRUCTURE),
            "structure_valid": float(stage >= QUESTIONER_STAGE_EXECUTION),
            "format": -1.0,
            "executable": 0.0,
            "answer_nonempty": 0.0,
            "certified": 0.0,
        },
    )


def _verification_failure_reward(result: VerifierResult, *, efficiency: float) -> float:
    reasons = set(result.rejection_reasons)
    if "ANSWER_LEAK" in reasons or "SHORTCUT_FOUND" in reasons:
        return -0.3
    if not result.executable or "EXECUTION_ERROR" in reasons:
        return -0.35
    if not result.answer_nonempty or not result.cardinality_valid:
        return -0.2
    if "SHORTCUT_UNKNOWN" in reasons:
        return -0.15
    # Redundant conditions and other post-execution verifier failures are closest to valid.
    return -0.1 - 0.02 * (1.0 - efficiency)


def challenger_reward(
    result: VerifierResult,
    *,
    pass_rate: float,
    cost: float,
    target_alignment: float | None = None,
) -> RewardBreakdown:
    if target_alignment is not None and not 0.0 <= target_alignment <= 1.0:
        raise ValueError("target_alignment must be between 0 and 1")
    validity = float(result.passed)
    efficiency = math.exp(-0.05 * cost)
    components = {
        "validity": validity,
        "frontier": frontier_reward(pass_rate) if validity else 0.0,
        "necessity": result.necessity_mean if validity else 0.0,
        "novelty": 0.5 * (result.novelty_structural + result.novelty_textual) if validity else 0.0,
        "shortcut_penalty": -float(result.shortcut_found is True),
        "answer_leak_penalty": -float(result.answer_leak),
        "efficiency": efficiency,
        "executable": float(result.executable),
        "answer_nonempty": float(result.answer_nonempty),
        "cardinality_valid": float(result.cardinality_valid),
        "certified": validity,
    }
    if not result.passed:
        components["reward_stage"] = float(QUESTIONER_STAGE_VERIFICATION)
        return RewardBreakdown(
            total=_verification_failure_reward(result, efficiency=efficiency),
            components=components,
        )
    if target_alignment is None:
        quality = (
            0.35 * components["frontier"]
            + 0.25 * components["necessity"]
            + 0.25 * components["novelty"]
            + 0.15 * components["efficiency"]
        )
    else:
        components["target_alignment"] = target_alignment
        quality = (
            0.30 * components["frontier"]
            + 0.20 * components["necessity"]
            + 0.20 * components["novelty"]
            + 0.15 * components["target_alignment"]
            + 0.15 * components["efficiency"]
        )
    components["quality"] = quality
    components["reward_stage"] = float(QUESTIONER_STAGE_CERTIFIED)
    total = 0.20 + 0.80 * quality
    return RewardBreakdown(total=total, components=components)


def frontier_gated_challenger_reward(
    result: VerifierResult,
    *,
    pass_rate: float,
    samples: int,
    cost: float,
    target_alignment: float | None = None,
    frontier_target: float = 0.5,
    frontier_sigma: float = 0.2,
) -> RewardBreakdown:
    """Questioner v2 reward with certification as a gate and difficulty as the driver.

    The legacy additive reward intentionally remains in :func:`challenger_reward` so
    old experiments are exactly reproducible.  This opt-in variant smooths the
    opponent success estimate with a Beta(1, 1) prior, then multiplies task quality
    by the frontier score.  Consequently novelty or efficiency cannot compensate
    for a task that is trivial or impossible for the frozen Solver.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError("pass_rate must be between 0 and 1")
    if target_alignment is not None and not 0.0 <= target_alignment <= 1.0:
        raise ValueError("target_alignment must be between 0 and 1")
    if not 0.0 <= frontier_target <= 1.0:
        raise ValueError("frontier_target must be between 0 and 1")

    validity = float(result.passed)
    efficiency = math.exp(-0.05 * cost)
    smoothed_pass_rate = (pass_rate * samples + 1.0) / (samples + 2.0)
    frontier = (
        frontier_reward(
            smoothed_pass_rate,
            target=frontier_target,
            sigma=frontier_sigma,
        )
        if validity
        else 0.0
    )
    components = {
        "validity": validity,
        "frontier": frontier,
        "opponent_pass_rate_smoothed": smoothed_pass_rate,
        "necessity": result.necessity_mean if validity else 0.0,
        "novelty": (
            0.5 * (result.novelty_structural + result.novelty_textual) if validity else 0.0
        ),
        "shortcut_penalty": -float(result.shortcut_found is True),
        "answer_leak_penalty": -float(result.answer_leak),
        "efficiency": efficiency,
        "executable": float(result.executable),
        "answer_nonempty": float(result.answer_nonempty),
        "cardinality_valid": float(result.cardinality_valid),
        "certified": validity,
        "reward_variant_frontier_v2": 1.0,
    }
    if not result.passed:
        components["reward_stage"] = float(QUESTIONER_STAGE_VERIFICATION)
        return RewardBreakdown(
            total=_verification_failure_reward(result, efficiency=efficiency),
            components=components,
        )

    alignment = target_alignment if target_alignment is not None else 1.0
    if target_alignment is not None:
        components["target_alignment"] = target_alignment
    quality = (
        0.35 * components["necessity"]
        + 0.25 * components["novelty"]
        + 0.20 * alignment
        + 0.20 * components["efficiency"]
    )
    frontier_quality = frontier * (0.5 + 0.5 * quality)
    components["quality"] = quality
    components["frontier_quality"] = frontier_quality
    components["reward_stage"] = float(QUESTIONER_STAGE_CERTIFIED)
    return RewardBreakdown(total=0.05 + 0.95 * frontier_quality, components=components)
