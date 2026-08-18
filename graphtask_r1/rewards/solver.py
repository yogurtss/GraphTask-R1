from __future__ import annotations

from graphtask_r1.evaluation import answer_metrics
from graphtask_r1.schema import AnswerSet, RewardBreakdown

from .challenger import questioner_rejection_reward


def solver_rejection_reward(reason_code: str) -> RewardBreakdown:
    """Give malformed Solver outputs the same ordered syntax-to-execution curriculum."""
    staged = questioner_rejection_reward(reason_code)
    return staged.model_copy(
        update={
            "components": {
                **staged.components,
                "answer_f1": 0.0,
                "exact_match": 0.0,
            }
        }
    )


def solver_outcome_reward(*, f1: float, exact_match: float) -> RewardBreakdown:
    """Reward executable attempts before smoothly increasing toward exact answers."""
    if not 0.0 <= f1 <= 1.0 or not 0.0 <= exact_match <= 1.0:
        raise ValueError("solver metrics must be between 0 and 1")
    total = 0.1 + 0.7 * f1 + 0.2 * exact_match
    return RewardBreakdown(
        total=total,
        components={
            "reward_stage": 6.0 if exact_match else 5.0,
            "json_valid": 1.0,
            "exact_output": 1.0,
            "schema_valid": 1.0,
            "structure_valid": 1.0,
            "format": 1.0,
            "executable": 1.0,
            "answer_f1": f1,
            "exact_match": exact_match,
        },
    )


def solver_reward(
    predicted: AnswerSet,
    gold: AnswerSet,
    *,
    search_calls: int,
    invalid_calls: int,
    free_turns: int = 3,
) -> RewardBreakdown:
    metrics = answer_metrics(predicted, gold)
    components = {
        "answer_f1": metrics["f1"],
        "exact_match": metrics["exact_match"],
        "turn_penalty": -0.02 * max(0, search_calls - free_turns),
        "invalid_penalty": -0.1 * invalid_calls,
    }
    return RewardBreakdown(total=sum(components.values()), components=components)
