from __future__ import annotations

from graphtask_r1.evaluation import answer_metrics
from graphtask_r1.schema import AnswerSet, RewardBreakdown


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
