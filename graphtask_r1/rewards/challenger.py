from __future__ import annotations

from graphtask_r1.rewards.frontier import frontier_reward
from graphtask_r1.schema import RewardBreakdown, VerifierResult


def challenger_reward(result: VerifierResult, *, pass_rate: float, cost: float) -> RewardBreakdown:
    validity = float(result.passed)
    components = {
        "validity": validity,
        "frontier": frontier_reward(pass_rate) if validity else 0.0,
        "necessity": result.necessity_mean if validity else 0.0,
        "novelty": 0.5 * (result.novelty_structural + result.novelty_textual) if validity else 0.0,
        "shortcut_penalty": -float(result.shortcut_found is True),
        "answer_leak_penalty": -float(result.answer_leak),
        "cost_penalty": -0.02 * cost,
    }
    total = (
        validity
        * (
            0.30 * components["frontier"]
            + 0.30 * components["necessity"]
            + 0.20 * components["novelty"]
            + 0.20
        )
        + components["shortcut_penalty"]
        + components["answer_leak_penalty"]
        + components["cost_penalty"]
    )
    return RewardBreakdown(total=total, components=components)
