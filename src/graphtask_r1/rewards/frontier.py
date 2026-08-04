from __future__ import annotations

import math


def frontier_reward(pass_rate: float, *, target: float = 0.5, sigma: float = 0.2) -> float:
    if not 0 <= pass_rate <= 1:
        raise ValueError("pass_rate must be in [0, 1]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-((pass_rate - target) ** 2) / (2 * sigma**2))
