from __future__ import annotations

import math


def normalize_advantages(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 1e-12:
        return [0.0] * len(values)
    std = math.sqrt(variance)
    return [(value - mean) / std for value in values]
