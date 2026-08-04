from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, model_validator


class RewardBreakdown(BaseModel):
    total: float
    components: dict[str, float]
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def finite(self) -> RewardBreakdown:
        values = [self.total, *self.components.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("reward values must be finite")
        return self


class VerifierResult(BaseModel):
    passed: bool
    executable: bool
    answer_nonempty: bool
    cardinality_valid: bool
    type_valid: bool
    semantic_equivalent: bool | None
    answer_leak: bool
    shortcut_found: bool | None
    necessity_mean: float = Field(ge=0.0, le=1.0)
    necessity_min: float = Field(ge=0.0, le=1.0)
    novelty_structural: float = Field(ge=0.0, le=1.0)
    novelty_textual: float = Field(ge=0.0, le=1.0)
    rejection_reasons: tuple[str, ...] = ()
    component_latency_ms: dict[str, float] = {}
