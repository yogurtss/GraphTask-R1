from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from swift.llm.template.register import get_template_meta, register_template
from swift.llm.template.template.qwen import Qwen3Template
from swift.llm.template.utils import ContextType
from swift.plugin import loss_scale_map
from swift.plugin.loss_scale.loss_scale import LossScale

from kqapro_grr.boundary import BOUNDARY_END, BOUNDARY_START


class ExecutionBoundaryLossScale(LossScale):  # type: ignore[misc]
    """Anchor the chosen response; mask the rejected response outside one operation."""

    is_binary = True

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer: Any | None = None

    def _tokenized_response(
        self,
        context: str,
        *,
        full_response: bool,
    ) -> tuple[list[list[int]], list[float]]:
        prefix, marked = context.split(BOUNDARY_START, 1)
        boundary, suffix = marked.split(BOUNDARY_END, 1)
        clean = prefix + boundary + suffix
        tokenizer = cast(Any, self.tokenizer)
        encoded = tokenizer(
            clean,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        input_ids = cast(list[int], encoded["input_ids"])
        offsets = cast(list[tuple[int, int]], encoded["offset_mapping"])
        boundary_start = len(prefix)
        boundary_end = boundary_start + len(boundary)
        weights = [
            1.0
            if full_response or (end > boundary_start and start < boundary_end)
            else 0.0
            for start, end in offsets
        ]
        return [[token_id] for token_id in input_ids], weights

    def __call__(
        self,
        context_list: list[Any],
        context_types: list[ContextType],
        messages: list[dict[str, Any]],
        **kwargs: object,
    ) -> tuple[list[Any], list[float]]:
        full_response = float(messages[-1].get("loss", 0.0)) == 1.0
        contexts: list[Any] = []
        weights: list[float] = []
        for context, context_type in zip(context_list, context_types, strict=True):
            if context_type == ContextType.RESPONSE:
                token_contexts, token_weights = self._tokenized_response(
                    cast(str, context),
                    full_response=full_response,
                )
                contexts.extend(token_contexts)
                weights.extend(token_weights)
            else:
                contexts.append(context)
                weights.append(
                    1.0
                    if full_response and context_type == ContextType.SUFFIX
                    else 0.0
                )
        return contexts, weights


class ExecutionBoundaryQwen3Template(Qwen3Template):  # type: ignore[misc]
    def init_processor(self, processor: Any) -> None:
        super().init_processor(processor)
        loss_scale = cast(ExecutionBoundaryLossScale, self.loss_scale)
        loss_scale.tokenizer = self.tokenizer


ExecutionBoundaryLossScale.name = "execution_boundary_asymmetric"
loss_scale_map["execution_boundary_asymmetric"] = ExecutionBoundaryLossScale

register_template(
    replace(
        get_template_meta("qwen3"),
        template_type="qwen3_execution_boundary",
        template_cls=ExecutionBoundaryQwen3Template,
    ),
    exist_ok=True,
)
