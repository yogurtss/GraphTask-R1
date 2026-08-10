"""verl v0.5 SFT dataset compatibility for nested Parquet conversations."""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

from graphtask_r1.training.json_compat import to_json_compatible


class GraphTaskMultiTurnSFTDataset(MultiTurnSFTDataset):  # type: ignore[misc]
    """Normalize pandas/Arrow arrays before Qwen's chat template serializes tool calls."""

    messages: list[Any]
    tools: list[Any] | None

    def _read_files_and_process(self) -> None:
        super()._read_files_and_process()
        self.messages = [to_json_compatible(messages) for messages in self.messages]
        if self.tools is not None:
            self.tools = [to_json_compatible(tools) for tools in self.tools]
