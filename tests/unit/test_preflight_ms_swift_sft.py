from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import preflight_ms_swift_sft


def test_preflight_streams_accepted_and_rejected_batches(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    caplog.set_level(logging.INFO)
    input_path = tmp_path / "input.parquet"
    accepted_path = tmp_path / "accepted.parquet"
    rejected_path = tmp_path / "rejected.parquet"
    summary_path = tmp_path / "summary.json"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"row-{index}"},
                {
                    "role": "assistant",
                    "content": "reject" if index % 10 == 0 else "ok",
                },
            ],
            "role": "solver",
            "task_id": f"task-{index}",
            "interaction_mode": "graphscript",
        }
        for index in range(70)
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_path, row_group_size=17)

    class FakeMaxLengthError(ValueError):
        pass

    class FakeTemplate:
        def set_mode(self, mode: str) -> None:
            assert mode == "train"

        def encode(self, row):
            if row["messages"][-1]["content"] == "reject":
                raise FakeMaxLengthError("Current length of row(50000)")
            return {"input_ids": [1, 2, 3], "length": 3}

    swift = types.ModuleType("swift")
    swift_llm = types.ModuleType("swift.llm")
    swift_llm.MaxLengthError = FakeMaxLengthError
    swift_llm.get_model_tokenizer = lambda *args, **kwargs: (None, object())
    swift_llm.get_template = lambda *args, **kwargs: FakeTemplate()
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.llm", swift_llm)
    monkeypatch.setattr(
        preflight_ms_swift_sft,
        "_arguments",
        lambda: argparse.Namespace(
            input=input_path,
            accepted_output=accepted_path,
            rejected_output=rejected_path,
            summary_output=summary_path,
            model="model",
            model_type="qwen3",
            template="qwen3",
            agent_template="hermes",
            max_length=32_768,
            require_all=False,
        ),
    )

    assert preflight_ms_swift_sft.main() == 0
    assert pq.ParquetFile(accepted_path).metadata.num_rows == 63
    assert pq.ParquetFile(rejected_path).metadata.num_rows == 7
    assert any("template_loading" in record.message for record in caplog.records)
    assert any(
        'operation="data.preflight_ms_swift_sft" phase="completed"' in record.message
        for record in caplog.records
    )
