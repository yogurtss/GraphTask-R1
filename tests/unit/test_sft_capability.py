from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from graphtask_r1.evaluation.kqapro_val import CompletionResult, KQAProModelConfig
from graphtask_r1.evaluation.sft_capability import (
    SFTCapabilityConfig,
    _questioner_artifact,
    visualize_sft_capability,
)
from graphtask_r1.utils import read_json, read_records


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = ["<bad & questioner>", '{"version":"0.3","ops":[]}']
        self.flushed = False

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        trace_id: str,
        seed: int,
    ) -> CompletionResult:
        self.calls.append({"messages": messages, "trace_id": trace_id, "seed": seed})
        return CompletionResult(content=self.responses.pop(0), completion_tokens=5)

    def flush(self) -> None:
        self.flushed = True


class FakeScorer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict[str, Any] | None,
    ) -> dict[str, float]:
        self.calls.append(
            {
                "data_source": data_source,
                "solution_str": solution_str,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
            }
        )
        if data_source == "graphtask/questioner":
            assert extra_info is not None
            assert extra_info["opponent_url"] == "http://probe-opponent"
            assert extra_info["opponent_samples"] == 2
            return {
                "score": -0.2625,
                "raw_score": -0.75,
                "reward_stage": 2.0,
                "certified": 0.0,
                "reject_extra_field": 1.0,
            }
        return {
            "score": 0.65,
            "raw_score": 1.0,
            "f1": 1.0,
            "exact_match": 1.0,
        }


def _write_rl_row(path: Path, *, role: str) -> None:
    source = f"graphtask/{role}"
    ground_truth = "{}" if role == "questioner" else '{"answers":[]}'
    row = {
        "data_source": source,
        "prompt": [
            {"role": "system", "content": f"You are the {role}."},
            {"role": "user", "content": f"Probe <{role}> & score it."},
        ],
        "ability": "probe",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "role": role,
            "role_weight": 0.35 if role == "questioner" else 0.65,
            "graph_snapshot": "toy-v1",
            "interaction_mode": "graphscript",
            "graphscript_version": "0.3",
            "task_id": f"{role}-task",
        },
        "uid": f"{role}:task",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row]), path)


def test_sft_capability_probe_generates_scores_and_live_html(tmp_path: Path) -> None:
    questioner = tmp_path / "questioner.parquet"
    solver = tmp_path / "solver.parquet"
    _write_rl_row(questioner, role="questioner")
    _write_rl_row(solver, role="solver")
    config = SFTCapabilityConfig(
        model=KQAProModelConfig(model_url="http://unused", model="sft-model"),
        questioner_input=questioner,
        solver_input=solver,
        opponent_url="http://probe-opponent",
        groups=1,
        candidates_per_role=1,
        opponent_samples=2,
        seed=7,
    )
    client = FakeClient()
    scorer = FakeScorer()
    output = tmp_path / "report"

    result = asyncio.run(
        visualize_sft_capability(
            config,
            output,
            client=client,
            scorer=scorer,
        )
    )

    assert result["groups"] == 1
    assert result["summary"]["roles"]["questioner"]["stage_counts"]["2"] == 1
    assert result["summary"]["roles"]["solver"]["f1"] == 1.0
    assert len(client.calls) == 2
    assert client.calls[0]["trace_id"] == "sft-capability:g000:questioner:c000"
    assert [call["data_source"] for call in scorer.calls] == [
        "graphtask/questioner",
        "graphtask/solver",
    ]
    assert client.flushed is False

    html = (output / "report.html").read_text()
    assert "SFT 后 Questioner / Solver 能力探针" in html
    assert "&lt;bad &amp; questioner&gt;" in html
    assert "EXTRA_FIELD" in html
    assert "<bad & questioner>" not in html
    assert len(read_records(output / "results.parquet")) == 2
    assert read_json(output / "summary.json")["roles"]["solver"]["exact_match"] == 1.0


def test_sft_capability_probe_requires_enough_role_rows(tmp_path: Path) -> None:
    questioner = tmp_path / "questioner.parquet"
    solver = tmp_path / "solver.parquet"
    _write_rl_row(questioner, role="questioner")
    _write_rl_row(solver, role="solver")
    config = SFTCapabilityConfig(
        model=KQAProModelConfig(model_url="http://unused", model="sft-model"),
        questioner_input=questioner,
        solver_input=solver,
        opponent_url="http://probe-opponent",
        groups=2,
        candidates_per_role=1,
    )

    try:
        asyncio.run(
            visualize_sft_capability(
                config,
                tmp_path / "report",
                client=FakeClient(),
                scorer=FakeScorer(),
            )
        )
    except ValueError as exc:
        assert "2 groups requested" in str(exc)
    else:
        raise AssertionError("expected insufficient role rows to fail")


def test_questioner_artifact_shows_certified_question_and_program_answer() -> None:
    completion = json.dumps(
        {
            "version": "0.3",
            "ops": [
                {
                    "op": "resolve_entity",
                    "query": "alice",
                    "match": "id",
                    "limit": 1,
                    "out": "h0",
                },
                {
                    "op": "follow",
                    "in": "h0",
                    "relation": "works_at",
                    "direction": "out",
                    "limit": 10,
                    "out": "h1",
                },
                {"op": "emit", "in": "h1"},
            ],
        }
    )

    artifact = _questioner_artifact(
        completion,
        {
            "graph_snapshot": "toy-v1",
            "interaction_mode": "graphscript",
            "topic_entity_ids": ["alice"],
            "allowed_relations": ["works_at"],
            "max_edge_visits": 10,
        },
        {},
    )

    assert artifact["generated_question"]
    assert artifact["generated_answers"] == ["acme"]
