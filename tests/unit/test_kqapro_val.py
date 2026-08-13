from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from graphtask_r1.evaluation.kqapro_val import (
    CompletionResult,
    KQAProModelConfig,
    KQAProValConfig,
    compare_kqapro_val_metrics,
    evaluate_kqapro_val,
    inspect_kqapro_val,
    visualize_kqapro_val,
)
from graphtask_r1.generation import certify_proposal
from graphtask_r1.graph import toy_graph
from graphtask_r1.schema import Entity, Hop, TaskProposal
from graphtask_r1.training.relations import build_relation_catalog
from graphtask_r1.utils import read_records, write_json, write_records


class FakeCompletionClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        trace_id: str,
        seed: int,
    ) -> CompletionResult:
        self.calls.append({"messages": messages, "trace_id": trace_id, "seed": seed})
        return CompletionResult(content=self.responses.pop(0), completion_tokens=7)

    def flush(self) -> None:
        return None


def _fixture(tmp_path: Path) -> tuple[Path, KQAProValConfig]:
    backend = toy_graph()
    task = certify_proposal(
        TaskProposal(
            topic_entities=("alice",),
            program=Hop(input=Entity(entity_id="alice"), relation="friend"),
        ),
        backend,
        graph_snapshot="toy-v1",
    ).model_copy(
        update={"split": "val", "question": "Who is <Alice>'s friend?"}
    )
    input_path = tmp_path / "val.parquet"
    write_records(input_path, [task.model_dump(mode="json")])
    catalog_path = tmp_path / "relations.json"
    build_relation_catalog([task], backend, catalog_path)
    model = KQAProModelConfig(model_url="http://unused", model="fake")
    return input_path, KQAProValConfig(
        model=model,
        input_path=input_path,
        relation_catalog=catalog_path,
        graph_snapshot="toy-v1",
        concurrency=3,
    )


def _script() -> str:
    return json.dumps(
        {
            "version": "0.3",
            "ops": [
                {
                    "op": "resolve_entity",
                    "query": "Alice",
                    "match": "exact",
                    "limit": 1,
                    "out": "h0",
                },
                {
                    "op": "follow",
                    "in": "h0",
                    "relation": "friend",
                    "direction": "out",
                    "limit": 10,
                    "out": "h1",
                },
                {"op": "emit", "in": "h1"},
            ],
        }
    )


def _all_entities_script() -> str:
    return json.dumps(
        {
            "version": "0.3",
            "ops": [
                {"op": "all_entities", "max_results": 100, "out": "h0"},
                {"op": "filter_type", "in": "h0", "type_id": "person", "out": "h1"},
                {"op": "emit", "in": "h1"},
            ],
        }
    )


def _filter_literal_script() -> str:
    return json.dumps(
        {
            "version": "0.3",
            "ops": [
                {
                    "op": "resolve_entity",
                    "query": "Alice",
                    "match": "exact",
                    "limit": 1,
                    "out": "h0",
                },
                {
                    "op": "follow",
                    "in": "h0",
                    "relation": "friend",
                    "direction": "out",
                    "limit": 10,
                    "out": "h1",
                },
                {
                    "op": "filter_literal",
                    "in": "h1",
                    "relation": "age",
                    "comparator": "eq",
                    "value": {"value": 29, "datatype": "quantity"},
                    "out": "h2",
                },
                {"op": "emit", "in": "h2"},
            ],
        }
    )


def test_single_model_eval_uses_tool_model_direct_fallback(tmp_path: Path) -> None:
    input_path, config = _fixture(tmp_path)
    import asyncio

    for stage in ("sft", "grpo"):
        summary = asyncio.run(
            evaluate_kqapro_val(
                input_path,
                tmp_path / stage,
                config,
                model_stage=stage,
                backend=toy_graph(),
                client=FakeCompletionClient(
                    ["not-json", '<answer>["Bob"]</answer>']
                ),
            )
        )
        assert summary["model_stage"] == stage
        assert summary["overall"]["exact_match"] == 1.0
        assert summary["overall"]["fallback_rate"] == 1.0
        assert summary["overall"]["tool_success_rate"] == 0.0
        rows = read_records(tmp_path / stage / "predictions.parquet")
        assert len(rows) == 1
        assert rows[0]["model"] == stage
        assert rows[0]["inference_mode"] == "direct_fallback"
        assert rows[0]["rejection_reason"]["code"] == "GRAPHSCRIPT_PARSE_FAILED"


def test_base_eval_never_attempts_graph_tool(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    input_path, config = _fixture(tmp_path)
    import asyncio

    caplog.set_level(logging.INFO, logger="graphtask_r1.progress")
    summary = asyncio.run(
        evaluate_kqapro_val(
            input_path,
            tmp_path / "base",
            config,
            model_stage="base",
            backend=toy_graph(),
            client=FakeCompletionClient(['<answer>["Bob"]</answer>']),
        )
    )
    row = read_records(tmp_path / "base/predictions.parquet")[0]
    assert summary["overall"]["exact_match"] == 1.0
    assert row["tool_attempted"] is False
    assert row["inference_mode"] == "direct"
    progress_messages = [record.message for record in caplog.records]
    assert any(
        'operation="evaluate.kqapro_val"' in message
        and 'phase="started"' in message
        and 'model_stage="base"' in message
        for message in progress_messages
    )
    assert any(
        'operation="evaluate.kqapro_val"' in message
        and 'phase="completed"' in message
        and "completed=1" in message
        and "correct=1" in message
        and 'bar="[████████████████████]"' in message
        for message in progress_messages
    )


def test_visualization_is_separate_bounded_and_html_safe(tmp_path: Path) -> None:
    input_path, config = _fixture(tmp_path)
    preview = inspect_kqapro_val(input_path, config, backend=toy_graph())
    assert preview[0]["gold_answers"] == ["Bob"]

    import asyncio

    result = asyncio.run(
        visualize_kqapro_val(
            input_path,
            tmp_path / "visualization",
            config,
            model_stage="grpo",
            limit=1,
            backend=toy_graph(),
            client=FakeCompletionClient([_script()]),
        )
    )

    html = (tmp_path / "visualization/paths.html").read_text()
    assert result["html"].endswith("paths.html")
    assert "Who is &lt;Alice&gt;&#x27;s friend?" in html
    assert "Who is <Alice>" not in html
    assert "resolve_entity" in html
    assert 'class="trace-view"' in html
    assert 'class="trace-graph"' in html
    assert "本步获取的节点" in html
    assert "点击图中的节点查看" in html
    assert len(result["results"]) == 1
    assert result["results"][0]["model"] == "grpo"
    assert len(result["results"][0]["execution_steps"]) == 3
    prediction = read_records(tmp_path / "visualization/predictions.parquet")[0]
    assert prediction["entity_details"]["bob"]["label"] == "Bob"
    assert prediction["relation_details"]["friend"]["relation_id"] == "friend"


def test_visualization_marks_all_entities_as_deferred_not_empty(tmp_path: Path) -> None:
    input_path, config = _fixture(tmp_path)
    import asyncio

    result = asyncio.run(
        visualize_kqapro_val(
            input_path,
            tmp_path / "all-entities",
            config,
            model_stage="sft",
            limit=1,
            backend=toy_graph(),
            client=FakeCompletionClient([_all_entities_script()]),
        )
    )

    steps = result["results"][0]["execution_steps"]
    assert steps[0]["output"]["state"] == "deferred"
    assert steps[0]["output"]["limit"] == 100
    assert steps[1]["output"]["state"] == "materialized"
    assert steps[1]["output"]["total_count"] == 3
    html = (tmp_path / "all-entities/paths.html").read_text()
    assert "延迟集合不是空集合" in html
    assert "已出现节点不会在后续步骤消失" in html
    assert "拖动节点可调整布局" in html


def test_visualization_connects_filter_property_to_candidate_node(tmp_path: Path) -> None:
    input_path, config = _fixture(tmp_path)
    assert config.relation_catalog is not None
    catalog = json.loads(config.relation_catalog.read_text())
    catalog.append(toy_graph().relation_info("age").model_dump(mode="json"))
    write_json(config.relation_catalog, catalog)
    import asyncio

    asyncio.run(
        visualize_kqapro_val(
            input_path,
            tmp_path / "filter-property",
            config,
            model_stage="grpo",
            limit=1,
            backend=toy_graph(),
            client=FakeCompletionClient([_filter_literal_script()]),
        )
    )

    prediction = read_records(tmp_path / "filter-property/predictions.parquet")[0]
    filter_step = prediction["execution_steps"][2]
    assert {
        (edge["subject"], edge["relation"], edge["object"])
        for edge in filter_step["new_evidence"]
    } == {("bob", "age", "29")}
    assert prediction["entity_details"]["bob"]["observed_properties"] == {
        "age": ["29"]
    }
    html = (tmp_path / "filter-property/paths.html").read_text()
    assert "filter_literal" in html
    assert "observed_properties" in html
    assert "process-result" in html


def test_compare_reads_separate_single_model_runs(tmp_path: Path) -> None:
    paths: list[Path] = []
    for stage, exact_match in (("base", 0.2), ("sft", 0.5), ("grpo", 0.6)):
        path = tmp_path / f"{stage}.json"
        write_json(
            path,
            {
                "dataset": "kqapro",
                "split": "val",
                "model_stage": stage,
                "graph_snapshot": "kqapro-v1",
                "input": "val/tasks.parquet",
                "examples": 100,
                "overall": {
                    "exact_match": exact_match,
                    "f1": exact_match + 0.1,
                    "precision": exact_match,
                    "recall": exact_match,
                    "tool_success_rate": 0.9 if stage != "base" else 0.0,
                    "fallback_rate": 0.1 if stage != "base" else 0.0,
                },
            },
        )
        paths.append(path)

    comparison = compare_kqapro_val_metrics(
        paths, output_path=tmp_path / "comparison.json"
    )

    assert comparison["stages"]["base"]["exact_match"] == 0.2
    assert comparison["delta_vs_base"]["sft"]["exact_match"] == 0.3
    assert comparison["delta_vs_base"]["grpo"]["f1"] == pytest.approx(0.4)
    assert (tmp_path / "comparison.json").exists()
