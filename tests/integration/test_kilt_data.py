import asyncio
import json
from pathlib import Path
from typing import Any

from graphtask_r1.data import prepare_kilt
from graphtask_r1.envs import SolverEnv
from graphtask_r1.graph import SQLiteGraphBackend, backend_from_snapshot
from graphtask_r1.graphscript import execute_graphscript, parse_graphscript
from graphtask_r1.schema import AnswerSet, BenchmarkExample, EpisodeInput, ToolCall
from graphtask_r1.training.opponent import FrozenSolverService
from graphtask_r1.utils import file_hash, read_records


def _write_kilt_fixture(path: Path) -> None:
    pages = [
        {
            "_id": "1",
            "wikipedia_id": "1",
            "wikipedia_title": "Alpha",
            "text": ["Alpha", "Alpha links to the city of Beta."],
            "anchors": [{"text": "Beta", "wikipedia_id": "2"}],
            "categories": "Letters,Examples",
            "history": {"revid": 1},
            "wikidata_info": {"wikidata_id": "Q1"},
        },
        {
            "_id": "2",
            "wikipedia_id": "2",
            "wikipedia_title": "Beta City",
            "text": ["Beta City", "Beta City is the expected answer."],
            "anchors": [{"text": "missing target"}],
            "categories": "Cities",
            "history": {"revid": 2},
            "wikidata_info": {},
        },
        {
            "_id": "3",
            "wikipedia_id": "3",
            "wikipedia_title": "Gamma",
            "text": ["Not included by the fixture limit."],
            "anchors": [],
            "categories": "",
            "history": {"revid": 3},
            "wikidata_info": {},
        },
    ]
    path.write_text("\n".join(json.dumps(page) for page in pages) + "\n")


def test_kilt_streaming_graph_and_passage_index(tmp_path: Path) -> None:
    source = tmp_path / "kilt_knowledgesource.json"
    _write_kilt_fixture(source)
    output = tmp_path / "processed"

    metrics = prepare_kilt(source, output, limit=2)
    assert metrics["pages"] == 2
    assert metrics["source_complete"] is False
    assert metrics["rejections"] == 1
    assert metrics["build"]["reused"] is False
    rejections = read_records(output / "rejections.parquet")
    assert rejections == [
        {
            "count": 1,
            "detail": "page_id=2",
            "index": 1,
            "reason_code": "MISSING_ANCHOR_TARGET",
        }
    ]

    backend = SQLiteGraphBackend(output / "graph.sqlite", snapshot_id="kilt-2019-08-01-v1")
    try:
        edges = backend.neighbors(["1"], direction="out", relation_ids=["wikipedia_link"])
        assert [edge.object for edge in edges] == ["2"]
        assert backend.entity_info("1").label == "Alpha"
        results = backend.search_text("expected answer", limit=2)
        assert results[0]["page_id"] == "2"
        assert results[0]["title"] == "Beta City"
        assert backend.resolve_entities("Alpha", match="exact", limit=2) == ("1",)
        assert backend.resolve_entities("expected answer", match="search", limit=2)[0] == "2"

        script = parse_graphscript(
            {
                "version": "0.2",
                "ops": [
                    {
                        "op": "search_passage",
                        "query": "expected answer",
                        "limit": 1,
                        "out": "h0",
                    },
                    {"op": "passage_pages", "in": "h0", "out": "h1"},
                    {"op": "require_unique", "in": "h1"},
                    {"op": "emit", "in": "h1"},
                ],
            }
        )
        execution = execute_graphscript(
            script,
            backend,
            allowed_relations=frozenset(),
            max_edge_visits=10,
        )
        assert execution.answers == AnswerSet.entities(["2"])
        assert execution.usage.passage_searches == 1

        env = SolverEnv(backend, max_turns=3)
        env.reset(
            EpisodeInput(
                task_id="openqa-no-topic",
                question="Which city is the expected answer?",
                topic_entity_ids=(),
                gold_answers=AnswerSet.entities(["Beta City"]),
            ),
            seed=7,
        )
        searched = env.step(
            ToolCall(
                name="text_search",
                arguments={"query": "expected answer", "limit": 2},
                trace_id="openqa-search-1",
            )
        )
        assert searched.done is False
        assert searched.observation.passages[0].page_id == "2"
        assert searched.observation.passages[0].title == "Beta City"
        json.dumps(env.snapshot())
    finally:
        backend.close()

    reused = prepare_kilt(source, output, limit=2)
    assert reused["build"]["reused"] is True


def test_complete_kilt_build_records_full_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "kilt_knowledgesource.json"
    _write_kilt_fixture(source)
    output = tmp_path / "processed"

    metrics = prepare_kilt(source, output, with_text_index=False)

    assert metrics["source_complete"] is True
    assert metrics["source_sha256"] == file_hash(source)
    assert metrics["source_prefix_sha256"] == file_hash(source)


def test_kilt_snapshot_factory_uses_instance_scoped_path(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "kilt_knowledgesource.json"
    _write_kilt_fixture(source)
    output = tmp_path / "processed"
    prepare_kilt(source, output, limit=1, with_text_index=False)
    monkeypatch.setenv("GRAPHTASK_KILT_DB", str(output / "graph.sqlite"))

    backend = backend_from_snapshot("kilt-2019-08-01-v1")
    try:
        assert backend.all_entities(limit=10) == ("1",)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def test_openqa_benchmark_without_topic_entity_uses_kilt_text_search(
    tmp_path: Path,
) -> None:
    source = tmp_path / "kilt_knowledgesource.json"
    _write_kilt_fixture(source)
    output = tmp_path / "processed"
    prepare_kilt(source, output, limit=2)
    backend = SQLiteGraphBackend(output / "graph.sqlite", snapshot_id="kilt-test-v1")

    class ScriptedService(FrozenSolverService):
        def __init__(self) -> None:
            super().__init__(
                model_url="http://unused",
                model="scripted",
                archive_path=tmp_path / "archive.sqlite",
                max_turns=3,
            )
            self.responses = iter(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "search-1",
                                "type": "function",
                                "function": {
                                    "name": "text_search",
                                    "arguments": json.dumps(
                                        {"query": "expected answer", "limit": 2}
                                    ),
                                },
                            }
                        ],
                    },
                    {"role": "assistant", "content": '<answer>["Beta City"]</answer>'},
                ]
            )

        async def _completion(
            self, messages: list[dict[str, Any]], *, use_tools: bool
        ) -> dict[str, Any]:
            del messages, use_tools
            return next(self.responses)

    service = ScriptedService()
    service.backends["kilt-test-v1"] = backend
    example = BenchmarkExample(
        example_id="openqa-no-topic",
        dataset="hotpotqa",
        split="test",
        question="Which city is described as the expected answer?",
        topic_entity_ids=(),
        gold_answers=AnswerSet.entities(["Beta City"]),
        answer_aliases=(("Beta City",),),
    )
    try:
        result = asyncio.run(
            service.solve(
                {
                    "example": example.model_dump(mode="json"),
                    "graph_snapshot": "kilt-test-v1",
                    "samples": 1,
                }
            )
        )
        assert result["pass_rate"] == 1.0
        assert result["mean_f1"] == 1.0
        assert result["mean_tool_calls"] == 1.0
    finally:
        backend.close()


def test_openqa_graphscript_v02_runs_without_topic_entity(tmp_path: Path) -> None:
    source = tmp_path / "kilt_knowledgesource.json"
    _write_kilt_fixture(source)
    output = tmp_path / "processed"
    prepare_kilt(source, output, limit=2)
    backend = SQLiteGraphBackend(output / "graph.sqlite", snapshot_id="kilt-test-v1")

    class ScriptedService(FrozenSolverService):
        def __init__(self) -> None:
            super().__init__(
                model_url="http://unused",
                model="scripted",
                archive_path=tmp_path / "archive.sqlite",
                max_turns=3,
                interaction_mode="graphscript",
            )

        async def _completion(
            self, messages: list[dict[str, Any]], *, use_tools: bool
        ) -> dict[str, Any]:
            assert use_tools is False
            assert "GraphScript v0.2" in messages[0]["content"]
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "version": "0.2",
                        "ops": [
                            {
                                "op": "search_passage",
                                "query": "expected answer",
                                "limit": 1,
                                "out": "h0",
                            },
                            {"op": "passage_pages", "in": "h0", "out": "h1"},
                            {"op": "require_unique", "in": "h1"},
                            {"op": "emit", "in": "h1"},
                        ],
                    }
                ),
            }

    service = ScriptedService()
    service.backends["kilt-test-v1"] = backend
    example = BenchmarkExample(
        example_id="openqa-code-no-topic",
        dataset="hotpotqa",
        split="test",
        question="Which city is described as the expected answer?",
        topic_entity_ids=(),
        gold_answers=AnswerSet.literals(["Beta City"]),
        answer_aliases=(("Beta City",),),
    )
    try:
        result = asyncio.run(
            service.solve(
                {
                    "example": example.model_dump(mode="json"),
                    "graph_snapshot": "kilt-test-v1",
                    "samples": 1,
                }
            )
        )
        assert result["pass_rate"] == 1.0
        assert result["mean_f1"] == 1.0
        assert result["mean_tool_calls"] == 0.0
        assert result["program_parse_rate"] == 1.0
        assert result["program_execution_rate"] == 1.0
        assert result["mean_program_operators"] == 4.0
        assert result["mean_passage_searches"] == 1.0
    finally:
        backend.close()
