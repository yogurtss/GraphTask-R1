from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from kqapro_grr.pipeline import (
    TARGET_CONTRACT,
    GRRConfig,
    _certify_task,
    build_replay_pairs,
    collect_fixed_incumbent_rollouts,
)

from graphtask_r1.dsl import canonical_signature
from graphtask_r1.evaluation.kqapro_val import CompletionResult
from graphtask_r1.graph import InMemoryGraphBackend, toy_graph
from graphtask_r1.graphscript import program_to_graphscript
from graphtask_r1.schema import (
    AnswerSet,
    Entity,
    EntityInfo,
    Hop,
    TaskTrainingRecord,
    Triple,
    Union,
    VerificationSummary,
)
from graphtask_r1.utils import read_json, write_json, write_records


class FakeCompletionClient:
    def __init__(self, responses: Mapping[str, str]) -> None:
        self.responses = dict(responses)
        self.trace_ids: list[str] = []
        self.seeds: list[int] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        trace_id: str,
        seed: int,
    ) -> CompletionResult:
        question = next(
            str(message["content"])
            for message in messages
            if message.get("role") == "user"
        )
        self.trace_ids.append(trace_id)
        self.seeds.append(seed)
        response = next(value for key, value in self.responses.items() if key in question)
        return CompletionResult(content=response, completion_tokens=17, cached=False)

    def flush(self) -> None:
        return None


def _tasks() -> list[TaskTrainingRecord]:
    return [
        TaskTrainingRecord(
            task_id="toy-friend",
            question="Who is Bob's friend?",
            topic_entities=(EntityInfo(entity_id="bob", label="Bob"),),
            program=Hop(input=Entity(entity_id="bob"), relation="friend"),
            gold_answers=AnswerSet.entities(["cara"]),
            verification=VerificationSummary(executable=True),
        ),
        TaskTrainingRecord(
            task_id="toy-work",
            question="Where does Alice work?",
            topic_entities=(EntityInfo(entity_id="alice", label="Alice"),),
            program=Hop(input=Entity(entity_id="alice"), relation="works_at"),
            gold_answers=AnswerSet.entities(["acme"]),
            verification=VerificationSummary(executable=True),
        ),
    ]


def _config(tmp_path: Path, output_name: str) -> GRRConfig:
    backend = toy_graph()
    tasks_path = tmp_path / "tasks.parquet"
    catalog_path = tmp_path / "relations.json"
    graph_path = tmp_path / "graph.sqlite"
    model_path = tmp_path / "model.safetensors"
    if not tasks_path.exists():
        write_records(
            tasks_path,
            (task.model_dump(mode="json") for task in _tasks()),
        )
        write_json(
            catalog_path,
            [value.model_dump(mode="json") for value in backend.all_relation_infos()],
        )
        graph_path.write_bytes(b"toy-graph")
        model_path.write_bytes(b"fixed-incumbent")
    return GRRConfig(
        repo_root=tmp_path,
        seed=4242,
        tasks=tasks_path,
        relation_catalog=catalog_path,
        graph_db=graph_path,
        model_artifact=model_path,
        output_dir=tmp_path / output_name,
        model_url="http://127.0.0.1:9999/v1",
        model_name="fixed-incumbent",
        max_completion_tokens=512,
        enable_thinking=False,
        probe_count=2,
        pair_count=2,
        max_pairs_per_boundary_family=None,
        concurrency=2,
        request_timeout_s=10.0,
        request_retries=1,
        max_follow_limit=100,
        max_edge_visits=100,
        max_returned_entities=100,
    )


def _responses() -> dict[str, str]:
    friend = program_to_graphscript(_tasks()[0].program, version="0.3").model_dump(
        mode="json", by_alias=True
    )
    for operation in friend["ops"]:
        if operation["op"] == "follow":
            operation["direction"] = "in"
    return {
        "Bob's friend": "<think>hidden</think>\n" + json.dumps(friend),
        "Alice work": "<think>hidden</think>\n{not-json",
    }


def test_fixed_rollout_to_task_local_pair_is_replayable(tmp_path: Path) -> None:
    backend = toy_graph()
    first_config = _config(tmp_path, "first")
    first_client = FakeCompletionClient(_responses())
    first = asyncio.run(
        collect_fixed_incumbent_rollouts(
            first_config,
            backend=backend,
            client=first_client,
        )
    )
    replay_config = _config(tmp_path, "replay")
    replay = asyncio.run(
        collect_fixed_incumbent_rollouts(
            replay_config,
            backend=backend,
            client=FakeCompletionClient(_responses()),
        )
    )

    assert first["probe_count"] == 2
    assert first["target_contract"] == TARGET_CONTRACT
    assert first["certified_probe_count"] == 2
    assert first["incorrect_rollout_count"] == 2
    assert first["reason_counts"] == {"NON_JSON": 1, "TOOL_WRONG_ANSWER": 1}
    assert first["stage_counts"] == {
        "predicted_parse": 1,
        "terminal_semantic": 1,
    }
    assert first["rollouts_sha256"] == replay["rollouts_sha256"]
    assert len(first_client.trace_ids) == 2
    assert len(set(first_client.trace_ids)) == 2
    assert len(set(first_client.seeds)) == 2

    pair_manifest = build_replay_pairs(first_config)
    assert pair_manifest["selected_pairs"] == 2
    assert pair_manifest["generation_mode"] == "task_local_fixed_incumbent_error_replay"
    assert pair_manifest["stage_counts"] == first["stage_counts"]
    train_rows = [
        json.loads(line)
        for line in Path(str(pair_manifest["train"])).read_text().splitlines()
    ]
    assert len({row["task_id"] for row in train_rows}) == 2
    assert all(row["messages"][-1]["role"] == "assistant" for row in train_rows)
    assert all(not row["rejected_response"].startswith("<think>") for row in train_rows)
    assert {row["grr_reason_code"] for row in train_rows} == {
        "NON_JSON",
        "TOOL_WRONG_ANSWER",
    }

    certificates = [
        json.loads(line)
        for line in Path(str(pair_manifest["certificates"])).read_text().splitlines()
    ]
    assert all(value["same_task"] for value in certificates)
    assert all(value["chosen_graph_certified"] for value in certificates)
    assert all(value["rejected_sampled_from_fixed_incumbent"] for value in certificates)
    stored_manifest = read_json(first_config.rollout_manifest_path)
    assert stored_manifest["model_name"] == "fixed-incumbent"
    rollout_rows = [
        json.loads(line)
        for line in first_config.rollouts_path.read_text(encoding="utf-8").splitlines()
    ]
    root_queries: set[str] = set()
    for row in rollout_rows:
        chosen = json.loads(row["chosen_graphscript"])
        roots = [op for op in chosen["ops"] if op["op"] == "resolve_entity"]
        assert all(op["match"] == "exact" for op in roots)
        root_queries.update(str(op["query"]) for op in roots)
    assert root_queries == {"Alice", "Bob"}


def test_certification_compiles_complete_same_label_candidate_set(
    tmp_path: Path,
) -> None:
    backend = InMemoryGraphBackend(
        (
            Triple(subject="alex-1", relation="works_at", object="acme"),
            Triple(subject="alex-2", relation="works_at", object="globex"),
        ),
        (
            EntityInfo(entity_id="alex-1", label="Alex"),
            EntityInfo(entity_id="alex-2", label="Alex"),
            EntityInfo(entity_id="acme", label="Acme"),
            EntityInfo(entity_id="globex", label="Globex"),
        ),
    )
    program = Hop(
        input=Union(
            inputs=(Entity(entity_id="alex-1"), Entity(entity_id="alex-2"))
        ),
        relation="works_at",
    )
    task = TaskTrainingRecord(
        task_id="same-label-candidate-set",
        question="Where do people named Alex work?",
        topic_entities=(EntityInfo(entity_id="alex-1", label="Alex"),),
        program=program,
        gold_answers=AnswerSet.entities(["acme", "globex"]),
        verification=VerificationSummary(executable=True),
    )

    certified, reason = _certify_task(
        task,
        rank=0,
        backend=backend,
        relations=backend.all_relation_infos(),
        config=_config(tmp_path, "candidate-set"),
    )

    assert reason == "CERTIFIED"
    assert certified is not None
    assert certified.chosen_program_signature == canonical_signature(program)
    chosen = json.loads(certified.chosen_graphscript)
    roots = [operation for operation in chosen["ops"] if operation["op"] == "resolve_entity"]
    assert roots == [
        {
            "op": "resolve_entity",
            "query": "Alex",
            "match": "exact",
            "limit": 2,
            "out": "h0",
        }
    ]


def test_certification_rejects_incomplete_same_label_candidate_set(
    tmp_path: Path,
) -> None:
    backend = InMemoryGraphBackend(
        (),
        (
            EntityInfo(entity_id="alex-1", label="Alex"),
            EntityInfo(entity_id="alex-2", label="Alex"),
            EntityInfo(entity_id="alex-3", label="Alex"),
        ),
    )
    task = TaskTrainingRecord(
        task_id="incomplete-same-label-candidate-set",
        question="Which Alex?",
        topic_entities=(EntityInfo(entity_id="alex-1", label="Alex"),),
        program=Union(
            inputs=(Entity(entity_id="alex-1"), Entity(entity_id="alex-2"))
        ),
        gold_answers=AnswerSet.entities(["alex-1", "alex-2"]),
        verification=VerificationSummary(executable=True),
    )

    certified, reason = _certify_task(
        task,
        rank=0,
        backend=backend,
        relations=(),
        config=_config(tmp_path, "incomplete-candidate-set"),
    )

    assert certified is None
    assert reason == "CHOSEN_UNOBSERVABLE_ENTITY_ID"


def test_terminal_correct_rollout_is_not_replayed(tmp_path: Path) -> None:
    config = _config(tmp_path, "correct-filter")
    correct_friend = program_to_graphscript(
        _tasks()[0].program, version="0.3"
    ).model_dump_json(by_alias=True)
    responses = _responses()
    responses["Bob's friend"] = correct_friend
    manifest = asyncio.run(
        collect_fixed_incumbent_rollouts(
            config,
            backend=toy_graph(),
            client=FakeCompletionClient(responses),
        )
    )
    assert manifest["reason_counts"] == {"NON_JSON": 1, "TERMINAL_CORRECT": 1}

    pairs = build_replay_pairs(config)
    assert pairs["selected_pairs"] == 1
    assert pairs["reason_counts"] == {"NON_JSON": 1}
