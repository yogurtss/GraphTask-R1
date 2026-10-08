from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from kqapro_ere.mutation import CounterfactualMutator
from kqapro_ere.pairs import load_residual_profile
from kqapro_ere.profile import mine_residual_profile

from graphtask_r1.graph import toy_graph
from graphtask_r1.graphscript import program_to_graphscript
from graphtask_r1.schema import (
    Entity,
    Hop,
    Intersect,
    TaskTrainingRecord,
    VerificationSummary,
)
from graphtask_r1.utils import write_json


def _rendered_prompt(question: str) -> str:
    return f"<|im_start|>user\nQuestion: {question}\nAllowed relation catalog:\n- friend: friend"


def _mine_single_completion(
    tmp_path: Path,
    *,
    task_id: str,
    question: str,
    reference_program_json: str,
    completion: str,
) -> dict[str, object]:
    backend = toy_graph()
    relations = tuple(backend.all_relation_infos())
    tasks_path = tmp_path / "solver.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [
                        {"role": "system", "content": "solver"},
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\nAllowed relation catalog:\n- friend: friend"
                            ),
                        },
                    ],
                    "extra_info": {
                        "task_id": task_id,
                        "reference_program_json": reference_program_json,
                        "allowed_relations": [value.relation_id for value in relations],
                        "topic_entity_ids": ["bob"],
                    },
                }
            ]
        ),
        tasks_path,
    )
    catalog_path = tmp_path / "relations.json"
    write_json(
        catalog_path,
        [relation.model_dump(mode="json") for relation in relations],
    )
    completions_path = tmp_path / "completions.jsonl"
    completions_path.write_text(
        json.dumps(
            {
                "step": ["1"],
                "prompt": [_rendered_prompt(question)],
                "completion": [completion],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return mine_residual_profile(
        tasks_path,
        completions_path,
        catalog_path,
        tmp_path / "profile",
        backend=backend,
        seed=4242,
        max_edge_visits=100,
        max_returned_entities=100,
    )


def test_latest_residual_profile_is_seeded_replayable_and_deterministic(
    tmp_path: Path,
) -> None:
    backend = toy_graph()
    relations = tuple(backend.all_relation_infos())
    relation_ids = [relation.relation_id for relation in relations]
    question = "Who is Bob's friend?"
    program = Hop(input=Entity(entity_id="bob"), relation="friend")
    correct_script = program_to_graphscript(program, version="0.3")
    wrong_payload = correct_script.model_dump(mode="json", by_alias=True)
    for operation in wrong_payload["ops"]:
        if operation["op"] == "follow":
            operation["direction"] = "in"
    wrong_script = json.dumps(wrong_payload, separators=(",", ":"))

    tasks_path = tmp_path / "solver.parquet"
    task_row = {
        "prompt": [
            {"role": "system", "content": "solver"},
            {
                "role": "user",
                "content": (f"Question: {question}\nAllowed relation catalog:\n- friend: friend"),
            },
        ],
        "extra_info": {
            "task_id": "toy-friend",
            "reference_program_json": program.model_dump_json(),
            "allowed_relations": relation_ids,
            "topic_entity_ids": ["bob"],
        },
    }
    pq.write_table(pa.Table.from_pylist([task_row]), tasks_path)

    catalog_path = tmp_path / "relations.json"
    write_json(
        catalog_path,
        [relation.model_dump(mode="json") for relation in relations],
    )
    completions_path = tmp_path / "completions.jsonl"
    records = [
        {
            "step": ["1"],
            "prompt": [_rendered_prompt(question)],
            "completion": [correct_script.model_dump_json(by_alias=True)],
        },
        {
            "step": ["2"],
            "prompt": [_rendered_prompt(question)],
            "completion": [wrong_script],
        },
    ]
    completions_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    first = mine_residual_profile(
        tasks_path,
        completions_path,
        catalog_path,
        tmp_path / "profile-a",
        backend=backend,
        seed=4242,
        max_edge_visits=100,
        max_returned_entities=100,
    )
    replay = mine_residual_profile(
        tasks_path,
        completions_path,
        catalog_path,
        tmp_path / "profile-b",
        backend=backend,
        seed=4242,
        max_edge_visits=100,
        max_returned_entities=100,
    )

    assert first["selection"] == "latest_completion_per_question"
    assert first["total_completion_observations"] == 2
    assert first["selected_latest_observations"] == 1
    assert first["reason_counts"] == {"RESIDUAL_LOCALIZED": 1}
    assert first["localized_residual_count"] == 1
    assert first["effective_signal_count"] == 1
    assert first["has_effective_signal"] is True
    assert first["effective_slot_counts"] == {"direction": 1}
    assert first["slot_priority"][0] == "direction"  # type: ignore[index]
    assert first["observed_values"] == {"direction": ["in"]}
    assert first["residuals_sha256"] == replay["residuals_sha256"]

    residual_path = Path(str(first["residuals"]))
    residual = json.loads(residual_path.read_text().strip())
    assert residual["step"] == 2
    assert residual["residual_slot"] == "direction"

    profile_path = tmp_path / "profile-a" / "residual_profile.json"
    loaded = load_residual_profile(profile_path)
    assert loaded.slot_priority[0] == "direction"
    assert loaded.observed_values == {"direction": ("in",)}
    outcome = CounterfactualMutator(
        backend,
        relations,
        seed=4242,
        max_edge_visits=100,
        max_returned_entities=100,
        observed_values=loaded.observed_values,
    ).mutate(
        # The gold positive remains the certified executed Program; the profile only
        # controls which one-slot counterfactual is tried first.
        TaskTrainingRecord(
            task_id="toy-friend",
            question=question,
            topic_entities=(),
            program=program,
            gold_answers=backend.execute_program(program),
            verification=VerificationSummary(executable=True),
        ),
        slot_priority=loaded.slot_priority,
    )
    assert outcome.certificate is not None
    assert outcome.certificate.edit.slot == "direction"
    assert outcome.certificate.edit.new_value == "in"


def test_cross_operator_residual_is_not_mislabeled_as_query(tmp_path: Path) -> None:
    completion = json.dumps(
        {
            "version": "0.3",
            "ops": [
                {"op": "all_entities", "max_results": 100, "out": "h0"},
                {"op": "filter_type", "in": "h0", "type_id": "company", "out": "h1"},
                {"op": "emit", "in": "h1"},
            ],
        }
    )
    profile = _mine_single_completion(
        tmp_path,
        task_id="toy-entity",
        question="Which entity is Bob?",
        reference_program_json=Entity(entity_id="bob").model_dump_json(),
        completion=completion,
    )

    assert profile["reason_counts"] == {"OPERATOR_MISMATCH": 1}
    assert profile["slot_counts"] == {}
    assert profile["localized_residual_count"] == 0
    assert profile["effective_signal_count"] == 0
    assert profile["has_effective_signal"] is False
    residual = json.loads(Path(str(profile["residuals"])).read_text().strip())
    divergence = residual["analysis"]["earliest_divergence"]
    assert divergence["gold_op"] == "resolve_entity"
    assert divergence["predicted_op"] == "all_entities"
    assert residual["reason_code"] == "OPERATOR_MISMATCH"
    assert residual["residual_slot"] is None
    assert residual["effective_signal"] is False


def test_intersect_union_residual_remains_effective_set_operator_signal(
    tmp_path: Path,
) -> None:
    program = Intersect(
        inputs=(Entity(entity_id="bob"), Entity(entity_id="cara")),
    )
    predicted = program_to_graphscript(program, version="0.3").model_dump(
        mode="json",
        by_alias=True,
    )
    for operation in predicted["ops"]:
        if operation["op"] == "intersect":
            operation["op"] = "union"
    profile = _mine_single_completion(
        tmp_path,
        task_id="toy-set",
        question="Which entity is both Bob and Cara?",
        reference_program_json=program.model_dump_json(),
        completion=json.dumps(predicted),
    )

    assert profile["reason_counts"] == {"RESIDUAL_LOCALIZED": 1}
    assert profile["slot_counts"] == {"set_operator": 1}
    assert profile["effective_slot_counts"] == {"set_operator": 1}
    assert profile["effective_signal_count"] == 1
    assert profile["observed_values"] == {"set_operator": ["union"]}
    assert profile["slot_priority"][0] == "set_operator"  # type: ignore[index]
