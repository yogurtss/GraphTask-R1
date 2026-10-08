from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kqapro_grr.boundary import (
    BOUNDARY_END,
    BOUNDARY_START,
    _certify_graft,
    _graft_operation,
    build_boundary_pairs,
    locate_execution_boundary,
)
from kqapro_grr.ms_swift_boundary_plugin import ExecutionBoundaryLossScale
from kqapro_grr.pipeline import TARGET_CONTRACT, collect_fixed_incumbent_rollouts
from swift.llm.template.utils import ContextType
from test_pipeline import FakeCompletionClient, _config, _responses

from graphtask_r1.graph import toy_graph
from graphtask_r1.graphscript import parse_graphscript


def _without_markers(value: str) -> str:
    return value.replace(BOUNDARY_START, "").replace(BOUNDARY_END, "")


def test_boundary_pair_masks_one_localized_operation(tmp_path: Path) -> None:
    config = _config(tmp_path, "boundary")
    asyncio.run(
        collect_fixed_incumbent_rollouts(
            config,
            backend=toy_graph(),
            client=FakeCompletionClient(_responses()),
        )
    )

    manifest = build_boundary_pairs(config, backend=toy_graph())

    assert (
        manifest["algorithm"]
        == "executor_certified_denotation_boundary_po_v2_candidate_set"
    )
    assert manifest["selected_pairs"] == 1
    assert manifest["unlocalizable_reason_counts"] == {"NON_JSON": 1}
    row = json.loads(
        Path(str(manifest["train"])).read_text(encoding="utf-8").splitlines()[0]
    )
    chosen = row["messages"][-1]["content"]
    rejected = row["rejected_response"]
    assert chosen.count(BOUNDARY_START) == chosen.count(BOUNDARY_END) == 1
    assert rejected.count(BOUNDARY_START) == rejected.count(BOUNDARY_END) == 1
    parse_graphscript(_without_markers(chosen))
    parse_graphscript(_without_markers(rejected))
    chosen_prefix, chosen_tail = chosen.split(BOUNDARY_START)
    chosen_boundary, chosen_suffix = chosen_tail.split(BOUNDARY_END)
    rejected_prefix, rejected_tail = rejected.split(BOUNDARY_START)
    rejected_boundary, rejected_suffix = rejected_tail.split(BOUNDARY_END)
    assert chosen_prefix == rejected_prefix
    assert chosen_suffix == rejected_suffix
    assert chosen_boundary != rejected_boundary

    certificate = json.loads(
        Path(str(manifest["certificates"]))
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert certificate["same_task"] is True
    assert certificate["chosen_loss_scope"] == "full_response_with_eos"
    assert certificate["rejected_loss_scope"].startswith("natural_tokens_overlapping")
    assert certificate["natural_response_tokenization_preserved"] is True
    assert manifest["target_contract"] == TARGET_CONTRACT


class _PairTokenizer:
    def __call__(self, value: str, **_: object) -> dict[str, object]:
        offsets = [(start, min(start + 2, len(value))) for start in range(0, len(value), 2)]
        return {
            "input_ids": [
                sum(ord(char) for char in value[start:end])
                for start, end in offsets
            ],
            "offset_mapping": offsets,
        }


def test_boundary_loss_scale_preserves_natural_tokens_and_is_asymmetric() -> None:
    value = f'prefix{BOUNDARY_START}{{"op":"follow"}}{BOUNDARY_END}suffix'
    loss_scale = ExecutionBoundaryLossScale()
    loss_scale.tokenizer = _PairTokenizer()
    contexts, weights = loss_scale(
        [value, "<eos>"],
        [ContextType.RESPONSE, ContextType.SUFFIX],
        [{"role": "assistant", "content": value}],
    )
    clean = _without_markers(value)
    natural_ids = _PairTokenizer()(clean)["input_ids"]

    assert [context[0] for context in contexts[:-1]] == natural_ids
    assert 0.0 in weights[:-1]
    assert 1.0 in weights[:-1]
    assert weights[-1] == 0.0

    positive_contexts, positive_weights = loss_scale(
        [value, "<eos>"],
        [ContextType.RESPONSE, ContextType.SUFFIX],
        [{"role": "assistant", "content": value, "loss": 1.0}],
    )
    assert [context[0] for context in positive_contexts[:-1]] == natural_ids
    assert positive_weights == [1.0] * len(positive_weights)


def test_ordered_operand_mapping_preserves_the_policy_role_swap(tmp_path: Path) -> None:
    backend = toy_graph()
    config = _config(tmp_path, "operand-swap")
    gold = {
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
                "op": "resolve_entity",
                "query": "Bob",
                "match": "exact",
                "limit": 1,
                "out": "h1",
            },
            {"op": "query_relation", "subject": "h0", "object": "h1", "out": "h2"},
            {"op": "emit", "in": "h2"},
        ],
    }
    rejected = json.loads(json.dumps(gold))
    rejected["ops"][2]["subject"] = "h1"
    rejected["ops"][2]["object"] = "h0"
    record: dict[str, object] = {
        "task_id": "ordered-swap",
        "chosen_graphscript": json.dumps(gold),
        "rejected_response": json.dumps(rejected),
        "chosen_answers": ["literal:friend"],
        "failure_stage": "predicted_execution",
        "reason_code": "EMPTY_RESULT",
    }
    relations = frozenset(value.relation_id for value in backend.all_relation_infos())

    location = locate_execution_boundary(
        record,
        backend=backend,
        allowed_relations=relations,
        config=config,
    )
    assert location is not None
    assert location.gold_step_index == location.predicted_step_index == 2
    assert dict(location.input_handle_mapping or ()) == {"h0": "h0", "h1": "h1"}
    graft, reason = _certify_graft(
        record,
        location,
        backend=backend,
        allowed_relations=relations,
        config=config,
    )
    assert reason == "CERTIFIED"
    assert graft is not None
    assert graft.grafted_operation["subject"] == "h1"
    assert graft.grafted_operation["object"] == "h0"


def test_graft_accepts_input_compatible_output_kind_confusion() -> None:
    gold = {
        "op": "query_attribute",
        "in": "h0",
        "attribute": "age",
        "out": "h1",
    }
    predicted = {
        "op": "follow",
        "in": "h7",
        "relation": "friend",
        "direction": "out",
        "limit": 10,
        "out": "h8",
    }

    grafted = _graft_operation(predicted, gold, (("h7", "h0"),))

    assert grafted is not None
    assert grafted["op"] == "follow"
    assert grafted["in"] == "h0"
    assert grafted["out"] == "h1"
