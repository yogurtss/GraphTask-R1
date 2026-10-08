from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel

from graphtask_r1.graph import GraphBackend
from graphtask_r1.graph.sqlite import SQLiteGraphBackend
from graphtask_r1.graphscript import (
    AllEntitiesOp,
    EmitOp,
    GraphScript,
    GraphScriptError,
    execute_graphscript,
    parse_graphscript,
)
from graphtask_r1.schema import AnswerSet
from graphtask_r1.training.prompts import role_prompt
from graphtask_r1.training.relations import load_relation_catalog
from graphtask_r1.utils import file_hash, stable_hash, write_json
from kqapro_grr.pipeline import (
    TARGET_CONTRACT,
    GRRConfig,
    _certify_task,
    _load_probe_tasks,
)

BOUNDARY_START = "<|graph_boundary_start|>"
BOUNDARY_END = "<|graph_boundary_end|>"


@dataclass(frozen=True)
class PrefixState:
    step_index: int
    operation: str
    output_kind: str
    signature: tuple[str, ...]
    alignable: bool = True


@dataclass(frozen=True)
class BoundaryLocation:
    gold_step_index: int
    predicted_step_index: int
    stage: str
    reason_code: str
    input_handle_mapping: tuple[tuple[str, str], ...] | None


@dataclass(frozen=True)
class CertifiedGraft:
    rejected_graphscript: dict[str, object]
    current_policy_operation: dict[str, object]
    grafted_operation: dict[str, object]
    rejected_stage: str
    reason_code: str
    answers: tuple[str, ...]


_ROOT_OPERATIONS = frozenset({"all_entities", "resolve_entity"})
_ENTITY_OPERATIONS = frozenset(
    {
        "follow",
        "filter_type",
        "filter_literal",
        "filter_qualifier",
        "select_among",
        "intersect",
        "union",
        "select_between",
    }
)


def _answer_signature(answers: AnswerSet) -> tuple[str, ...]:
    return tuple(sorted(f"{answer.kind}:{answer.value}" for answer in answers.answers))


def _semantic_operation_signature(operation: BaseModel) -> str:
    payload = operation.model_dump(mode="json", by_alias=True)
    for field in ("out", "in", "inputs", "left", "right", "subject", "object"):
        payload.pop(field, None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prefix_script(script: GraphScript, step_index: int) -> GraphScript:
    operation = script.ops[step_index]
    if isinstance(operation, EmitOp):
        return GraphScript(version=script.version, ops=script.ops[: step_index + 1])
    output_handle = operation.model_dump(mode="json", by_alias=True)["out"]
    assert isinstance(output_handle, str)
    return GraphScript.model_validate(
        {
            "version": script.version,
            "ops": [
                *(
                    value.model_dump(mode="json", by_alias=True)
                    for value in script.ops[: step_index + 1]
                ),
                {"op": "emit", "in": output_handle},
            ],
        }
    )


def _execute_prefixes(
    script: GraphScript,
    *,
    backend: GraphBackend,
    allowed_relations: frozenset[str],
    config: GRRConfig,
    trace_id: str,
) -> tuple[tuple[PrefixState, ...], int | None, str | None]:
    states: list[PrefixState] = []
    for step_index, operation in enumerate(script.ops):
        if isinstance(operation, EmitOp):
            continue
        if isinstance(operation, AllEntitiesOp):
            states.append(
                PrefixState(
                    step_index=step_index,
                    operation=operation.op,
                    output_kind="entity",
                    signature=(f"deferred:all_entities:{operation.max_results}",),
                )
            )
            continue
        try:
            execution = execute_graphscript(
                _prefix_script(script, step_index),
                backend,
                allowed_relations=allowed_relations,
                max_edge_visits=config.max_edge_visits,
                max_returned_entities=config.max_returned_entities,
                trace_id=f"{trace_id}:prefix:{step_index}",
            )
        except GraphScriptError as exc:
            if exc.reason_code == "EMPTY_RESULT":
                states.append(
                    PrefixState(
                        step_index=step_index,
                        operation=operation.op,
                        output_kind=_OPERATION_CONTRACTS[operation.op][1],
                        signature=(
                            "EMPTY_RESULT",
                            _semantic_operation_signature(operation),
                        ),
                    )
                )
                continue
            if exc.reason_code == "BUDGET_EXCEEDED":
                states.append(
                    PrefixState(
                        step_index=step_index,
                        operation=operation.op,
                        output_kind=_OPERATION_CONTRACTS[operation.op][1],
                        signature=("BUDGET_EXCEEDED",),
                        alignable=False,
                    )
                )
                continue
            return tuple(states), step_index, exc.reason_code
        states.append(
            PrefixState(
                step_index=step_index,
                operation=operation.op,
                output_kind=_OPERATION_CONTRACTS[operation.op][1],
                signature=_answer_signature(execution.answers),
            )
        )
    return tuple(states), None, None


def _exact_alignment(
    predicted: tuple[PrefixState, ...],
    gold: tuple[PrefixState, ...],
) -> tuple[tuple[int, int], ...]:
    def matches(predicted_position: int, gold_position: int) -> bool:
        predicted_state = predicted[predicted_position]
        gold_state = gold[gold_position]
        return (
            predicted_state.alignable
            and gold_state.alignable
            and predicted_state.operation == gold_state.operation
            and predicted_state.signature == gold_state.signature
        )

    lengths = [[0] * (len(gold) + 1) for _ in range(len(predicted) + 1)]
    for predicted_position in range(1, len(predicted) + 1):
        for gold_position in range(1, len(gold) + 1):
            if matches(predicted_position - 1, gold_position - 1):
                lengths[predicted_position][gold_position] = (
                    lengths[predicted_position - 1][gold_position - 1] + 1
                )
            else:
                lengths[predicted_position][gold_position] = max(
                    lengths[predicted_position - 1][gold_position],
                    lengths[predicted_position][gold_position - 1],
                )

    pairs: list[tuple[int, int]] = []
    predicted_position = len(predicted)
    gold_position = len(gold)
    while predicted_position and gold_position:
        if matches(predicted_position - 1, gold_position - 1):
            pairs.append((predicted_position - 1, gold_position - 1))
            predicted_position -= 1
            gold_position -= 1
        elif (
            lengths[predicted_position - 1][gold_position]
            >= lengths[predicted_position][gold_position - 1]
        ):
            predicted_position -= 1
        else:
            gold_position -= 1
    return tuple(reversed(pairs))


def _divergent_steps(
    predicted: tuple[PrefixState, ...],
    gold: tuple[PrefixState, ...],
) -> tuple[int, int | None]:
    pairs = _exact_alignment(predicted, gold)
    exact_gold_positions = {gold_position for _, gold_position in pairs}
    gold_position = next(
        (
            position
            for position in range(len(gold))
            if position not in exact_gold_positions
        ),
        len(gold) - 1,
    )
    before = [
        predicted_position
        for predicted_position, paired_gold_position in pairs
        if paired_gold_position < gold_position
    ]
    after = [
        predicted_position
        for predicted_position, paired_gold_position in pairs
        if paired_gold_position > gold_position
    ]
    lower = max(before, default=-1) + 1
    upper = min(after, default=len(predicted))
    predicted_position = lower if lower < upper else None
    return (
        gold[gold_position].step_index,
        (
            predicted[predicted_position].step_index
            if predicted_position is not None
            else None
        ),
    )


def _operation_payload(script: GraphScript, step_index: int) -> dict[str, object]:
    return cast(
        dict[str, object],
        script.ops[step_index].model_dump(mode="json", by_alias=True),
    )


def _input_handles(operation: dict[str, object]) -> tuple[str, ...]:
    if "in" in operation:
        return (str(operation["in"]),)
    if "inputs" in operation:
        values = operation["inputs"]
        assert isinstance(values, list)
        return tuple(str(value) for value in values)
    if "left" in operation:
        return (str(operation["left"]), str(operation["right"]))
    if "subject" in operation:
        return (str(operation["subject"]), str(operation["object"]))
    return ()


def _handle_states(
    script: GraphScript,
    states: tuple[PrefixState, ...],
) -> dict[str, PrefixState]:
    result: dict[str, PrefixState] = {}
    for state in states:
        output = _operation_payload(script, state.step_index).get("out")
        if output is not None:
            result[str(output)] = state
    return result


def _input_handle_mapping(
    predicted: GraphScript,
    gold: GraphScript,
    predicted_states: tuple[PrefixState, ...],
    gold_states: tuple[PrefixState, ...],
    *,
    predicted_step_index: int,
    gold_step_index: int,
) -> tuple[tuple[str, str], ...] | None:
    predicted_inputs = _input_handles(
        _operation_payload(predicted, predicted_step_index)
    )
    gold_inputs = _input_handles(_operation_payload(gold, gold_step_index))
    if not predicted_inputs and not gold_inputs:
        return ()
    if len(predicted_inputs) != len(gold_inputs):
        return None
    predicted_by_handle = _handle_states(predicted, predicted_states)
    gold_by_handle = _handle_states(gold, gold_states)
    predicted_unique = tuple(dict.fromkeys(predicted_inputs))
    gold_unique = tuple(dict.fromkeys(gold_inputs))
    if len(predicted_unique) != len(gold_unique):
        return None

    def state_key(state: PrefixState) -> tuple[str, tuple[str, ...]]:
        signature = (
            ("EMPTY_RESULT",)
            if state.signature and state.signature[0] == "EMPTY_RESULT"
            else state.signature
        )
        return state.output_kind, signature

    candidates: dict[str, tuple[str, ...]] = {}
    for predicted_handle in predicted_unique:
        predicted_state = predicted_by_handle.get(predicted_handle)
        if predicted_state is None or not predicted_state.alignable:
            return None
        matches = tuple(
            gold_handle
            for gold_handle in gold_unique
            if (
                gold_by_handle.get(gold_handle) is not None
                and gold_by_handle[gold_handle].alignable
                and state_key(gold_by_handle[gold_handle]) == state_key(predicted_state)
            )
        )
        if not matches:
            return None
        if state_key(predicted_state)[1] == ("EMPTY_RESULT",) and len(matches) != 1:
            return None
        candidates[predicted_handle] = matches

    matchings: list[tuple[tuple[str, str], ...]] = []

    def search(
        position: int,
        used_gold: frozenset[str],
        current: tuple[tuple[str, str], ...],
    ) -> None:
        if position == len(predicted_unique):
            matchings.append(current)
            return
        predicted_handle = predicted_unique[position]
        for gold_handle in candidates[predicted_handle]:
            if gold_handle not in used_gold:
                search(
                    position + 1,
                    used_gold | {gold_handle},
                    (*current, (predicted_handle, gold_handle)),
                )

    search(0, frozenset(), ())
    if not matchings:
        return None
    predicted_op = str(
        _operation_payload(predicted, predicted_step_index).get("op")
    )
    if len(matchings) > 1 and predicted_op not in {"intersect", "union"}:
        return None
    return min(matchings)


def locate_execution_boundary(
    record: dict[str, object],
    *,
    backend: GraphBackend,
    allowed_relations: frozenset[str],
    config: GRRConfig,
) -> BoundaryLocation | None:
    try:
        raw_predicted = json.loads(str(record["rejected_response"]))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw_predicted, dict) or not isinstance(raw_predicted.get("ops"), list):
        return None
    if not raw_predicted["ops"] or not all(
        isinstance(operation, dict) for operation in raw_predicted["ops"]
    ):
        return None

    gold = parse_graphscript(str(record["chosen_graphscript"]))
    gold_prefixes, gold_failure, _ = _execute_prefixes(
        gold,
        backend=backend,
        allowed_relations=allowed_relations,
        config=config,
        trace_id=f"boundary:gold:{record['task_id']}",
    )
    assert gold_failure is None

    try:
        predicted = parse_graphscript(
            raw_predicted,
            max_follow_limit=config.max_follow_limit,
        )
    except GraphScriptError:
        return None

    predicted_prefixes, predicted_failure, execution_reason = _execute_prefixes(
        predicted,
        backend=backend,
        allowed_relations=allowed_relations,
        config=config,
        trace_id=f"boundary:predicted:{record['task_id']}",
    )
    gold_step, predicted_divergence = _divergent_steps(
        predicted_prefixes,
        gold_prefixes,
    )
    if predicted_failure is not None:
        predicted_step = (
            predicted_divergence
            if predicted_divergence is not None
            else predicted_failure
        )
        return BoundaryLocation(
            gold_step_index=gold_step,
            predicted_step_index=predicted_step,
            stage="execution",
            reason_code=execution_reason or str(record["reason_code"]),
            input_handle_mapping=_input_handle_mapping(
                predicted,
                gold,
                predicted_prefixes,
                gold_prefixes,
                predicted_step_index=predicted_step,
                gold_step_index=gold_step,
            ),
        )

    stage = (
        "execution"
        if str(record["failure_stage"]) == "predicted_execution"
        else "denotation"
    )
    predicted_step = (
        predicted_divergence
        if predicted_divergence is not None
        else min(gold_step, len(predicted.ops) - 1)
    )
    return BoundaryLocation(
        gold_step_index=gold_step,
        predicted_step_index=predicted_step,
        stage=stage,
        reason_code=str(record["reason_code"]),
        input_handle_mapping=_input_handle_mapping(
            predicted,
            gold,
            predicted_prefixes,
            gold_prefixes,
            predicted_step_index=predicted_step,
            gold_step_index=gold_step,
        ),
    )


_OPERATION_CONTRACTS: dict[str, tuple[tuple[str, ...], str]] = {
    "all_entities": ((), "entity"),
    "resolve_entity": ((), "entity"),
    "follow": (("entity",), "entity"),
    "filter_type": (("entity",), "entity"),
    "filter_literal": (("entity",), "entity"),
    "filter_qualifier": (("entity",), "entity"),
    "select_among": (("entity",), "entity"),
    "intersect": (("entity", "entity"), "entity"),
    "union": (("entity", "entity"), "entity"),
    "select_between": (("entity", "entity"), "entity"),
    "count": (("entity",), "answer"),
    "query_attribute": (("entity",), "answer"),
    "query_attribute_under_condition": (("entity",), "answer"),
    "query_attribute_qualifier": (("entity",), "answer"),
    "query_relation": (("entity", "entity"), "answer"),
    "query_relation_qualifier": (("entity", "entity"), "answer"),
    "verify": (("answer",), "answer"),
}


def _with_input_handles(
    operation: dict[str, object],
    handles: tuple[str, ...],
) -> dict[str, object]:
    grafted = deepcopy(operation)
    if "in" in grafted:
        grafted["in"] = handles[0]
    elif "inputs" in grafted:
        grafted["inputs"] = list(handles)
    elif "left" in grafted:
        grafted["left"], grafted["right"] = handles
    elif "subject" in grafted:
        grafted["subject"], grafted["object"] = handles
    return grafted


def _input_kinds(operation_name: str, handles: tuple[str, ...]) -> tuple[str, ...] | None:
    contract = _OPERATION_CONTRACTS.get(operation_name)
    if contract is None:
        return None
    if operation_name in {"intersect", "union"}:
        return ("entity",) * len(handles)
    return contract[0]


def _graft_operation(
    current_policy_operation: dict[str, object],
    gold_operation: dict[str, object],
    input_handle_mapping: tuple[tuple[str, str], ...],
) -> dict[str, object] | None:
    predicted_op = str(current_policy_operation.get("op"))
    gold_op = str(gold_operation.get("op"))
    predicted_handles = _input_handles(current_policy_operation)
    gold_handles = _input_handles(gold_operation)
    predicted_input_kinds = _input_kinds(predicted_op, predicted_handles)
    gold_input_kinds = _input_kinds(gold_op, gold_handles)
    if predicted_input_kinds is None or predicted_input_kinds != gold_input_kinds:
        return None
    mapping = dict(input_handle_mapping)
    if any(
        handle not in mapping for handle in predicted_handles
    ):
        return None
    grafted = _with_input_handles(
        current_policy_operation,
        tuple(mapping[handle] for handle in predicted_handles),
    )
    grafted["out"] = gold_operation["out"]
    return grafted if grafted != gold_operation else None


def _certify_graft(
    record: dict[str, object],
    location: BoundaryLocation,
    *,
    backend: GraphBackend,
    allowed_relations: frozenset[str],
    config: GRRConfig,
) -> tuple[CertifiedGraft | None, str]:
    gold = parse_graphscript(str(record["chosen_graphscript"]))
    predicted = parse_graphscript(
        str(record["rejected_response"]),
        max_follow_limit=config.max_follow_limit,
    )
    gold_payload = cast(
        dict[str, object],
        gold.model_dump(mode="json", by_alias=True),
    )
    predicted_payload = cast(
        dict[str, object],
        predicted.model_dump(mode="json", by_alias=True),
    )
    gold_operations = cast(list[dict[str, object]], gold_payload["ops"])
    predicted_operations = cast(list[dict[str, object]], predicted_payload["ops"])
    gold_operation = gold_operations[location.gold_step_index]
    current_policy_operation = predicted_operations[location.predicted_step_index]
    if location.input_handle_mapping is None:
        return None, "UNALIGNED_INPUT_STATE"
    grafted_operation = _graft_operation(
        current_policy_operation,
        gold_operation,
        location.input_handle_mapping,
    )
    if grafted_operation is None:
        return None, "INCOMPATIBLE_OPERATION_BOUNDARY"

    rejected_payload = deepcopy(gold_payload)
    rejected_operations = cast(list[dict[str, object]], rejected_payload["ops"])
    rejected_operations[location.gold_step_index] = grafted_operation
    try:
        rejected = parse_graphscript(
            rejected_payload,
            max_follow_limit=config.max_follow_limit,
        )
    except GraphScriptError:
        return None, "GRAFT_INVALID_GRAPHSCRIPT"

    chosen_execution = execute_graphscript(
        gold,
        backend,
        allowed_relations=allowed_relations,
        max_edge_visits=config.max_edge_visits,
        max_returned_entities=config.max_returned_entities,
        trace_id=f"boundary:graft:chosen:{record['task_id']}",
    )
    chosen_answers = _answer_signature(chosen_execution.answers)
    recorded_answers = tuple(
        sorted(str(value) for value in cast(list[object], record["chosen_answers"]))
    )
    assert chosen_answers == recorded_answers
    try:
        rejected_execution = execute_graphscript(
            rejected,
            backend,
            allowed_relations=allowed_relations,
            max_edge_visits=config.max_edge_visits,
            max_returned_entities=config.max_returned_entities,
            trace_id=f"boundary:graft:rejected:{record['task_id']}",
        )
    except GraphScriptError as exc:
        return (
            CertifiedGraft(
                rejected_graphscript=rejected_payload,
                current_policy_operation=current_policy_operation,
                grafted_operation=grafted_operation,
                rejected_stage="execution",
                reason_code=exc.reason_code,
                answers=(),
            ),
            "CERTIFIED",
        )
    rejected_answers = _answer_signature(rejected_execution.answers)
    if rejected_answers == chosen_answers:
        return None, "GRAFT_TERMINAL_EQUIVALENT"
    return (
        CertifiedGraft(
            rejected_graphscript=rejected_payload,
            current_policy_operation=current_policy_operation,
            grafted_operation=grafted_operation,
            rejected_stage="terminal_semantic",
            reason_code="TOOL_WRONG_ANSWER",
            answers=rejected_answers,
        ),
        "CERTIFIED",
    )


def _marked_graphscript(raw_script: dict[str, object], step_index: int) -> str:
    operations = raw_script["ops"]
    assert isinstance(operations, list)
    serialized = [
        json.dumps(operation, ensure_ascii=False, separators=(",", ":"))
        for operation in operations
    ]
    serialized[step_index] = (
        f"{BOUNDARY_START}{serialized[step_index]}{BOUNDARY_END}"
    )
    version = json.dumps(raw_script.get("version", "0.3"), ensure_ascii=False)
    return f'{{"version":{version},"ops":[{",".join(serialized)}]}}'


def _boundary_family(record: dict[str, object], location: BoundaryLocation) -> str:
    gold = parse_graphscript(str(record["chosen_graphscript"]))
    operation = gold.ops[location.gold_step_index].op
    if operation in _ROOT_OPERATIONS:
        return "entity_root"
    if operation in _ENTITY_OPERATIONS:
        return "entity_transform"
    return "answer_projection"


def _select_family_balanced_pairs(
    eligible: list[tuple[dict[str, object], BoundaryLocation, CertifiedGraft]],
    *,
    pair_count: int,
    max_per_family: int | None,
) -> list[tuple[dict[str, object], BoundaryLocation, CertifiedGraft]]:
    selected: list[tuple[dict[str, object], BoundaryLocation, CertifiedGraft]] = []
    family_counts: Counter[str] = Counter()
    for candidate in eligible:
        family = _boundary_family(candidate[0], candidate[1])
        if max_per_family is not None and family_counts[family] >= max_per_family:
            continue
        selected.append(candidate)
        family_counts[family] += 1
        if len(selected) == pair_count:
            break
    return selected


def build_boundary_pairs(
    config: GRRConfig,
    *,
    backend: GraphBackend | None = None,
) -> dict[str, object]:
    effective_backend = backend or SQLiteGraphBackend(
        config.graph_db,
        snapshot_id="kqapro-v1",
    )
    relations = load_relation_catalog(config.relation_catalog)
    allowed_relations = frozenset(relation.relation_id for relation in relations)
    rollout_manifest = json.loads(config.rollout_manifest_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in config.rollouts_path.read_text(encoding="utf-8").splitlines()
    ]
    tasks = {task.task_id: task for task in _load_probe_tasks(config)}
    target_contract_rejection_counts: Counter[str] = Counter()
    failures: list[dict[str, object]] = []
    for record in sorted(records, key=lambda value: int(value["task_rank"])):
        if record["reason_code"] == "TERMINAL_CORRECT":
            continue
        task = tasks[str(record["task_id"])]
        certified, reason = _certify_task(
            task,
            rank=int(record["task_rank"]),
            backend=effective_backend,
            relations=relations,
            config=config,
        )
        if certified is None:
            target_contract_rejection_counts[reason] += 1
            continue
        recertified = dict(record)
        recertified["chosen_graphscript"] = certified.chosen_graphscript
        recertified["chosen_program_signature"] = certified.chosen_program_signature
        recertified["chosen_answers"] = list(certified.chosen_answers)
        failures.append(recertified)

    eligible: list[tuple[dict[str, object], BoundaryLocation, CertifiedGraft]] = []
    localization_rejection_counts: Counter[str] = Counter()
    graft_rejection_counts: Counter[str] = Counter()
    for record in failures:
        location = locate_execution_boundary(
            record,
            backend=effective_backend,
            allowed_relations=allowed_relations,
            config=config,
        )
        if location is None:
            localization_rejection_counts[str(record["reason_code"])] += 1
            continue
        graft, reason = _certify_graft(
            record,
            location,
            backend=effective_backend,
            allowed_relations=allowed_relations,
            config=config,
        )
        if graft is None:
            graft_rejection_counts[reason] += 1
            continue
        eligible.append((record, location, graft))
    selected = _select_family_balanced_pairs(
        eligible,
        pair_count=config.pair_count,
        max_per_family=config.max_pairs_per_boundary_family,
    )

    output_dir = config.output_dir / "boundary-pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    control_path = output_dir / "train_control.jsonl"
    certificate_path = output_dir / "boundary_certificates.jsonl"
    stage_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    boundary_family_counts: Counter[str] = Counter()
    with (
        train_path.open("w", encoding="utf-8") as train_stream,
        control_path.open("w", encoding="utf-8") as control_stream,
        certificate_path.open("w", encoding="utf-8") as certificate_stream,
    ):
        for record, location, graft in selected:
            raw_gold = cast(
                dict[str, object],
                parse_graphscript(str(record["chosen_graphscript"])).model_dump(
                    mode="json",
                    by_alias=True,
                ),
            )
            chosen = _marked_graphscript(raw_gold, location.gold_step_index)
            rejected = _marked_graphscript(
                graft.rejected_graphscript,
                location.gold_step_index,
            )
            messages: list[dict[str, object]] = [
                dict(message)
                for message in role_prompt(
                    "solver",
                    f"Question: {record['question']}",
                    interaction_mode="graphscript",
                    relation_catalog=relations,
                    graphscript_version="0.3",
                )
            ]
            messages.append(
                {"role": "assistant", "content": chosen, "loss": 1.0}
            )
            pair: dict[str, object] = {
                "messages": messages,
                "rejected_response": rejected,
                "task_id": str(record["task_id"]),
                "boundary_family": _boundary_family(record, location),
                "boundary_stage": location.stage,
                "boundary_reason_code": location.reason_code,
            }
            control_messages = deepcopy(messages)
            control_messages[-1]["content"] = chosen.replace(
                BOUNDARY_START, ""
            ).replace(BOUNDARY_END, "")
            control_pair: dict[str, object] = {
                **pair,
                "messages": control_messages,
                "rejected_response": rejected.replace(
                    BOUNDARY_START, ""
                ).replace(BOUNDARY_END, ""),
            }
            gold_operations = cast(list[dict[str, object]], raw_gold["ops"])
            certificate = {
                "task_id": str(record["task_id"]),
                "task_rank": int(str(record["task_rank"])),
                "sample_seed": int(str(record["sample_seed"])),
                "stage": location.stage,
                "reason_code": location.reason_code,
                "gold_step_index": location.gold_step_index,
                "current_policy_step_index": location.predicted_step_index,
                "grafted_step_index": location.gold_step_index,
                "boundary_family": _boundary_family(record, location),
                "gold_operation": str(
                    gold_operations[location.gold_step_index]["op"]
                ),
                "grafted_operation": str(graft.grafted_operation["op"]),
                "gold_operation_sha256": stable_hash(
                    gold_operations[location.gold_step_index]
                ),
                "current_policy_operation_sha256": stable_hash(
                    graft.current_policy_operation
                ),
                "grafted_operation_sha256": stable_hash(graft.grafted_operation),
                "rejected_stage": graft.rejected_stage,
                "rejected_reason_code": graft.reason_code,
                "rejected_answers": list(graft.answers),
                "same_task": True,
                "current_policy_negative": True,
                "gold_graph_certified": True,
                "single_operation_graft": True,
                "common_prefix": True,
                "common_suffix": True,
                "chosen_loss_scope": "full_response_with_eos",
                "rejected_loss_scope": (
                    "natural_tokens_overlapping_earliest_divergent_operation"
                ),
                "natural_response_tokenization_preserved": True,
            }
            train_stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
            control_stream.write(json.dumps(control_pair, ensure_ascii=False) + "\n")
            certificate_stream.write(
                json.dumps(certificate, ensure_ascii=False, sort_keys=True) + "\n"
            )
            stage_counts[location.stage] += 1
            reason_counts[location.reason_code] += 1
            boundary_family_counts[_boundary_family(record, location)] += 1

    manifest: dict[str, object] = {
        "algorithm": "executor_certified_denotation_boundary_po_v2_candidate_set",
        "generation_mode": (
            "current_policy_same_task_denotation_aligned_single_operation_graft"
        ),
        "seed": config.seed,
        "requested_pairs": config.pair_count,
        "eligible_pairs": len(eligible),
        "selected_pairs": len(selected),
        "selected_task_ids": [
            str(record["task_id"]) for record, _, _ in selected
        ],
        "stage_counts": dict(sorted(stage_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "boundary_family_counts": dict(sorted(boundary_family_counts.items())),
        "max_pairs_per_boundary_family": config.max_pairs_per_boundary_family,
        "unlocalizable_reason_counts": dict(
            sorted(localization_rejection_counts.items())
        ),
        "graft_rejection_counts": dict(sorted(graft_rejection_counts.items())),
        "target_contract": TARGET_CONTRACT,
        "target_contract_rejection_counts": dict(
            sorted(target_contract_rejection_counts.items())
        ),
        "rollout_manifest": str(config.rollout_manifest_path),
        "rollout_manifest_sha256": file_hash(config.rollout_manifest_path),
        "rollouts_sha256": file_hash(config.rollouts_path),
        "model_artifact_sha256": rollout_manifest["model_artifact_sha256"],
        "chosen_loss_scope": "full_response_with_eos",
        "rejected_loss_scope": (
            "natural_tokens_overlapping_earliest_denotation_divergent_operation"
        ),
        "natural_response_tokenization_preserved": True,
        "train": str(train_path),
        "train_sha256": file_hash(train_path),
        "control_train": str(control_path),
        "control_train_sha256": file_hash(control_path),
        "certificates": str(certificate_path),
        "certificates_sha256": file_hash(certificate_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
