from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TypedDict

from graphtask_r1.graph import GraphBackend


class SeedEntityContext(TypedDict):
    entity_id: str
    label: str
    type_ids: list[str]
    outgoing_relation_ids: list[str]
    incoming_relation_ids: list[str]


def build_questioner_seed_context(
    backend: GraphBackend,
    entity_ids: Sequence[str],
    *,
    allowed_relations: frozenset[str],
    max_neighbor_facts: int = 200,
    max_relation_ids: int = 64,
) -> list[SeedEntityContext]:
    """Build bounded, answer-free seed metadata for a Questioner prompt."""
    if max_neighbor_facts < 1:
        raise ValueError("max_neighbor_facts must be positive")
    if max_relation_ids < 1:
        raise ValueError("max_relation_ids must be positive")
    contexts: list[SeedEntityContext] = []
    for entity_id in entity_ids:
        info = backend.entity_info(entity_id)
        facts = backend.neighbors(
            [entity_id], direction="both", limit=max_neighbor_facts
        )
        outgoing = sorted(
            {
                fact.relation
                for fact in facts
                if fact.subject == entity_id and fact.relation in allowed_relations
            }
        )[:max_relation_ids]
        incoming = sorted(
            {
                fact.relation
                for fact in facts
                if fact.object == entity_id and fact.relation in allowed_relations
            }
        )[:max_relation_ids]
        contexts.append(
            {
                "entity_id": entity_id,
                "label": info.label,
                "type_ids": list(info.type_ids),
                "outgoing_relation_ids": outgoing,
                "incoming_relation_ids": incoming,
            }
        )
    return contexts


def render_questioner_seed_payload(
    contexts: Sequence[Mapping[str, object]],
) -> str:
    if not contexts:
        raise ValueError("Questioner seed context cannot be empty")
    lines = [
        "Construct one certified task rooted exactly in the seed entities below.",
        "Seed metadata exposes only labels, types, and bounded relation IDs; "
        "it does not expose answers.",
    ]
    for index, context in enumerate(contexts):
        entity_id = str(context.get("entity_id", "")).strip()
        if not entity_id:
            raise ValueError("Questioner seed context requires entity_id")
        label = str(context.get("label", entity_id))
        type_ids = _string_values(context.get("type_ids"))
        outgoing = _string_values(context.get("outgoing_relation_ids"))
        incoming = _string_values(context.get("incoming_relation_ids"))
        lines.extend(
            [
                f"Seed {index + 1}:",
                f"- entity_id: {entity_id}",
                f"- label: {label}",
                f"- type_ids: {_display(type_ids)}",
                f"- observed_outgoing_relation_ids: {_display(outgoing)}",
                f"- observed_incoming_relation_ids: {_display(incoming)}",
                "- required_root_op: "
                + '{"op":"resolve_entity","query":'
                + _json_string(entity_id)
                + f',"match":"id","limit":1,"out":"h{index}"}}',
            ]
        )
    return "\n".join(lines)


def _string_values(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [str(item) for item in value]


def _display(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "(none observed within bound)"


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
