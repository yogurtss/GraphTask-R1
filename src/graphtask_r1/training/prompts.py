from __future__ import annotations

from typing import Literal

from graphtask_r1.schema import RelationInfo

InteractionMode = Literal["tool", "graphscript"]


QUESTIONER_SYSTEM_PROMPT = """You are the Questioner in graph self-play. Explore only with the
provided privileged graph tools. Construct a typed executable program whose answer is non-empty.
The canonical question is rendered externally from the verified program. Return exactly:
<task>{\"topic_entities\": [\"...\"], \"program\": {...}, \"paraphrase\": null}</task>
Never use an all_entities root and never invent or include gold answers."""

SOLVER_SYSTEM_PROMPT = """You are the Solver in graph self-play. You cannot see the gold program
or answer. Use only search and inspect_entity results, then return exactly one JSON list inside
<answer>...</answer>. Do not answer from parametric memory when graph evidence is unavailable."""


QUESTIONER_GRAPHSCRIPT_PROMPT = """You are the Questioner in graph self-play. Produce one bounded
GraphScript v0.1 program rooted at $seed. Output exactly one JSON object and no prose or markdown.
The only valid shape is start -> follow -> follow -> require_unique -> emit. Use only relation IDs
from the provided catalog, use handles h0/h1/h2, and never include or guess the gold answer."""


SOLVER_GRAPHSCRIPT_PROMPT = """You are the Solver in graph self-play. Infer a bounded two-hop
GraphScript v0.1 program from the question and topic seed. Output exactly one JSON object and no
prose or markdown. The only valid shape is start -> follow -> follow -> require_unique -> emit.
Use only relation IDs from the provided catalog; program execution, not parametric recall, supplies
the answer."""


def relation_catalog_text(relations: tuple[RelationInfo, ...]) -> str:
    if not relations:
        return ""
    lines = [f"- {relation.relation_id}: {relation.label}" for relation in relations]
    return "\nAllowed relation catalog:\n" + "\n".join(lines)


def role_prompt(
    role: str,
    payload: str,
    *,
    interaction_mode: InteractionMode = "tool",
    relation_catalog: tuple[RelationInfo, ...] = (),
) -> list[dict[str, str]]:
    if role == "questioner":
        system = (
            QUESTIONER_SYSTEM_PROMPT
            if interaction_mode == "tool"
            else QUESTIONER_GRAPHSCRIPT_PROMPT
        )
    elif role == "solver":
        system = (
            SOLVER_SYSTEM_PROMPT if interaction_mode == "tool" else SOLVER_GRAPHSCRIPT_PROMPT
        )
    else:
        raise ValueError(f"unknown role: {role}")
    content = payload + relation_catalog_text(relation_catalog)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]
