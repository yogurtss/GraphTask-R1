from __future__ import annotations

from typing import Literal

from graphtask_r1.schema import RelationInfo

InteractionMode = Literal["tool", "graphscript"]
GraphScriptVersion = Literal["0.1", "0.2"]


QUESTIONER_SYSTEM_PROMPT = """You are the Questioner in graph self-play. Explore only with the
provided privileged graph tools. Construct a typed executable program whose answer is non-empty.
The canonical question is rendered externally from the verified program. Return exactly:
<task>{\"topic_entities\": [\"...\"], \"program\": {...}, \"paraphrase\": null}</task>
Never use an all_entities root and never invent or include gold answers."""

SOLVER_SYSTEM_PROMPT = """You are the Solver in graph self-play. You cannot see the gold program
or answer. Use only graph_search, text_search, and inspect_entity results that are available in the
current episode, then return exactly one JSON list inside <answer>...</answer>. Do not answer from
parametric memory when retrieved evidence is unavailable."""


QUESTIONER_GRAPHSCRIPT_PROMPT = """You are the Questioner in graph self-play. Produce one bounded
GraphScript v0.1 program rooted at $seed. Output exactly one JSON object and no prose or markdown.
The only valid shape is start -> follow -> follow -> require_unique -> emit. Use only relation IDs
from the provided catalog, use handles h0/h1/h2, and never include or guess the gold answer."""


SOLVER_GRAPHSCRIPT_PROMPT = """You are the Solver in graph self-play. Infer a bounded two-hop
GraphScript v0.1 program from the question and topic seed. Output exactly one JSON object and no
prose or markdown. The only valid shape is start -> follow -> follow -> require_unique -> emit.
Use only relation IDs from the provided catalog; program execution, not parametric recall, supplies
the answer."""


QUESTIONER_GRAPHSCRIPT_V02_PROMPT = """You are the Questioner in graph self-play. Produce one
bounded, typed GraphScript v0.2 JSON program. The program may resolve entities, search passages,
traverse allowed relations, combine or filter entity handles, query attributes or relations,
select extrema, count, and emit an answer. Output exactly one JSON object with no prose or markdown.
Never embed or emit a guessed gold answer; program execution must supply the answer."""


SOLVER_GRAPHSCRIPT_V02_PROMPT = """You are the Solver in graph self-play. Compile the natural
language question into one bounded, typed GraphScript v0.2 JSON program. No topic seed is
guaranteed: use resolve_entity or search_passage when needed, convert passage results with
passage_pages, and use only relations in the provided catalog. Output exactly one JSON object with
no prose or markdown.
The executor, not a free-form answer or parametric recall, must produce the answer."""


GRAPHSCRIPT_V02_GRAMMAR = """
Use at most 64 operations and SSA handles h0..h63. Valid signatures are:
all_entities(max_results, out) [must be immediately restricted before materialization];
resolve_entity(query, match=id|exact|search, limit, out); search_passage(query, limit, max_chars,
out); passage_pages(in, out); start(entity="$seed", out); follow(in, relation, direction=out|in,
limit, out); intersect(inputs, out); union(inputs, out); filter_type(in, type_id, out);
filter_literal(in, relation, comparator, value={value,datatype,unit}, out); count(in, out);
query_attribute(in, attribute, out); query_relation(subject, object, out);
select_between(left, right, attribute, mode=min|max, out); select_among(in, attribute,
mode=min|max, out); require_unique(in); emit(in). The first operation must be start,
all_entities, resolve_entity, or search_passage, and the last must be emit."""


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
    graphscript_version: GraphScriptVersion = "0.1",
) -> list[dict[str, str]]:
    if role == "questioner":
        if interaction_mode == "tool":
            system = QUESTIONER_SYSTEM_PROMPT
        else:
            system = (
                QUESTIONER_GRAPHSCRIPT_PROMPT
                if graphscript_version == "0.1"
                else QUESTIONER_GRAPHSCRIPT_V02_PROMPT
            )
    elif role == "solver":
        if interaction_mode == "tool":
            system = SOLVER_SYSTEM_PROMPT
        else:
            system = (
                SOLVER_GRAPHSCRIPT_PROMPT
                if graphscript_version == "0.1"
                else SOLVER_GRAPHSCRIPT_V02_PROMPT
            )
    else:
        raise ValueError(f"unknown role: {role}")
    if interaction_mode == "graphscript" and graphscript_version == "0.2":
        system += GRAPHSCRIPT_V02_GRAMMAR
    content = payload + relation_catalog_text(relation_catalog)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]
