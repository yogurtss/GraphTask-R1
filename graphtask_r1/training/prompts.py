from __future__ import annotations

from typing import Literal

from graphtask_r1.schema import RelationInfo

InteractionMode = Literal["tool", "graphscript"]
GraphScriptVersion = Literal["0.1", "0.2", "0.3"]
QuestionerContract = Literal["program", "question_program"]


QUESTIONER_SYSTEM_PROMPT = """You are the Questioner in graph self-play. Explore only with the
provided privileged graph tools. Construct a typed executable program whose answer is non-empty.
The canonical question is rendered externally from the verified program. Return exactly:
<task>{\"topic_entities\": [\"...\"], \"program\": {...}, \"paraphrase\": null}</task>
Never use an all_entities root and never invent or include gold answers."""


QUESTIONER_QUESTION_PROGRAM_TOOL_PROMPT = """You are the Questioner in graph self-play. Construct
both one natural-language question and one typed executable program whose answer is non-empty.
Return exactly one object and no other text:
<task>{"question":"...","topic_entities":["..."],"program":{...}}</task>
The question must describe the program and must not contain or reveal its executed answer. Never use
an all_entities root; program execution, not a guessed answer, supplies gold."""

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
the answer. Follow this exact JSON schema, replacing RELATION_ID_1 and RELATION_ID_2 with catalog
IDs:
{"version":"0.1","ops":[{"op":"start","entity":"$seed","out":"h0"},{"op":"follow","in":"h0","relation":"RELATION_ID_1","direction":"out","limit":8,"out":"h1"},{"op":"follow","in":"h1","relation":"RELATION_ID_2","direction":"out","limit":8,"out":"h2"},{"op":"require_unique","in":"h2"},{"op":"emit","in":"h2"}]}"""


QUESTIONER_GRAPHSCRIPT_V02_PROMPT = """You are the Questioner in graph self-play. Produce one
bounded, typed KILT GraphScript v0.2 JSON program. It may resolve entities, search passages,
traverse allowed Wikipedia relations, combine handles, and emit an answer. Output exactly one JSON
object with no prose or markdown. Never embed a guessed gold answer; execution supplies it."""


SOLVER_GRAPHSCRIPT_V02_PROMPT = """You are the Solver in graph self-play. Compile the natural
language question into one bounded, typed GraphScript v0.2 JSON program. No topic seed is
guaranteed: use resolve_entity or search_passage when needed, convert passages with passage_pages,
and use only relations in the provided KILT catalog. Output exactly one JSON object with no prose.
The executor, not a free-form answer or parametric recall, must produce the answer."""


GRAPHSCRIPT_V02_GRAMMAR = """
Use at most 64 operations and SSA handles h0..h63. Valid signatures are:
all_entities(max_results, out) [must be immediately restricted before materialization];
resolve_entity(query, match=id|exact|search, limit, out); search_passage(query, limit, max_chars,
out); passage_pages(in, out); start(entity="$seed", out); follow(in, relation, direction=out|in,
limit, out); intersect(inputs, out); union(inputs, out); filter_type(in, type_id, out);
filter_literal(in, relation, comparator, value={value,datatype,unit}, out);
count(in, out); query_attribute(in, attribute, out); query_relation(subject, object, out);
select_between(left, right, attribute, mode=min|max, out); select_among(in, attribute,
mode=min|max, out); require_unique(in); emit(in). The first operation must be start,
all_entities, resolve_entity, or search_passage, and the last must be emit."""


QUESTIONER_GRAPHSCRIPT_V03_PROMPT = """You are the Questioner in structured graph self-play.
Produce one bounded, typed GraphScript JSON program. Output exactly
{"version":"0.3","ops":[...]} with no prose, markdown, or additional top-level fields. Root the
program in every provided seed using the required resolve_entity operation with the exact entity ID
and match="id"; never use all_entities or a different root. Traverse only observed/allowed relation
IDs, include only fields listed in the operation signatures, and end with exactly one emit. Never
embed a guessed gold answer; execution supplies it."""


QUESTIONER_QUESTION_PROGRAM_PROMPT = """You are the Questioner in structured graph self-play.
Produce both one natural-language question and one bounded, typed GraphScript program that answers
that question. Output exactly one JSON object in this shape and no prose or markdown:
{"question":"...","program":{"version":"$version","ops":[...]}}
The question must not contain or reveal the executed answer. The program must use only the provided
seed entities and allowed relation IDs; program execution, not a guessed answer, supplies gold."""


QUESTIONER_QUESTION_PROGRAM_V01_GRAMMAR = """
GraphScript v0.1 must have exactly this shape: start(entity="$seed", out="h0"), then two
follow operations producing h1 and h2, then require_unique(in="h2"), then emit(in="h2"). Use
only h0/h1/h2 and relation IDs from the provided catalog. Follow this exact JSON schema, replacing
the question and both RELATION_ID placeholders:
{"question":"QUESTION","program":{"version":"0.1","ops":[{"op":"start","entity":"$seed","out":"h0"},{"op":"follow","in":"h0","relation":"RELATION_ID_1","direction":"out","limit":8,"out":"h1"},{"op":"follow","in":"h1","relation":"RELATION_ID_2","direction":"out","limit":8,"out":"h2"},{"op":"require_unique","in":"h2"},{"op":"emit","in":"h2"}]}}"""


SOLVER_GRAPHSCRIPT_V03_PROMPT = """You are the Solver in structured graph self-play. Compile the
natural-language question into one bounded, typed GraphScript JSON program. Output exactly
{"version":"0.3","ops":[...]} with no prose, markdown, or additional top-level fields. Start with
resolve_entity or a bounded all_entities candidate set, use only allowed relation/qualifier IDs,
include only fields listed in the operation signatures, and end with exactly one emit. Program
execution, not a free-form answer, must produce the answer."""


GRAPHSCRIPT_V03_GRAMMAR = """
Operator contract: use at most 64 operations and SSA handles h0..h63. Valid signatures are:
all_entities(max_results, out) [immediately restrict before materialization];
resolve_entity(query, match=id|exact|search, limit, out); follow(in, relation, direction=out|in,
limit, out); intersect(inputs, out); union(inputs, out); filter_type(in, type_id, out);
filter_literal(in, relation, comparator, value={value,datatype,unit}, out);
filter_qualifier(in, qualifier, comparator, value={value,datatype,unit}, out); count(in, out);
query_attribute(in, attribute, out); query_attribute_under_condition(in, attribute, qualifier,
qualifier_value, out); query_attribute_qualifier(in, attribute, attribute_value, qualifier, out);
query_relation(subject, object, out); query_relation_qualifier(subject, object, relation, qualifier,
out); verify(in, comparator, value={value,datatype,unit}, out); select_between(left, right,
attribute, mode=min|max, out); select_among(in, attribute, mode=min|max, out); emit(in). The first
operation must be all_entities or resolve_entity, and the last must be emit."""


QUESTIONER_GRAPHSCRIPT_V03_GRAMMAR = """
Questioner operator contract: use at most 64 operations and SSA handles h0..h63. Valid signatures
are: resolve_entity(query, match=id, limit, out); follow(in, relation, direction=out|in, limit,
out); intersect(inputs, out); union(inputs, out); filter_type(in, type_id, out);
filter_literal(in, relation, comparator, value={value,datatype,unit}, out);
filter_qualifier(in, qualifier, comparator, value={value,datatype,unit}, out); count(in, out);
query_attribute(in, attribute, out); query_attribute_under_condition(in, attribute, qualifier,
qualifier_value, out); query_attribute_qualifier(in, attribute, attribute_value, qualifier, out);
query_relation(subject, object, out); query_relation_qualifier(subject, object, relation, qualifier,
out); verify(in, comparator, value={value,datatype,unit}, out); select_between(left, right,
attribute, mode=min|max, out); select_among(in, attribute, mode=min|max, out); emit(in). Every root
must be a required seed resolve_entity operation, and the last operation must be emit."""


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
    questioner_contract: QuestionerContract = "program",
) -> list[dict[str, str]]:
    if role == "questioner":
        if interaction_mode == "tool":
            system = (
                QUESTIONER_QUESTION_PROGRAM_TOOL_PROMPT
                if questioner_contract == "question_program"
                else QUESTIONER_SYSTEM_PROMPT
            )
        else:
            system = (
                QUESTIONER_QUESTION_PROGRAM_PROMPT.replace("$version", graphscript_version)
                if questioner_contract == "question_program"
                else {
                    "0.1": QUESTIONER_GRAPHSCRIPT_PROMPT,
                    "0.2": QUESTIONER_GRAPHSCRIPT_V02_PROMPT,
                    "0.3": QUESTIONER_GRAPHSCRIPT_V03_PROMPT,
                }[graphscript_version]
            )
    elif role == "solver":
        if interaction_mode == "tool":
            system = SOLVER_SYSTEM_PROMPT
        else:
            system = {
                "0.1": SOLVER_GRAPHSCRIPT_PROMPT,
                "0.2": SOLVER_GRAPHSCRIPT_V02_PROMPT,
                "0.3": SOLVER_GRAPHSCRIPT_V03_PROMPT,
            }[graphscript_version]
    else:
        raise ValueError(f"unknown role: {role}")
    if (
        interaction_mode == "graphscript"
        and graphscript_version == "0.1"
        and role == "questioner"
        and questioner_contract == "question_program"
    ):
        system += QUESTIONER_QUESTION_PROGRAM_V01_GRAMMAR
    if interaction_mode == "graphscript" and graphscript_version == "0.2":
        system += GRAPHSCRIPT_V02_GRAMMAR
    if interaction_mode == "graphscript" and graphscript_version == "0.3":
        system += (
            QUESTIONER_GRAPHSCRIPT_V03_GRAMMAR
            if role == "questioner"
            else GRAPHSCRIPT_V03_GRAMMAR
        )
    content = payload + relation_catalog_text(relation_catalog)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]
