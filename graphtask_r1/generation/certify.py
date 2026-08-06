from __future__ import annotations

from graphtask_r1.dsl import canonical_signature, compile_sparql, operator_tags, program_cost
from graphtask_r1.generation.verbalizer import verbalize
from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    AllEntities,
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    Program,
    TaskCertificate,
    TaskProposal,
    TaskProvenance,
    Union,
    VerificationSummary,
)
from graphtask_r1.utils import stable_hash
from graphtask_r1.verification import verify_task


def certify_proposal(
    proposal: TaskProposal,
    backend: GraphBackend,
    *,
    graph_snapshot: str,
    round_index: int | None = None,
) -> TaskCertificate:
    validate_proposal(proposal)
    question = verbalize(proposal.program, backend)
    verification = verify_task(question, proposal.program, backend)
    if not verification.passed:
        raise ValueError("proposal rejected: " + ",".join(verification.rejection_reasons))
    answers = backend.execute_program(proposal.program)
    signature = canonical_signature(proposal.program)
    task_id = "gt_selfplay_" + stable_hash([graph_snapshot, signature, question, round_index])[:20]
    witnesses = backend.extract_witness(proposal.program, answers)
    return TaskCertificate(
        task_id=task_id,
        source="selfplay",
        source_id=task_id,
        split="train",
        graph_snapshot=graph_snapshot,
        question=question,
        topic_entities=tuple(backend.entity_info(value) for value in proposal.topic_entities),
        program=proposal.program,
        sparql=compile_sparql(proposal.program),
        gold_answers=answers,
        witness_facts=tuple(
            sorted(
                {fact for witness in witnesses for fact in witness.facts},
                key=lambda fact: fact.sort_key(),
            )
        ),
        program_signature=signature,
        program_cost=program_cost(proposal.program),
        operator_tags=operator_tags(proposal.program),
        verification=VerificationSummary(
            executable=True,
            semantic_equivalent=True,
            necessity_mean=verification.necessity_mean,
            necessity_min=verification.necessity_min,
            shortcut_found=verification.shortcut_found,
            answer_leak=verification.answer_leak,
        ),
        provenance=TaskProvenance(dataset="selfplay", converter_version="proposal-v1"),
        generation={
            "graph_snapshot": graph_snapshot,
            "round": round_index,
            "proposed_paraphrase": proposal.paraphrase,
        },
    )


def validate_proposal(proposal: TaskProposal) -> None:
    expected_topics = _topic_ids(proposal.program)
    if not expected_topics:
        raise ValueError("Questioner proposals must be rooted in explicit seed entities")
    if tuple(sorted(proposal.topic_entities)) != expected_topics:
        raise ValueError("proposal topic_entities do not match program entity roots")


def _topic_ids(program: Program) -> tuple[str, ...]:
    if isinstance(program, Entity):
        return (program.entity_id,)
    if isinstance(program, AllEntities):
        return ()
    if isinstance(program, Intersect | Union):
        return tuple(sorted({value for branch in program.inputs for value in _topic_ids(branch)}))
    if isinstance(program, Hop | FilterType | FilterLiteral | Count):
        return _topic_ids(program.input)
    raise TypeError(type(program).__name__)
