from __future__ import annotations

import time

from graphtask_r1.dsl import necessity_scores
from graphtask_r1.graph import GraphBackend, InMemoryGraphBackend, SQLiteGraphBackend
from graphtask_r1.schema import Program, SelectBetween, VerifierResult
from graphtask_r1.verification.lexical import answer_leak
from graphtask_r1.verification.shortcut import bounded_shortcut_search


def verify_task(
    question: str,
    program: Program,
    backend: GraphBackend,
    *,
    min_answers: int = 1,
    max_answers: int = 20,
    necessity_min_threshold: float = 0.0,
    reject_shortcuts: bool = True,
    shortcut_budget: int = 1000,
) -> VerifierResult:
    started = time.perf_counter()
    reasons: list[str] = []
    try:
        remote_answers = backend.execute_program(program)
        working_backend = backend
        if not isinstance(backend, InMemoryGraphBackend | SQLiteGraphBackend):
            graph_slice = backend.materialize(program)
            if not graph_slice.complete:
                reasons.append("INCOMPLETE_SLICE")
            working_backend = InMemoryGraphBackend(
                graph_slice.triples, graph_slice.entities, graph_slice.relations
            )
            local_answers = working_backend.execute_program(program)
            if local_answers != remote_answers:
                reasons.append("INCOMPLETE_SLICE")
        answers = remote_answers
        executable = True
    except (TypeError, ValueError, KeyError, RuntimeError):
        answers = None
        executable = False
        reasons.append("EXECUTION_ERROR")
    execution_ms = (time.perf_counter() - started) * 1000
    if answers is None:
        return VerifierResult(
            passed=False,
            executable=False,
            answer_nonempty=False,
            cardinality_valid=False,
            type_valid=False,
            semantic_equivalent=None,
            answer_leak=False,
            shortcut_found=None,
            necessity_mean=0,
            necessity_min=0,
            novelty_structural=1,
            novelty_textual=1,
            rejection_reasons=tuple(reasons),
            component_latency_ms={"execution": execution_ms},
        )
    answer_nonempty = bool(answers.answers)
    cardinality_valid = min_answers <= len(answers.answers) <= max_answers
    if not answer_nonempty:
        reasons.append("EMPTY_ANSWER")
    if not cardinality_valid:
        reasons.append("INVALID_CARDINALITY")
    necessity_started = time.perf_counter()
    necessity_mean, necessity_min, _ = necessity_scores(program, working_backend)
    necessity_ms = (time.perf_counter() - necessity_started) * 1000
    if necessity_min < necessity_min_threshold:
        reasons.append("REDUNDANT_CONDITION")
    explicit_selection = isinstance(program, SelectBetween)
    shortcut_started = time.perf_counter()
    shortcut = bounded_shortcut_search(program, working_backend, max_candidates=shortcut_budget)
    shortcut_ms = (time.perf_counter() - shortcut_started) * 1000
    if shortcut.found is True and reject_shortcuts and not explicit_selection:
        reasons.append("SHORTCUT_FOUND")
    if shortcut.found is None and not explicit_selection:
        reasons.append("SHORTCUT_UNKNOWN")
    leaked = answer_leak(question, answers, working_backend) and not explicit_selection
    if leaked:
        reasons.append("ANSWER_LEAK")
    return VerifierResult(
        passed=not reasons,
        executable=executable,
        answer_nonempty=answer_nonempty,
        cardinality_valid=cardinality_valid,
        type_valid=True,
        semantic_equivalent=None,
        answer_leak=leaked,
        shortcut_found=shortcut.found,
        necessity_mean=necessity_mean,
        necessity_min=necessity_min,
        novelty_structural=1,
        novelty_textual=1,
        rejection_reasons=tuple(reasons),
        component_latency_ms={
            "execution": execution_ms,
            "necessity": necessity_ms,
            "shortcut": shortcut_ms,
        },
    )
