import asyncio
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import graphtask_r1.training.evidence_selfplay_runner as evidence_runner
from graphtask_r1.data import prepare_kilt
from graphtask_r1.evaluation import KILTQAMetrics, kilt_qa_metrics
from graphtask_r1.experiments import (
    AnswerPrediction,
    EvidenceABConfig,
    EvidenceAgentDecision,
    EvidenceSelfPlayConfig,
    HyperlinkEvidenceFlow,
    SingleRetrievalBaseline,
    benchmark_examples_from_records,
    certify_counterfactual_pair,
    counterfactual_retrieval_examples,
    evaluate_evidence_ab,
    evaluate_interactive_evidence,
    export_evidence_selfplay_round,
    generate_evidence_challenges,
    parse_answer_prediction,
    parse_evidence_agent_decision,
    run_evidence_selfplay_round,
    train_counterfactual_reranker,
)
from graphtask_r1.graph import SQLiteGraphBackend
from graphtask_r1.graphscript import (
    BackendEvidenceRetriever,
    CounterfactualEvidenceRetriever,
    EvidenceEpisode,
    EvidenceFlowSession,
    EvidenceTurn,
    execute_proofscript,
    execute_searchscript,
    parse_proofscript,
    parse_search_action,
    parse_searchscript,
)
from graphtask_r1.graphscript.evidence_flow import EvidenceRetriever
from graphtask_r1.rewards import (
    classify_evidence_residual,
    evidence_causal_curriculum_trace,
    evidence_ecp_answer_gated_reward,
    evidence_ecp_causal_reward,
    evidence_ecp_curriculum_reward,
    evidence_ecp_protocol_gated_reward,
    evidence_ecp_solver_reward,
    evidence_frontier_reward,
    evidence_proof_potential_trace,
    evidence_search_r1_em_reward,
    evidence_solver_reward,
)
from graphtask_r1.schema import (
    AnswerSet,
    BenchmarkExample,
    EvidenceProvenance,
    PassageHit,
)
from graphtask_r1.training.evidence_selfplay_rl import (
    build_promoted_solver_curriculum,
    compute_evidence_questioner_score,
    compute_evidence_solver_score,
    export_evidence_rl_round,
)
from graphtask_r1.training.evidence_selfplay_runner import (
    EvidenceRLTrainConfig,
    evidence_selfplay_plan,
    run_evidence_selfplay_update,
)
from graphtask_r1.training.ms_swift_reward import compute_score


def _write_kilt_fixture(path: Path) -> None:
    pages = [
        {
            "_id": "1",
            "wikipedia_id": "1",
            "wikipedia_title": "Alpha",
            "text": ["Alpha", "Alpha links to the city in the next article."],
            "anchors": [{"text": "the city", "wikipedia_id": "2"}],
            "categories": "Examples",
            "history": {"revid": 1},
            "wikidata_info": {},
        },
        {
            "_id": "2",
            "wikipedia_id": "2",
            "wikipedia_title": "Beta City",
            "text": ["Beta City", "Beta City is the expected answer."],
            "anchors": [],
            "categories": "Cities",
            "history": {"revid": 2},
            "wikidata_info": {},
        },
    ]
    path.write_text("\n".join(json.dumps(page) for page in pages) + "\n")


@pytest.fixture
def evidence_backend(tmp_path: Path) -> SQLiteGraphBackend:
    source = tmp_path / "kilt.json"
    output = tmp_path / "processed"
    _write_kilt_fixture(source)
    prepare_kilt(source, output)
    backend = SQLiteGraphBackend(output / "graph.sqlite", snapshot_id="kilt-test-v1")
    yield backend
    backend.close()


def _proof(backend: SQLiteGraphBackend):
    script = parse_proofscript(
        {
            "version": "0.4-proof",
            "ops": [
                {"op": "page", "page_id": "1", "out": "h0"},
                {"op": "paragraph", "in": "h0", "paragraph_id": 1, "out": "h1"},
                {
                    "op": "follow_anchor",
                    "in": "h0",
                    "target_page_id": "2",
                    "out": "h2",
                },
                {"op": "paragraph", "in": "h2", "paragraph_id": 1, "out": "h3"},
                {"op": "span", "in": "h3", "start": 0, "end": 9, "out": "h4"},
                {"op": "join_evidence", "inputs": ["h1", "h3"], "out": "h5"},
                {"op": "emit", "answer": "h4", "evidence": "h5"},
            ],
        }
    )
    return execute_proofscript(script, backend)


def _step(session: EvidenceFlowSession, action: dict[str, object]) -> None:
    session.step(parse_search_action({"version": "0.4", "action": action}))


def _complete_episode(backend: SQLiteGraphBackend, *, include_second_hop: bool, answer: str):
    session = EvidenceFlowSession(
        "Which city is reached from Alpha?", BackendEvidenceRetriever(backend)
    )
    _step(
        session,
        {"op": "retrieve", "query": "Alpha next article", "limit": 1, "out": "h0"},
    )
    handles = ["h0"]
    keys = ["1:1"]
    if include_second_hop:
        _step(
            session,
            {
                "op": "expand",
                "in": "h0",
                "query": "expected answer city",
                "limit": 2,
                "out": "h1",
            },
        )
        handles.append("h1")
        keys.append("2:1")
    _step(
        session,
        {
            "op": "select_evidence",
            "inputs": handles,
            "passage_keys": keys,
            "out": "h2",
        },
    )
    _step(session, {"op": "answer", "value": answer, "evidence": "h2"})
    return session.episode


def test_proofscript_and_evidence_flow_recover_two_hop_answer(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    proof = _proof(evidence_backend)
    episode = _complete_episode(evidence_backend, include_second_hop=True, answer="Beta City")

    assert proof.answers == AnswerSet.literals(["Beta City"])
    assert proof.passage_keys == ("1:1", "2:1")
    assert episode.done is True
    assert {value.passage_key for value in episode.final_evidence} == {"1:1", "2:1"}
    assert classify_evidence_residual(proof, episode, (("Beta City",),)).kind == "success"
    assert evidence_solver_reward(proof, episode, (("Beta City",),)).total == 0.95
    json.dumps(episode.model_dump(mode="json"))


def test_ecp_reward_is_joint_first_and_preserves_proof_progress(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    proof = _proof(evidence_backend)
    success = _complete_episode(evidence_backend, include_second_hop=True, answer="Beta City")
    wrong_answer = _complete_episode(evidence_backend, include_second_hop=True, answer="Gamma")
    trace = evidence_proof_potential_trace(proof, success)
    success_reward = evidence_ecp_solver_reward(proof, success, (("Beta City",),))
    failure_reward = evidence_ecp_solver_reward(proof, wrong_answer, (("Beta City",),))

    assert trace.potentials == pytest.approx((0.55, 0.94, 1.0, 1.0))
    assert trace.deltas == pytest.approx((0.55, 0.39, 0.06, 0.0))
    assert success_reward.components["joint_outcome"] == 0.7
    assert success_reward.total > 0.7
    assert failure_reward.total <= 0.3
    assert success_reward.total > failure_reward.total

    unrelated = EvidenceEpisode(
        question="unrelated",
        turns=(
            EvidenceTurn(
                index=0,
                action={"op": "retrieve"},
                passages=(
                    PassageHit(
                        page_id="9",
                        paragraph_id=0,
                        title="Noise",
                        text="Irrelevant passage",
                        score=0.0,
                    ),
                ),
            ),
        ),
    )
    assert evidence_proof_potential_trace(proof, unrelated).final == 0.0


def test_ecp_v2_gates_proof_credit_by_answer_correctness(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    proof = _proof(evidence_backend)
    success = _complete_episode(evidence_backend, include_second_hop=True, answer="Beta City")
    wrong_answer = _complete_episode(evidence_backend, include_second_hop=True, answer="Gamma")

    success_reward = evidence_ecp_answer_gated_reward(proof, success, (("Beta City",),))
    failure_reward = evidence_ecp_answer_gated_reward(
        proof, wrong_answer, (("Beta City",),)
    )

    assert success_reward.total > 0.95
    assert success_reward.metadata["answer_gate"] == 1.0
    assert failure_reward.total == 0.0
    assert failure_reward.components["answer_gated_proof_auc"] == 0.0


def test_ecp_v3_requires_exact_answer_and_complete_protocol(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    proof = _proof(evidence_backend)
    success = _complete_episode(evidence_backend, include_second_hop=True, answer="Beta City")
    partial = _complete_episode(evidence_backend, include_second_hop=True, answer="Beta")
    no_expand = _complete_episode(
        evidence_backend, include_second_hop=False, answer="Beta City"
    )

    success_reward = evidence_ecp_protocol_gated_reward(proof, success, (("Beta City",),))
    partial_reward = evidence_ecp_protocol_gated_reward(proof, partial, (("Beta City",),))
    no_expand_reward = evidence_ecp_protocol_gated_reward(
        proof, no_expand, (("Beta City",),)
    )

    assert success_reward.total > 0.95
    assert success_reward.metadata["protocol_complete"] is True
    assert partial_reward.total == 0.0
    assert no_expand_reward.total == 0.0
    assert no_expand_reward.metadata["protocol_complete"] is False


def test_ecp_v4_requires_full_proof_selection_after_expand(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    proof = _proof(evidence_backend)
    success = _complete_episode(evidence_backend, include_second_hop=True, answer="Beta City")
    reordered = success.model_copy(
        update={"turns": (success.turns[0], success.turns[2], success.turns[1], success.turns[3])}
    )
    missing_hop = _complete_episode(
        evidence_backend, include_second_hop=False, answer="Beta City"
    )

    success_reward = evidence_ecp_causal_reward(proof, success, (("Beta City",),))
    reordered_reward = evidence_ecp_causal_reward(proof, reordered, (("Beta City",),))
    missing_reward = evidence_ecp_causal_reward(proof, missing_hop, (("Beta City",),))

    assert success_reward.total > 0.99
    assert success_reward.metadata["causal_order"] is True
    assert success_reward.metadata["proof_complete"] is True
    assert reordered_reward.components["causal_proof_certificate"] == 0.0
    assert missing_reward.components["causal_proof_certificate"] == 0.0


def test_ecp_v5_transfers_grounding_curriculum_without_weakening_certificate(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    proof = _proof(evidence_backend)
    success = _complete_episode(evidence_backend, include_second_hop=True, answer="Beta City")
    expanded_only = EvidenceEpisode(
        question=success.question,
        turns=success.turns[:2],
    )
    reordered = success.model_copy(
        update={"turns": (success.turns[0], success.turns[2], success.turns[1], success.turns[3])}
    )

    trace = evidence_causal_curriculum_trace(proof, success)
    expanded_reward = evidence_ecp_curriculum_reward(
        proof, expanded_only, (("Beta City",),)
    )
    reordered_reward = evidence_ecp_curriculum_reward(
        proof, reordered, (("Beta City",),)
    )
    success_reward = evidence_ecp_curriculum_reward(proof, success, (("Beta City",),))

    assert trace.potentials == pytest.approx((0.20, 0.55, 1.0, 1.0))
    assert trace.deltas == pytest.approx((0.20, 0.35, 0.45, 0.0))
    assert expanded_reward.total == pytest.approx(0.0825)
    assert expanded_reward.components["causal_proof_certificate"] == 0.0
    assert reordered_reward.components["causal_proof_certificate"] == 0.0
    assert success_reward.total == pytest.approx(1.0)
    assert success_reward.metadata["causal_certificate"] is True


def test_search_r1_baseline_uses_only_normalized_answer_em(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    proof = _proof(evidence_backend)
    exact_without_provenance = EvidenceEpisode(
        question="Which city is reached from Alpha?",
        turns=(),
        final_answer=AnswerSet.literals(["the Beta City"]),
        done=False,
    )
    partial = EvidenceEpisode(
        question="Which city is reached from Alpha?",
        turns=(),
        final_answer=AnswerSet.literals(["Beta"]),
        done=False,
    )

    exact_reward = evidence_search_r1_em_reward(
        proof, exact_without_provenance, (("Beta City",),)
    )
    partial_reward = evidence_search_r1_em_reward(proof, partial, (("Beta City",),))

    assert exact_reward.total == 1.0
    assert exact_reward.components == {"answer_exact_match": 1.0}
    assert partial_reward.total == 0.0


def test_dual_residual_localizes_retrieval_and_reasoning_failures(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    proof = _proof(evidence_backend)
    retrieval_failure = _complete_episode(
        evidence_backend, include_second_hop=False, answer="Beta City"
    )
    reasoning_failure = _complete_episode(evidence_backend, include_second_hop=True, answer="Gamma")

    retrieval = classify_evidence_residual(proof, retrieval_failure, (("Beta City",),))
    reasoning = classify_evidence_residual(proof, reasoning_failure, (("Beta City",),))
    assert retrieval.kind == "retrieval"
    assert retrieval.missing_passage_keys == ("2:1",)
    assert reasoning.kind == "reasoning"
    assert reasoning.missing_passage_keys == ()


def test_kilt_metrics_and_two_dimensional_frontier() -> None:
    gold = (
        (
            EvidenceProvenance(page_id="1", paragraph_id=1),
            EvidenceProvenance(page_id="2", paragraph_id=1),
        ),
    )
    metrics = kilt_qa_metrics(
        AnswerSet.literals(["the Beta City"]),
        gold[0],
        (("Beta City",),),
        gold,
    )

    assert metrics.answer_exact_match == 1.0
    assert metrics.provenance_rprecision == 1.0
    assert metrics.kilt_f1 == 1.0
    assert evidence_frontier_reward(0.5, 0.5) == 1.0
    assert evidence_frontier_reward(0.0, 0.0) < evidence_frontier_reward(0.3, 0.3)


class _TwoHopRetriever(EvidenceRetriever):
    def __init__(self) -> None:
        self.first = PassageHit(
            page_id="1",
            paragraph_id=1,
            title="Alpha",
            text="Alpha points to another article.",
            score=1.0,
        )
        self.second = PassageHit(
            page_id="2",
            paragraph_id=1,
            title="Beta City",
            text="Beta City is the expected answer.",
            score=1.0,
        )

    def retrieve(
        self, query: str, *, limit: int, trace_id: str | None = None
    ) -> tuple[PassageHit, ...]:
        del query, trace_id
        return (self.first,)[:limit]

    def expand(
        self,
        passages: tuple[PassageHit, ...],
        query: str,
        *,
        limit: int,
        trace_id: str | None = None,
    ) -> tuple[PassageHit, ...]:
        del passages, query, trace_id
        return (self.second,)[:limit]


class _SharedExtractiveAnswerer:
    def answer(
        self,
        question: str,
        passages: tuple[PassageHit, ...],
        *,
        trace_id: str,
    ) -> AnswerPrediction:
        del question, trace_id
        has_answer = any("expected answer" in passage.text for passage in passages)
        answer = "Beta City" if has_answer else "unknown"
        keys = tuple(f"{value.page_id}:{value.paragraph_id}" for value in passages)
        return AnswerPrediction(answer=AnswerSet.literals([answer]), passage_keys=keys)


def test_paired_baseline_shows_hyperlink_evidence_flow_delta() -> None:
    retriever = _TwoHopRetriever()
    answerer = _SharedExtractiveAnswerer()
    config = EvidenceABConfig(retrieve_k=1, expand_k=1, context_k=2)
    example = BenchmarkExample(
        example_id="hotpot-smoke-1",
        dataset="hotpotqa",
        split="dev",
        question="Which city is reached through the article linked from Alpha?",
        topic_entity_ids=(),
        gold_answers=AnswerSet.literals(["Beta City"]),
        answer_aliases=(("Beta City",),),
        gold_provenance=(
            (
                EvidenceProvenance(page_id="1", paragraph_id=1),
                EvidenceProvenance(page_id="2", paragraph_id=1),
            ),
        ),
    )
    report = evaluate_evidence_ab(
        [example],
        SingleRetrievalBaseline(retriever, answerer, config),
        HyperlinkEvidenceFlow(retriever, answerer, config),
    )

    assert report.baseline["kilt_f1"] == 0.0
    assert report.evidence_flow["kilt_f1"] == 1.0
    assert report.delta["answer_f1"] == 1.0
    assert report.delta["provenance_rprecision"] == 0.5
    assert report.per_example[0].baseline_prediction.answer == AnswerSet.literals(["unknown"])
    assert report.per_example[0].evidence_flow_prediction.answer == AnswerSet.literals(
        ["Beta City"]
    )


def test_small_model_json_prediction_parser() -> None:
    prediction = parse_answer_prediction(
        '<think>brief</think>\n{"answer":"Beta City","passage_keys":["1:1","2:1"]}'
    )

    assert prediction.answer == AnswerSet.literals(["Beta City"])
    assert prediction.passage_keys == ("1:1", "2:1")
    assert prediction.rejection_reason is None


def test_interactive_decision_parser_accepts_hermes_tool_and_answer() -> None:
    tool = parse_evidence_agent_decision(
        '<tool_call>{"name":"text_search","arguments":{"query":"Alpha"}}</tool_call>'
    )
    answer = parse_evidence_agent_decision('{"answer":"Beta City"}')

    multi = parse_evidence_agent_decision(
        '<tool_call>{"name":"text_search","arguments":{"query":"Alpha"}}</tool_call>'
        '<tool_call>{"name":"expand_evidence","arguments":{"query":"Beta"}}</tool_call>'
    )
    assert tool.tool_name == "text_search"
    assert tool.arguments == {"query": "Alpha"}
    assert len(tool.tool_calls) == 1
    assert [call.tool_name for call in multi.tool_calls] == [
        "text_search",
        "expand_evidence",
    ]
    assert answer == EvidenceAgentDecision(kind="answer", answer="Beta City")


class _ScriptedInteractivePolicy:
    def __init__(self) -> None:
        self.index = 0

    def decide(
        self,
        messages: tuple[dict[str, object], ...],
        tools: tuple[dict[str, object], ...],
        *,
        trace_id: str,
    ) -> EvidenceAgentDecision:
        del messages, tools, trace_id
        decisions = (
            EvidenceAgentDecision(
                kind="tool",
                tool_name="text_search",
                arguments={"query": "Alpha", "limit": 1},
            ),
            EvidenceAgentDecision(
                kind="tool",
                tool_name="expand_evidence",
                arguments={"query": "Beta City", "limit": 1},
            ),
            EvidenceAgentDecision(
                kind="tool",
                tool_name="select_evidence",
                arguments={"passage_keys": ["1:1", "2:1"]},
            ),
            EvidenceAgentDecision(kind="answer", answer="Beta City"),
        )
        decision = decisions[self.index]
        self.index += 1
        return decision


def test_interactive_evaluator_uses_training_evidence_protocol() -> None:
    example = BenchmarkExample(
        example_id="interactive-1",
        dataset="hotpotqa",
        split="dev",
        question="Which city is reached through Alpha?",
        topic_entity_ids=(),
        gold_answers=AnswerSet.literals(["Beta City"]),
        answer_aliases=(("Beta City",),),
        gold_provenance=(
            (
                EvidenceProvenance(page_id="1", paragraph_id=1),
                EvidenceProvenance(page_id="2", paragraph_id=1),
            ),
        ),
    )
    report = evaluate_interactive_evidence(
        [example],
        retriever=_TwoHopRetriever(),
        policy=_ScriptedInteractivePolicy(),
        max_turns=4,
    )

    assert report.summary["kilt_f1"] == 1.0
    assert report.residuals == {"joint_success": 1}
    assert [turn.action["op"] for turn in report.per_example[0].episode.turns] == [
        "retrieve",
        "expand",
        "select_evidence",
        "answer",
    ]


class _AlwaysWrongAnswerer:
    def answer(
        self,
        question: str,
        passages: tuple[PassageHit, ...],
        *,
        trace_id: str,
    ) -> AnswerPrediction:
        del question, trace_id
        return AnswerPrediction(
            answer=AnswerSet.literals(["unknown"]),
            passage_keys=tuple(f"{passage.page_id}:{passage.paragraph_id}" for passage in passages),
        )


def test_evidence_selfplay_questioner_certifies_and_exports_reasoning_residuals(
    evidence_backend: SQLiteGraphBackend,
    tmp_path: Path,
) -> None:
    config = EvidenceSelfPlayConfig(seed=7, candidate_limit=2, train_ratio=0.5)
    challenges = generate_evidence_challenges(evidence_backend, config)
    assert len(challenges) == 1
    assert challenges[0].target_page_id == "2"
    assert challenges[0].category == "Cities"

    result = run_evidence_selfplay_round(
        challenges,
        backend=evidence_backend,
        retriever=BackendEvidenceRetriever(evidence_backend),
        answerer=_AlwaysWrongAnswerer(),
        config=config,
    )
    assert result.reasoning_residuals + result.retrieval_residuals == 1
    assert result.successes == 0

    if result.reasoning_residuals:
        duplicated = result.model_copy(update={"records": result.records * 2})
        summary = export_evidence_selfplay_round(
            duplicated,
            tmp_path / "selfplay",
            config=config,
        )
        assert summary["train_reasoning_residuals"] == 1
        assert (tmp_path / "selfplay" / "train.parquet").exists()


def test_searchscript_executes_as_bounded_replayable_solver_policy(
    evidence_backend: SQLiteGraphBackend,
) -> None:
    script = parse_searchscript(
        {
            "version": "0.4",
            "actions": [
                {"op": "retrieve", "query": "Alpha", "limit": 1, "out": "h0"},
                {
                    "op": "expand",
                    "in": "h0",
                    "query": "Cities",
                    "limit": 1,
                    "out": "h1",
                },
                {
                    "op": "select_evidence",
                    "inputs": ["h0", "h1"],
                    "passage_keys": ["1:0", "2:0"],
                    "out": "h2",
                },
                {"op": "answer", "value": "Beta City", "evidence": "h2"},
            ],
        }
    )

    episode = execute_searchscript(
        script,
        "Which city is linked from Alpha?",
        BackendEvidenceRetriever(evidence_backend),
        trace_id="unit-searchscript",
    )

    assert episode.done is True
    assert episode.final_answer == AnswerSet.literals(["Beta City"])
    assert {item.passage_key for item in episode.final_evidence} == {"1:0", "2:0"}


def test_direct_rl_export_and_rewards_need_no_sft(
    evidence_backend: SQLiteGraphBackend,
    tmp_path: Path,
) -> None:
    config = EvidenceSelfPlayConfig(seed=11, candidate_limit=2, train_ratio=0.5)
    challenges = generate_evidence_challenges(evidence_backend, config)
    result = run_evidence_selfplay_round(
        challenges,
        backend=evidence_backend,
        retriever=BackendEvidenceRetriever(evidence_backend),
        answerer=_AlwaysWrongAnswerer(),
        config=config,
    )
    duplicated = result.model_copy(update={"records": result.records * 2})
    output = tmp_path / "rl"
    summary = export_evidence_rl_round(
        duplicated,
        output,
        seed=11,
        train_ratio=0.5,
    )

    assert summary["uses_sft"] is False
    assert summary["questioner_algorithm"] == "reinforce_plus_plus"
    assert summary["solver_algorithm"] == "grpo"
    assert summary["solver_reward_variant"] == "additive_v1"
    assert (output / "records.parquet").exists()
    evaluation = summary["evaluation"]
    assert isinstance(evaluation, dict)
    validation = evaluation["validation"]
    assert isinstance(validation, dict)
    assert validation["examples"] == 1
    assert set(validation["delta"]) == set(KILTQAMetrics.model_fields)
    assert (output / "evaluation.json").exists()
    solver_row = pq.read_table(output / "solver_train.parquet").to_pylist()[0]
    questioner_row = pq.read_table(output / "questioner_train.parquet").to_pylist()[0]
    solution = json.dumps(
        {
            "version": "0.4",
            "actions": [
                {"op": "retrieve", "query": "Alpha", "limit": 1, "out": "h0"},
                {
                    "op": "expand",
                    "in": "h0",
                    "query": "Cities",
                    "limit": 1,
                    "out": "h1",
                },
                {
                    "op": "select_evidence",
                    "inputs": ["h0", "h1"],
                    "passage_keys": ["1:0", "2:0"],
                    "out": "h2",
                },
                {"op": "answer", "value": "Beta City", "evidence": "h2"},
            ],
        }
    )
    solver_score = compute_evidence_solver_score(
        solution,
        solver_row["extra_info"],
        backend=evidence_backend,
    )
    integrated_solver_score = asyncio.run(
        compute_score(
            "graphtask/evidence_solver",
            solution,
            "{}",
            solver_row["extra_info"],
            backend=evidence_backend,
        )
    )
    challenge_id = next(iter(questioner_row["extra_info"]["candidate_frontiers"]))
    questioner_score = compute_evidence_questioner_score(
        json.dumps({"version": "0.4-questioner", "challenge_id": challenge_id}),
        questioner_row["extra_info"],
        backend=evidence_backend,
    )
    one_hop_info = {
        **solver_row["extra_info"],
        "solver_rollout": {
            "calls": 1,
            "valid_calls": 1,
            "early_answer_attempts": 2,
            "observed_passages": [
                {
                    "page_id": "1",
                    "paragraph_id": 0,
                    "title": "Alpha",
                    "text": "Alpha",
                    "score": 1.0,
                }
            ],
            "selected_passage_keys": [],
            "evidence_actions": [{"op": "retrieve", "passages": []}],
        },
    }
    expanded_info = {
        **solver_row["extra_info"],
        "solver_rollout": {
            "calls": 2,
            "valid_calls": 2,
            "early_answer_attempts": 1,
            "observed_passages": [
                {
                    "page_id": "1",
                    "paragraph_id": 0,
                    "title": "Alpha",
                    "text": "Alpha",
                    "score": 1.0,
                },
                {
                    "page_id": "2",
                    "paragraph_id": 0,
                    "title": "Beta City",
                    "text": "Beta City",
                    "score": 1.0,
                },
            ],
            "selected_passage_keys": [],
            "evidence_actions": [
                {"op": "retrieve", "passages": []},
                {"op": "expand", "passages": []},
            ],
        },
    }
    one_hop_score = compute_evidence_solver_score(
        '{"answer":"Beta City"}', one_hop_info, backend=evidence_backend
    )
    expanded_score = compute_evidence_solver_score(
        '{"answer":"Beta City"}', expanded_info, backend=evidence_backend
    )
    search_r1_score = compute_evidence_solver_score(
        '{"answer":"the Beta City"}',
        {**one_hop_info, "evidence_reward_variant": "search_r1_em"},
        backend=evidence_backend,
    )

    assert solver_score["residual_success"] == 1.0
    assert solver_row["extra_info"]["evidence_reward_variant"] == "additive_v1"
    assert integrated_solver_score == solver_score
    assert solver_score["score"] > 0.9
    assert questioner_score["proof_validity"] == 0.2
    assert questioner_score["score"] > 0.0
    assert one_hop_score["score"] < 0.05
    assert one_hop_score["proof_completion"] == 0.0
    assert expanded_score["score"] > 0.5
    assert expanded_score["proof_completion"] == 1.0
    assert search_r1_score["score"] == 1.0
    assert search_r1_score["answer_exact_match"] == 1.0

    validation_id = duplicated.records[0].challenge.example.example_id
    nested_rows = [record.model_dump(mode="json") for record in duplicated.records]
    selected = benchmark_examples_from_records(
        nested_rows,
        example_ids=frozenset({validation_id}),
        limit=1,
    )
    assert tuple(example.example_id for example in selected) == (validation_id,)

    ecp_output = tmp_path / "ecp-rl"
    ecp_summary = export_evidence_rl_round(
        duplicated,
        ecp_output,
        seed=11,
        train_ratio=0.5,
        solver_reward_variant="ecp_v1",
    )
    ecp_row = pq.read_table(ecp_output / "solver_train.parquet").to_pylist()[0]
    assert ecp_summary["solver_reward_variant"] == "ecp_v1"
    assert ecp_row["extra_info"]["evidence_reward_variant"] == "ecp_v1"


def test_counterfactual_pair_executes_one_edge_intervention(
    tmp_path: Path,
) -> None:
    source = tmp_path / "counterfactual-kilt.json"
    pages = [
        {
            "_id": "1",
            "wikipedia_id": "1",
            "wikipedia_title": "Alpha",
            "text": ["Alpha", "Alpha links to two places."],
            "anchors": [
                {"text": "city", "wikipedia_id": "2"},
                {"text": "village", "wikipedia_id": "3"},
            ],
            "categories": "Examples",
            "history": {"revid": 1},
            "wikidata_info": {},
        },
        {
            "_id": "2",
            "wikipedia_id": "2",
            "wikipedia_title": "Beta City",
            "text": ["Beta City", "Beta City is a city."],
            "anchors": [],
            "categories": "Cities",
            "history": {"revid": 2},
            "wikidata_info": {},
        },
        {
            "_id": "3",
            "wikipedia_id": "3",
            "wikipedia_title": "Gamma Village",
            "text": ["Gamma Village", "Gamma Village is a village."],
            "anchors": [],
            "categories": "Villages",
            "history": {"revid": 3},
            "wikidata_info": {},
        },
    ]
    source.write_text("\n".join(json.dumps(page) for page in pages) + "\n")
    processed = tmp_path / "counterfactual-processed"
    prepare_kilt(source, processed)
    backend = SQLiteGraphBackend(processed / "graph.sqlite", snapshot_id="counterfactual-v1")
    try:
        challenges = generate_evidence_challenges(
            backend,
            EvidenceSelfPlayConfig(seed=3, candidate_limit=3),
        )
        by_target = {challenge.target_page_id: challenge for challenge in challenges}
        pair = certify_counterfactual_pair(
            by_target["2"],
            by_target["3"],
            backend=backend,
        )
        rows = counterfactual_retrieval_examples([pair], backend=backend)
        reranker = train_counterfactual_reranker(rows, backend=backend, seed=3)
        retrieved = BackendEvidenceRetriever(backend).retrieve(
            "Alpha", limit=1, trace_id="reranker-test:retrieve"
        )
        reranked = CounterfactualEvidenceRetriever(backend, reranker).expand(
            retrieved,
            rows[0].query,
            limit=2,
            trace_id="reranker-test:expand",
        )
    finally:
        backend.close()

    assert pair.changed_operation_index == 2
    assert pair.shared_passage_keys == ("1:0",)
    assert pair.positive_only_passage_keys == ("2:0",)
    assert pair.counterfactual_only_passage_keys == ("3:0",)
    assert rows[0].positive_passage.page_id == "2"
    assert rows[0].hard_negative_passages[0].page_id == "3"
    assert reranker.token_weights["cities"] > 0.0
    assert reranked[0].page_id == "2"


def test_direct_rl_plan_alternates_grpo_then_reinforce_without_sft(
    tmp_path: Path,
) -> None:
    plan = evidence_selfplay_plan(
        EvidenceRLTrainConfig(
            model_path="local/qwen3-0.6b",
            graph_db=tmp_path / "graph.sqlite",
            round_data=tmp_path / "round-data",
            output_dir=tmp_path / "updates",
            actor_gpus="0",
        )
    )

    assert plan["ms_swift_version"] == "3.10.3"
    assert plan["uses_sft"] is False
    assert plan["shared_policy_update_order"] == ["solver", "questioner"]
    phases = plan["phases"]
    assert isinstance(phases, list)
    assert isinstance(phases[0], dict)
    assert isinstance(phases[1], dict)
    assert phases[0]["algorithm"] == "grpo"
    assert phases[1]["algorithm"] == "reinforce_plus_plus"


def test_questioner_archive_promotes_online_choice_and_frontier_fallback() -> None:
    def solver_row(challenge_id: str) -> dict[str, object]:
        return {
            "data_source": "graphtask/evidence_solver",
            "prompt": [],
            "ability": "certified_evidence_search",
            "reward_model": {"style": "rule", "ground_truth": "{}"},
            "extra_info": {"task_id": challenge_id},
            "uid": f"old:{challenge_id}",
        }

    questioner_rows = [
        {
            "extra_info": {
                "candidate_frontiers": {
                    "a": {
                        "evidence_recovery_rate": 0.5,
                        "conditional_answer_rate": 0.5,
                        "novelty": 1.0,
                    },
                    "b": {
                        "evidence_recovery_rate": 0.0,
                        "conditional_answer_rate": 0.0,
                        "novelty": 1.0,
                    },
                },
                "frontier_target": 0.5,
                "frontier_sigma": 0.2,
            }
        },
        {
            "extra_info": {
                "candidate_frontiers": {
                    "c": {
                        "evidence_recovery_rate": 0.5,
                        "conditional_answer_rate": 0.5,
                        "novelty": 1.0,
                    }
                },
                "frontier_target": 0.5,
                "frontier_sigma": 0.2,
            }
        },
    ]
    events = [
        {
            "prompt": [
                '{"candidate_challenges":[{"challenge_id":"a"},'
                '{"challenge_id":"b"}]}'
            ],
            "completion": [
                '{"version":"0.4-questioner","challenge_id":"b"}'
            ],
            "GraphTaskReward": [0.4],
        }
    ]

    rows, manifest = build_promoted_solver_curriculum(
        questioner_rows,
        [solver_row(value) for value in ("a", "b", "c")],
        events,
        target_size=4,
        seed=9,
    )

    assert len(rows) == 4
    assert manifest["rollout_promotions"] == 1
    assert manifest["fallback_promotions"] == 1
    selections = manifest["selections"]
    assert isinstance(selections, list)
    assert [item["challenge_id"] for item in selections] == ["b", "c"]
    assert all(row["uid"].startswith("evidence-solver-r2:") for row in rows)

    mixed_rows, mixed_manifest = build_promoted_solver_curriculum(
        questioner_rows,
        [solver_row(value) for value in ("a", "b", "c")],
        events,
        target_size=5,
        seed=9,
        retain_all_solver_rows=True,
    )
    assert len(mixed_rows) == 5
    assert {row["extra_info"]["task_id"] for row in mixed_rows} == {"a", "b", "c"}
    assert mixed_manifest["retains_all_solver_rows"] is True
    assert mixed_manifest["coverage_replays"] == 1


def test_solver_only_plan_skips_questioner_phase(tmp_path: Path) -> None:
    plan = evidence_selfplay_plan(
        EvidenceRLTrainConfig(
            model_path="local/qwen3-0.6b",
            initial_adapter=tmp_path / "solver-v1",
            graph_db=tmp_path / "graph.sqlite",
            round_data=tmp_path / "round-data",
            output_dir=tmp_path / "updates",
            policy_topology="role_separated",
            train_questioner=False,
        )
    )

    assert plan["phase_order"] == ["solver"]
    assert plan["shared_policy_update_order"] == []
    phases = plan["phases"]
    assert isinstance(phases, list)
    assert len(phases) == 1
    assert phases[0]["role"] == "solver"


def test_role_separated_update_reuses_solver_without_initializing_questioner_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_db = tmp_path / "graph.sqlite"
    graph_db.touch()
    round_data = tmp_path / "round-data"
    round_data.mkdir()
    (round_data / "solver_train.parquet").touch()
    (round_data / "questioner_train.parquet").touch()
    train_script = tmp_path / "train.sh"
    train_script.touch()
    solver_adapter = tmp_path / "solver-checkpoint"
    solver_adapter.mkdir()
    (solver_adapter / "adapter_model.safetensors").touch()
    calls: list[dict[str, str]] = []

    def fake_run(
        _command: list[str], *, cwd: Path, env: dict[str, str], check: bool
    ) -> None:
        assert cwd == train_script.parent.parent
        assert check is True
        calls.append(env)
        checkpoint = Path(env["OUTPUT_DIR"]) / "v0" / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").touch()

    monkeypatch.setattr(evidence_runner.subprocess, "run", fake_run)
    result = run_evidence_selfplay_update(
        EvidenceRLTrainConfig(
            model_path="local/qwen3-0.6b",
            graph_db=graph_db,
            round_data=round_data,
            output_dir=tmp_path / "updates",
            train_script=train_script,
            existing_solver_adapter=solver_adapter,
            policy_topology="role_separated",
        )
    )

    assert len(calls) == 1
    assert calls[0]["RL_ALGORITHM"] == "reinforce_plus_plus"
    assert "LORA_ADAPTER_PATH" not in calls[0]
    assert result["shared_policy_update_order"] == []
    assert result["final_adapter"] == str(solver_adapter)
    assert result["role_adapters"]["solver"] == str(solver_adapter)
    assert result["role_adapters"]["questioner"].endswith("checkpoint-1")
