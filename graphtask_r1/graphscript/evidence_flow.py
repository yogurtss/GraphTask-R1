from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from graphtask_r1.envs.text_search import execute_text_search
from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import AnswerSet, EvidenceProvenance, PassageHit

_HANDLE = re.compile(r"^h(?:[0-9]|[1-5][0-9]|6[0-3])$")


class EvidenceFlowError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ProofPageOp(_FrozenModel):
    op: Literal["page"] = "page"
    page_id: str = Field(min_length=1)
    out: str


class ProofParagraphOp(_FrozenModel):
    op: Literal["paragraph"] = "paragraph"
    input_handle: str = Field(alias="in")
    paragraph_id: int = Field(ge=0)
    out: str


class ProofFollowAnchorOp(_FrozenModel):
    op: Literal["follow_anchor"] = "follow_anchor"
    input_handle: str = Field(alias="in")
    target_page_id: str = Field(min_length=1)
    out: str


class ProofSpanOp(_FrozenModel):
    op: Literal["span"] = "span"
    input_handle: str = Field(alias="in")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    out: str


class ProofPageTitleOp(_FrozenModel):
    op: Literal["page_title"] = "page_title"
    input_handle: str = Field(alias="in")
    out: str


class ProofJoinEvidenceOp(_FrozenModel):
    op: Literal["join_evidence"] = "join_evidence"
    inputs: tuple[str, ...] = Field(min_length=1)
    out: str


class ProofEmitOp(_FrozenModel):
    op: Literal["emit"] = "emit"
    answer: str
    evidence: str


ProofOperation: TypeAlias = Annotated[
    ProofPageOp
    | ProofParagraphOp
    | ProofFollowAnchorOp
    | ProofSpanOp
    | ProofPageTitleOp
    | ProofJoinEvidenceOp
    | ProofEmitOp,
    Field(discriminator="op"),
]


class ProofScript(_FrozenModel):
    version: Literal["0.4-proof"] = "0.4-proof"
    ops: tuple[ProofOperation, ...] = Field(min_length=2, max_length=64)


class RetrieveAction(_FrozenModel):
    op: Literal["retrieve"] = "retrieve"
    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=3, ge=1, le=10)
    out: str


class ExpandAction(_FrozenModel):
    op: Literal["expand"] = "expand"
    input_handle: str = Field(alias="in")
    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=3, ge=1, le=10)
    out: str


class SelectEvidenceAction(_FrozenModel):
    op: Literal["select_evidence"] = "select_evidence"
    inputs: tuple[str, ...] = Field(min_length=1)
    passage_keys: tuple[str, ...] = Field(min_length=1)
    out: str


class EvidenceAnswerAction(_FrozenModel):
    op: Literal["answer"] = "answer"
    value: str = Field(min_length=1)
    evidence: str


SearchAction: TypeAlias = Annotated[
    RetrieveAction | ExpandAction | SelectEvidenceAction | EvidenceAnswerAction,
    Field(discriminator="op"),
]


class SearchActionEnvelope(_FrozenModel):
    version: Literal["0.4"] = "0.4"
    action: SearchAction


class SearchScript(_FrozenModel):
    """A bounded, replayable search policy emitted by the Solver."""

    version: Literal["0.4"] = "0.4"
    actions: tuple[SearchAction, ...] = Field(min_length=2, max_length=8)


class EvidenceTurn(_FrozenModel):
    index: int = Field(ge=0)
    action: dict[str, Any]
    passages: tuple[PassageHit, ...] = ()
    selected_evidence: tuple[EvidenceProvenance, ...] = ()


class EvidenceEpisode(_FrozenModel):
    version: Literal["0.4"] = "0.4"
    question: str = Field(min_length=1)
    turns: tuple[EvidenceTurn, ...]
    final_answer: AnswerSet = AnswerSet()
    final_evidence: tuple[EvidenceProvenance, ...] = ()
    done: bool = False

    def retrieved_provenance(self) -> tuple[EvidenceProvenance, ...]:
        values: list[EvidenceProvenance] = []
        for turn in self.turns:
            values.extend(_passage_provenance(passage) for passage in turn.passages)
        return tuple(dict.fromkeys(values))


class ProofExecution(_FrozenModel):
    answers: AnswerSet
    evidence: tuple[EvidenceProvenance, ...]
    passage_keys: tuple[str, ...]


def evidence_solver_prompt(question: str) -> list[dict[str, str]]:
    """Shared prompt for training and protocol-matched interactive evaluation."""

    return [
        {
            "role": "system",
            "content": (
                "You are the Solver. For every certified task, complete this evidence "
                "protocol: (1) text_search, (2) expand_evidence for the linked second "
                "hop, (3) select_evidence by copying exact returned passage_key values, "
                'then (4) emit only JSON {"answer":"..."}. Never answer before step 3 '
                "and never invent passage keys. The answer is the Wikipedia page title "
                "supported by the selected target passage; it is never a passage_key, "
                "page_id, or URL."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]


@runtime_checkable
class PassageAddressBackend(Protocol):
    def get_passage(
        self, page_id: str, paragraph_id: int, *, max_chars: int = 4_000
    ) -> dict[str, Any]: ...


@runtime_checkable
class PagePassagesBackend(Protocol):
    def page_passages(
        self,
        page_ids: list[str],
        *,
        max_chars: int = 2_000,
        limit_per_page: int = 3,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class EvidenceRetriever(Protocol):
    def retrieve(
        self, query: str, *, limit: int, trace_id: str | None = None
    ) -> tuple[PassageHit, ...]: ...

    def expand(
        self,
        passages: tuple[PassageHit, ...],
        query: str,
        *,
        limit: int,
        trace_id: str | None = None,
    ) -> tuple[PassageHit, ...]: ...


class BackendEvidenceRetriever:
    """Sparse KILT retriever with hyperlink expansion.

    A dense/hybrid implementation can satisfy the same ``EvidenceRetriever``
    protocol without changing the GraphScript runtime.
    """

    def __init__(self, backend: GraphBackend) -> None:
        self.backend = backend

    def retrieve(
        self, query: str, *, limit: int, trace_id: str | None = None
    ) -> tuple[PassageHit, ...]:
        return execute_text_search(
            self.backend,
            query,
            limit=limit,
            trace_id=trace_id,
        )

    def expand(
        self,
        passages: tuple[PassageHit, ...],
        query: str,
        *,
        limit: int,
        trace_id: str | None = None,
    ) -> tuple[PassageHit, ...]:
        del trace_id
        if not isinstance(self.backend, PagePassagesBackend):
            raise EvidenceFlowError(
                "PASSAGE_LOOKUP_UNAVAILABLE", "backend cannot load hyperlink page passages"
            )
        source_ids = sorted({passage.page_id for passage in passages})
        triples = self.backend.neighbors(
            source_ids,
            direction="out",
            relation_ids=["wikipedia_link"],
            limit=max(100, limit),
        )
        target_ids = sorted({triple.object for triple in triples})
        candidates = tuple(
            PassageHit.model_validate(value)
            for value in self.backend.page_passages(target_ids, limit_per_page=3)
        )
        query_tokens = _tokens(query)
        page_types = {
            page_id: " ".join(self.backend.entity_info(page_id).type_ids) for page_id in target_ids
        }
        ranked = sorted(
            candidates,
            key=lambda passage: (
                -_lexical_overlap(
                    query_tokens,
                    _tokens(
                        f"{passage.title} {passage.text} {page_types.get(passage.page_id, '')}"
                    ),
                ),
                passage.page_id,
                passage.paragraph_id,
            ),
        )
        return tuple(ranked[:limit])


class CounterfactualRerankerState(_FrozenModel):
    """Serializable token weights learned from executable proof interventions."""

    version: Literal["ecp-counterfactual-reranker-v1"] = (
        "ecp-counterfactual-reranker-v1"
    )
    seed: int
    examples: int = Field(ge=1)
    epochs: int = Field(ge=1)
    token_weights: dict[str, float]


class CounterfactualEvidenceRetriever:
    """Rerank hyperlink expansion with counterfactual pairwise token weights."""

    def __init__(
        self,
        backend: GraphBackend,
        state: CounterfactualRerankerState,
        *,
        candidate_limit: int = 100,
    ) -> None:
        self.backend = backend
        self.state = state
        self.candidate_limit = candidate_limit
        self.base = BackendEvidenceRetriever(backend)

    @classmethod
    def from_path(
        cls,
        backend: GraphBackend,
        path: Path,
        *,
        candidate_limit: int = 100,
    ) -> CounterfactualEvidenceRetriever:
        state = CounterfactualRerankerState.model_validate_json(path.read_text())
        return cls(backend, state, candidate_limit=candidate_limit)

    def retrieve(
        self, query: str, *, limit: int, trace_id: str | None = None
    ) -> tuple[PassageHit, ...]:
        return self.base.retrieve(query, limit=limit, trace_id=trace_id)

    def expand(
        self,
        passages: tuple[PassageHit, ...],
        query: str,
        *,
        limit: int,
        trace_id: str | None = None,
    ) -> tuple[PassageHit, ...]:
        candidates = self.base.expand(
            passages,
            query,
            limit=max(limit, self.candidate_limit),
            trace_id=trace_id,
        )
        query_tokens = _tokens(query)
        ranked = sorted(
            candidates,
            key=lambda passage: (
                -self._score(query_tokens, passage),
                passage.page_id,
                passage.paragraph_id,
            ),
        )
        return tuple(ranked[:limit])

    def _score(self, query_tokens: frozenset[str], passage: PassageHit) -> float:
        type_text = " ".join(self.backend.entity_info(passage.page_id).type_ids)
        passage_tokens = _tokens(f"{passage.title} {passage.text} {type_text}")
        overlap = query_tokens & passage_tokens
        learned = sum(self.state.token_weights.get(token, 0.0) for token in overlap)
        return learned + _lexical_overlap(query_tokens, passage_tokens)


@dataclass(frozen=True)
class _ProofHandle:
    pages: tuple[str, ...] = ()
    passages: tuple[PassageHit, ...] = ()
    answer: str | None = None
    evidence: tuple[EvidenceProvenance, ...] = ()


def parse_proofscript(value: str | dict[str, Any]) -> ProofScript:
    return (
        ProofScript.model_validate_json(value)
        if isinstance(value, str)
        else ProofScript.model_validate(value)
    )


def parse_search_action(value: str | dict[str, Any]) -> SearchActionEnvelope:
    return (
        SearchActionEnvelope.model_validate_json(value)
        if isinstance(value, str)
        else SearchActionEnvelope.model_validate(value)
    )


def parse_searchscript(value: str | dict[str, Any]) -> SearchScript:
    return (
        SearchScript.model_validate_json(value)
        if isinstance(value, str)
        else SearchScript.model_validate(value)
    )


def execute_proofscript(script: ProofScript, backend: GraphBackend) -> ProofExecution:
    if not isinstance(backend, PassageAddressBackend):
        raise EvidenceFlowError(
            "PASSAGE_LOOKUP_UNAVAILABLE", "ProofScript requires addressable passages"
        )
    handles: dict[str, _ProofHandle] = {}
    emitted: ProofExecution | None = None
    for op in script.ops:
        if isinstance(op, ProofPageOp):
            _define(op.out, handles)
            backend.entity_info(op.page_id)
            handles[op.out] = _ProofHandle(pages=(op.page_id,))
        elif isinstance(op, ProofParagraphOp):
            source = _get(op.input_handle, handles)
            if len(source.pages) != 1:
                raise EvidenceFlowError("TYPE_MISMATCH", "paragraph requires one page")
            passage = PassageHit.model_validate(
                backend.get_passage(source.pages[0], op.paragraph_id)
            )
            _define(op.out, handles)
            handles[op.out] = _ProofHandle(
                passages=(passage,), evidence=(_passage_provenance(passage),)
            )
        elif isinstance(op, ProofFollowAnchorOp):
            source = _get(op.input_handle, handles)
            if len(source.pages) != 1:
                raise EvidenceFlowError("TYPE_MISMATCH", "follow_anchor requires one page")
            links = backend.neighbors(
                [source.pages[0]],
                direction="out",
                relation_ids=["wikipedia_link"],
                limit=10_000,
            )
            if op.target_page_id not in {link.object for link in links}:
                raise EvidenceFlowError(
                    "ANCHOR_NOT_FOUND",
                    f"{source.pages[0]} does not link to {op.target_page_id}",
                )
            _define(op.out, handles)
            handles[op.out] = _ProofHandle(pages=(op.target_page_id,))
        elif isinstance(op, ProofSpanOp):
            source = _get(op.input_handle, handles)
            if len(source.passages) != 1:
                raise EvidenceFlowError("TYPE_MISMATCH", "span requires one passage")
            passage = source.passages[0]
            value = passage.text[op.start : op.end]
            if not value:
                raise EvidenceFlowError("EMPTY_ANSWER", "span emitted an empty answer")
            _define(op.out, handles)
            handles[op.out] = _ProofHandle(
                answer=value,
                evidence=(
                    EvidenceProvenance(
                        page_id=passage.page_id,
                        paragraph_id=passage.paragraph_id,
                        title=passage.title,
                        start_character=op.start,
                        end_character=op.end,
                    ),
                ),
            )
        elif isinstance(op, ProofPageTitleOp):
            source = _get(op.input_handle, handles)
            if len(source.pages) != 1:
                raise EvidenceFlowError("TYPE_MISMATCH", "page_title requires one page")
            _define(op.out, handles)
            handles[op.out] = _ProofHandle(answer=backend.entity_info(source.pages[0]).label)
        elif isinstance(op, ProofJoinEvidenceOp):
            evidence = tuple(
                dict.fromkeys(
                    item for handle in op.inputs for item in _get(handle, handles).evidence
                )
            )
            _define(op.out, handles)
            handles[op.out] = _ProofHandle(evidence=evidence)
        elif isinstance(op, ProofEmitOp):
            answer = _get(op.answer, handles).answer
            evidence = _get(op.evidence, handles).evidence
            if answer is None:
                raise EvidenceFlowError("TYPE_MISMATCH", "emit answer handle is not textual")
            if not evidence:
                raise EvidenceFlowError("EMPTY_EVIDENCE", "emit requires evidence")
            emitted = ProofExecution(
                answers=AnswerSet.literals([answer]),
                evidence=evidence,
                passage_keys=tuple(value.passage_key for value in evidence),
            )
    if emitted is None:
        raise EvidenceFlowError("MISSING_EMIT", "ProofScript did not emit an answer")
    return emitted


class EvidenceFlowSession:
    def __init__(
        self,
        question: str,
        retriever: EvidenceRetriever,
        *,
        max_turns: int = 8,
        trace_id: str = "evidence-flow",
    ) -> None:
        self.question = question
        self.retriever = retriever
        self.max_turns = max_turns
        self.trace_id = trace_id
        self._passage_handles: dict[str, tuple[PassageHit, ...]] = {}
        self._evidence_handles: dict[str, tuple[EvidenceProvenance, ...]] = {}
        self._turns: list[EvidenceTurn] = []
        self._answer = AnswerSet()
        self._final_evidence: tuple[EvidenceProvenance, ...] = ()
        self._done = False

    @property
    def episode(self) -> EvidenceEpisode:
        return EvidenceEpisode(
            question=self.question,
            turns=tuple(self._turns),
            final_answer=self._answer,
            final_evidence=self._final_evidence,
            done=self._done,
        )

    def step(self, envelope: SearchActionEnvelope) -> EvidenceTurn:
        if self._done:
            raise EvidenceFlowError("EPISODE_DONE", "answer was already emitted")
        if len(self._turns) >= self.max_turns:
            raise EvidenceFlowError("TURN_BUDGET_EXCEEDED", "search turn budget exhausted")
        action = envelope.action
        passages: tuple[PassageHit, ...] = ()
        selected: tuple[EvidenceProvenance, ...] = ()
        trace_id = f"{self.trace_id}:{len(self._turns)}"
        if isinstance(action, RetrieveAction):
            _define_action_handle(action.out, self._passage_handles, self._evidence_handles)
            passages = self.retriever.retrieve(action.query, limit=action.limit, trace_id=trace_id)
            self._passage_handles[action.out] = passages
        elif isinstance(action, ExpandAction):
            _define_action_handle(action.out, self._passage_handles, self._evidence_handles)
            source = self._passage_handles.get(action.input_handle)
            if source is None:
                raise EvidenceFlowError(
                    "INVALID_HANDLE", f"unknown passage handle: {action.input_handle}"
                )
            passages = self.retriever.expand(
                source, action.query, limit=action.limit, trace_id=trace_id
            )
            self._passage_handles[action.out] = passages
        elif isinstance(action, SelectEvidenceAction):
            _define_action_handle(action.out, self._passage_handles, self._evidence_handles)
            candidates = {
                _passage_key(passage): passage
                for handle in action.inputs
                for passage in self._required_passages(handle)
            }
            selected = tuple(
                _passage_provenance(candidates[key])
                for key in action.passage_keys
                if key in candidates
            )
            if not selected:
                raise EvidenceFlowError(
                    "EVIDENCE_NOT_OBSERVED", "selected evidence was not returned by retrieval"
                )
            self._evidence_handles[action.out] = tuple(dict.fromkeys(selected))
        elif isinstance(action, EvidenceAnswerAction):
            selected = self._evidence_handles.get(action.evidence, ())
            if not selected:
                raise EvidenceFlowError(
                    "INVALID_HANDLE", f"unknown evidence handle: {action.evidence}"
                )
            self._answer = AnswerSet.literals([action.value])
            self._final_evidence = selected
            self._done = True
        turn = EvidenceTurn(
            index=len(self._turns),
            action=action.model_dump(mode="json", by_alias=True),
            passages=passages,
            selected_evidence=selected,
        )
        self._turns.append(turn)
        return turn

    def _required_passages(self, handle: str) -> tuple[PassageHit, ...]:
        passages = self._passage_handles.get(handle)
        if passages is None:
            raise EvidenceFlowError("INVALID_HANDLE", f"unknown passage handle: {handle}")
        return passages


def execute_searchscript(
    script: SearchScript,
    question: str,
    retriever: EvidenceRetriever,
    *,
    trace_id: str = "evidence-flow",
) -> EvidenceEpisode:
    """Execute a complete Solver policy without exposing the certified proof."""

    session = EvidenceFlowSession(
        question,
        retriever,
        max_turns=len(script.actions),
        trace_id=trace_id,
    )
    for action in script.actions:
        session.step(SearchActionEnvelope(action=action))
    episode = session.episode
    if not episode.done:
        raise EvidenceFlowError("MISSING_ANSWER", "SearchScript did not emit an answer")
    return episode


def _passage_key(passage: PassageHit) -> str:
    return f"{passage.page_id}:{passage.paragraph_id}"


def _passage_provenance(passage: PassageHit) -> EvidenceProvenance:
    return EvidenceProvenance(
        page_id=passage.page_id,
        paragraph_id=passage.paragraph_id,
        title=passage.title,
    )


def _define(handle: str, handles: dict[str, _ProofHandle]) -> None:
    if not _HANDLE.fullmatch(handle):
        raise EvidenceFlowError("INVALID_HANDLE", f"invalid handle: {handle}")
    if handle in handles:
        raise EvidenceFlowError("DUPLICATE_HANDLE", f"duplicate handle: {handle}")


def _get(handle: str, handles: dict[str, _ProofHandle]) -> _ProofHandle:
    if handle not in handles:
        raise EvidenceFlowError("INVALID_HANDLE", f"unknown handle: {handle}")
    return handles[handle]


def _define_action_handle(
    handle: str,
    passages: dict[str, tuple[PassageHit, ...]],
    evidence: dict[str, tuple[EvidenceProvenance, ...]],
) -> None:
    if not _HANDLE.fullmatch(handle):
        raise EvidenceFlowError("INVALID_HANDLE", f"invalid handle: {handle}")
    if handle in passages or handle in evidence:
        raise EvidenceFlowError("DUPLICATE_HANDLE", f"duplicate handle: {handle}")


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _lexical_overlap(query: frozenset[str], passage: frozenset[str]) -> float:
    return len(query & passage) / len(query) if query else 0.0


_PROOF_ADAPTER: TypeAdapter[ProofOperation] = TypeAdapter(ProofOperation)
_SEARCH_ACTION_ADAPTER: TypeAdapter[SearchAction] = TypeAdapter(SearchAction)


def parse_proof_operation(value: dict[str, Any]) -> ProofOperation:
    return _PROOF_ADAPTER.validate_python(value)


def parse_evidence_action(value: dict[str, Any]) -> SearchAction:
    return _SEARCH_ACTION_ADAPTER.validate_python(value)
