from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any, Literal, cast

from graphtask_r1.experiments.interactive_evidence import (
    EvidenceAgentDecision,
    EvidenceAgentToolCall,
)
from graphtask_r1.experiments.kilt_evidence_ab import AnswerPrediction
from graphtask_r1.schema import AnswerSet, PassageHit


class EvidenceAnswerError(ValueError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class TransformersEvidenceAnswerer:
    """Deterministic local-HF answerer suitable for the Qwen3-0.6B smoke run."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 128,
        device_map: str = "auto",
    ) -> None:
        transformers = importlib.import_module("transformers")
        self._torch = importlib.import_module("torch")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map=device_map,
        )
        if adapter_path is not None:
            peft = importlib.import_module("peft")
            self._model = peft.PeftModel.from_pretrained(
                self._model,
                str(adapter_path),
            )
        self._model.eval()
        self.max_new_tokens = max_new_tokens

    def answer(
        self,
        question: str,
        passages: tuple[PassageHit, ...],
        *,
        trace_id: str,
    ) -> AnswerPrediction:
        prompt = evidence_answer_prompt(question, passages, trace_id=trace_id)
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        completion = self._tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        try:
            return parse_answer_prediction(completion)
        except EvidenceAnswerError as exc:
            return AnswerPrediction(
                answer=AnswerSet(),
                rejection_reason=exc.reason_code,
            )


class TransformersInteractiveEvidencePolicy:
    """Local deterministic policy for protocol-matched evidence evaluation."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 256,
        device_map: str = "auto",
    ) -> None:
        transformers = importlib.import_module("transformers")
        self._torch = importlib.import_module("torch")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map=device_map,
        )
        if adapter_path is not None:
            peft = importlib.import_module("peft")
            self._model = peft.PeftModel.from_pretrained(self._model, str(adapter_path))
        self._model.eval()
        self.max_new_tokens = max_new_tokens

    def decide(
        self,
        messages: tuple[dict[str, Any], ...],
        tools: tuple[dict[str, object], ...],
        *,
        trace_id: str,
    ) -> EvidenceAgentDecision:
        del trace_id
        rendered = self._tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        completion = self._tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return parse_evidence_agent_decision(completion)


def parse_answer_prediction(completion: str) -> AnswerPrediction:
    start = completion.find("{")
    end = completion.rfind("}")
    if start < 0 or end < start:
        raise EvidenceAnswerError("ANSWER_JSON_INVALID", completion[:500])
    try:
        value: Any = json.loads(completion[start : end + 1])
    except json.JSONDecodeError as exc:
        raise EvidenceAnswerError("ANSWER_JSON_INVALID", str(exc)) from exc
    if not isinstance(value, dict):
        raise EvidenceAnswerError("ANSWER_SCHEMA_INVALID", "prediction must be an object")
    raw_answer = value.get("answer")
    answers = raw_answer if isinstance(raw_answer, list) else [raw_answer]
    if not all(isinstance(answer, str | int | float) for answer in answers):
        raise EvidenceAnswerError("ANSWER_SCHEMA_INVALID", "answer must be scalar or list")
    raw_keys = value.get("passage_keys", [])
    if not isinstance(raw_keys, list) or not all(isinstance(key, str) for key in raw_keys):
        raise EvidenceAnswerError("ANSWER_SCHEMA_INVALID", "passage_keys must be strings")
    return AnswerPrediction(
        answer=AnswerSet.literals([str(answer) for answer in answers]),
        passage_keys=tuple(raw_keys),
    )


def parse_evidence_agent_decision(completion: str) -> EvidenceAgentDecision:
    """Parse Qwen/Hermes tool calls or the final answer without repairing them."""

    tool_matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", completion, re.S)
    candidate = tool_matches[0] if tool_matches else completion
    start = candidate.find("{")
    if start < 0:
        raise EvidenceAnswerError("AGENT_JSON_INVALID", completion[:500])
    try:
        payload, end = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        raise EvidenceAnswerError("AGENT_JSON_INVALID", str(exc)) from exc
    if candidate[start + end :].strip() and not tool_matches:
        raise EvidenceAnswerError("AGENT_EXTRA_TEXT", candidate[start + end :][:500])
    if not isinstance(payload, dict):
        raise EvidenceAnswerError("AGENT_SCHEMA_INVALID", "decision must be an object")
    if isinstance(payload.get("answer"), str):
        answer = str(payload["answer"]).strip()
        if not answer:
            raise EvidenceAnswerError("AGENT_SCHEMA_INVALID", "answer cannot be empty")
        return EvidenceAgentDecision(kind="answer", answer=answer)
    calls = tuple(_parse_agent_tool_call(value) for value in (tool_matches or (candidate,)))
    first = calls[0]
    return EvidenceAgentDecision(
        kind="tool",
        tool_name=first.tool_name,
        arguments=first.arguments,
        tool_calls=calls,
    )


def _parse_agent_tool_call(value: str) -> EvidenceAgentToolCall:
    try:
        payload: Any = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvidenceAnswerError("AGENT_JSON_INVALID", str(exc)) from exc
    if not isinstance(payload, dict):
        raise EvidenceAnswerError("AGENT_SCHEMA_INVALID", "tool call must be an object")
    raw_function = payload.get("function")
    function = raw_function if isinstance(raw_function, dict) else payload
    name = function.get("name", function.get("tool_name"))
    if name not in {"text_search", "expand_evidence", "select_evidence"}:
        raise EvidenceAnswerError("AGENT_TOOL_INVALID", str(name))
    arguments = function.get("arguments", payload.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise EvidenceAnswerError("AGENT_ARGUMENTS_INVALID", str(exc)) from exc
    if not isinstance(arguments, dict):
        raise EvidenceAnswerError("AGENT_ARGUMENTS_INVALID", "arguments must be an object")
    return EvidenceAgentToolCall(
        tool_name=cast(
            Literal["text_search", "expand_evidence", "select_evidence"], name
        ),
        arguments=arguments,
    )


def evidence_answer_prompt(
    question: str, passages: tuple[PassageHit, ...], *, trace_id: str
) -> str:
    context = "\n\n".join(
        f"[{passage.page_id}:{passage.paragraph_id}] {passage.title}\n{passage.text}"
        for passage in passages
    )
    return (
        "Answer the question only from the evidence passages. Select every passage needed "
        "for the answer. The entity used to locate a source article is usually a bridge, "
        "not the final answer. Compare all passage titles and choose the subject that "
        "satisfies every constraint in the question. Copy passage keys exactly as the full "
        "strings inside square brackets. Return exactly one JSON object with this schema: "
        '{"answer":"short answer","passage_keys":["page_id:paragraph_id"]}. '
        f"Trace: {trace_id}\nQuestion: {question}\n\nEvidence:\n{context}"
    )
