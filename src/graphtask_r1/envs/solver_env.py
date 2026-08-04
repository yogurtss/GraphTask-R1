from __future__ import annotations

import copy
import random
from typing import Any

from graphtask_r1.graph import GraphBackend
from graphtask_r1.schema import (
    AnswerSet,
    EpisodeInput,
    Observation,
    StepResult,
    ToolCall,
)


class SolverEnv:
    def __init__(
        self,
        backend: GraphBackend,
        *,
        max_turns: int = 8,
        max_invalid_calls: int = 2,
        max_observation_triples: int = 100,
    ) -> None:
        self.backend = backend
        self.max_turns = max_turns
        self.max_invalid_calls = max_invalid_calls
        self.max_observation_triples = max_observation_triples
        self._state: dict[str, Any] = {}

    def reset(self, sample: EpisodeInput, seed: int) -> Observation:
        random.Random(seed)  # explicit seed validation; environment itself is deterministic
        self._state = {
            "sample": sample.model_dump(mode="json"),
            "seed": seed,
            "step": 0,
            "invalid_calls": 0,
            "done": False,
            "calls": [],
            "observations": [],
            "final_answers": {"answers": []},
        }
        observation = Observation(step=0, message=sample.question)
        self._state["observations"].append(observation.model_dump(mode="json"))
        return observation

    def step(self, action: ToolCall) -> StepResult:
        if not self._state:
            raise RuntimeError("reset must be called before step")
        if self._state["done"]:
            raise RuntimeError("episode is already complete")
        self._state["step"] += 1
        self._state["calls"].append(action.model_dump(mode="json"))
        step = int(self._state["step"])
        try:
            if action.name == "search":
                entity_ids = [str(value) for value in action.arguments["entity_ids"]]
                direction = str(action.arguments.get("direction", "both"))
                relation_ids_raw = action.arguments.get("relation_ids")
                relation_ids = (
                    [str(value) for value in relation_ids_raw]
                    if isinstance(relation_ids_raw, list)
                    else None
                )
                triples = self.backend.neighbors(
                    entity_ids,
                    direction=direction,
                    relation_ids=relation_ids,
                    limit=self.max_observation_triples + 1,
                    trace_id=action.trace_id,
                )
                warnings: tuple[str, ...] = ()
                if len(triples) > self.max_observation_triples:
                    triples = triples[: self.max_observation_triples]
                    warnings = ("TRUNCATED",)
                observation = Observation(
                    step=step, message="search results", triples=tuple(triples), warnings=warnings
                )
            elif action.name == "inspect_entity":
                entity_id = str(action.arguments["entity_id"])
                observation = Observation(
                    step=step,
                    message="entity details",
                    entity=self.backend.entity_info(entity_id),
                )
            elif action.name == "final_answer":
                raw = action.arguments.get("answers", [])
                if isinstance(raw, str | int | float):
                    raw = [raw]
                sample = EpisodeInput.model_validate(self._state["sample"])
                if sample.gold_answers.answers and sample.gold_answers.answers[0].kind == "count":
                    answers = AnswerSet.count(int(raw[0])) if raw else AnswerSet()
                else:
                    answers = AnswerSet.entities([str(value) for value in raw])
                self._state["final_answers"] = answers.model_dump(mode="json")
                self._state["done"] = True
                observation = Observation(step=step, message="answer submitted", done=True)
            else:
                raise ValueError(f"unsupported tool: {action.name}")
        except (KeyError, TypeError, ValueError) as exc:
            self._state["invalid_calls"] += 1
            exhausted = int(self._state["invalid_calls"]) >= self.max_invalid_calls
            self._state["done"] = exhausted
            observation = Observation(
                step=step,
                message=f"tool error: {exc}",
                done=exhausted,
                warnings=("INVALID_CALL",),
            )
        if step >= self.max_turns and not self._state["done"]:
            self._state["done"] = True
            observation = observation.model_copy(
                update={"done": True, "warnings": (*observation.warnings, "MAX_TURNS")}
            )
        self._state["observations"].append(observation.model_dump(mode="json"))
        return StepResult(observation=observation, done=bool(self._state["done"]))

    def snapshot(self) -> dict[str, object]:
        return copy.deepcopy(self._state)

    def restore(self, state: dict[str, object]) -> None:
        EpisodeInput.model_validate(state["sample"])
        self._state = copy.deepcopy(state)

    @property
    def final_answers(self) -> AnswerSet:
        return AnswerSet.model_validate(self._state.get("final_answers", {"answers": []}))
