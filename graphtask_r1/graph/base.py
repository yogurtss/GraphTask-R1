from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from graphtask_r1.graph.overlay import GraphOverlay
from graphtask_r1.schema import (
    AnswerSet,
    EntityInfo,
    GraphSlice,
    Program,
    RelationInfo,
    Triple,
    Witness,
)


class GraphBackend(Protocol):
    def all_entities(self, *, limit: int) -> tuple[str, ...]: ...

    def neighbors(
        self,
        entity_ids: Sequence[str],
        *,
        direction: str,
        relation_ids: Sequence[str] | None = None,
        limit: int = 100,
        trace_id: str | None = None,
    ) -> list[Triple]: ...

    def execute_program(self, program: Program) -> AnswerSet: ...

    def execute_sparql(self, sparql: str) -> AnswerSet: ...

    def entity_info(self, entity_id: str) -> EntityInfo: ...

    def relation_info(self, relation_id: str) -> RelationInfo: ...

    def extract_witness(self, program: Program, answers: AnswerSet) -> list[Witness]: ...

    def materialize(
        self, program: Program, *, max_nodes: int = 10_000, max_edges: int = 50_000
    ) -> GraphSlice: ...

    def with_overlay(self, overlay: GraphOverlay) -> GraphBackend: ...
