from __future__ import annotations

import random
from dataclasses import dataclass

from graphtask_r1.dsl import canonical_signature
from graphtask_r1.graph import InMemoryGraphBackend
from graphtask_r1.schema import (
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    LiteralValue,
    Program,
)


@dataclass(frozen=True)
class SampleRecord:
    index: int
    program: Program | None
    reason_code: str | None


class ProgramSampler:
    def __init__(self, backend: InMemoryGraphBackend, *, seed: int) -> None:
        self.backend = backend
        self.rng = random.Random(seed)

    def _hop(self, depth: int) -> Program | None:
        entities = sorted(
            {t.subject for t in self.backend.triples} | {t.object for t in self.backend.triples}
        )
        current: Program = Entity(entity_id=self.rng.choice(entities))
        for _ in range(depth):
            values = self.backend.execute_program(current).entity_ids()
            candidates = self.backend.neighbors(values, direction="both", limit=100)
            if not candidates:
                return None
            triple = self.rng.choice(candidates)
            if triple.subject in values:
                current = Hop(input=current, relation=triple.relation, direction="out")
            else:
                current = Hop(input=current, relation=triple.relation, direction="in")
        return current

    def _intersection(self) -> Program | None:
        grouped: dict[str, list[tuple[str, str]]] = {}
        for triple in self.backend.triples:
            grouped.setdefault(triple.object, []).append((triple.subject, triple.relation))
        viable = [(target, incoming) for target, incoming in grouped.items() if len(incoming) >= 2]
        if not viable:
            return None
        _, incoming = self.rng.choice(viable)
        left, right = self.rng.sample(incoming, 2)
        return Intersect(
            inputs=(
                Hop(input=Entity(entity_id=left[0]), relation=left[1], direction="out"),
                Hop(input=Entity(entity_id=right[0]), relation=right[1], direction="out"),
            )
        )

    def sample(self, count: int) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        families = ("hop1", "hop2", "filter_type", "filter_literal", "intersect", "count")
        for index in range(count):
            family = families[index % len(families)]
            base = self._hop(1 if family not in {"hop2"} else 2)
            program: Program | None = base
            if family == "filter_type" and base is not None:
                values = self.backend.execute_program(base).entity_ids()
                types = sorted(
                    {t for value in values for t in self.backend.entity_info(value).type_ids}
                )
                program = FilterType(input=base, type_id=self.rng.choice(types)) if types else None
            elif family == "filter_literal" and base is not None:
                values = self.backend.execute_program(base).entity_ids()
                literal_edges = [
                    t
                    for t in self.backend.triples
                    if t.subject in values and t.object.replace(".", "", 1).isdigit()
                ]
                if literal_edges:
                    edge = self.rng.choice(literal_edges)
                    program = FilterLiteral(
                        input=base,
                        relation=edge.relation,
                        comparator="ge",
                        value=LiteralValue(value=float(edge.object), datatype="number"),
                    )
                else:
                    program = None
            elif family == "intersect":
                program = self._intersection()
            elif family == "count" and base is not None:
                program = Count(input=base)
            reason = None
            if program is None:
                reason = "NO_VALID_EXTENSION"
            elif not self.backend.execute_program(program).answers:
                reason = "EMPTY_PARTIAL_PROGRAM"
                program = None
            records.append(SampleRecord(index=index, program=program, reason_code=reason))
        return records

    def signatures(self, count: int) -> tuple[str, ...]:
        return tuple(
            canonical_signature(record.program)
            for record in self.sample(count)
            if record.program is not None
        )
