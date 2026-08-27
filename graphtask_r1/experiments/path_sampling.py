from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import pyarrow.parquet as pq

from graphtask_r1.dsl import canonical_signature
from graphtask_r1.generation import verbalize
from graphtask_r1.graph import GraphBackend
from graphtask_r1.graphscript import (
    execute_graphscript,
    graphscript_operators,
    program_to_graphscript,
)
from graphtask_r1.schema import (
    Count,
    Entity,
    FilterLiteral,
    FilterType,
    Hop,
    Intersect,
    LiteralValue,
    Program,
    QueryAttribute,
    QueryRelation,
    SelectAmong,
    SelectBetween,
    TaskTrainingRecord,
    Union,
    Verify,
)
from graphtask_r1.utils import ProgressLogger
from graphtask_r1.verification import verify_task

SamplingStrategy = Literal["naive_path", "bounded_path", "family_balanced"]

FAMILY_ORDER = (
    "path",
    "count",
    "filter_type",
    "filter_literal",
    "query_attribute",
    "verify",
    "intersect",
    "union",
    "query_relation",
    "select_among",
    "select_between",
)


@dataclass(frozen=True)
class PathSamplingConfig:
    seed: int = 42
    max_depth: int = 3
    neighbor_limit: int = 200
    branch_trials: int = 24
    max_prefix_entities: int = 100
    min_answers: int = 1
    max_answers: int = 20
    follow_limit: int = 100
    max_edge_visits: int = 200
    max_returned_entities: int = 1_000
    bounded_retries: int = 4
    family_retries: int = 16

    def __post_init__(self) -> None:
        positive = {
            "max_depth": self.max_depth,
            "neighbor_limit": self.neighbor_limit,
            "branch_trials": self.branch_trials,
            "max_prefix_entities": self.max_prefix_entities,
            "min_answers": self.min_answers,
            "max_answers": self.max_answers,
            "follow_limit": self.follow_limit,
            "max_edge_visits": self.max_edge_visits,
            "max_returned_entities": self.max_returned_entities,
            "bounded_retries": self.bounded_retries,
            "family_retries": self.family_retries,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"positive sampling limits required: {', '.join(invalid)}")
        if self.min_answers > self.max_answers:
            raise ValueError("min_answers cannot exceed max_answers")


@dataclass(frozen=True)
class ReferenceProfile:
    rows: int
    seed_entities: tuple[str, ...]
    operator_counts: dict[str, int]
    relation_counts: dict[str, int]
    terminal_counts: dict[str, int]
    graphscript_lengths: tuple[int, ...]

    def summary(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "unique_seed_entities": len(self.seed_entities),
            "operators": self.operator_counts,
            "unique_operators": len(self.operator_counts),
            "relations": len(self.relation_counts),
            "terminals": self.terminal_counts,
            "graphscript_length": _distribution(self.graphscript_lengths),
        }


@dataclass(frozen=True)
class SampleCandidate:
    attempt: int
    strategy: SamplingStrategy
    family: str
    topic_entities: tuple[str, ...]
    program: Program
    question: str
    answers: tuple[str, ...]
    graphscript: dict[str, object]
    graphscript_length: int
    operators: tuple[str, ...]
    relations: tuple[str, ...]
    strict_certified: bool
    strict_rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "strategy": self.strategy,
            "family": self.family,
            "topic_entities": list(self.topic_entities),
            "program": self.program.model_dump(mode="json"),
            "program_signature": canonical_signature(self.program),
            "question": self.question,
            "answers": list(self.answers),
            "graphscript": self.graphscript,
            "graphscript_length": self.graphscript_length,
            "operators": list(self.operators),
            "relations": list(self.relations),
            "strict_certified": self.strict_certified,
            "strict_rejection_reasons": list(self.strict_rejection_reasons),
        }


@dataclass(frozen=True)
class SamplingExperiment:
    strategy: SamplingStrategy
    attempts: int
    proposal_trials: int
    constructed: int
    sampling_successes: int
    strict_certification_successes: int
    candidates: tuple[SampleCandidate, ...]
    rejection_counts: dict[str, int]
    family_attempts: dict[str, int]

    def report(self, reference: ReferenceProfile | None = None) -> dict[str, object]:
        operator_counts: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        family_successes: Counter[str] = Counter()
        lengths: list[int] = []
        for candidate in self.candidates:
            operator_counts.update(candidate.operators)
            relation_counts.update(candidate.relations)
            family_successes[candidate.family] += 1
            lengths.append(candidate.graphscript_length)
        target_operators = (
            set(reference.operator_counts)
            if reference is not None
            else set(graphscript_operators("0.3"))
        )
        covered_operators = set(operator_counts) & target_operators
        family_metrics = {
            family: {
                "attempts": attempts,
                "successes": family_successes[family],
                "success_rate": family_successes[family] / attempts if attempts else 0.0,
            }
            for family, attempts in sorted(self.family_attempts.items())
        }
        report: dict[str, object] = {
            "strategy": self.strategy,
            "attempts": self.attempts,
            "proposal_trials": self.proposal_trials,
            "constructed": self.constructed,
            "sampling_successes": self.sampling_successes,
            "sampling_success_rate": self.sampling_successes / self.attempts,
            "strict_certification_successes": self.strict_certification_successes,
            "strict_certification_rate": self.strict_certification_successes / self.attempts,
            "rejections": self.rejection_counts,
            "families": family_metrics,
            "operators": dict(sorted(operator_counts.items())),
            "operator_coverage": {
                "covered": len(covered_operators),
                "target": len(target_operators),
                "rate": len(covered_operators) / len(target_operators) if target_operators else 1.0,
                "missing": sorted(target_operators - covered_operators),
            },
            "unique_relations": len(relation_counts),
            "graphscript_length": _distribution(lengths),
        }
        if reference is not None:
            reference_total = sum(reference.relation_counts.values())
            covered_mass = sum(
                count
                for relation, count in reference.relation_counts.items()
                if relation in relation_counts
            )
            report["reference_relation_coverage"] = {
                "unique_covered": len(set(relation_counts) & set(reference.relation_counts)),
                "unique_target": len(reference.relation_counts),
                "unique_rate": (
                    len(set(relation_counts) & set(reference.relation_counts))
                    / len(reference.relation_counts)
                    if reference.relation_counts
                    else 1.0
                ),
                "weighted_rate": covered_mass / reference_total if reference_total else 1.0,
            }
            low, high = _reference_length_band(reference.graphscript_lengths)
            report["reference_length_band"] = {
                "p10": low,
                "p90": high,
                "within_rate": (
                    sum(low <= length <= high for length in lengths) / len(lengths)
                    if lengths
                    else 0.0
                ),
            }
        return report


class ExperimentalPathSampler:
    """Rule sampler kept separate from the production Questioner/self-play path."""

    def __init__(
        self,
        backend: GraphBackend,
        *,
        seed_entities: tuple[str, ...],
        allowed_relations: frozenset[str],
        config: PathSamplingConfig,
    ) -> None:
        if not seed_entities:
            raise ValueError("seed_entities cannot be empty")
        if not allowed_relations:
            raise ValueError("allowed_relations cannot be empty")
        self.backend = backend
        self.seed_entities = tuple(sorted(set(seed_entities)))
        self.allowed_relations = allowed_relations
        self.config = config

    def run(
        self,
        *,
        strategy: SamplingStrategy,
        attempts: int,
        progress: ProgressLogger | None = None,
    ) -> SamplingExperiment:
        if attempts < 1:
            raise ValueError("attempts must be positive")
        rng = random.Random(self.config.seed)
        candidates: list[SampleCandidate] = []
        rejections: Counter[str] = Counter()
        family_attempts: Counter[str] = Counter()
        proposal_trials = 0
        constructed = 0
        strict_certified = 0
        if progress is not None:
            progress.start(strategy=strategy, seed=self.config.seed)
        for attempt in range(attempts):
            family = self._family(strategy, attempt, rng)
            family_attempts[family] += 1
            retries = {
                "naive_path": 1,
                "bounded_path": self.config.bounded_retries,
                "family_balanced": self.config.family_retries,
            }[strategy]
            for _ in range(retries):
                proposal_trials += 1
                try:
                    program = self._construct(strategy, family, rng)
                    constructed += 1
                    candidate = self._evaluate(program, strategy, family, attempt)
                except SamplingRejection as exc:
                    rejections[exc.reason] += 1
                    continue
                candidates.append(candidate)
                strict_certified += int(candidate.strict_certified)
                break
            if progress is not None:
                completed = attempt + 1
                progress.update(
                    completed,
                    strategy=strategy,
                    family=family,
                    proposal_trials=proposal_trials,
                    constructed=constructed,
                    candidates=len(candidates),
                    strict_certified=strict_certified,
                    rejected_trials=sum(rejections.values()),
                    sampling_success_rate=round(len(candidates) / completed, 4),
                    strict_certification_rate=round(strict_certified / completed, 4),
                )
        if progress is not None:
            progress.finish(
                attempts,
                strategy=strategy,
                proposal_trials=proposal_trials,
                constructed=constructed,
                candidates=len(candidates),
                strict_certified=strict_certified,
                rejected_trials=sum(rejections.values()),
                sampling_success_rate=round(len(candidates) / attempts, 4),
                strict_certification_rate=round(strict_certified / attempts, 4),
            )
        return SamplingExperiment(
            strategy=strategy,
            attempts=attempts,
            proposal_trials=proposal_trials,
            constructed=constructed,
            sampling_successes=len(candidates),
            strict_certification_successes=strict_certified,
            candidates=tuple(candidates),
            rejection_counts=dict(sorted(rejections.items())),
            family_attempts=dict(sorted(family_attempts.items())),
        )

    def _family(self, strategy: SamplingStrategy, attempt: int, rng: random.Random) -> str:
        del rng
        if strategy != "family_balanced":
            return "path"
        return FAMILY_ORDER[attempt % len(FAMILY_ORDER)]

    def _construct(
        self,
        strategy: SamplingStrategy,
        family: str,
        rng: random.Random,
    ) -> Program:
        if strategy == "naive_path":
            return self._path(rng, constrained=False, min_entities=1)
        if family == "query_relation":
            return self._query_relation(rng)
        if family == "intersect":
            return self._intersection(rng)
        base = self._path(
            rng,
            constrained=True,
            min_entities=2 if family in {"select_among", "select_between"} else 1,
        )
        if family == "path":
            return base
        if family == "count":
            return Count(input=base)
        if family == "filter_type":
            return self._filter_type(base, rng)
        if family == "filter_literal":
            return self._filter_literal(base, rng)
        if family == "query_attribute":
            return self._query_attribute(base, rng)
        if family == "verify":
            query, value = self._query_attribute_with_value(base, rng)
            return Verify(input=query, comparator="eq", value=value)
        if family == "union":
            return self._union(base, rng)
        if family == "select_among":
            return self._select_among(base, rng)
        if family == "select_between":
            return self._select_between(base, rng)
        raise SamplingRejection("UNSUPPORTED_FAMILY")

    def _path(
        self,
        rng: random.Random,
        *,
        constrained: bool,
        min_entities: int,
    ) -> Program:
        seed = rng.choice(self.seed_entities)
        program: Program = Entity(entity_id=seed)
        depth = rng.randint(1, self.config.max_depth)
        for _ in range(depth):
            answers = self.backend.execute_program(program)
            entity_ids = answers.entity_ids()
            if not entity_ids:
                raise SamplingRejection("EMPTY_PREFIX")
            moves = self._moves(entity_ids)
            if not moves:
                raise SamplingRejection("NO_PATH_EXTENSION")
            rng.shuffle(moves)
            selected: Program | None = None
            trials = moves if constrained else moves[:1]
            for relation, direction in trials[: self.config.branch_trials]:
                candidate = Hop(input=program, relation=relation, direction=direction)
                if not constrained:
                    selected = candidate
                    break
                try:
                    size = len(self.backend.execute_program(candidate).answers)
                except (KeyError, RuntimeError, TypeError, ValueError):
                    continue
                if min_entities <= size <= self.config.max_prefix_entities:
                    selected = candidate
                    break
            if selected is None:
                raise SamplingRejection("NO_BOUNDED_EXTENSION")
            program = selected
        return program

    def _moves(self, entity_ids: tuple[str, ...]) -> list[tuple[str, Literal["out", "in"]]]:
        values = frozenset(entity_ids)
        facts = self.backend.neighbors(
            entity_ids,
            direction="both",
            limit=self.config.neighbor_limit,
            trace_id="path-sampling",
        )
        moves: set[tuple[str, Literal["out", "in"]]] = set()
        for fact in facts:
            if fact.relation not in self.allowed_relations:
                continue
            if fact.subject in values:
                moves.add((fact.relation, "out"))
            if fact.object in values:
                moves.add((fact.relation, "in"))
        return sorted(moves)

    def _entity_answers(self, program: Program) -> tuple[str, ...]:
        answers = self.backend.execute_program(program)
        entities = answers.entity_ids()
        if len(entities) != len(answers.answers):
            raise SamplingRejection("ENTITY_RESULT_REQUIRED")
        return entities

    def _filter_type(self, base: Program, rng: random.Random) -> Program:
        candidates: list[tuple[str, str]] = []
        for entity_id in self._entity_answers(base):
            candidates.extend(
                (entity_id, type_id)
                for type_id in self.backend.entity_info(entity_id).type_ids
            )
        if not candidates:
            raise SamplingRejection("NO_TYPE_EXTENSION")
        rng.shuffle(candidates)
        for _, type_id in candidates[: self.config.branch_trials]:
            program = FilterType(input=base, type_id=type_id)
            if self.backend.execute_program(program).answers:
                return program
        raise SamplingRejection("NO_TYPE_EXTENSION")

    def _attribute_candidates(self, base: Program) -> list[tuple[str, str, str]]:
        entities = self._entity_answers(base)
        facts = self.backend.attribute_facts(
            entities,
            limit=self.config.neighbor_limit,
            trace_id="path-sampling-attributes",
        )
        entity_set = frozenset(entities)
        return [
            (fact.subject, fact.relation, fact.object)
            for fact in facts
            if fact.subject in entity_set and fact.relation in self.allowed_relations
        ]

    def _filter_literal(self, base: Program, rng: random.Random) -> Program:
        facts = self._attribute_candidates(base)
        if not facts:
            raise SamplingRejection("NO_LITERAL_EXTENSION")
        rng.shuffle(facts)
        for _, relation, value in facts[: self.config.branch_trials]:
            program = FilterLiteral(
                input=base,
                relation=relation,
                comparator="eq",
                value=LiteralValue(value=value, datatype="string"),
            )
            try:
                if self.backend.execute_program(program).answers:
                    return program
            except (KeyError, RuntimeError, TypeError, ValueError):
                continue
        raise SamplingRejection("NO_LITERAL_EXTENSION")

    def _query_attribute(self, base: Program, rng: random.Random) -> Program:
        query, _ = self._query_attribute_with_value(base, rng)
        return query

    def _query_attribute_with_value(
        self, base: Program, rng: random.Random
    ) -> tuple[QueryAttribute, LiteralValue]:
        facts = self._attribute_candidates(base)
        if not facts:
            raise SamplingRejection("NO_ATTRIBUTE_EXTENSION")
        rng.shuffle(facts)
        for _, attribute, value in facts[: self.config.branch_trials]:
            program = QueryAttribute(input=base, attribute=attribute)
            try:
                if self.backend.execute_program(program).answers:
                    return program, LiteralValue(value=value, datatype="string")
            except (KeyError, RuntimeError, TypeError, ValueError):
                continue
        raise SamplingRejection("NO_ATTRIBUTE_EXTENSION")

    def _query_relation(self, rng: random.Random) -> Program:
        seed = rng.choice(self.seed_entities)
        facts = self.backend.neighbors(
            [seed], direction="both", limit=self.config.neighbor_limit, trace_id="path-relation"
        )
        facts = [fact for fact in facts if fact.relation in self.allowed_relations]
        if not facts:
            raise SamplingRejection("NO_RELATION_EXTENSION")
        fact = rng.choice(facts)
        return QueryRelation(
            subject=Entity(entity_id=fact.subject),
            object=Entity(entity_id=fact.object),
        )

    def _intersection(self, rng: random.Random) -> Program:
        pivot = rng.choice(self.seed_entities)
        facts = self.backend.neighbors(
            [pivot], direction="both", limit=self.config.neighbor_limit, trace_id="path-intersect"
        )
        branches: list[Program] = []
        for fact in facts:
            if fact.relation not in self.allowed_relations:
                continue
            if fact.object == pivot and fact.subject != pivot:
                branches.append(
                    Hop(
                        input=Entity(entity_id=fact.subject),
                        relation=fact.relation,
                        direction="out",
                    )
                )
            if fact.subject == pivot and fact.object != pivot:
                branches.append(
                    Hop(input=Entity(entity_id=fact.object), relation=fact.relation, direction="in")
                )
        rng.shuffle(branches)
        for left_index, left in enumerate(branches[: self.config.branch_trials]):
            for right in branches[left_index + 1 : self.config.branch_trials]:
                if _entity_roots(left) == _entity_roots(right):
                    continue
                program = Intersect(inputs=(left, right))
                try:
                    size = len(self.backend.execute_program(program).answers)
                except (KeyError, RuntimeError, TypeError, ValueError):
                    continue
                if self.config.min_answers <= size <= self.config.max_answers:
                    return program
        raise SamplingRejection("NO_INTERSECTION")

    def _union(self, base: Program, rng: random.Random) -> Program:
        roots = _entity_roots(base)
        if len(roots) != 1:
            raise SamplingRejection("UNION_REQUIRES_ONE_ROOT")
        seed = roots[0]
        moves = self._moves((seed,))
        rng.shuffle(moves)
        branches = [
            Hop(input=Entity(entity_id=seed), relation=relation, direction=direction)
            for relation, direction in moves[: self.config.branch_trials]
        ]
        for left_index, left in enumerate(branches):
            for right in branches[left_index + 1 :]:
                program = Union(inputs=(left, right))
                size = len(self.backend.execute_program(program).answers)
                if self.config.min_answers <= size <= self.config.max_answers:
                    return program
        raise SamplingRejection("NO_UNION")

    def _select_among(self, base: Program, rng: random.Random) -> Program:
        facts = self._attribute_candidates(base)
        by_attribute: dict[str, set[str]] = defaultdict(set)
        for entity_id, attribute, _ in facts:
            by_attribute[attribute].add(entity_id)
        attributes = [key for key, entities in by_attribute.items() if len(entities) >= 2]
        rng.shuffle(attributes)
        for attribute in attributes[: self.config.branch_trials]:
            program = SelectAmong(input=base, attribute=attribute, mode=rng.choice(("min", "max")))
            try:
                self.backend.execute_program(program)
            except (KeyError, RuntimeError, TypeError, ValueError):
                continue
            return program
        raise SamplingRejection("NO_ORDERABLE_ATTRIBUTE")

    def _select_between(self, base: Program, rng: random.Random) -> Program:
        facts = self._attribute_candidates(base)
        by_attribute: dict[str, set[str]] = defaultdict(set)
        for entity_id, attribute, _ in facts:
            by_attribute[attribute].add(entity_id)
        options = [
            (attribute, sorted(entities))
            for attribute, entities in by_attribute.items()
            if len(entities) >= 2
        ]
        rng.shuffle(options)
        for attribute, entities in options[: self.config.branch_trials]:
            left_id, right_id = rng.sample(entities, 2)
            program = SelectBetween(
                left=Entity(entity_id=left_id),
                right=Entity(entity_id=right_id),
                attribute=attribute,
                mode=rng.choice(("min", "max")),
            )
            try:
                self.backend.execute_program(program)
            except (KeyError, RuntimeError, TypeError, ValueError):
                continue
            return program
        raise SamplingRejection("NO_ORDERABLE_ATTRIBUTE")

    def _evaluate(
        self,
        program: Program,
        strategy: SamplingStrategy,
        family: str,
        attempt: int,
    ) -> SampleCandidate:
        try:
            script = program_to_graphscript(
                program,
                version="0.3",
                follow_limit=self.config.follow_limit,
            )
            execution = execute_graphscript(
                script,
                self.backend,
                allowed_relations=self.allowed_relations,
                max_edge_visits=self.config.max_edge_visits,
                max_returned_entities=self.config.max_returned_entities,
                trace_id=f"path-sampling-{strategy}-{attempt}",
            )
            certified_answers = self.backend.execute_program(program)
        except Exception as exc:
            raise SamplingRejection("EXECUTION_ERROR") from exc
        if execution.answers != certified_answers:
            raise SamplingRejection("BOUNDED_UNBOUNDED_MISMATCH")
        answer_count = len(certified_answers.answers)
        if not self.config.min_answers <= answer_count <= self.config.max_answers:
            raise SamplingRejection("INVALID_CARDINALITY")
        question = verbalize(program, self.backend)
        verification = verify_task(
            question,
            program,
            self.backend,
            min_answers=self.config.min_answers,
            max_answers=self.config.max_answers,
        )
        raw_script = cast(dict[str, object], script.model_dump(mode="json", by_alias=True))
        operators = tuple(op.op for op in script.ops)
        relations = tuple(sorted(_relations(program)))
        return SampleCandidate(
            attempt=attempt,
            strategy=strategy,
            family=family,
            topic_entities=_entity_roots(program),
            program=program,
            question=question,
            answers=tuple(str(answer.value) for answer in certified_answers.answers),
            graphscript=raw_script,
            graphscript_length=len(script.ops),
            operators=operators,
            relations=relations,
            strict_certified=verification.passed,
            strict_rejection_reasons=verification.rejection_reasons,
        )


class SamplingRejection(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def load_reference_profile(path: Path, *, max_topic_entities: int = 4) -> ReferenceProfile:
    if max_topic_entities < 1:
        raise ValueError("max_topic_entities must be positive")
    table = pq.read_table(path, columns=["record_json"])
    seeds: set[str] = set()
    operators: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    lengths: list[int] = []
    rows = 0
    for raw in table["record_json"].to_pylist():
        task = TaskTrainingRecord.model_validate_json(str(raw))
        if not 0 < len(task.topic_entities) <= max_topic_entities:
            continue
        try:
            script = program_to_graphscript(task.program, version="0.3")
        except ValueError:
            continue
        rows += 1
        seeds.update(entity.entity_id for entity in task.topic_entities)
        operators.update(op.op for op in script.ops)
        relations.update(_relations(task.program))
        terminals[task.program.op] += 1
        lengths.append(len(script.ops))
    if not rows:
        raise ValueError(f"no explicit-root reference tasks found in {path}")
    return ReferenceProfile(
        rows=rows,
        seed_entities=tuple(sorted(seeds)),
        operator_counts=dict(sorted(operators.items())),
        relation_counts=dict(sorted(relations.items())),
        terminal_counts=dict(sorted(terminals.items())),
        graphscript_lengths=tuple(lengths),
    )


def write_experiment(
    output_dir: Path,
    *,
    config: PathSamplingConfig,
    reference: ReferenceProfile,
    experiments: tuple[SamplingExperiment, ...],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [experiment.report(reference) for experiment in experiments]
    for experiment, report in zip(experiments, reports, strict=True):
        strategy_dir = output_dir / experiment.strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (strategy_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
            for candidate in experiment.candidates:
                handle.write(json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
    comparison: dict[str, object] = {
        "config": asdict(config),
        "reference": reference.summary(),
        "strategies": reports,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparison


def _entity_roots(program: Program) -> tuple[str, ...]:
    if isinstance(program, Entity):
        return (program.entity_id,)
    if isinstance(program, Intersect | Union):
        return tuple(
            sorted({value for branch in program.inputs for value in _entity_roots(branch)})
        )
    if isinstance(program, QueryRelation):
        return tuple(sorted({*_entity_roots(program.subject), *_entity_roots(program.object)}))
    if isinstance(program, SelectBetween):
        return tuple(sorted({*_entity_roots(program.left), *_entity_roots(program.right)}))
    child = getattr(program, "input", None)
    if child is None:
        return ()
    return _entity_roots(cast(Program, child))


def _relations(program: Program) -> set[str]:
    values: set[str] = set()
    dumped = program.model_dump(mode="json")

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key in ("relation", "attribute", "qualifier"):
                value = node.get(key)
                if isinstance(value, str):
                    values.add(value)
            for value in node.values():
                visit(value)
        elif isinstance(node, list | tuple):
            for value in node:
                visit(value)

    visit(dumped)
    return values


def _distribution(values: list[int] | tuple[int, ...]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": _percentile(ordered, 0.9),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _reference_length_band(values: tuple[int, ...]) -> tuple[int, int]:
    ordered = sorted(values)
    return math.floor(_percentile(ordered, 0.1)), math.ceil(_percentile(ordered, 0.9))


def _percentile(ordered: list[int], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
