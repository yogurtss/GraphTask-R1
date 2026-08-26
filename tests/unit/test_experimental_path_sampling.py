from graphtask_r1.experiments.path_sampling import ExperimentalPathSampler, PathSamplingConfig
from graphtask_r1.graph import toy_graph


def _sampler(seed: int = 7) -> ExperimentalPathSampler:
    graph = toy_graph()
    return ExperimentalPathSampler(
        graph,
        seed_entities=graph.all_entities(limit=100),
        allowed_relations=frozenset(
            relation.relation_id for relation in graph.all_relation_infos()
        ),
        config=PathSamplingConfig(seed=seed, max_depth=2, neighbor_limit=100),
    )


def test_experimental_sampler_is_reproducible_and_isolated() -> None:
    first = _sampler().run(strategy="bounded_path", attempts=40)
    second = _sampler().run(strategy="bounded_path", attempts=40)
    assert [candidate.to_dict() for candidate in first.candidates] == [
        candidate.to_dict() for candidate in second.candidates
    ]
    assert first.sampling_successes > 0


def test_family_balanced_improves_operator_coverage_over_path_only() -> None:
    path = _sampler().run(strategy="bounded_path", attempts=110)
    balanced = _sampler().run(strategy="family_balanced", attempts=110)
    path_operators = {
        operator for candidate in path.candidates for operator in candidate.operators
    }
    balanced_operators = {
        operator for candidate in balanced.candidates for operator in candidate.operators
    }
    assert path_operators < balanced_operators
    assert balanced.sampling_successes > 0
    assert all(1 <= len(candidate.answers) <= 20 for candidate in balanced.candidates)
