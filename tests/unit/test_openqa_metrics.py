from types import SimpleNamespace

from graphtask_r1.evaluation import normalize_openqa_answer, openqa_alias_metrics
from graphtask_r1.schema import Answer, AnswerSet
from graphtask_r1.training.opponent import FrozenSolverService


def test_openqa_normalization_matches_coevokg_convention() -> None:
    assert normalize_openqa_answer("  The, City-of Paris! ") == "cityof paris"


def test_openqa_aliases_are_alternatives_not_required_set_members() -> None:
    predicted = AnswerSet(answers=(Answer(value="the answer", kind="literal"),))
    metrics = openqa_alias_metrics(predicted, (("Answer", "The Answer"),))
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}


def test_solver_benchmark_path_uses_alias_metrics_only_when_present() -> None:
    predicted = AnswerSet.entities(["Edinburgh"])
    task = SimpleNamespace(
        answer_aliases=(("Edinburgh", "City of Edinburgh"),),
        gold_answers=AnswerSet(answers=(Answer(value="City of Edinburgh", kind="literal"),)),
    )
    assert FrozenSolverService._answer_metrics(task, predicted)["exact_match"] == 1.0
