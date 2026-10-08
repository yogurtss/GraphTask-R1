from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_boundary_pairs.py"
SPEC = importlib.util.spec_from_file_location("grr_score_boundary_pairs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hardness_selection_takes_top_scores_with_balanced_family_caps() -> None:
    scores = [
        {
            "task_id": f"{family}-{rank}",
            "boundary_family": family,
            "confusion_margin": float(10 - rank + family_index),
        }
        for family_index, family in enumerate(("root", "transform", "answer"))
        for rank in range(5)
    ]

    selected = MODULE.select_family_balanced_hard_boundaries(
        scores,
        keep_total=10,
        keep_per_family=4,
    )

    counts = Counter(str(value["boundary_family"]) for value in selected)
    assert len(selected) == 10
    assert set(counts.values()) == {3, 4}
    assert all(count <= 4 for count in counts.values())
    assert all(f"{family}-0" in {str(value["task_id"]) for value in selected} for family in counts)


def test_hardness_selection_excludes_nonpositive_confusion_margins() -> None:
    scores = [
        {
            "task_id": task_id,
            "boundary_family": family,
            "confusion_margin": margin,
        }
        for task_id, family, margin in (
            ("root-hard", "root", 0.4),
            ("root-easy", "root", -0.1),
            ("answer-hard", "answer", 0.2),
            ("answer-tie", "answer", 0.0),
        )
    ]

    selected = MODULE.select_family_balanced_hard_boundaries(
        scores,
        keep_total=2,
        keep_per_family=1,
    )

    assert {str(value["task_id"]) for value in selected} == {
        "root-hard",
        "answer-hard",
    }
