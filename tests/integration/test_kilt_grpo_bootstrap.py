import json
from pathlib import Path

import pyarrow.parquet as pq

from graphtask_r1.data import audit_records, bootstrap_kilt_grpo, prepare_kilt
from graphtask_r1.schema import TaskCertificate
from graphtask_r1.training.ms_swift_data import convert_rl_row
from graphtask_r1.utils import read_records


def _write_training_graph(path: Path) -> None:
    links = {
        "1": ("Alpha", ["2", "3"]),
        "2": ("Beta", ["4", "5"]),
        "3": ("Gamma", ["4", "6"]),
        "4": ("Delta", ["1"]),
        "5": ("Epsilon", ["1"]),
        "6": ("Zeta", ["2"]),
    }
    pages = []
    for page_id, (title, targets) in links.items():
        pages.append(
            {
                "_id": page_id,
                "wikipedia_id": page_id,
                "wikipedia_title": title,
                "text": [title, f"{title} is a fixture page."],
                "anchors": [
                    {"text": links[target][0], "wikipedia_id": target} for target in targets
                ],
                "categories": "Fixture pages",
                "history": {"revid": int(page_id)},
                "wikidata_info": {"wikidata_id": f"Q{page_id}"},
            }
        )
    path.write_text("\n".join(json.dumps(page) for page in pages) + "\n")


def test_kilt_bootstrap_exports_replayable_solver_grpo(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "kilt_knowledgesource.json"
    _write_training_graph(source)
    graph_output = tmp_path / "graph"
    prepare_kilt(source, graph_output, with_text_index=False)
    monkeypatch.setenv("GRAPHTASK_KILT_DB", str(graph_output / "graph.sqlite"))

    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = {
        "count": 6,
        "seed": 17,
        "pool_limit": 6,
        "max_attempts": 500,
        "min_degree": 1,
        "max_degree": 10,
        "val_ratio": 1 / 3,
    }
    first_metrics = bootstrap_kilt_grpo(first, **kwargs)
    second_metrics = bootstrap_kilt_grpo(second, **kwargs)

    assert first_metrics == second_metrics
    assert first_metrics["complete"] is True
    assert first_metrics["bootstrap_version"] == "kilt-certified-grpo-v2"
    assert first_metrics["interaction_mode"] == "graphscript"
    assert first_metrics["graphscript_version"] == "0.2"
    assert first_metrics["splits"] == {"train": 4, "val": 2}
    for split, expected in (("train", 4), ("val", 2)):
        task_path = first / split / "tasks.parquet"
        assert audit_records(task_path, kind="task")["passed"]
        task_rows = read_records(task_path)
        trace_rows = read_records(first / split / "traces.parquet")
        grpo_rows = pq.read_table(first / split / "solver_grpo.parquet").to_pylist()
        assert len(task_rows) == len(trace_rows) == len(grpo_rows) == expected
        assert task_rows == read_records(second / split / "tasks.parquet")
        assert trace_rows == read_records(second / split / "traces.parquet")
        assert grpo_rows == pq.read_table(second / split / "solver_grpo.parquet").to_pylist()
        traces = {trace["task_id"]: trace for trace in trace_rows}
        for task_row, grpo_row in zip(task_rows, grpo_rows, strict=True):
            task = TaskCertificate.model_validate(task_row)
            assert task.source == "kilt_bootstrap"
            assert task.graph_snapshot == "kilt-2019-08-01-v1"
            assert traces[task.task_id]["final_answers"] == task_row["gold_answers"]
            assert grpo_row["data_source"] == "graphtask/solver"
            assert grpo_row["extra_info"]["graph_snapshot"] == task.graph_snapshot
            converted = convert_rl_row(grpo_row)
            assert converted["uid"] == f"solver:{task.task_id}"
            assert grpo_row["extra_info"]["interaction_mode"] == "graphscript"
            assert grpo_row["extra_info"]["graphscript_version"] == "0.2"
            assert "Topic entities" not in grpo_row["prompt"][1]["content"]
            assert "tools" not in converted
            assert grpo_row["extra_info"]["text_search_enabled"] is True
            assert "tools_kwargs" not in grpo_row

    rejection_rows = read_records(first / "rejections.parquet")
    assert rejection_rows
    assert all(row["reason_code"] for row in rejection_rows)
