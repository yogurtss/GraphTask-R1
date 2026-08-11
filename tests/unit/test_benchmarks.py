import json
from pathlib import Path

from graphtask_r1.data import audit_records, prepare_benchmark
from graphtask_r1.utils import read_records


def test_webqsp_adapter_and_heldout_denylist(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = {
        "Questions": [
            {
                "QuestionId": "w1",
                "RawQuestion": "Who is connected?",
                "Parses": [
                    {
                        "TopicEntityMid": "m.seed",
                        "Sparql": "SELECT ?x WHERE { ns:m.seed ns:people.friend ?x }",
                        "Answers": [{"AnswerArgument": "m.answer", "EntityName": "Answer"}],
                    }
                ],
            }
        ]
    }
    (raw / "WebQSP.test.json").write_text(json.dumps(payload))
    output = tmp_path / "processed"
    metrics = prepare_benchmark("webqsp", raw, output, workers=2)
    assert metrics["examples"] == 1
    example = read_records(output / "test" / "examples.parquet")[0]
    assert example["topic_entity_ids"] == ["m.seed"]
    assert json.loads((output / "heldout_topic_entities.json").read_text()) == ["m.seed"]
    assert metrics["workers"] == 2


def _ssp_row(data_source: str, index: int, question: str, targets: list[str]) -> dict:
    return {
        "data_source": data_source,
        "reward_model": {"ground_truth": {"target": targets}},
        "extra_info": {"index": index, "question": question, "split": "test"},
    }


def test_ssp_adapter_filters_buckets_and_preserves_alias_groups(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        _ssp_row("searchR1_hotpotqa", 1, "Hotpot question?", ["The Answer", "Answer"]),
        _ssp_row("searchR1_triviaqa", 2, "Trivia question?", ["Edinburgh", "Edinburg"]),
        _ssp_row("searchR1_musique", 3, "Excluded?", ["No"]),
    ]
    (raw / "test.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    output = tmp_path / "processed"
    metrics = prepare_benchmark(
        "ssp",
        raw,
        output,
        include_datasets=("hotpotqa", "triviaqa"),
        workers=2,
    )

    assert metrics["examples"] == 2
    assert metrics["datasets"] == {"hotpotqa": 1, "triviaqa": 1}
    examples = read_records(output / "test" / "examples.parquet")
    assert examples[0]["answer_aliases"] == [["The Answer", "Answer"]]
    assert examples[0]["gold_answers"]["answers"] == [
        {"kind": "literal", "label": None, "value": "The Answer"}
    ]
    assert examples[0]["topic_entity_ids"] == []
    assert audit_records(output / "test" / "examples.parquet", kind="benchmark")["passed"]


def test_kilt_hotpot_and_official_trivia_formats(tmp_path: Path) -> None:
    hotpot_raw = tmp_path / "hotpot"
    hotpot_raw.mkdir()
    hotpot = {
        "id": "hp1",
        "input": "Which city?",
        "output": [
            {
                "answer": "Paris",
                "provenance": [{"wikipedia_id": 22989, "title": "Paris"}],
            },
            {"answer": "City of Paris", "provenance": []},
        ],
    }
    (hotpot_raw / "hotpot-dev-kilt.jsonl").write_text(json.dumps(hotpot) + "\n")
    hotpot_output = tmp_path / "hotpot-processed"
    prepare_benchmark("hotpotqa", hotpot_raw, hotpot_output)
    hotpot_example = read_records(hotpot_output / "dev" / "examples.parquet")[0]
    assert hotpot_example["topic_entity_ids"] == ["22989"]
    assert hotpot_example["answer_aliases"] == [["Paris", "City of Paris"]]

    trivia_raw = tmp_path / "trivia"
    trivia_raw.mkdir()
    trivia = {
        "Data": [
            {
                "QuestionId": "t1",
                "Question": "Which city?",
                "Answer": {"Value": "Paris", "Aliases": ["Paris", "City of Paris"]},
            }
        ]
    }
    (trivia_raw / "trivia-test.json").write_text(json.dumps(trivia))
    trivia_output = tmp_path / "trivia-processed"
    prepare_benchmark("triviaqa", trivia_raw, trivia_output)
    trivia_example = read_records(trivia_output / "test" / "examples.parquet")[0]
    assert trivia_example["answer_aliases"] == [["Paris", "City of Paris"]]
