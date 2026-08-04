import json
from pathlib import Path

from graphtask_r1.data import prepare_benchmark
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
    metrics = prepare_benchmark("webqsp", raw, output)
    assert metrics["examples"] == 1
    example = read_records(output / "test" / "examples.parquet")[0]
    assert example["topic_entity_ids"] == ["m.seed"]
    assert json.loads((output / "heldout_topic_entities.json").read_text()) == ["m.seed"]
