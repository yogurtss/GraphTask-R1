import json

import pytest
from pydantic import ValidationError

from graphtask_r1.schema import Entity, Hop, Intersect, parse_program, program_to_dict


def test_program_json_round_trip() -> None:
    program = Intersect(
        inputs=(
            Hop(input=Entity(entity_id="alice"), relation="works_at"),
            Hop(input=Entity(entity_id="bob"), relation="works_at"),
        )
    )
    encoded = json.loads(json.dumps(program_to_dict(program)))
    assert parse_program(encoded) == program


def test_invalid_intersection_has_clear_error() -> None:
    with pytest.raises(ValidationError, match="at least two"):
        Intersect(inputs=(Entity(entity_id="alice"),))
