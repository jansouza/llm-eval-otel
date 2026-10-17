"""The JSON schema subset the judges' outputs are checked against."""

from typing import Any

import pytest

from llm_eval_otel.judge.schema import is_valid

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "label": {"type": "string", "enum": ["pass", "fail"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "label"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("value", "expected", "valid"),
    [
        ({}, "object", True),
        ([], "object", False),
        ([1], "array", True),
        ("x", "string", True),
        (1, "string", False),
        (True, "boolean", True),
        (1, "boolean", False),
        (4, "integer", True),
        (4.0, "integer", True),  # some local models write 4.0 for 4
        (4.5, "integer", False),
        (True, "integer", False),
        (0.5, "number", True),
        (2, "number", True),
        (False, "number", False),
        (None, "null", True),
        (0, "null", False),
        ("x", "unknown", False),
    ],
)
def test_types(value: Any, expected: str, valid: bool) -> None:
    assert is_valid(value, {"type": expected}) is valid


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ({"score": 3, "label": "pass"}, True),
        ({"score": 3, "label": "pass", "tags": ["a", "b"]}, True),
        ({"score": 3}, False),  # required
        ({"score": 3, "label": "pass", "extra": 1}, False),  # additionalProperties
        ({"score": 0, "label": "pass"}, False),  # minimum
        ({"score": 6, "label": "pass"}, False),  # maximum
        ({"score": 3, "label": "maybe"}, False),  # enum
        ({"score": 3, "label": "pass", "tags": ["a", 1]}, False),  # items
        ("not an object", False),
    ],
)
def test_object_schema(value: Any, valid: bool) -> None:
    assert is_valid(value, SCHEMA) is valid


def test_no_type_accepts_anything_else_unchecked() -> None:
    assert is_valid({"anything": 1}, {})
    assert is_valid([1, "x"], {"type": "array"})  # no items: elements unchecked
