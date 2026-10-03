"""Check a judge's output against the JSON schema it was asked to follow.

Covers the subset the judges' schemas use: ``type``, ``properties``, ``required``,
``additionalProperties: false``, ``enum``, ``minimum``, ``maximum`` and ``items``. The
output is checked in every response format mode, because not every OpenAI-compatible
server enforces the schema.
"""

from collections.abc import Mapping
from typing import Any


def _type_ok(value: Any, expected: str) -> bool:
    match expected:
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case "string":
            return isinstance(value, str)
        case "boolean":
            return isinstance(value, bool)
        case "integer":
            # Some local models write 4.0 for 4.
            return (isinstance(value, int) and not isinstance(value, bool)) or (
                isinstance(value, float) and value.is_integer()
            )
        case "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        case "null":
            return value is None
    return False


def is_valid(value: Any, schema: Mapping[str, Any]) -> bool:
    expected = schema.get("type")
    if expected is not None and not _type_ok(value, expected):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "minimum" in schema and value < schema["minimum"]:
        return False
    if "maximum" in schema and value > schema["maximum"]:
        return False
    if isinstance(value, dict):
        properties: Mapping[str, Any] = schema.get("properties", {})
        if any(name not in value for name in schema.get("required", ())):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        return all(is_valid(value[name], sub) for name, sub in properties.items() if name in value)
    if isinstance(value, list) and "items" in schema:
        return all(is_valid(item, schema["items"]) for item in value)
    return True
