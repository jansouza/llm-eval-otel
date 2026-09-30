"""Last line of defense: no raw sensitive value leaves the service.

Every string attribute goes through the PII and secret detectors before the SDK
sees it. A match replaces the whole value with ``[REDACTED]``. This covers
third-party evaluators, such as an LLM judge quoting the text in its explanation.
"""

from collections.abc import Mapping, Sequence

from llm_eval_otel.evaluators.base import AttributeValue
from llm_eval_otel.evaluators.pii import find_pii
from llm_eval_otel.evaluators.secrets import find_secrets

REDACTED = "[REDACTED]"


def is_sensitive(value: str) -> bool:
    return bool(find_pii(value) or find_secrets(value))


def _clean_str(value: str) -> tuple[str, int]:
    return (REDACTED, 1) if is_sensitive(value) else (value, 0)


def sanitize(attributes: Mapping[str, AttributeValue]) -> tuple[dict[str, AttributeValue], int]:
    """Return the sanitized attributes and how many values were redacted."""
    clean: dict[str, AttributeValue] = {}
    redactions = 0
    for key, value in attributes.items():
        if isinstance(value, str):
            clean[key], n = _clean_str(value)
            redactions += n
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            items: list[str] = []
            is_str_list = all(isinstance(v, str) for v in value)
            if is_str_list:
                for item in value:
                    cleaned, n = _clean_str(str(item))
                    items.append(cleaned)
                    redactions += n
                clean[key] = items
            else:
                clean[key] = value
        else:
            clean[key] = value
    return clean, redactions
