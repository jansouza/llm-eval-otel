"""``output_format``: the response is valid JSON when the client asked for JSON.

Syntax only: the pinned semconv has no attribute with the requested schema. Markdown
fences are not stripped: with ``gen_ai.output.type = json`` the provider ran in JSON mode,
and a fence means the mode was not honored.
"""

import json
from typing import Final

from llm_eval_otel import semconv
from llm_eval_otel.evaluators.base import (
    LOCATION_OUTPUT,
    EvaluationResult,
    EvaluatorKind,
    GenAIInteraction,
    Message,
)

ERROR_SYNTAX: Final = "syntax"
ERROR_EMPTY: Final = "empty"
ERROR_TRUNCATED: Final = "truncated"  # syntax error with finish_reason=length


def _has_text_part(message: Message) -> bool:
    return not message.parts or any(p.type == semconv.PART_TEXT for p in message.parts)


def json_error(text: str) -> tuple[str, str] | None:
    """``(error kind, detail)`` for invalid JSON, or None.

    The detail comes from ``JSONDecodeError.msg`` and ``.pos``, never ``.doc`` (the
    evaluated text), so it is safe to put in the explanation.
    """
    if not text.strip():
        return ERROR_EMPTY, "empty"
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        # Some messages already end in "at" ("Unterminated string starting at").
        at = "" if exc.msg.endswith(" at") else " at"
        return ERROR_SYNTAX, f"{exc.msg}{at} char {exc.pos}"
    return None


class OutputFormatValidator:
    name = "output_format"
    kind = EvaluatorKind.HEURISTIC
    timeout_s = 0.0  # 0 = use LLM_EVAL_TIMEOUT_S
    sample_rate = 1.0
    max_chars: int | None = None

    def applies_to(self, interaction: GenAIInteraction) -> bool:
        return interaction.output_type == semconv.OUTPUT_TYPE_JSON and any(
            _has_text_part(m) for m in interaction.output_messages
        )

    async def evaluate(self, interaction: GenAIInteraction) -> EvaluationResult:
        outputs = [m for m in interaction.output_messages if _has_text_part(m)]
        errors = [error for m in outputs if (error := json_error(m.text_of(semconv.PART_TEXT)))]
        total = len(outputs)
        score = (total - len(errors)) / total
        if not errors:
            return EvaluationResult(
                score=score,
                label=semconv.LABEL_PASS,
                explanation=f"valid_json={total} of {total} ({LOCATION_OUTPUT})",
            )
        kind, detail = errors[0]
        truncated = semconv.FINISH_REASON_LENGTH in interaction.finish_reasons
        if truncated and kind == ERROR_SYNTAX:
            kind = ERROR_TRUNCATED
        explanation = f"invalid_json={len(errors)} of {total} ({LOCATION_OUTPUT}): {detail}"
        if truncated:
            explanation += f", finish_reason={semconv.FINISH_REASON_LENGTH}"
        return EvaluationResult(
            score=score,
            label=semconv.LABEL_FAIL,
            explanation=explanation,
            attributes={semconv.LLM_EVAL_OUTPUT_FORMAT_ERROR: kind},
        )
