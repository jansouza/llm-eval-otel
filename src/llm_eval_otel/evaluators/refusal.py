"""``refusal``: responses in which the model declines the request.

``fail`` here means the model refused, not that it misbehaved: refusing an abusive
request is the right behavior. Dashboards read the refusal rate per model and service.
"""

import re
import unicodedata
from typing import Final

from llm_eval_otel import semconv
from llm_eval_otel.evaluators.base import (
    EvaluationResult,
    EvaluatorKind,
    GenAIInteraction,
)
from llm_eval_otel.evaluators.refusal_phrases import PHRASES

SOURCE_PHRASE: Final = "phrase"
SOURCE_FINISH_REASON: Final = "finish_reason"

# Refusals open the response; looking only at the start avoids matching a phrase in the
# middle of a normal answer.
HEAD_CHARS: Final = 300

# Curly and modifier-letter apostrophes (U+2019, U+2018, U+02BC) become "'".
_APOSTROPHES = str.maketrans(dict.fromkeys((0x2019, 0x2018, 0x02BC), "'"))
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, no accents, straight apostrophes, single spaces."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_APOSTROPHES))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _SPACES.sub(" ", stripped).casefold()


def refusal_language(text: str) -> str | None:
    """The language of the first refusal phrase at the start of ``text``, if any."""
    head = normalize(text[:HEAD_CHARS])
    for language, pattern in PHRASES.items():
        if pattern.search(head):
            return language
    return None


def _output_texts(interaction: GenAIInteraction) -> list[str]:
    """Text parts only: reasoning and tool calls don't reach the user."""
    texts = (m.text_of(semconv.PART_TEXT) for m in interaction.output_messages)
    return [t for t in texts if t]


class RefusalDetector:
    name = "refusal"
    kind = EvaluatorKind.HEURISTIC
    timeout_s = 0.0  # 0 = use LLM_EVAL_TIMEOUT_S
    sample_rate = 1.0
    max_chars: int | None = None

    def applies_to(self, interaction: GenAIInteraction) -> bool:
        return bool(interaction.finish_reasons) or bool(_output_texts(interaction))

    async def evaluate(self, interaction: GenAIInteraction) -> EvaluationResult:
        # The provider's filter is the stronger signal and needs no text.
        filtered = interaction.finish_reasons.count(semconv.FINISH_REASON_CONTENT_FILTER)
        if filtered:
            return EvaluationResult(
                score=0.0,
                label=semconv.LABEL_FAIL,
                explanation=f"refusal={filtered} (output), source={SOURCE_FINISH_REASON}",
                attributes={semconv.LLM_EVAL_REFUSAL_SOURCE: SOURCE_FINISH_REASON},
            )
        languages = [
            language
            for text in _output_texts(interaction)
            if (language := refusal_language(text)) is not None
        ]
        if not languages:
            return EvaluationResult(score=1.0, label=semconv.LABEL_PASS, explanation="no refusal")
        return EvaluationResult(
            score=0.0,
            label=semconv.LABEL_FAIL,
            explanation=(
                f"refusal={len(languages)} (output), source={SOURCE_PHRASE}, lang={languages[0]}"
            ),
            attributes={
                semconv.LLM_EVAL_REFUSAL_SOURCE: SOURCE_PHRASE,
                semconv.LLM_EVAL_REFUSAL_LANGUAGE: languages[0],
            },
        )
