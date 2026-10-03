"""The evaluator contract.

Evaluators know nothing about OpenTelemetry: they receive an already extracted
interaction and return a result. ``emit`` turns results into telemetry.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from llm_eval_otel.semconv import PART_TEXT

AttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)


@dataclass(frozen=True)
class PartSpan:
    """Where one part of a message sits in :attr:`Message.text` (offsets, not a copy)."""

    type: str  # text | reasoning | tool_call | tool_call_response
    start: int
    end: int


@dataclass(frozen=True)
class Message:
    role: str  # system | user | assistant | tool | other
    text: str  # text, tool_call, tool_call_response and reasoning parts, joined by "\n"
    # Empty when the whole text is a single text part (the common case) or the split is
    # unknown; text_of() treats both as one text part.
    parts: tuple[PartSpan, ...] = ()

    def text_of(self, *types: str) -> str:
        """Only the parts of the given types, joined by ``"\\n"``."""
        if not self.parts:
            return self.text if PART_TEXT in types else ""
        return "\n".join(self.text[p.start : p.end] for p in self.parts if p.type in types)


@dataclass(frozen=True)
class GenAIInteraction:
    trace_id: bytes  # 16 bytes, as received in OTLP
    span_id: bytes  # 8 bytes
    parent_span_id: bytes | None
    trace_flags: int
    service_name: str | None  # resource service.name of the application
    operation_name: str  # gen_ai.operation.name
    provider_name: str | None  # gen_ai.provider.name
    request_model: str | None  # gen_ai.request.model
    response_id: str | None  # gen_ai.response.id
    system_instructions: list[Message]
    input_messages: list[Message]  # only what is new in this turn
    output_messages: list[Message]
    # traceloop.association.properties.*, keyed without the prefix
    association_properties: Mapping[str, str] = field(default_factory=dict)
    output_type: str | None = None  # gen_ai.output.type: text | json | image | speech
    finish_reasons: tuple[str, ...] = ()  # gen_ai.response.finish_reasons, one per output
    # Up to the last few user/assistant text messages before this turn, capped in size, so a
    # judge can read a follow-up question ("and in English?"). Heuristics ignore it.
    context_messages: list[Message] = field(default_factory=list)


class EvaluatorKind(StrEnum):
    HEURISTIC = "heuristic"
    MODEL = "model"
    LLM_JUDGE = "llm_judge"


@dataclass(frozen=True)
class EvaluationResult:
    score: float | None  # 0.0 to 1.0, higher is better; None when exempt or on error
    label: str | None  # pass | fail | exempt
    explanation: str | None  # never contains evaluated content
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)  # llm_eval.* keys only
    error_type: str | None = None


@runtime_checkable
class Evaluator(Protocol):
    name: str  # becomes gen_ai.evaluation.name
    kind: EvaluatorKind
    timeout_s: float
    sample_rate: float  # 1.0 = every span; heuristics stay at 1.0
    max_chars: int | None  # None = whole text; heuristics stay at None

    def applies_to(self, interaction: GenAIInteraction) -> bool: ...

    async def evaluate(self, interaction: GenAIInteraction) -> EvaluationResult: ...


# Where a finding appeared, used in explanations.
LOCATION_SYSTEM = "system"
LOCATION_INPUT = "input"
LOCATION_OUTPUT = "output"


def texts_by_location(interaction: GenAIInteraction) -> list[tuple[str, str]]:
    """Every text to scan, tagged with where it came from."""
    return [
        *((LOCATION_SYSTEM, m.text) for m in interaction.system_instructions),
        *((LOCATION_INPUT, m.text) for m in interaction.input_messages),
        *((LOCATION_OUTPUT, m.text) for m in interaction.output_messages),
    ]


def summarize_findings(counts: Mapping[tuple[str, str], int]) -> str:
    """Build an explanation from (type, location) counts only, e.g. ``cpf=1 (input)``.

    The template never includes the matched value, so the explanation is safe by
    construction.
    """
    return ", ".join(f"{kind}={n} ({location})" for (kind, location), n in counts.items())
