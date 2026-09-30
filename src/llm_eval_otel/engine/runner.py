"""Run the enabled evaluators on one interaction."""

import asyncio
import dataclasses
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llm_eval_otel import semconv
from llm_eval_otel.evaluators.base import (
    EvaluationResult,
    Evaluator,
    EvaluatorKind,
    GenAIInteraction,
    Message,
)

log = logging.getLogger(__name__)

ERROR_TIMEOUT = "timeout"
EXEMPT_PREFIX = "exempt service; "


@dataclass(frozen=True)
class EvaluationRecord:
    """One evaluator's result on one interaction, plus what ``emit`` needs around it."""

    interaction: GenAIInteraction
    evaluator_name: str
    evaluator_kind: str
    result: EvaluationResult
    start_ns: int
    end_ns: int
    truncated: bool = False


def sampled(trace_id: bytes, rate: float) -> bool:
    """Same rule as the OTel ProbabilitySampler: the last 7 bytes of the TraceID are R."""
    randomness = int.from_bytes(trace_id[-7:], "big")
    threshold = round((1 - rate) * 2**56)
    return randomness >= threshold


def truncate(interaction: GenAIInteraction, max_chars: int) -> tuple[GenAIInteraction, bool]:
    """Keep at most ``max_chars`` characters, in order: system, input, output."""
    budget = max_chars
    truncated = False

    def cut(messages: list[Message]) -> list[Message]:
        nonlocal budget, truncated
        kept = []
        for message in messages:
            if budget <= 0:
                truncated = True
                break
            if len(message.text) > budget:
                message = Message(message.role, message.text[:budget])
                truncated = True
            budget -= len(message.text)
            kept.append(message)
        return kept

    result = dataclasses.replace(
        interaction,
        system_instructions=cut(interaction.system_instructions),
        input_messages=cut(interaction.input_messages),
        output_messages=cut(interaction.output_messages),
    )
    return result, truncated


def exempt(result: EvaluationResult) -> EvaluationResult:
    """Keep what was found, drop the score, label it ``exempt``."""
    if result.label == semconv.LABEL_FAIL and result.explanation:
        found = result.explanation
    else:
        found = "no findings"
    return dataclasses.replace(
        result, score=None, label=semconv.LABEL_EXEMPT, explanation=EXEMPT_PREFIX + found
    )


def _run_in_thread(evaluator: Evaluator, interaction: GenAIInteraction) -> EvaluationResult:
    return asyncio.run(evaluator.evaluate(interaction))


class Runner:
    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        *,
        default_timeout_s: float,
        sample_rates: Mapping[str, float] | None = None,
        exceptions: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.evaluators = list(evaluators)
        self.default_timeout_s = default_timeout_s
        self.sample_rates = dict(sample_rates or {})
        self.exceptions = {
            service: frozenset(names) for service, names in (exceptions or {}).items()
        }

    def sample_rate(self, evaluator: Evaluator) -> float:
        return self.sample_rates.get(evaluator.name, evaluator.sample_rate)

    def timeout(self, evaluator: Evaluator) -> float:
        return evaluator.timeout_s if evaluator.timeout_s > 0 else self.default_timeout_s

    def is_exempt(self, evaluator: Evaluator, interaction: GenAIInteraction) -> bool:
        if interaction.service_name is None:
            return False
        return evaluator.name in self.exceptions.get(interaction.service_name, ())

    def select(self, interaction: GenAIInteraction) -> list[Evaluator]:
        return [
            e
            for e in self.evaluators
            if e.applies_to(interaction) and sampled(interaction.trace_id, self.sample_rate(e))
        ]

    async def run(self, interaction: GenAIInteraction) -> list[EvaluationRecord]:
        selected = self.select(interaction)
        return list(await asyncio.gather(*(self._run_one(e, interaction) for e in selected)))

    async def _run_one(
        self, evaluator: Evaluator, interaction: GenAIInteraction
    ) -> EvaluationRecord:
        truncated = False
        if evaluator.max_chars is not None:
            interaction_in, truncated = truncate(interaction, evaluator.max_chars)
        else:
            interaction_in = interaction

        start_ns = time.time_ns()
        try:
            if evaluator.kind == EvaluatorKind.HEURISTIC:
                # CPU-bound: keep the event loop free so ingestion and 429s keep answering.
                call = asyncio.to_thread(_run_in_thread, evaluator, interaction_in)
            else:
                call = evaluator.evaluate(interaction_in)
            result = await asyncio.wait_for(call, timeout=self.timeout(evaluator))
        except TimeoutError:
            result = EvaluationResult(None, None, None, error_type=ERROR_TIMEOUT)
        except Exception as exc:  # an evaluator failure must not stop the others
            # Class name only: exception messages may quote evaluated content.
            log.warning("evaluator %s failed: %s", evaluator.name, type(exc).__name__)
            result = EvaluationResult(None, None, None, error_type=type(exc).__name__)
        end_ns = time.time_ns()

        if result.error_type is None and self.is_exempt(evaluator, interaction):
            result = exempt(result)
        return EvaluationRecord(
            interaction=interaction,
            evaluator_name=evaluator.name,
            evaluator_kind=str(evaluator.kind),
            result=result,
            start_ns=start_ns,
            end_ns=end_ns,
            truncated=truncated,
        )
