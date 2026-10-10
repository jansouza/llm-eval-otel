"""Run the enabled evaluators on one interaction."""

import asyncio
import dataclasses
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from llm_eval_otel import semconv
from llm_eval_otel.evaluators.base import (
    JUDGE_KINDS,
    BatchEvaluator,
    EvaluationResult,
    Evaluator,
    EvaluatorKind,
    GenAIInteraction,
    Message,
    PartSpan,
)
from llm_eval_otel.judge.client import JudgeCall, recording

log = logging.getLogger(__name__)

ERROR_TIMEOUT = semconv.ERROR_TIMEOUT
EXEMPT_PREFIX = "exempt service; "
NOT_EVALUATED = "not evaluated"


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
    judge_calls: tuple[JudgeCall, ...] = ()


@dataclass(frozen=True)
class Job:
    """Evaluators chosen for an interaction: sampled, not exempt, input already cut.

    One evaluator, or a batch of :class:`BatchEvaluator` with the same ``batch_key`` and
    ``max_chars`` that one call answers.
    """

    evaluators: tuple[Evaluator, ...]
    interaction: GenAIInteraction  # as extracted; the records and the events refer to it
    evaluator_input: GenAIInteraction  # cut to the evaluators' max_chars
    truncated: bool


class JobLane(Protocol):
    def offer(self, job: Job) -> bool: ...


def lane_of(evaluator: Evaluator) -> str:
    """The lane an evaluator asks for (``lane``), or its kind."""
    lane = getattr(evaluator, "lane", None)
    return lane if isinstance(lane, str) else str(evaluator.kind)


def batch_key(evaluator: Evaluator) -> tuple[str, int | None] | None:
    """Evaluators with the same key, chosen for the same interaction, share one job."""
    if isinstance(evaluator, BatchEvaluator):
        return evaluator.batch_key, evaluator.max_chars
    return None


def ref(interaction: GenAIInteraction) -> str:
    """How log lines name an interaction: IDs and metadata, never content."""
    return (
        f"trace={interaction.trace_id.hex()} span={interaction.span_id.hex()} "
        f"service={interaction.service_name}"
    )


def job_name(job: Job) -> str:
    """How log lines name a job: its evaluators, ``+``-joined for a batch."""
    return "+".join(e.name for e in job.evaluators)


def sampled(trace_id: bytes, rate: float) -> bool:
    """Same rule as the OTel ProbabilitySampler: the last 7 bytes of the TraceID are R."""
    randomness = int.from_bytes(trace_id[-7:], "big")
    threshold = round((1 - rate) * 2**56)
    return randomness >= threshold


def _cut_message(message: Message, size: int) -> Message:
    parts = tuple(
        PartSpan(p.type, p.start, min(p.end, size)) for p in message.parts if p.start < size
    )
    return Message(message.role, message.text[:size], parts)


def truncate(interaction: GenAIInteraction, max_chars: int) -> tuple[GenAIInteraction, bool]:
    """Keep at most ``max_chars`` characters, in order: output, input, system.

    The output comes first because it is what evaluators with a limit (judges, classifiers)
    judge; the system instructions come last because they repeat on every call.
    """
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
                message = _cut_message(message, budget)
                truncated = True
            budget -= len(message.text)
            kept.append(message)
        return kept

    output = cut(interaction.output_messages)
    inputs = cut(interaction.input_messages)
    system = cut(interaction.system_instructions)
    result = dataclasses.replace(
        interaction, system_instructions=system, input_messages=inputs, output_messages=output
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


def log_result(
    name: str,
    job: Job,
    result: EvaluationResult,
    duration_ms: float,
    calls: Sequence[JudgeCall],
) -> None:
    """DEBUG: label, score and timings; never the explanation, which may be a judge's own words.

    Errors are DEBUG too: the summary counts them and a run of them is logged once as failing.
    """
    if not log.isEnabledFor(logging.DEBUG):
        return
    if result.error_type is not None:
        outcome = f"error={result.error_type}"
    else:
        outcome = f"label={result.label} score={result.score}"
    judge = ""
    if calls:
        tokens = [t for call in calls if (t := call.tokens_used) is not None]
        judge = f" judge_calls={len(calls)} tokens={sum(tokens) if tokens else 'unknown'}"
    log.debug(
        "%s %s: %s in %.1f ms%s%s",
        name,
        ref(job.interaction),
        outcome,
        duration_ms,
        " truncated" if job.truncated else "",
        judge,
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
        # Lanes, by name, that run their jobs apart instead of holding the queue worker.
        self.lanes: dict[str, JobLane] = {}

    def add_lane(self, name: str, lane: JobLane) -> None:
        self.lanes[name] = lane

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

    def prepare(self, evaluator: Evaluator, interaction: GenAIInteraction) -> Job:
        return self.prepare_batch([evaluator], interaction)

    def prepare_batch(self, evaluators: Sequence[Evaluator], interaction: GenAIInteraction) -> Job:
        """A batch shares one max_chars (part of its key), so one cut input serves all."""
        max_chars = evaluators[0].max_chars
        if max_chars is None:
            return Job(tuple(evaluators), interaction, interaction, truncated=False)
        evaluator_input, truncated = truncate(interaction, max_chars)
        return Job(tuple(evaluators), interaction, evaluator_input, truncated)

    async def run(self, interaction: GenAIInteraction) -> list[EvaluationRecord]:
        """Records of the evaluators run here; lane jobs are emitted by their lane."""
        records = []
        jobs = []
        batches: dict[tuple[str, int | None], list[Evaluator]] = {}
        for evaluator in self.select(interaction):
            if evaluator.kind in JUDGE_KINDS and self.is_exempt(evaluator, interaction):
                # Running a judge for an exempt service would pay to send content out.
                log.debug(
                    "%s %s: exempt service, judge not called", evaluator.name, ref(interaction)
                )
                records.append(self._exempt_without_call(evaluator, interaction))
                continue
            key = batch_key(evaluator)
            if key is None:
                jobs.append(self.prepare(evaluator, interaction))
            else:
                batches.setdefault(key, []).append(evaluator)
        jobs += [self.prepare_batch(batch, interaction) for batch in batches.values()]

        inline = []
        for job in jobs:
            lane_name = lane_of(job.evaluators[0])
            lane = self.lanes.get(lane_name)
            if lane is None:
                inline.append(job)
            elif lane.offer(job):  # a full lane drops and counts it; the heuristics go on
                log.debug(
                    "%s %s: queued in the %s lane", job_name(job), ref(interaction), lane_name
                )
        for job_records in await asyncio.gather(*(self.execute(job) for job in inline)):
            records += job_records
        return records

    def _exempt_without_call(
        self, evaluator: Evaluator, interaction: GenAIInteraction
    ) -> EvaluationRecord:
        now = time.time_ns()
        return EvaluationRecord(
            interaction=interaction,
            evaluator_name=evaluator.name,
            evaluator_kind=str(evaluator.kind),
            result=EvaluationResult(None, semconv.LABEL_EXEMPT, EXEMPT_PREFIX + NOT_EVALUATED),
            start_ns=now,
            end_ns=now,
        )

    async def _evaluate(self, job: Job) -> list[EvaluationResult]:
        evaluators = job.evaluators
        first = evaluators[0]
        if len(evaluators) > 1:
            assert isinstance(first, BatchEvaluator)  # only those are batched
            batch = [e for e in evaluators if isinstance(e, BatchEvaluator)]
            timeout = max(self.timeout(e) for e in evaluators)
            results = await asyncio.wait_for(
                type(first).evaluate_batch(batch, job.evaluator_input), timeout=timeout
            )
            if len(results) != len(evaluators):
                raise ValueError("evaluate_batch must return one result per evaluator")
            return results
        if first.kind == EvaluatorKind.HEURISTIC:
            # CPU-bound: keep the event loop free so ingestion and 429s keep answering.
            call = asyncio.to_thread(_run_in_thread, first, job.evaluator_input)
        else:
            call = first.evaluate(job.evaluator_input)
        return [await asyncio.wait_for(call, timeout=self.timeout(first))]

    async def execute(self, job: Job) -> list[EvaluationRecord]:
        """One record per evaluator in the job.

        A batch makes one call, so its judge calls go on the first record only: the call's
        span and client metrics appear once per request.
        """
        start_ns = time.time_ns()
        with recording() as judge_calls:
            try:
                results = await self._evaluate(job)
            except TimeoutError:
                results = [EvaluationResult(None, None, None, error_type=ERROR_TIMEOUT)] * len(
                    job.evaluators
                )
            except Exception as exc:  # an evaluator failure must not stop the others
                # Class name only: exception messages may quote evaluated content.
                error = EvaluationResult(None, None, None, error_type=type(exc).__name__)
                results = [error] * len(job.evaluators)
        end_ns = time.time_ns()

        records = []
        for n, (evaluator, result) in enumerate(zip(job.evaluators, results, strict=True)):
            if result.error_type is None and self.is_exempt(evaluator, job.interaction):
                result = exempt(result)
            calls = tuple(judge_calls) if n == 0 else ()
            log_result(evaluator.name, job, result, (end_ns - start_ns) / 1e6, calls)
            records.append(
                EvaluationRecord(
                    interaction=job.interaction,
                    evaluator_name=evaluator.name,
                    evaluator_kind=str(evaluator.kind),
                    result=result,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    truncated=job.truncated,
                    judge_calls=calls,
                )
            )
        return records
