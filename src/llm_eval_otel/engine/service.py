"""Wires extraction, the queue, the lanes, the runner and the emitter together."""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.resources import SERVICE_NAME

from llm_eval_otel import semconv
from llm_eval_otel.config import Settings
from llm_eval_otel.emit.emitter import Emitter
from llm_eval_otel.emit.sdk import Telemetry
from llm_eval_otel.engine.activity import Activity
from llm_eval_otel.engine.lanes import Lane, TokenBudget
from llm_eval_otel.engine.queue import DedupCache, EvaluationQueue, QueueFull
from llm_eval_otel.engine.runner import EvaluationRecord, Runner
from llm_eval_otel.evaluators.base import Evaluator, EvaluatorKind
from llm_eval_otel.extract.genai import extract

log = logging.getLogger(__name__)

# Upper bound on dedup keys kept in memory, whatever the TTL.
DEDUP_MAX_KEYS = 200_000


@dataclass(frozen=True)
class IngestOutcome:
    received: int
    queued: int
    skipped: dict[str, int]


class Service:
    def __init__(
        self, settings: Settings, telemetry: Telemetry, evaluators: Sequence[Evaluator]
    ) -> None:
        self.settings = settings
        self.telemetry = telemetry
        own_name = telemetry.tracer_provider.resource.attributes.get(SERVICE_NAME)
        self.service_name = own_name if isinstance(own_name, str) else None
        self.emitter = Emitter(
            telemetry,
            emit_spans=settings.emit_spans,
            association_exclude=settings.association_exclude,
        )
        self.activity = Activity()
        self.runner = Runner(
            evaluators,
            default_timeout_s=settings.timeout_s,
            sample_rates=settings.sample_rates,
            exceptions=settings.exceptions,
        )
        judge_lane = str(EvaluatorKind.LLM_JUDGE)
        tokens_per_minute = settings.judge_tokens_per_minute
        self.judge_lane = Lane(
            judge_lane,
            self.runner.execute,
            self._sink,
            max_size=settings.judge_queue_max,
            concurrency=settings.judge_max_concurrency,
            budget=TokenBudget(tokens_per_minute) if tokens_per_minute else None,
            output_tokens=settings.judge_max_output_tokens,
            on_drop=self._record_drop,
            on_size_change=lambda delta: self.emitter.lane_size.add(
                delta, {semconv.LLM_EVAL_LANE: judge_lane}
            ),
        )
        self.runner.add_lane(EvaluatorKind.LLM_JUDGE, self.judge_lane)
        self.lanes = [self.judge_lane]
        self.queue = EvaluationQueue(
            self.runner,
            self._sink,
            max_size=settings.queue_max,
            workers=settings.workers,
            dedup=DedupCache(settings.dedup_ttl_s, DEDUP_MAX_KEYS),
            on_size_change=lambda delta: self.emitter.queue_size.add(delta),
        )
        self.accepting = False
        self.rejecting = False  # answering 429 because the queue is full
        self._summary_task: asyncio.Task[None] | None = None

    def _sink(self, records: list[EvaluationRecord]) -> None:
        self.activity.add_records(records)
        self.emitter.emit_all(records)

    def _record_drop(self, evaluation_name: str, reason: str) -> None:
        self.activity.add_drop(evaluation_name, reason)
        self.emitter.record_drop(evaluation_name, reason)

    @property
    def ready(self) -> bool:
        # Only the main queue: a backed-up judge lane drops work, it doesn't push back.
        return self.accepting and self.queue.fill_ratio <= 0.9

    def start(self) -> None:
        self.queue.start()
        for lane in self.lanes:
            lane.start()
        if self.settings.log_summary_interval_s > 0:
            self._summary_task = asyncio.create_task(
                self._summarize(self.settings.log_summary_interval_s), name="llm-eval-summary"
            )
        self.accepting = True

    def log_summary(self) -> None:
        queues = [f"queue={self.queue.size}/{self.queue.max_size}"]
        queues += [f"{lane.name}={lane.size}/{lane.max_size}" for lane in self.lanes]
        self.activity.log_summary(" ".join(queues))

    async def _summarize(self, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            self.log_summary()

    def ingest(self, request: ExportTraceServiceRequest) -> IngestOutcome:
        """Extract and enqueue; raises :class:`QueueFull` when there is no room."""
        result = extract(request, self.service_name)
        self.emitter.spans_received.add(result.received)
        for reason, count in result.skipped.items():
            self.emitter.spans_skipped.add(count, {semconv.LLM_EVAL_SKIP_REASON: reason})
        try:
            duplicates = self.queue.submit(result.interactions)
        except QueueFull:
            if not self.rejecting:
                self.rejecting = True
                log.warning(
                    "queue full (%d interactions): answering 429 until it drains",
                    self.queue.max_size,
                )
            raise
        if self.rejecting and len(result.interactions) > duplicates:
            # Something got in; a batch with nothing new would pass a full queue too.
            self.rejecting = False
            log.info("queue accepting again (%d/%d)", self.queue.size, self.queue.max_size)
        if duplicates:
            self.emitter.spans_skipped.add(
                duplicates, {semconv.LLM_EVAL_SKIP_REASON: semconv.SKIP_DUPLICATE}
            )
        skipped = dict(result.skipped)
        if duplicates:
            skipped[semconv.SKIP_DUPLICATE] = duplicates
        outcome = IngestOutcome(
            received=result.received,
            queued=len(result.interactions) - duplicates,
            skipped=skipped,
        )
        self.activity.add_batch(outcome.received, outcome.queued, skipped)
        log.debug(
            "batch: received=%d queued=%d skipped=%s queue=%d/%d",
            outcome.received,
            outcome.queued,
            skipped or "none",
            self.queue.size,
            self.queue.max_size,
        )
        return outcome

    def record_rejected(self, status: int) -> None:
        self.activity.add_rejected(status)

    def record_invalid_payload(self) -> None:
        self.emitter.spans_skipped.add(
            1, {semconv.LLM_EVAL_SKIP_REASON: semconv.SKIP_INVALID_PAYLOAD}
        )

    async def drain(self, timeout_s: float | None = None) -> bool:
        """The main queue first (it feeds the lanes), then the lanes, within one timeout."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout_s is None else loop.time() + timeout_s

        def remaining() -> float | None:
            return None if deadline is None else max(0.0, deadline - loop.time())

        if not await self.queue.drain(remaining()):
            log.warning("drain timed out with %d interactions queued", self.queue.size)
            return False
        for lane in self.lanes:
            if not await lane.drain(remaining()):
                log.warning("drain timed out with %d %s jobs queued", lane.size, lane.name)
                return False
        return True

    async def stop_workers(self) -> None:
        """Cancel every worker; lane jobs still pending are counted as shutdown drops."""
        if self._summary_task is not None:
            self._summary_task.cancel()
            await asyncio.gather(self._summary_task, return_exceptions=True)
            self._summary_task = None
        await self.queue.stop()
        for lane in self.lanes:
            await lane.stop()

    async def shutdown(self) -> None:
        """Stop accepting, drain for up to the drain timeout, stop the workers, flush the SDK."""
        self.accepting = False
        if await self.drain(self.settings.drain_timeout_s):
            log.info("queue and lanes drained")
        await self.stop_workers()
        self.log_summary()  # what happened since the last one, shutdown drops included
        self.telemetry.force_flush()
        self.telemetry.shutdown()
