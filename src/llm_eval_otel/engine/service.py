"""Wires extraction, the queue, the runner and the emitter together."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from llm_eval_otel import semconv
from llm_eval_otel.config import Settings
from llm_eval_otel.emit.emitter import Emitter
from llm_eval_otel.emit.sdk import Telemetry
from llm_eval_otel.engine.queue import DedupCache, EvaluationQueue
from llm_eval_otel.engine.runner import Runner
from llm_eval_otel.evaluators.base import Evaluator
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
        self.emitter = Emitter(
            telemetry,
            emit_spans=settings.emit_spans,
            association_exclude=settings.association_exclude,
        )
        self.runner = Runner(
            evaluators,
            default_timeout_s=settings.timeout_s,
            sample_rates=settings.sample_rates,
            exceptions=settings.exceptions,
        )
        self.queue = EvaluationQueue(
            self.runner,
            self.emitter.emit_all,
            max_size=settings.queue_max,
            workers=settings.workers,
            dedup=DedupCache(settings.dedup_ttl_s, DEDUP_MAX_KEYS),
            on_size_change=lambda delta: self.emitter.queue_size.add(delta),
        )
        self.accepting = False

    @property
    def ready(self) -> bool:
        return self.accepting and self.queue.fill_ratio <= 0.9

    def start(self) -> None:
        self.queue.start()
        self.accepting = True

    def ingest(self, request: ExportTraceServiceRequest) -> IngestOutcome:
        """Extract and enqueue; raises :class:`QueueFull` when there is no room."""
        result = extract(request)
        self.emitter.spans_received.add(result.received)
        for reason, count in result.skipped.items():
            self.emitter.spans_skipped.add(count, {semconv.LLM_EVAL_SKIP_REASON: reason})
        duplicates = self.queue.submit(result.interactions)
        if duplicates:
            self.emitter.spans_skipped.add(
                duplicates, {semconv.LLM_EVAL_SKIP_REASON: semconv.SKIP_DUPLICATE}
            )
        skipped = dict(result.skipped)
        if duplicates:
            skipped[semconv.SKIP_DUPLICATE] = duplicates
        return IngestOutcome(
            received=result.received,
            queued=len(result.interactions) - duplicates,
            skipped=skipped,
        )

    def record_invalid_payload(self) -> None:
        self.emitter.spans_skipped.add(
            1, {semconv.LLM_EVAL_SKIP_REASON: semconv.SKIP_INVALID_PAYLOAD}
        )

    async def drain(self, timeout_s: float | None = None) -> bool:
        return await self.queue.drain(timeout_s)

    async def shutdown(self) -> None:
        """Stop accepting, drain the queue for up to the drain timeout, flush the SDK."""
        self.accepting = False
        if not await self.drain(self.settings.drain_timeout_s):
            log.warning("drain timed out with %d interactions queued", self.queue.size)
        await self.queue.stop()
        self.telemetry.force_flush()
        self.telemetry.shutdown()
