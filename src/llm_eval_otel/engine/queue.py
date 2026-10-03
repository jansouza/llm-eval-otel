"""Bounded in-memory queue, deduplication and the workers that drain it."""

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable

from llm_eval_otel.engine.runner import EvaluationRecord, Runner, ref
from llm_eval_otel.evaluators.base import GenAIInteraction

log = logging.getLogger(__name__)

Sink = Callable[[list[EvaluationRecord]], Awaitable[None] | None]


class DedupCache:
    """(TraceID, SpanID) keys seen within the TTL, as an LRU bounded by ``max_size``.

    Collector retries resend whole batches; this keeps a resent span from being
    evaluated twice. It is per instance: across replicas the Collector routes by
    TraceID with the ``loadbalancing`` exporter.
    """

    def __init__(self, ttl_s: float, max_size: int, clock: Callable[[], float] = time.monotonic):
        self.ttl_s = ttl_s
        self.max_size = max_size
        self._clock = clock
        self._seen: OrderedDict[tuple[bytes, bytes], float] = OrderedDict()

    def _evict(self, now: float) -> None:
        while self._seen:
            key, expires = next(iter(self._seen.items()))
            if expires > now and len(self._seen) <= self.max_size:
                break
            del self._seen[key]

    def seen(self, key: tuple[bytes, bytes]) -> bool:
        now = self._clock()
        self._evict(now)
        expires = self._seen.get(key)
        return expires is not None and expires > now

    def add(self, key: tuple[bytes, bytes]) -> None:
        now = self._clock()
        self._seen[key] = now + self.ttl_s
        self._seen.move_to_end(key)
        self._evict(now)

    def __len__(self) -> int:
        return len(self._seen)


class QueueFull(Exception):
    pass


class EvaluationQueue:
    def __init__(
        self,
        runner: Runner,
        sink: Sink,
        *,
        max_size: int,
        workers: int,
        dedup: DedupCache,
        on_size_change: Callable[[int], None] | None = None,
    ) -> None:
        self.runner = runner
        self.sink = sink
        self.max_size = max_size
        self.workers = workers
        self.dedup = dedup
        self._on_size_change = on_size_change or (lambda delta: None)
        self._queue: asyncio.Queue[GenAIInteraction] = asyncio.Queue(maxsize=max_size)
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def fill_ratio(self) -> float:
        return self.size / self.max_size

    def submit(self, interactions: Iterable[GenAIInteraction]) -> int:
        """Enqueue new interactions; return how many were duplicates.

        Raises :class:`QueueFull` when the queue fills up. Interactions enqueued
        before that stay queued and are recognized as duplicates when the
        Collector retries the batch.
        """
        duplicates = 0
        for interaction in interactions:
            key = (interaction.trace_id, interaction.span_id)
            if self.dedup.seen(key):
                duplicates += 1
                continue
            try:
                self._queue.put_nowait(interaction)
            except asyncio.QueueFull:
                raise QueueFull from None
            self.dedup.add(key)
            self._on_size_change(1)
        return duplicates

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._work(), name=f"llm-eval-worker-{n}")
            for n in range(self.workers)
        ]

    async def _work(self) -> None:
        while True:
            interaction = await self._queue.get()
            self._on_size_change(-1)
            start = time.perf_counter()
            try:
                records = await self.runner.run(interaction)
                outcome = self.sink(records)
                if outcome is not None:
                    await outcome
                log.debug(
                    "interaction %s operation=%s model=%s: %d results in %.1f ms",
                    ref(interaction),
                    interaction.operation_name,
                    interaction.request_model,
                    len(records),
                    (time.perf_counter() - start) * 1000,
                )
            except Exception as exc:
                log.error(
                    "failed to process interaction %s: %s", ref(interaction), type(exc).__name__
                )
            finally:
                self._queue.task_done()

    async def drain(self, timeout_s: float | None = None) -> bool:
        """Wait until every queued interaction is processed; False on timeout."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout_s)
        except TimeoutError:
            return False
        return True

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
