"""Execution lanes: slow evaluators run apart from the main queue's workers.

A judge call takes seconds. If it ran in the queue worker, it would set the heuristics'
throughput, and a full queue would answer 429 for them too. The runner offers judge jobs to
a lane without waiting; the lane has its own bounded queue and a fixed number of workers,
which caps the concurrent calls. A full lane drops the job and counts it: the main queue
alone pushes back on the Collector, and the heuristics keep seeing every span.
"""

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable

from llm_eval_otel import semconv
from llm_eval_otel.engine.queue import Sink
from llm_eval_otel.engine.runner import EvaluationRecord, Job, ref
from llm_eval_otel.evaluators.base import GenAIInteraction

log = logging.getLogger(__name__)

Execute = Callable[[Job], Awaitable[EvaluationRecord]]
OnDrop = Callable[[str, str], None]  # (evaluation name, drop reason)

# A rough token count for a reservation, settled later with the usage the server reports.
CHARS_PER_TOKEN = 4


def estimate_tokens(interaction: GenAIInteraction, output_tokens: int) -> int:
    chars = sum(
        len(m.text)
        for messages in (
            interaction.system_instructions,
            interaction.context_messages,
            interaction.input_messages,
            interaction.output_messages,
        )
        for m in messages
    )
    return math.ceil(chars / CHARS_PER_TOKEN) + output_tokens


class TokenBudget:
    """A token bucket refilled at ``tokens_per_minute``, holding at most one minute's worth.

    A call reserves an estimate before it starts and settles with what the server reports
    afterwards, so the balance can go negative after an underestimate; new calls wait for
    it to refill.
    """

    def __init__(self, tokens_per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.capacity = float(tokens_per_minute)
        self.per_second = tokens_per_minute / 60
        self._clock = clock
        self._tokens = self.capacity
        self._updated = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.per_second)
        self._updated = now

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    def reserve(self, tokens: int) -> bool:
        self._refill()
        if tokens > self._tokens:
            return False
        self._tokens -= tokens
        return True

    def settle(self, reserved: int, used: int) -> None:
        self._refill()
        self._tokens = min(self.capacity, self._tokens + reserved - used)


class Lane:
    def __init__(
        self,
        name: str,
        execute: Execute,
        sink: Sink,
        *,
        max_size: int,
        concurrency: int,
        budget: TokenBudget | None = None,
        output_tokens: int = 0,
        on_drop: OnDrop | None = None,
        on_size_change: Callable[[int], None] | None = None,
    ) -> None:
        self.name = name
        self.execute = execute
        self.sink = sink
        self.max_size = max_size
        self.concurrency = concurrency
        self.budget = budget
        self.output_tokens = output_tokens
        self._on_drop = on_drop or (lambda name, reason: None)
        self._on_size_change = on_size_change or (lambda delta: None)
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=max_size)
        self._tasks: list[asyncio.Task[None]] = []
        self._dropping: set[str] = set()  # drop reasons already logged, until they clear

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def offer(self, job: Job) -> bool:
        """Queue the job without waiting; when the lane is full, drop and count it."""
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            self._drop(job, semconv.DROP_LANE_FULL)
            return False
        self._on_size_change(1)
        self._cleared(semconv.DROP_LANE_FULL, still_tight=self.size > self.max_size // 2)
        return True

    def _drop(self, job: Job, reason: str) -> None:
        """One WARNING when drops start; each drop is DEBUG and counted in the summary."""
        log.debug(
            "%s %s: dropped from the %s lane (%s)",
            job.evaluator.name,
            ref(job.interaction),
            self.name,
            reason,
        )
        if reason not in self._dropping:
            self._dropping.add(reason)
            log.warning("%s lane dropping evaluations (%s)", self.name, reason)
        self._on_drop(job.evaluator.name, reason)

    def _cleared(self, reason: str, *, still_tight: bool) -> None:
        # Cleared only with room to spare, so a lane hovering at its limit doesn't flap.
        if reason in self._dropping and not still_tight:
            self._dropping.discard(reason)
            log.info("%s lane no longer dropping evaluations (%s)", self.name, reason)

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._work(), name=f"llm-eval-{self.name}-{n}")
            for n in range(self.concurrency)
        ]

    async def _work(self) -> None:
        while True:
            job = await self._queue.get()
            self._on_size_change(-1)
            try:
                await self._process(job)
            except asyncio.CancelledError:
                self._on_drop(job.evaluator.name, semconv.DROP_SHUTDOWN)
                raise
            except Exception as exc:
                log.error("failed to process a %s job: %s", self.name, type(exc).__name__)
            finally:
                self._queue.task_done()

    async def _process(self, job: Job) -> None:
        reserved = 0
        if self.budget is not None:
            reserved = estimate_tokens(job.evaluator_input, self.output_tokens)
            if not self.budget.reserve(reserved):
                self._drop(job, semconv.DROP_BUDGET)
                return
            self._cleared(
                semconv.DROP_BUDGET, still_tight=self.budget.available < self.budget.capacity / 2
            )
        record = await self.execute(job)
        if self.budget is not None:
            used = [u for call in record.judge_calls if (u := call.tokens_used) is not None]
            # Without reported usage, the estimate stands.
            self.budget.settle(reserved, sum(used) if used else reserved)
        outcome = self.sink([record])
        if outcome is not None:
            await outcome

    async def drain(self, timeout_s: float | None = None) -> bool:
        """Wait until every queued job is processed; False on timeout."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout_s)
        except TimeoutError:
            return False
        return True

    async def stop(self) -> None:
        """Cancel the workers; what was running or still queued counts as a shutdown drop."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        pending = self._queue.qsize()
        while not self._queue.empty():
            job = self._queue.get_nowait()
            self._on_size_change(-1)
            self._on_drop(job.evaluator.name, semconv.DROP_SHUTDOWN)
            self._queue.task_done()
        if pending:
            # One line, not one per job: a backed-up lane can hold thousands.
            log.warning("dropped %d queued %s jobs at shutdown", pending, self.name)
