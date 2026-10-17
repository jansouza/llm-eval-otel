"""The deduplication cache and the main queue's workers."""

import asyncio
import logging

import pytest
from judge_fakes import chat

from llm_eval_otel.engine.queue import DedupCache, EvaluationQueue, Sink
from llm_eval_otel.engine.runner import EvaluationRecord, Runner
from llm_eval_otel.evaluators.pii import PIIDetector

KEY_A, KEY_B, KEY_C = ((bytes([n]) * 16, bytes([n]) * 8) for n in (1, 2, 3))


def test_dedup_forgets_keys_after_the_ttl() -> None:
    now = 0.0
    cache = DedupCache(ttl_s=10, max_size=100, clock=lambda: now)
    cache.add(KEY_A)
    now = 9.0
    assert cache.seen(KEY_A)
    now = 10.0
    assert not cache.seen(KEY_A)
    assert len(cache) == 0


def test_dedup_evicts_the_oldest_key_beyond_max_size() -> None:
    cache = DedupCache(ttl_s=60, max_size=2, clock=lambda: 0.0)
    for key in (KEY_A, KEY_B, KEY_C):
        cache.add(key)
    assert len(cache) == 2
    assert not cache.seen(KEY_A)
    assert cache.seen(KEY_B) and cache.seen(KEY_C)


def queue(sink: Sink) -> EvaluationQueue:
    runner = Runner([PIIDetector()], default_timeout_s=1)
    dedup = DedupCache(ttl_s=60, max_size=100)
    return EvaluationQueue(runner, sink, max_size=10, workers=1, dedup=dedup)


async def test_an_async_sink_is_awaited() -> None:
    sunk: list[EvaluationRecord] = []

    async def sink(records: list[EvaluationRecord]) -> None:
        await asyncio.sleep(0)
        sunk.extend(records)

    q = queue(sink)
    q.start()
    q.submit([chat()])
    assert await q.drain(1)
    await q.stop()
    assert [r.evaluator_name for r in sunk] == ["pii_detection"]


async def test_a_failing_interaction_is_logged_and_the_worker_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sunk: list[EvaluationRecord] = []
    calls = 0

    def sink(records: list[EvaluationRecord]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("529.982.247-25")  # a message that could quote content
        sunk.extend(records)

    q = queue(sink)
    q.start()
    q.submit([chat(trace_id=b"\x01" * 16), chat(trace_id=b"\x02" * 16)])
    assert await q.drain(1)
    await q.stop()
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].endswith(": RuntimeError")
    assert "529.982.247-25" not in caplog.text
    assert len(sunk) == 1


async def test_drain_times_out_while_a_job_runs() -> None:
    async def slow_sink(records: list[EvaluationRecord]) -> None:
        await asyncio.sleep(1)

    q = queue(slow_sink)
    q.start()
    q.submit([chat()])
    assert not await q.drain(0.05)
    await q.stop()
