"""The judge lane: the heuristics never wait for it, and what it can't run is counted."""

import asyncio
import time

from conftest import OtelMemory, ServiceFactory
from judge_fakes import FakeJudgeClient, FakeSystemOneClient, chat, jev_checks, relevance
from opentelemetry._logs import SeverityNumber
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from otlp import chat_request

from llm_eval_otel.engine.lanes import Lane, TokenBudget, estimate_tokens
from llm_eval_otel.engine.runner import EvaluationRecord, Job, Runner
from llm_eval_otel.engine.service import Service
from llm_eval_otel.evaluators.base import EvaluationResult
from llm_eval_otel.evaluators.pii import PIIDetector
from llm_eval_otel.evaluators.secrets import SecretDetector
from llm_eval_otel.judge.client import JudgeCall

EVENT = "gen_ai.evaluation.result"
DROPPED = "llm_eval.evaluations.dropped"


def requests(n: int) -> list[ExportTraceServiceRequest]:
    return [chat_request("Qual o horário?", span_id=bytes([i + 1]) * 8) for i in range(n)]


def judge_service(
    make_service: ServiceFactory, client: FakeJudgeClient, **overrides: object
) -> Service:
    overrides.setdefault("sample_rates", {"relevance": 1.0})
    return make_service([PIIDetector(), SecretDetector(), relevance(client)], **overrides)


async def test_relevance_event_on_the_evaluated_span(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    client = FakeJudgeClient(score=2, reason="the answer is about something else")
    service = judge_service(make_service, client)
    service.ingest(chat_request("Qual o horário de atendimento?"))
    await service.drain()

    [event] = otel_memory.events(EVENT, name="relevance")
    attrs = dict(event.log_record.attributes or {})
    assert event.log_record.severity_number == SeverityNumber.WARN
    assert attrs["gen_ai.evaluation.score.value"] == 0.25
    assert attrs["gen_ai.evaluation.score.label"] == "fail"
    assert attrs["gen_ai.evaluation.explanation"] == "the answer is about something else"
    assert attrs["llm_eval.evaluation.type"] == "llm_judge"
    assert attrs["llm_eval.judge.model"] == "fake-judge-1"
    assert attrs["llm_eval.judge.raw_score"] == 2
    [evaluate] = otel_memory.spans("evaluate relevance")
    [chat_span] = otel_memory.spans("chat fake-judge-1")
    assert chat_span.parent is not None
    assert chat_span.parent.span_id == evaluate.context.span_id
    assert otel_memory.counter("llm_eval.lane.size", {"llm_eval.lane": "llm_judge"}) == 0


async def test_heuristics_do_not_wait_for_a_slow_judge(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    client = FakeJudgeClient(delay_s=2)
    service = judge_service(make_service, client, llm_judge_max_concurrency=2)
    started = time.monotonic()
    for request in requests(20):
        service.ingest(request)
    assert await service.queue.drain(1.0)  # the main queue empties without the judge
    assert time.monotonic() - started < 1.0
    assert len(otel_memory.events(EVENT, name="pii_detection")) == 20
    assert otel_memory.events(EVENT, name="relevance") == []
    await service.stop_workers()
    # The two running calls and the 18 queued jobs were cut by the shutdown.
    assert otel_memory.counter(DROPPED, {"llm_eval.drop.reason": "shutdown"}) == 20
    assert otel_memory.counter("llm_eval.lane.size") == 0


async def test_full_lane_drops_and_counts_without_429(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    client = FakeJudgeClient(delay_s=0.2)
    service = judge_service(
        make_service, client, llm_judge_queue_max=2, llm_judge_max_concurrency=1
    )
    for request in requests(10):
        service.ingest(request)  # QueueFull would raise here
    await service.drain(10)
    judged = len(otel_memory.events(EVENT, name="relevance"))
    full = otel_memory.counter(DROPPED, {"llm_eval.drop.reason": "lane_full"})
    assert full >= 1 and judged + full == 10
    assert len(otel_memory.events(EVENT, name="pii_detection")) == 10
    assert otel_memory.counter(DROPPED, {"gen_ai.evaluation.name": "relevance"}) == full
    drops = [m for m in otel_memory.caplog.messages if "dropped from the llm_judge lane" in m]
    assert len(drops) == full and all(m.endswith("(lane_full)") for m in drops)
    # One WARNING when the drops start, not one per drop.
    warnings = [r.message for r in otel_memory.caplog.records if r.levelname == "WARNING"]
    assert warnings == ["llm_judge lane dropping evaluations (lane_full)"]
    assert service.activity.results["relevance"]["dropped_lane_full"] == full


async def test_judge_timeout_is_an_error_event(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    evaluator = relevance(FakeJudgeClient(delay_s=1))
    evaluator.timeout_s = 0.05
    service = make_service([PIIDetector(), evaluator], sample_rates={"relevance": 1.0})
    service.ingest(chat_request("Qual o horário?"))
    await service.drain(5)
    [event] = otel_memory.events(EVENT, name="relevance")
    assert event.log_record.severity_number == SeverityNumber.ERROR
    assert (event.log_record.attributes or {})["error.type"] == "timeout"
    assert len(otel_memory.events(EVENT, name="pii_detection")) == 1


async def test_exempt_service_never_calls_the_judge(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    client = FakeJudgeClient()
    service = judge_service(make_service, client, exceptions={"bank-chatbot": ["relevance"]})
    service.ingest(chat_request("Qual o horário?", service_name="bank-chatbot"))
    await service.drain()
    assert client.received == []
    [event] = otel_memory.events(EVENT, name="relevance")
    attrs = event.log_record.attributes or {}
    assert attrs["gen_ai.evaluation.score.label"] == "exempt"
    assert attrs["gen_ai.evaluation.explanation"] == "exempt service; not evaluated"
    assert "gen_ai.evaluation.score.value" not in attrs
    assert otel_memory.spans("chat fake-judge-1") == []


async def test_no_budget_drops_without_calling(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    client = FakeJudgeClient()
    # 100 tokens per minute: less than one call's output allowance.
    service = judge_service(make_service, client, llm_judge_tokens_per_minute=100)
    for request in requests(3):
        service.ingest(request)
    await service.drain()
    assert client.received == []
    assert otel_memory.counter(DROPPED, {"llm_eval.drop.reason": "budget"}) == 3
    assert otel_memory.events(EVENT, name="relevance") == []


async def test_shutdown_drains_the_lane_after_the_queue(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    service = judge_service(make_service, FakeJudgeClient(delay_s=0.05))
    for request in requests(5):
        service.ingest(request)
    await service.shutdown()
    assert len(otel_memory.events(EVENT, name="relevance")) == 5
    assert otel_memory.counter(DROPPED) == 0


async def test_shutdown_counts_what_the_drain_timeout_left(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    service = judge_service(
        make_service, FakeJudgeClient(delay_s=5), drain_timeout_s=0.2, llm_judge_max_concurrency=1
    )
    for request in requests(3):
        service.ingest(request)
    await service.shutdown()
    assert otel_memory.counter(DROPPED, {"llm_eval.drop.reason": "shutdown"}) == 3
    assert len(otel_memory.events(EVENT, name="pii_detection")) == 3


def test_token_budget_refills_and_settles() -> None:
    now = [0.0]
    budget = TokenBudget(600, clock=lambda: now[0])  # 10 tokens per second
    assert budget.reserve(500)
    assert not budget.reserve(200)
    budget.settle(reserved=500, used=300)  # the call used less than reserved
    assert budget.available == 300
    now[0] = 60
    assert budget.available == 600  # never above one minute's worth
    budget.settle(reserved=0, used=900)  # an underestimate can go negative
    assert budget.available == -300
    now[0] = 90
    assert budget.available == 0


def call_with(input_tokens: int | None, output_tokens: int | None) -> JudgeCall:
    return JudgeCall("openai", "m", None, None, 0, 1, None, input_tokens, output_tokens)


async def test_lane_settles_the_budget_with_reported_usage() -> None:
    interaction = chat("x" * 400, "y" * 400)  # 200 tokens estimated + 100 for the output
    budget = TokenBudget(10_000)
    usage: list[JudgeCall] = []

    async def execute(job: Job) -> list[EvaluationRecord]:
        return [
            EvaluationRecord(
                job.interaction,
                "relevance",
                "llm_judge",
                EvaluationResult(1.0, "pass", None),
                0,
                1,
                judge_calls=tuple(usage),
            )
        ]

    lane = Lane(
        "llm_judge",
        execute,
        lambda records: None,
        max_size=10,
        concurrency=1,
        budget=budget,
        output_tokens=100,
    )
    job = Runner([], default_timeout_s=1).prepare(relevance(), interaction)
    assert estimate_tokens(interaction, 100) == 300

    usage[:] = [call_with(40, 10)]
    lane.start()
    lane.offer(job)
    await lane.drain(1)
    assert 9_940 <= budget.available <= 9_960  # 50 used, not the 300 reserved

    usage[:] = [call_with(None, None)]  # no usage reported: the estimate stands
    lane.offer(job)
    await lane.drain(1)
    assert 9_640 <= budget.available <= 9_665
    await lane.stop()


async def test_lane_jobs_run_concurrently_up_to_the_limit() -> None:
    running = 0
    peak = 0

    async def execute(job: Job) -> list[EvaluationRecord]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        return [
            EvaluationRecord(
                job.interaction, "r", "llm_judge", EvaluationResult(1, "pass", None), 0, 1
            )
        ]

    lane = Lane("llm_judge", execute, lambda records: None, max_size=100, concurrency=3)
    job = Runner([], default_timeout_s=1).prepare(relevance(), chat())
    lane.start()
    for _ in range(12):
        lane.offer(job)
    await lane.drain(5)
    await lane.stop()
    assert peak == 3


# --- The jev_judge lane -----------------------------------------------------------------

JEV_RATES = {
    "relevance": 1.0,
    "jev_relevance": 1.0,
    "jev_refusal": 1.0,
    "jev_toxicity": 1.0,
    "jev_prompt_injection": 1.0,
}


def both_judges(
    make_service: ServiceFactory,
    judge: FakeJudgeClient,
    jev: FakeSystemOneClient,
    **overrides: object,
) -> Service:
    overrides.setdefault("sample_rates", JEV_RATES)
    return make_service([PIIDetector(), relevance(judge), *jev_checks(jev)], **overrides)


async def test_jev_lane_exists_only_with_jev_checks(make_service: ServiceFactory) -> None:
    assert make_service([PIIDetector(), relevance()]).jev_lane is None
    service = make_service([PIIDetector(), *jev_checks()])
    assert service.jev_lane is not None
    assert [lane.name for lane in service.lanes] == ["llm_judge", "jev_judge"]


async def test_a_stuck_judge_does_not_drop_jev_checks(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    service = both_judges(
        make_service,
        FakeJudgeClient(delay_s=5),
        FakeSystemOneClient(),
        llm_judge_queue_max=1,
        llm_judge_max_concurrency=1,
        drain_timeout_s=1,
    )
    for request in requests(10):
        service.ingest(request)
    assert service.jev_lane is not None
    assert await service.queue.drain(2) and await service.jev_lane.drain(2)
    for name in JEV_RATES:
        if name != "relevance":
            assert len(otel_memory.events(EVENT, name=name)) == 10
            assert otel_memory.counter(DROPPED, {"gen_ai.evaluation.name": name}) == 0
    assert otel_memory.counter(DROPPED, {"gen_ai.evaluation.name": "relevance"}) >= 8
    assert otel_memory.counter("llm_eval.lane.size", {"llm_eval.lane": "jev_judge"}) == 0


async def test_a_stuck_jev_does_not_drop_the_judge(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    service = both_judges(
        make_service,
        FakeJudgeClient(),
        FakeSystemOneClient(delay_s=5),
        jev_judge_queue_max=1,
        jev_judge_max_concurrency=1,
    )
    for request in requests(10):
        service.ingest(request)
    assert await service.queue.drain(2) and await service.judge_lane.drain(2)
    assert len(otel_memory.events(EVENT, name="relevance")) == 10
    assert otel_memory.counter(DROPPED, {"gen_ai.evaluation.name": "relevance"}) == 0
    # Two jobs in the lane (one running, one queued), eight requests dropped: four checks each.
    full = {"llm_eval.drop.reason": "lane_full"}
    assert otel_memory.counter(DROPPED, {**full, "gen_ai.evaluation.name": "jev_refusal"}) == 8
    assert otel_memory.counter(DROPPED, full) == 32
    warnings = [r.message for r in otel_memory.caplog.records if r.levelname == "WARNING"]
    assert warnings == ["jev_judge lane dropping evaluations (lane_full)"]
    await service.stop_workers()
    shutdown = {"llm_eval.drop.reason": "shutdown"}
    assert otel_memory.counter(DROPPED, shutdown) == 8  # the two jobs, four checks each


async def test_jev_budget_drops_the_whole_batch_without_calling(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    client = FakeSystemOneClient()
    service = make_service(
        [PIIDetector(), *jev_checks(client)],
        sample_rates=JEV_RATES,
        jev_judge_tokens_per_minute=100,
    )
    service.ingest(chat_request("Qual o horário?"))
    await service.drain()
    assert client.received == []
    assert otel_memory.counter(DROPPED, {"llm_eval.drop.reason": "budget"}) == 4


async def test_lane_settles_a_batch_once() -> None:
    interaction = chat("x" * 400, "y" * 400)  # 200 tokens estimated
    budget = TokenBudget(10_000)
    checks = jev_checks()

    async def execute(job: Job) -> list[EvaluationRecord]:
        call = JudgeCall("typesafe", "jev", None, None, 0, 1, None, 90, 10)
        return [
            EvaluationRecord(
                job.interaction,
                e.name,
                "llm_judge",
                EvaluationResult(1.0, "pass", None),
                0,
                1,
                judge_calls=(call,) if n == 0 else (),
            )
            for n, e in enumerate(job.evaluators)
        ]

    sunk: list[EvaluationRecord] = []
    lane = Lane(
        "jev_judge",
        execute,
        sunk.extend,
        max_size=10,
        concurrency=1,
        budget=budget,
        output_tokens=128,
    )
    job = Runner([], default_timeout_s=1).prepare_batch(checks, interaction)
    lane.start()
    lane.offer(job)
    await lane.drain(1)
    await lane.stop()
    assert len(sunk) == 4
    assert 9_895 <= budget.available <= 9_905  # 100 used, not the 200 + 4 x 128 reserved
