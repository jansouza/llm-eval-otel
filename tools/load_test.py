"""Measure evaluator latency and per-process throughput.

1. Latency: p50/p99 of each built-in evaluator on a 10 KB text, called directly.
   ``system_prompt_leak`` compares 10 KB of system instructions with 10 KB of output.
2. Throughput: starts the real service as a subprocess, posts OTLP batches to it,
   and counts the evaluation events it exports to a local fake Collector. Throughput
   is inference spans fully evaluated (both heuristics) per second.
3. With ``--judge slow`` or ``--judge down``, ``relevance`` is on too, against the fake
   judge server answering after ``--judge-delay`` seconds, or against a closed port. The
   heuristics' throughput should match the run without a judge: the judge has its own lane.
4. With ``--jev fake``, ``slow`` or ``down``, the four ``jev_*`` checks are on too, against the
   fake server's System One API (at once, or after ``--jev-delay`` seconds) or a closed port.
   They have their own lane: ``--judge down --jev fake`` should drop no Jev check.

    uv run python tools/load_test.py --spans 5000 --text-kb 10
    uv run python tools/load_test.py --spans 3000 --judge slow --judge-delay 5
    uv run python tools/load_test.py --spans 3000 --judge down --jev fake --skip-latency
"""

import argparse
import asyncio
import dataclasses
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from fake_judge_server import FakeJudgeServer
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

from llm_eval_otel.evaluators.base import Evaluator, GenAIInteraction, Message
from llm_eval_otel.evaluators.output_format import OutputFormatValidator
from llm_eval_otel.evaluators.pii import PIIDetector
from llm_eval_otel.evaluators.prompt_leak import SystemPromptLeakDetector
from llm_eval_otel.evaluators.refusal import RefusalDetector
from llm_eval_otel.evaluators.secrets import SecretDetector

PROSE = (
    "o cliente pediu ajuda com a fatura do mês passado e perguntou sobre o prazo de entrega "
    "the assistant answered with the order status and a link to the tracking page "
    "config deploy pipeline token rotation password policy api key region bucket"
)
WORDS = PROSE.split()
FINDINGS = [
    "CPF 529.982.247-25",
    "maria@example.com",
    "AKIAIOSFODNN7EXAMPLE",
    "4111 1111 1111 1111",
    "CNPJ 11.222.333/0001-81",
    "(11) 98765-4321",
    "pix 123e4567-e89b-42d3-a456-426614174000",
]


def filler(size: int, rng: random.Random) -> str:
    out: list[str] = []
    length = 0
    while length < size:
        word = rng.choice(WORDS)
        out.append(word)
        length += len(word) + 1
    return " ".join(out)[:size]


def sample_text(size: int, rng: random.Random) -> str:
    """Mostly clean prose with one finding per ~3 KB, like a real prompt with some PII."""
    text = filler(size, rng)
    for n in range(max(1, size // 3000)):
        pos = rng.randrange(len(text))
        text = text[:pos] + " " + FINDINGS[n % len(FINDINGS)] + " " + text[pos:]
    return text[:size]


# --- 1. Evaluator latency ----------------------------------------------------------


def evaluator_latency(text_kb: int, runs: int) -> dict[str, tuple[float, float]]:
    rng = random.Random(1)
    size = text_kb * 1024
    interaction = GenAIInteraction(
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        parent_span_id=None,
        trace_flags=1,
        service_name="load",
        operation_name="chat",
        provider_name=None,
        request_model=None,
        response_id=None,
        system_instructions=[],
        input_messages=[Message("user", sample_text(size, rng))],
        output_messages=[],
    )
    output = Message("assistant", filler(size, rng))
    system = [Message("system", filler(size, rng))]
    # Each evaluator gets the content it reads, 10 KB of it.
    cases: list[tuple[Evaluator, GenAIInteraction]] = [
        (PIIDetector(), interaction),
        (SecretDetector(), interaction),
        (RefusalDetector(), dataclasses.replace(interaction, output_messages=[output])),
        (
            SystemPromptLeakDetector(),
            dataclasses.replace(interaction, system_instructions=system, output_messages=[output]),
        ),
        (
            OutputFormatValidator(),
            dataclasses.replace(
                interaction,
                output_type="json",
                output_messages=[Message("assistant", json.dumps({"text": output.text}))],
            ),
        ),
    ]
    results = {}
    for evaluator, case in cases:
        assert evaluator.applies_to(case), evaluator.name
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            asyncio.run(evaluator.evaluate(case))
            timings.append(time.perf_counter() - start)
        timings.sort()
        results[evaluator.name] = (
            statistics.median(timings) * 1000,
            timings[int(len(timings) * 0.99) - 1] * 1000,
        )
    return results


# --- 2. Throughput -----------------------------------------------------------------


HEURISTICS = frozenset({"pii_detection", "secret_detection"})
JEV_CHECKS = ("jev_relevance", "jev_refusal", "jev_toxicity", "jev_prompt_injection")


class FakeCollector(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, Handler)
        self.events = 0  # heuristic evaluation events
        self.judge_events: Counter[str] = Counter()  # judge evaluation events, by name
        self.judge_errors: Counter[str] = Counter()
        # latest llm_eval.evaluations.dropped, by (evaluation name, reason)
        self.dropped: dict[tuple[str, str], int] = {}
        self.lock = threading.Lock()

    def drops(self, names: Iterable[str]) -> dict[str, int]:
        by_reason: Counter[str] = Counter()
        for (name, reason), n in self.dropped.items():
            if name in names:
                by_reason[reason] += n
        return dict(by_reason)


def attribute(record: Any, key: str) -> str:
    return next((kv.value.string_value for kv in record.attributes if kv.key == key), "")


class Handler(BaseHTTPRequestHandler):
    server: FakeCollector

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.path == "/v1/logs":
            request = ExportLogsServiceRequest.FromString(body)
            records = [
                record
                for rl in request.resource_logs
                for sl in rl.scope_logs
                for record in sl.log_records
                if record.event_name == "gen_ai.evaluation.result"
            ]
            names = [attribute(r, "gen_ai.evaluation.name") for r in records]
            with self.server.lock:
                for record, name in zip(records, names, strict=True):
                    if name in HEURISTICS:
                        self.server.events += 1
                        continue
                    self.server.judge_events[name] += 1
                    if attribute(record, "error.type"):
                        self.server.judge_errors[name] += 1
        elif self.path == "/v1/metrics":
            metrics = ExportMetricsServiceRequest.FromString(body)
            for rm in metrics.resource_metrics:
                for sm in rm.scope_metrics:
                    for metric in sm.metrics:
                        if metric.name != "llm_eval.evaluations.dropped":
                            continue
                        with self.server.lock:
                            for point in metric.sum.data_points:
                                name = attribute(point, "gen_ai.evaluation.name")
                                reason = attribute(point, "llm_eval.drop.reason")
                                self.server.dropped[name, reason] = point.as_int
        self.send_response(200)
        self.send_header("content-type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


def kv(key: str, value: str) -> KeyValue:
    return KeyValue(key=key, value=AnyValue(string_value=value))


def batch(start: int, size: int, text_kb: int, rng: random.Random) -> bytes:
    spans = []
    for n in range(start, start + size):
        content = sample_text(text_kb * 1024, rng).replace('"', "'")
        span = Span(
            trace_id=rng.randbytes(16),
            span_id=n.to_bytes(8, "big"),
            name="chat",
            flags=1,
        )
        span.attributes.extend(
            [
                kv("gen_ai.operation.name", "chat"),
                kv("gen_ai.provider.name", "openai"),
                kv("gen_ai.request.model", "gpt-4o-mini"),
                kv(
                    "gen_ai.input.messages",
                    '[{"role":"user","parts":[{"type":"text","content":"' + content + '"}]}]',
                ),
                kv(
                    "gen_ai.output.messages",
                    '[{"role":"assistant","parts":[{"type":"text","content":"ok"}]}]',
                ),
            ]
        )
        spans.append(span)
    rs = ResourceSpans(scope_spans=[ScopeSpans(spans=spans)])
    rs.resource.attributes.append(kv("service.name", "load-test"))
    return ExportTraceServiceRequest(resource_spans=[rs]).SerializeToString()


def post(url: str, body: bytes) -> int:
    request = urllib.request.Request(
        url, data=body, headers={"content-type": "application/x-protobuf"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status)
    except urllib.error.HTTPError as err:
        return err.code


def wait_ready(url: str, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("service did not become ready")


DOWN = "http://127.0.0.1:9"  # nothing listens there: every call fails fast


def fake_server(mode: str, delay_s: float) -> FakeJudgeServer | None:
    if mode not in ("fake", "slow"):
        return None
    server = FakeJudgeServer(("127.0.0.1", 0), delay_s=delay_s if mode == "slow" else 0.0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def judge_env(
    judge: str,
    judge_delay_s: float,
    judge_rate: float,
    jev: str,
    jev_delay_s: float,
    jev_rate: float,
) -> tuple[dict[str, str], list[FakeJudgeServer]]:
    """The judges' settings, and the fake servers started for them."""
    evaluators = ["pii_detection", "secret_detection"]
    rates: dict[str, float] = {}
    env: dict[str, str] = {}
    servers = []
    if judge != "none":
        server = fake_server(judge, judge_delay_s)
        servers += [server] if server else []
        evaluators.append("relevance")
        rates["relevance"] = judge_rate
        env |= {
            "LLM_EVAL_LLM_JUDGE_MODEL": "fake-judge-1",
            "LLM_EVAL_LLM_JUDGE_BASE_URL": server.base_url if server else f"{DOWN}/v1",
            "OPENAI_API_KEY": "unused",
        }
    if jev != "none":
        server = fake_server(jev, jev_delay_s)
        servers += [server] if server else []
        evaluators += JEV_CHECKS
        rates |= dict.fromkeys(JEV_CHECKS, jev_rate)
        env |= {
            "LLM_EVAL_JEV_JUDGE_MODEL": "jev-1.13.0",
            "LLM_EVAL_JEV_JUDGE_BASE_URL": server.root_url if server else DOWN,
            "TYPESAFE_API_KEY": "unused",
        }
    if rates:
        env["LLM_EVAL_EVALUATORS"] = ",".join(evaluators)
        env["LLM_EVAL_SAMPLE_RATES"] = ",".join(f"{k}={v}" for k, v in rates.items())
    return env, servers


def throughput(
    spans: int, batch_size: int, text_kb: int, workers: int, judge: dict[str, str]
) -> tuple[float, int, FakeCollector]:
    collector = FakeCollector(("127.0.0.1", 0))
    threading.Thread(target=collector.serve_forever, daemon=True).start()
    service_port = 14318
    env = {
        **os.environ,
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{collector.server_address[1]}",
        "OTEL_BLRP_SCHEDULE_DELAY": "200",
        "OTEL_BSP_SCHEDULE_DELAY": "200",
        "OTEL_BLRP_MAX_QUEUE_SIZE": "100000",
        "OTEL_BSP_MAX_QUEUE_SIZE": "100000",
        "OTEL_METRIC_EXPORT_INTERVAL": "1000",
        "LLM_EVAL_HTTP_PORT": str(service_port),
        "LLM_EVAL_WORKERS": str(workers),
        **judge,
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "llm_eval_otel.main"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    retries_429 = 0
    try:
        base = f"http://127.0.0.1:{service_port}"
        wait_ready(f"{base}/readyz")
        rng = random.Random(3)
        bodies = [batch(n, batch_size, text_kb, rng) for n in range(0, spans, batch_size)]
        start = time.perf_counter()
        for body in bodies:
            while (status := post(f"{base}/v1/traces", body)) == 429:
                retries_429 += 1
                time.sleep(0.5)
            if status != 200:
                raise RuntimeError(f"unexpected status {status}")
        expected = 2 * spans
        while collector.events < expected:
            time.sleep(0.05)
        elapsed = time.perf_counter() - start
        time.sleep(3)  # let the judge lane work, and one more metric export land
    finally:
        process.terminate()
        process.wait(timeout=60)  # the shutdown drains the judge lane for up to 30 s
        collector.shutdown()
    return spans / elapsed, retries_429, collector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--text-kb", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--latency-runs", type=int, default=1000)
    parser.add_argument("--judge", choices=["none", "slow", "down"], default="none")
    parser.add_argument("--judge-delay", type=float, default=5.0, help="seconds per judge call")
    parser.add_argument("--judge-rate", type=float, default=0.05, help="relevance sample rate")
    parser.add_argument("--jev", choices=["none", "fake", "slow", "down"], default="none")
    parser.add_argument("--jev-delay", type=float, default=0.5, help="seconds per Jev call")
    parser.add_argument("--jev-rate", type=float, default=0.1, help="each jev_* check's rate")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    if not args.skip_latency:
        print(f"Evaluator latency on {args.text_kb} KB of text ({args.latency_runs} runs):")
        for name, (p50, p99) in evaluator_latency(args.text_kb, args.latency_runs).items():
            print(f"  {name:18s} p50 {p50:6.2f} ms   p99 {p99:6.2f} ms")

    judge, servers = judge_env(
        args.judge, args.judge_delay, args.judge_rate, args.jev, args.jev_delay, args.jev_rate
    )
    try:
        rate, retries, collector = throughput(
            args.spans, args.batch, args.text_kb, args.workers, judge
        )
    finally:
        for server in servers:
            server.shutdown()
    print(
        f"Throughput: {rate:.0f} spans/s per process "
        f"({args.spans} spans, {args.text_kb} KB each, batches of {args.batch}, "
        f"{args.workers} workers, {retries} retries after 429)"
    )
    if args.judge != "none":
        delay = f", {args.judge_delay:g} s per call" if args.judge == "slow" else ""
        print(
            f"Judge {args.judge}{delay}, relevance at {args.judge_rate:g}: "
            f"{collector.judge_events['relevance']} events "
            f"({collector.judge_errors['relevance']} with error.type), "
            f"dropped {collector.drops(['relevance']) or 0}"
        )
    if args.jev != "none":
        delay = f", {args.jev_delay:g} s per call" if args.jev == "slow" else ""
        events = sum(collector.judge_events[name] for name in JEV_CHECKS)
        errors = sum(collector.judge_errors[name] for name in JEV_CHECKS)
        print(
            f"Jev {args.jev}{delay}, each check at {args.jev_rate:g}: {events} events "
            f"({errors} with error.type), dropped {collector.drops(JEV_CHECKS) or 0}"
        )


if __name__ == "__main__":
    main()
