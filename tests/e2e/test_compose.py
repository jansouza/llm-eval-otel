"""End to end: docker compose up, one round of synthetic spans, read the Collector's file export.

    uv run pytest tests/e2e -m e2e

Needs Docker with the compose plugin, and no LLM API key: the relevance judge is the fake
server in tools/. Set LLM_EVAL_IMAGE to test a prebuilt image.
"""

import json
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "deploy" / "docker-compose.yaml"
TIMEOUT_S = 180
# Inference spans the generator sends in one round: clean, PII, secret, two for the tool
# call flow, three conversation turns, exempt service, OpenLLMetry, CNPJ/phone/PIX,
# refusal, system prompt leak, two in JSON mode, and a relevant and an off-topic answer.
INFERENCE_SPANS = 17
# Events per evaluator: each one skips the spans it doesn't apply to.
EXPECTED_EVENTS = {
    "pii_detection": INFERENCE_SPANS,
    "secret_detection": INFERENCE_SPANS,
    "refusal": INFERENCE_SPANS - 2,  # the tool call flow has no output text
    "system_prompt_leak": 3,  # the leak case and the two JSON calls
    "output_format": 2,  # the two JSON calls
    "relevance": INFERENCE_SPANS - 2,  # sampled at 1.0; the tool call flow has no answer
}
TOTAL_EVENTS = sum(EXPECTED_EVENTS.values())
JUDGE_MODEL = "fake-judge-1"
# PII and credentials: masked before the judge, and never in the output.
DETECTABLE = [
    "529.982.247-25",
    "52998224725",
    "maria@example.com",
    "joao@example.com",
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
    "11.222.333/0001-81",
    "11222333000181",
    "98765-4321",
    "123e4567-e89b-42d3-a456-426614174000",
]
# Other content the output must not carry (the judge may read it: it is not PII).
SENSITIVE = [*DETECTABLE, "PORTO-ALFA-77", "ouvidoria", "PED-58213"]


def attr_value(value: dict[str, Any]) -> Any:
    if "arrayValue" in value:
        return [attr_value(v) for v in value["arrayValue"].get("values", [])]
    for kind in ("stringValue", "boolValue", "doubleValue"):
        if kind in value:
            return value[kind]
    if "intValue" in value:
        return int(value["intValue"])
    return None


def attrs(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {a["key"]: attr_value(a["value"]) for a in items or []}


@dataclass
class Output:
    raw: str = ""
    judge_requests: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> "Output":
        out = cls()
        judge = path.with_name("judge-requests.jsonl")
        if judge.exists():
            out.judge_requests = judge.read_text()
        if not path.exists():
            return out
        out.raw = path.read_text()
        for line in out.raw.splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            for rl in data.get("resourceLogs", []):
                for sl in rl.get("scopeLogs", []):
                    for record in sl.get("logRecords", []):
                        if record.get("eventName") == "gen_ai.evaluation.result":
                            out.events.append({**record, "attrs": attrs(record.get("attributes"))})
            for rs in data.get("resourceSpans", []):
                service = attrs(rs.get("resource", {}).get("attributes"))["service.name"]
                for ss in rs.get("scopeSpans", []):
                    for span in ss.get("spans", []):
                        out.spans.append({**span, "service": service})
            for rm in data.get("resourceMetrics", []):
                for sm in rm.get("scopeMetrics", []):
                    for metric in sm.get("metrics", []):
                        out.metrics.setdefault(metric["name"], []).append(metric)
        return out

    def find(self, name: str, service: str) -> list[dict[str, Any]]:
        return [
            e
            for e in self.events
            if e["attrs"].get("gen_ai.evaluation.name") == name
            and e["attrs"].get("llm_eval.source.service.name") == service
        ]

    def last_sum(self, metric: str) -> int:
        """Latest cumulative value of a counter, summed over its attribute sets."""
        points = [
            p for m in self.metrics.get(metric, []) for p in m.get("sum", {}).get("dataPoints", [])
        ]
        latest: dict[str, int] = {}
        for p in sorted(points, key=lambda p: int(p["timeUnixNano"])):
            latest[json.dumps(p.get("attributes", []), sort_keys=True)] = int(p.get("asInt", 0))
        return sum(latest.values())

    def complete(self) -> bool:
        return (
            len(self.events) >= TOTAL_EVENTS
            and self.last_sum("llm_eval.spans.received") >= INFERENCE_SPANS
            and self.last_sum("llm_eval.evaluations") >= TOTAL_EVENTS
        )


def compose(*args: str, env: dict[str, str]) -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), *args], check=True, env=env, cwd=ROOT
    )


@pytest.fixture(scope="module")
def output(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Output]:
    out_dir = tmp_path_factory.mktemp("e2e-output")
    out_dir.chmod(0o777)  # the Collector runs as a non-root user
    env = {**os.environ, "E2E_OUTPUT_DIR": str(out_dir), "GENERATOR_EVERY": "0"}
    build = [] if os.environ.get("LLM_EVAL_IMAGE") else ["--build"]
    compose("up", "-d", *build, env=env)
    try:
        deadline = time.monotonic() + TIMEOUT_S
        result = Output()
        while time.monotonic() < deadline:
            result = Output.read(out_dir / "eval-output.jsonl")
            if result.complete():
                break
            time.sleep(3)
        time.sleep(6)  # one more metric export interval, so counters settle
        yield Output.read(out_dir / "eval-output.jsonl")
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "logs", "--no-color", "llm-eval-otel"],
            env=env,
            cwd=ROOT,
        )
        compose("down", "-v", "--remove-orphans", env=env)


def test_events_per_evaluator(output: Output) -> None:
    counts: dict[str, int] = {}
    for e in output.events:
        name = e["attrs"]["gen_ai.evaluation.name"]
        counts[name] = counts.get(name, 0) + 1
    assert counts == EXPECTED_EVENTS


def test_expected_labels(output: Output) -> None:
    labels = {
        (e["attrs"]["llm_eval.source.service.name"], e["attrs"]["gen_ai.evaluation.name"]): []
        for e in output.events
    }
    for e in output.events:
        key = (e["attrs"]["llm_eval.source.service.name"], e["attrs"]["gen_ai.evaluation.name"])
        labels[key].append(e["attrs"]["gen_ai.evaluation.score.label"])
    # support-bot: clean, PII, three turns with the CPF only in the first, CNPJ/phone/PIX,
    # refusal and leak.
    assert sorted(labels["support-bot", "pii_detection"]) == 3 * ["fail"] + 5 * ["pass"]
    # devops-bot: AWS key in the prompt; tool flow = clean first call, token in the second.
    assert sorted(labels["devops-bot", "secret_detection"]) == ["fail", "fail", "pass"]
    assert labels["bank-chatbot", "pii_detection"] == ["exempt"]
    assert labels["legacy-bot", "pii_detection"] == ["fail"]  # OpenLLMetry format
    assert sorted(labels["support-bot", "refusal"]) == ["fail"] + 7 * ["pass"]
    assert labels["support-bot", "system_prompt_leak"] == ["fail"]
    assert labels["orders-api", "system_prompt_leak"] == ["pass", "pass"]
    assert sorted(labels["orders-api", "output_format"]) == ["fail", "pass"]
    # The fake judge: the off-topic answer fails, every other answer passes.
    assert sorted(labels["store-bot", "relevance"]) == ["fail", "pass"]
    assert labels["support-bot", "relevance"] == 8 * ["pass"]
    assert labels["bank-chatbot", "relevance"] == ["exempt"]


def test_relevance_events(output: Output) -> None:
    [off_topic] = [
        e
        for e in output.find("relevance", "store-bot")
        if e["attrs"]["gen_ai.evaluation.score.label"] == "fail"
    ]
    attrs = off_topic["attrs"]
    assert attrs["gen_ai.evaluation.score.value"] == 0.0
    assert attrs["llm_eval.judge.raw_score"] == 1
    assert attrs["llm_eval.judge.model"] == JUDGE_MODEL
    assert attrs["llm_eval.evaluation.type"] == "llm_judge"
    assert attrs["gen_ai.evaluation.explanation"].startswith("fake judge:")
    [exempt] = output.find("relevance", "bank-chatbot")
    assert exempt["attrs"]["gen_ai.evaluation.explanation"] == "exempt service; not evaluated"


def test_judge_calls_are_spans_without_content(output: Output) -> None:
    evaluate_spans = {s["spanId"]: s for s in output.spans if s["name"] == "evaluate relevance"}
    calls = [s for s in output.spans if s["name"] == f"chat {JUDGE_MODEL}"]
    assert len(calls) == EXPECTED_EVENTS["relevance"] - 1  # the exempt one makes no call
    for call in calls:
        assert call["parentSpanId"] in evaluate_spans
        call_attrs = attrs(call.get("attributes"))
        assert call_attrs["server.address"] == "fake-judge"
        assert call_attrs["gen_ai.usage.input_tokens"] > 0
        assert not [k for k in call_attrs if k.startswith(("gen_ai.input", "gen_ai.output"))]
    assert "gen_ai.client.token.usage" in output.metrics
    assert "gen_ai.client.operation.duration" in output.metrics


def test_judge_received_no_detectable_value(output: Output) -> None:
    # One request per judged span: the exempt service's span never reached the judge.
    assert len(output.judge_requests.splitlines()) == EXPECTED_EVENTS["relevance"] - 1
    for value in DETECTABLE:
        assert value not in output.judge_requests
    assert "[CPF]" in output.judge_requests


def test_new_evaluator_attributes(output: Output) -> None:
    [new_pii] = [
        e
        for e in output.find("pii_detection", "support-bot")
        if "cnpj" in e["attrs"]["gen_ai.evaluation.explanation"]
    ]
    assert new_pii["attrs"]["llm_eval.pii.types"] == ["cnpj", "phone", "pix_key"]
    [refusal] = [
        e
        for e in output.find("refusal", "support-bot")
        if e["attrs"].get("llm_eval.refusal.source")
    ]
    assert refusal["attrs"]["llm_eval.refusal.language"] == "pt"
    [leak] = output.find("system_prompt_leak", "support-bot")
    assert leak["attrs"]["llm_eval.prompt_leak.longest_run"] >= 20
    [invalid] = [
        e
        for e in output.find("output_format", "orders-api")
        if e["attrs"]["gen_ai.evaluation.score.label"] == "fail"
    ]
    assert invalid["attrs"]["llm_eval.output_format.error"] == "truncated"


def test_severity_and_explanations(output: Output) -> None:
    [exempt] = output.find("pii_detection", "bank-chatbot")
    assert exempt["attrs"]["gen_ai.evaluation.explanation"] == "exempt service; cpf=1 (input)"
    assert "gen_ai.evaluation.score.value" not in exempt["attrs"]
    assert exempt["severityNumber"] == 9  # INFO
    tool_call = [
        e
        for e in output.find("secret_detection", "devops-bot")
        if "github_token" in e["attrs"]["gen_ai.evaluation.explanation"]
    ]
    assert len(tool_call) == 1
    # Found in the tool result (input) and in the tool call arguments (output).
    explanation = tool_call[0]["attrs"]["gen_ai.evaluation.explanation"]
    assert explanation == "github_token=1 (input), github_token=1 (output)"
    assert tool_call[0]["severityNumber"] == 13  # WARN


def test_child_span_is_linked_to_the_event(output: Output) -> None:
    for event in output.events:
        name = event["attrs"]["gen_ai.evaluation.name"]
        matches = [
            s
            for s in output.spans
            if s["name"] == f"evaluate {name}"
            and s["traceId"] == event["traceId"]
            and s["parentSpanId"] == event["spanId"]
        ]
        assert len(matches) == 1, event


def test_metrics_exported(output: Output) -> None:
    assert output.last_sum("llm_eval.evaluations") == TOTAL_EVENTS
    assert output.last_sum("llm_eval.spans.received") == INFERENCE_SPANS
    assert "llm_eval.evaluation.score" in output.metrics
    assert "llm_eval.evaluation.duration" in output.metrics


def test_evaluator_output_does_not_loop_back(output: Output) -> None:
    services = {e["attrs"].get("llm_eval.source.service.name") for e in output.events}
    assert "llm-eval-otel" not in services
    assert {s["service"] for s in output.spans} == {"llm-eval-otel"}


def test_no_sensitive_value_in_output(output: Output) -> None:
    for value in SENSITIVE:
        assert value not in output.raw
