from dataclasses import replace

import pytest
from conftest import OtelMemory
from opentelemetry._logs import SeverityNumber
from opentelemetry.trace import SpanKind, StatusCode

from llm_eval_otel.emit.emitter import Emitter
from llm_eval_otel.emit.sanitize import REDACTED, sanitize
from llm_eval_otel.engine.runner import EvaluationRecord
from llm_eval_otel.evaluators.base import EvaluationResult, GenAIInteraction, Message

TRACE_ID = bytes.fromhex("4bf92f3577b34da6a3ce929d0e0e4736")
SPAN_ID = bytes.fromhex("00f067aa0ba902b7")


def record(
    result: EvaluationResult, name: str = "pii_detection", truncated: bool = False
) -> EvaluationRecord:
    interaction = GenAIInteraction(
        trace_id=TRACE_ID,
        span_id=SPAN_ID,
        parent_span_id=None,
        trace_flags=1,
        service_name="support-bot",
        operation_name="chat",
        provider_name="openai",
        request_model="gpt-4o-mini",
        response_id="chatcmpl-123",
        system_instructions=[],
        input_messages=[Message("user", "x")],
        output_messages=[],
    )
    return EvaluationRecord(interaction, name, "heuristic", result, 1_000, 3_000_000, truncated)


def test_event_attributes_severity_and_ids(otel_memory: OtelMemory) -> None:
    emitter = Emitter(otel_memory.telemetry)
    emitter.emit(
        record(EvaluationResult(0.0, "fail", "cpf=1 (input)", {"llm_eval.pii.types": ["cpf"]}))
    )

    [event] = otel_memory.events("gen_ai.evaluation.result")
    log = event.log_record
    assert log.trace_id == int.from_bytes(TRACE_ID, "big")
    assert log.span_id == int.from_bytes(SPAN_ID, "big")
    assert log.severity_number == SeverityNumber.WARN
    assert dict(log.attributes or {}) == {
        "gen_ai.evaluation.name": "pii_detection",
        "gen_ai.evaluation.score.value": 0.0,
        "gen_ai.evaluation.score.label": "fail",
        "gen_ai.evaluation.explanation": "cpf=1 (input)",
        "gen_ai.response.id": "chatcmpl-123",
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "llm_eval.source.service.name": "support-bot",
        "llm_eval.evaluation.type": "heuristic",
        "llm_eval.pii.types": ("cpf",),
    }

    [span] = otel_memory.spans("evaluate pii_detection")
    assert span.kind == SpanKind.INTERNAL
    assert span.context.trace_id == int.from_bytes(TRACE_ID, "big")
    assert span.parent is not None and span.parent.span_id == int.from_bytes(SPAN_ID, "big")
    assert span.start_time == 1_000 and span.end_time == 3_000_000
    assert span.attributes == log.attributes


def test_severity_follows_result(otel_memory: OtelMemory) -> None:
    emitter = Emitter(otel_memory.telemetry)
    emitter.emit(record(EvaluationResult(1.0, "pass", "no findings"), name="a"))
    emitter.emit(record(EvaluationResult(None, "exempt", "exempt service; no findings"), name="b"))
    emitter.emit(record(EvaluationResult(None, None, None, error_type="timeout"), name="c"))
    severities = {
        (e.log_record.attributes or {})["gen_ai.evaluation.name"]: e.log_record.severity_number
        for e in otel_memory.events("gen_ai.evaluation.result")
    }
    assert severities == {
        "a": SeverityNumber.INFO,
        "b": SeverityNumber.INFO,
        "c": SeverityNumber.ERROR,
    }

    [error_span] = otel_memory.spans("evaluate c")
    assert error_span.status.status_code == StatusCode.ERROR
    assert error_span.attributes is not None
    assert error_span.attributes["error.type"] == "timeout"
    assert "gen_ai.evaluation.score.value" not in error_span.attributes


def test_metrics(otel_memory: OtelMemory) -> None:
    emitter = Emitter(otel_memory.telemetry)
    emitter.emit(record(EvaluationResult(0.0, "fail", "x")))
    emitter.emit(record(EvaluationResult(1.0, "pass", "y")))
    emitter.emit(record(EvaluationResult(None, "exempt", "z")))
    emitter.emit(record(EvaluationResult(None, None, None, error_type="timeout")))

    count = otel_memory.counter
    assert count("llm_eval.evaluations", {"gen_ai.evaluation.score.label": "fail"}) == 1
    assert count("llm_eval.evaluations", {"gen_ai.evaluation.score.label": "pass"}) == 1
    assert count("llm_eval.evaluations", {"gen_ai.evaluation.score.label": "exempt"}) == 1
    assert count("llm_eval.evaluations", {"error.type": "timeout"}) == 1
    assert count("llm_eval.evaluations", {"llm_eval.source.service.name": "support-bot"}) == 4
    # Neither exempt nor error reach the score histogram.
    assert otel_memory.histogram_count("llm_eval.evaluation.score") == 2
    assert otel_memory.histogram_count("llm_eval.evaluation.duration") == 4

    otel_memory.serialize_all()
    metrics = "\n".join(otel_memory._metrics_json)
    for high_cardinality in (TRACE_ID.hex(), "chatcmpl-123"):
        assert high_cardinality not in metrics


def test_association_properties(otel_memory: OtelMemory) -> None:
    rec = record(EvaluationResult(0.0, "fail", "x"))
    rec = replace(
        rec,
        interaction=replace(
            rec.interaction,
            association_properties={"scenario": "pix", "correlation_id": "req-8f3a"},
        ),
    )
    Emitter(otel_memory.telemetry, association_exclude=["correlation_id"]).emit(rec)

    # Event and span keep every property, so the trace can be found by any of them.
    [event] = otel_memory.events("gen_ai.evaluation.result")
    attrs = event.log_record.attributes or {}
    assert attrs["traceloop.association.properties.scenario"] == "pix"
    assert attrs["traceloop.association.properties.correlation_id"] == "req-8f3a"
    [span] = otel_memory.spans("evaluate pii_detection")
    assert span.attributes == event.log_record.attributes

    # Metrics drop the excluded per-request key.
    scenario = {"traceloop.association.properties.scenario": "pix"}
    assert otel_memory.counter("llm_eval.evaluations", scenario) == 1
    assert otel_memory.histogram_count("llm_eval.evaluation.score", scenario) == 1
    otel_memory.serialize_all()
    assert "req-8f3a" not in "\n".join(otel_memory._metrics_json)


def test_evaluator_attributes_outside_llm_eval_are_dropped(otel_memory: OtelMemory) -> None:
    Emitter(otel_memory.telemetry).emit(
        record(EvaluationResult(1.0, "pass", None, {"gen_ai.made_up": "x", "llm_eval.ok": 1}))
    )
    [event] = otel_memory.events("gen_ai.evaluation.result")
    attrs = event.log_record.attributes or {}
    assert "gen_ai.made_up" not in attrs
    assert attrs["llm_eval.ok"] == 1


def test_truncated_flag(otel_memory: OtelMemory) -> None:
    Emitter(otel_memory.telemetry).emit(record(EvaluationResult(1.0, "pass", None), truncated=True))
    [event] = otel_memory.events("gen_ai.evaluation.result")
    assert (event.log_record.attributes or {})["llm_eval.content.truncated"] is True


def test_child_span_can_be_disabled(otel_memory: OtelMemory) -> None:
    Emitter(otel_memory.telemetry, emit_spans=False).emit(
        record(EvaluationResult(1.0, "pass", None))
    )
    assert otel_memory.span_exporter.get_finished_spans() == ()
    assert len(otel_memory.events("gen_ai.evaluation.result")) == 1


def test_sanitizer_redacts_third_party_explanation(otel_memory: OtelMemory) -> None:
    leaky = EvaluationResult(
        0.2,
        "fail",
        "the user wrote 529.982.247-25 and sk-proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv",
        {"llm_eval.quotes": ["ok", "mail me at a@b.co"]},
    )
    Emitter(otel_memory.telemetry).emit(record(leaky, name="judge"))
    [event] = otel_memory.events("gen_ai.evaluation.result")
    attrs = event.log_record.attributes or {}
    assert attrs["gen_ai.evaluation.explanation"] == REDACTED
    assert attrs["llm_eval.quotes"] == ("ok", REDACTED)
    assert (
        otel_memory.counter("llm_eval.sanitizer.redactions", {"gen_ai.evaluation.name": "judge"})
        == 2
    )
    dump = otel_memory.serialize_all()
    for leaked in ("529.982.247-25", "a@b.co", "sk-proj-"):
        assert leaked not in dump


def test_sanitize_leaves_clean_values() -> None:
    clean, n = sanitize({"a": "cpf=1 (input)", "b": 1.0, "c": True, "d": ["cpf", "email"]})
    assert n == 0
    assert clean == {"a": "cpf=1 (input)", "b": 1.0, "c": True, "d": ["cpf", "email"]}


@pytest.mark.parametrize(
    "value",
    [
        # Response IDs
        "chatcmpl-9xYzAbC123dEf456GhI789jKl0",
        "chatcmpl-AQx1b2C3d4E5f6G7h8I9j0K1l2M3n",
        "msg_01XFDUDYJgAACzvnptvVoYEL",
        "resp_68af2c1e8a3c81908b2b3e1a0f0f0a2b",
        # Model names
        "gpt-4o-mini-2024-07-18",
        "claude-sonnet-4-5-20250929",
        "gemini-2.5-flash",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        # The service's own explanations
        "cpf=1 (input), cnpj=1 (input), phone=2 (output), pix_key=1 (input)",
        "refusal=1 (output), source=phrase, lang=pt",
        "coverage=0.42, longest_run=37 words (output)",
        "invalid_json=1 of 2 (output): Expecting ',' delimiter at char 132",
        "invalid_json=1 of 1 (output): Unterminated string starting at char 10, "
        "finish_reason=length",
    ],
)
def test_sanitize_keeps_legitimate_values(value: str) -> None:
    assert sanitize({"a": value}) == ({"a": value}, 0)


def test_sanitize_redacts_new_pii_types() -> None:
    values = {
        "cnpj": "12.ABC.345/01DE-35",
        "phone": "(11) 98765-4321",
        "pix": "pix 123e4567-e89b-42d3-a456-426614174000",
    }
    assert sanitize(values) == ({k: REDACTED for k in values}, 3)
