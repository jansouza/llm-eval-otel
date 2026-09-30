import pytest

from llm_eval_otel import semconv
from llm_eval_otel.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.http_port == 4318
    assert s.evaluators == ["pii_detection", "secret_detection"]
    assert s.queue_max == 10_000
    assert s.max_request_bytes == 16_777_216
    assert s.dedup_ttl_s == 600
    assert s.emit_spans is True
    assert s.association_exclude == ["correlation_id"]
    assert s.pii_types == ["cpf", "cnpj", "email", "credit_card", "phone", "pix_key"]
    assert not s.tls_enabled


def test_parses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_EVALUATORS", "pii_detection, relevance")
    monkeypatch.setenv("LLM_EVAL_SAMPLE_RATES", "relevance=0.05")
    monkeypatch.setenv(
        "LLM_EVAL_EXCEPTIONS",
        '{"bank-chatbot": ["pii_detection"], "devops-assistant": ["secret_detection"]}',
    )
    monkeypatch.setenv("LLM_EVAL_EMIT_SPANS", "false")
    monkeypatch.setenv("LLM_EVAL_ASSOCIATION_EXCLUDE", "correlation_id, session_id")
    s = Settings()
    assert s.association_exclude == ["correlation_id", "session_id"]
    assert s.evaluators == ["pii_detection", "relevance"]
    assert s.sample_rates == {"relevance": 0.05}
    assert s.exceptions == {
        "bank-chatbot": ["pii_detection"],
        "devops-assistant": ["secret_detection"],
    }
    assert s.emit_spans is False


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_exceptions(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LLM_EVAL_EXCEPTIONS", value)
    assert Settings().exceptions == {}


def test_rejects_invalid_exceptions_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_EXCEPTIONS", "bank-chatbot=pii_detection")
    with pytest.raises(ValueError, match="exceptions"):
        Settings()


def test_rejects_rate_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_SAMPLE_RATES", "relevance=1.5")
    with pytest.raises(ValueError, match="between 0 and 1"):
        Settings()


def test_semconv_names_snapshot() -> None:
    """Names follow semantic-conventions-genai e57c543; a change here must be deliberate."""
    assert {
        name: value
        for name, value in vars(semconv).items()
        if name.isupper() and isinstance(value, str)
    } == {
        "GEN_AI_OPERATION_NAME": "gen_ai.operation.name",
        "GEN_AI_PROVIDER_NAME": "gen_ai.provider.name",
        "GEN_AI_SYSTEM": "gen_ai.system",
        "GEN_AI_REQUEST_MODEL": "gen_ai.request.model",
        "GEN_AI_RESPONSE_ID": "gen_ai.response.id",
        "GEN_AI_SYSTEM_INSTRUCTIONS": "gen_ai.system_instructions",
        "GEN_AI_INPUT_MESSAGES": "gen_ai.input.messages",
        "GEN_AI_OUTPUT_MESSAGES": "gen_ai.output.messages",
        "GEN_AI_OUTPUT_TYPE": "gen_ai.output.type",
        "GEN_AI_RESPONSE_FINISH_REASONS": "gen_ai.response.finish_reasons",
        "MESSAGE_FINISH_REASON": "finish_reason",
        "OUTPUT_TYPE_JSON": "json",
        "OUTPUT_TYPE_TEXT": "text",
        "FINISH_REASON_LENGTH": "length",
        "FINISH_REASON_CONTENT_FILTER": "content_filter",
        "OPENLLMETRY_PATTERN": (
            r"^gen_ai\.(prompt|completion)\.(\d+)\.(role|content|finish_reason)$"
        ),
        "OPENLLMETRY_STRUCTURED_OUTPUT_SCHEMA": "gen_ai.request.structured_output_schema",
        "TRACELOOP_ASSOCIATION_PREFIX": "traceloop.association.properties.",
        "PART_TEXT": "text",
        "PART_REASONING": "reasoning",
        "PART_TOOL_CALL": "tool_call",
        "PART_TOOL_CALL_RESPONSE": "tool_call_response",
        "SERVICE_NAME": "service.name",
        "EVENT_EVALUATION_RESULT": "gen_ai.evaluation.result",
        "SPAN_NAME_PREFIX": "evaluate",
        "GEN_AI_EVALUATION_NAME": "gen_ai.evaluation.name",
        "GEN_AI_EVALUATION_SCORE_VALUE": "gen_ai.evaluation.score.value",
        "GEN_AI_EVALUATION_SCORE_LABEL": "gen_ai.evaluation.score.label",
        "GEN_AI_EVALUATION_EXPLANATION": "gen_ai.evaluation.explanation",
        "ERROR_TYPE": "error.type",
        "LLM_EVAL_SOURCE_SERVICE_NAME": "llm_eval.source.service.name",
        "LLM_EVAL_EVALUATION_TYPE": "llm_eval.evaluation.type",
        "LLM_EVAL_PII_TYPES": "llm_eval.pii.types",
        "LLM_EVAL_SECRET_TYPES": "llm_eval.secret.types",
        "LLM_EVAL_REFUSAL_SOURCE": "llm_eval.refusal.source",
        "LLM_EVAL_REFUSAL_LANGUAGE": "llm_eval.refusal.language",
        "LLM_EVAL_PROMPT_LEAK_COVERAGE": "llm_eval.prompt_leak.coverage",
        "LLM_EVAL_PROMPT_LEAK_LONGEST_RUN": "llm_eval.prompt_leak.longest_run",
        "LLM_EVAL_OUTPUT_FORMAT_ERROR": "llm_eval.output_format.error",
        "LLM_EVAL_CONTENT_TRUNCATED": "llm_eval.content.truncated",
        "LLM_EVAL_SKIP_REASON": "llm_eval.skip.reason",
        "LLM_EVAL_ATTRIBUTE_PREFIX": "llm_eval.",
        "LABEL_PASS": "pass",
        "LABEL_FAIL": "fail",
        "LABEL_EXEMPT": "exempt",
        "METRIC_EVALUATIONS": "llm_eval.evaluations",
        "METRIC_EVALUATION_SCORE": "llm_eval.evaluation.score",
        "METRIC_EVALUATION_DURATION": "llm_eval.evaluation.duration",
        "METRIC_SPANS_RECEIVED": "llm_eval.spans.received",
        "METRIC_SPANS_SKIPPED": "llm_eval.spans.skipped",
        "METRIC_QUEUE_SIZE": "llm_eval.queue.size",
        "METRIC_SANITIZER_REDACTIONS": "llm_eval.sanitizer.redactions",
        "SKIP_NOT_INFERENCE": "not_inference",
        "SKIP_NO_CONTENT": "no_content",
        "SKIP_DUPLICATE": "duplicate",
        "SKIP_INVALID_PAYLOAD": "invalid_payload",
    }
