"""Turn evaluation records into an event, an optional child span and metrics.

For judge evaluators, each call to the judge also becomes a ``{operation} {model}`` span
(``chat`` for the OpenAI-compatible judge, ``system_one`` for Jev) under the
``evaluate {name}`` span, plus the GenAI client metrics. Those carry model, endpoint,
tokens and finish reason, never content. A Jev batch is one call: it hangs under the first
check's ``evaluate`` span.
"""

from collections.abc import Collection, Iterable

from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
)

from llm_eval_otel import semconv
from llm_eval_otel.emit.sanitize import sanitize
from llm_eval_otel.emit.sdk import Telemetry
from llm_eval_otel.engine.runner import EvaluationRecord
from llm_eval_otel.evaluators.base import AttributeValue
from llm_eval_otel.judge.client import JudgeCall

SCOPE = "llm_eval_otel"

_DURATION_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

# Attributes copied from the evaluated span onto metrics: bounded by apps and models.
# Association properties are added too, minus the per-request keys (association_exclude).
# The evaluation type follows from the name, so it adds no series; it lets a dashboard
# select the judges without listing them.
_METRIC_KEYS = (
    semconv.GEN_AI_EVALUATION_NAME,
    semconv.LLM_EVAL_EVALUATION_TYPE,
    semconv.LLM_EVAL_SOURCE_SERVICE_NAME,
    semconv.GEN_AI_PROVIDER_NAME,
    semconv.GEN_AI_REQUEST_MODEL,
)


def severity(record: EvaluationRecord) -> SeverityNumber:
    if record.result.error_type is not None:
        return SeverityNumber.ERROR
    if record.result.label == semconv.LABEL_FAIL:
        return SeverityNumber.WARN
    return SeverityNumber.INFO


def event_attributes(record: EvaluationRecord) -> dict[str, AttributeValue]:
    i, r = record.interaction, record.result
    attrs: dict[str, AttributeValue] = {
        semconv.GEN_AI_EVALUATION_NAME: record.evaluator_name,
        semconv.GEN_AI_OPERATION_NAME: i.operation_name,
        semconv.LLM_EVAL_EVALUATION_TYPE: record.evaluator_kind,
    }
    optional: dict[str, AttributeValue | None] = {
        semconv.GEN_AI_EVALUATION_SCORE_VALUE: r.score,
        semconv.GEN_AI_EVALUATION_SCORE_LABEL: r.label,
        semconv.GEN_AI_EVALUATION_EXPLANATION: r.explanation,
        semconv.GEN_AI_RESPONSE_ID: i.response_id,
        semconv.GEN_AI_PROVIDER_NAME: i.provider_name,
        semconv.GEN_AI_REQUEST_MODEL: i.request_model,
        semconv.ERROR_TYPE: r.error_type,
        semconv.LLM_EVAL_SOURCE_SERVICE_NAME: i.service_name,
    }
    attrs.update({k: v for k, v in optional.items() if v is not None})
    attrs.update(
        {semconv.TRACELOOP_ASSOCIATION_PREFIX + k: v for k, v in i.association_properties.items()}
    )
    # Evaluators may add only llm_eval.* keys; gen_ai.* belongs to the semconv.
    attrs.update(
        {k: v for k, v in r.attributes.items() if k.startswith(semconv.LLM_EVAL_ATTRIBUTE_PREFIX)}
    )
    if record.truncated:
        attrs[semconv.LLM_EVAL_CONTENT_TRUNCATED] = True
    return attrs


def judge_call_attributes(call: JudgeCall) -> dict[str, AttributeValue]:
    """GenAI client span attributes, without gen_ai.input/output.messages: no content."""
    attrs: dict[str, AttributeValue] = {
        semconv.GEN_AI_OPERATION_NAME: call.operation_name,
        semconv.GEN_AI_PROVIDER_NAME: call.provider_name,
        semconv.GEN_AI_REQUEST_MODEL: call.request_model,
    }
    optional: dict[str, AttributeValue | None] = {
        semconv.GEN_AI_RESPONSE_MODEL: call.response_model,
        semconv.SERVER_ADDRESS: call.server_address,
        semconv.SERVER_PORT: call.server_port,
        semconv.GEN_AI_USAGE_INPUT_TOKENS: call.input_tokens,
        semconv.GEN_AI_USAGE_OUTPUT_TOKENS: call.output_tokens,
        semconv.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: call.cache_read_tokens,
        semconv.GEN_AI_RESPONSE_FINISH_REASONS: [call.finish_reason]
        if call.finish_reason
        else None,
        semconv.ERROR_TYPE: call.error_type,
    }
    attrs.update({k: v for k, v in optional.items() if v is not None})
    return attrs


# Judge call attributes that go on the GenAI client metrics.
_JUDGE_METRIC_KEYS = (
    semconv.GEN_AI_OPERATION_NAME,
    semconv.GEN_AI_PROVIDER_NAME,
    semconv.GEN_AI_REQUEST_MODEL,
    semconv.GEN_AI_RESPONSE_MODEL,
    semconv.SERVER_ADDRESS,
    semconv.SERVER_PORT,
)


class Emitter:
    def __init__(
        self,
        telemetry: Telemetry,
        *,
        emit_spans: bool = True,
        association_exclude: Collection[str] = (),
    ) -> None:
        self.emit_spans = emit_spans
        # Association properties go on the metrics too, except these keys
        self.metric_association_skip = frozenset(
            semconv.TRACELOOP_ASSOCIATION_PREFIX + key for key in association_exclude
        )
        self.tracer = telemetry.tracer_provider.get_tracer(SCOPE)
        self.logger = telemetry.logger_provider.get_logger(SCOPE)
        meter = telemetry.meter_provider.get_meter(SCOPE)

        self.evaluations = meter.create_counter(
            semconv.METRIC_EVALUATIONS, unit="{evaluation}", description="Evaluations run"
        )
        self.score = meter.create_histogram(
            semconv.METRIC_EVALUATION_SCORE,
            unit="1",
            description="Evaluation scores, 0 to 1, higher is better",
            explicit_bucket_boundaries_advisory=semconv.SCORE_BUCKETS,
        )
        self.duration = meter.create_histogram(
            semconv.METRIC_EVALUATION_DURATION,
            unit="s",
            description="Time spent in one evaluator",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS,
        )
        self.redactions = meter.create_counter(
            semconv.METRIC_SANITIZER_REDACTIONS,
            unit="{value}",
            description="Attribute values replaced by the sanitizer",
        )
        self.spans_received = meter.create_counter(
            semconv.METRIC_SPANS_RECEIVED, unit="{span}", description="Spans received"
        )
        self.spans_skipped = meter.create_counter(
            semconv.METRIC_SPANS_SKIPPED, unit="{span}", description="Spans not evaluated"
        )
        self.queue_size = meter.create_up_down_counter(
            semconv.METRIC_QUEUE_SIZE, unit="{interaction}", description="Interactions queued"
        )
        self.dropped = meter.create_counter(
            semconv.METRIC_EVALUATIONS_DROPPED,
            unit="{evaluation}",
            description="Evaluations sampled but not run: lane full, no token budget, shutdown",
        )
        self.lane_size = meter.create_up_down_counter(
            semconv.METRIC_LANE_SIZE, unit="{evaluation}", description="Evaluations in a lane"
        )
        self.client_token_usage = meter.create_histogram(
            semconv.METRIC_CLIENT_TOKEN_USAGE,
            unit="{token}",
            description="Tokens used by the judge's calls",
            explicit_bucket_boundaries_advisory=semconv.TOKEN_USAGE_BUCKETS,
        )
        self.client_operation_duration = meter.create_histogram(
            semconv.METRIC_CLIENT_OPERATION_DURATION,
            unit="s",
            description="Duration of the judge's calls",
            explicit_bucket_boundaries_advisory=semconv.OPERATION_DURATION_BUCKETS,
        )

    def record_drop(self, evaluation_name: str, reason: str) -> None:
        self.dropped.add(
            1,
            {semconv.GEN_AI_EVALUATION_NAME: evaluation_name, semconv.LLM_EVAL_DROP_REASON: reason},
        )

    def emit_all(self, records: Iterable[EvaluationRecord]) -> None:
        for record in records:
            self.emit(record)

    def emit(self, record: EvaluationRecord) -> None:
        attrs, redactions = sanitize(event_attributes(record))
        name = record.evaluator_name
        if redactions:
            self.redactions.add(redactions, {semconv.GEN_AI_EVALUATION_NAME: name})

        i = record.interaction
        parent = SpanContext(
            trace_id=int.from_bytes(i.trace_id, "big"),
            span_id=int.from_bytes(i.span_id, "big"),
            is_remote=True,
            trace_flags=TraceFlags(i.trace_flags & 0xFF),
        )
        ctx = trace.set_span_in_context(NonRecordingSpan(parent))

        self.logger.emit(
            event_name=semconv.EVENT_EVALUATION_RESULT,
            context=ctx,
            severity_number=severity(record),
            attributes=attrs,
        )

        evaluate_ctx = None
        if self.emit_spans:
            span = self.tracer.start_span(
                f"{semconv.SPAN_NAME_PREFIX} {name}",
                context=ctx,
                kind=SpanKind.INTERNAL,
                attributes=attrs,
                start_time=record.start_ns,
            )
            if record.result.error_type is not None:
                span.set_status(Status(StatusCode.ERROR))
            span.end(end_time=record.end_ns)
            evaluate_ctx = trace.set_span_in_context(span)

        self._record_metrics(record, attrs)
        for call in record.judge_calls:
            self._emit_judge_call(name, call, evaluate_ctx)

    def _emit_judge_call(self, evaluation_name: str, call: JudgeCall, ctx: Context | None) -> None:
        attrs, _ = sanitize(judge_call_attributes(call))
        if ctx is not None:
            span = self.tracer.start_span(
                f"{call.operation_name} {call.request_model}",
                context=ctx,
                kind=SpanKind.CLIENT,
                attributes=attrs,
                start_time=call.start_ns,
            )
            if call.error_type is not None:
                span.set_status(Status(StatusCode.ERROR))
            span.end(end_time=call.end_ns)

        metric_attrs = {k: attrs[k] for k in _JUDGE_METRIC_KEYS if k in attrs}
        metric_attrs[semconv.GEN_AI_EVALUATION_NAME] = evaluation_name
        duration_attrs = dict(metric_attrs)
        if call.error_type is not None:
            duration_attrs[semconv.ERROR_TYPE] = attrs[semconv.ERROR_TYPE]
        self.client_operation_duration.record((call.end_ns - call.start_ns) / 1e9, duration_attrs)
        for token_type, tokens in (
            (semconv.TOKEN_TYPE_INPUT, call.input_tokens),
            (semconv.TOKEN_TYPE_OUTPUT, call.output_tokens),
        ):
            if tokens is not None:
                self.client_token_usage.record(
                    tokens, {**metric_attrs, semconv.GEN_AI_TOKEN_TYPE: token_type}
                )

    def _record_metrics(self, record: EvaluationRecord, attrs: dict[str, AttributeValue]) -> None:
        r = record.result
        common = {k: attrs[k] for k in _METRIC_KEYS if k in attrs}
        common.update(
            {
                k: v
                for k, v in attrs.items()
                if k.startswith(semconv.TRACELOOP_ASSOCIATION_PREFIX)
                and k not in self.metric_association_skip
            }
        )

        counter_attrs = dict(common)
        if semconv.GEN_AI_EVALUATION_SCORE_LABEL in attrs:
            counter_attrs[semconv.GEN_AI_EVALUATION_SCORE_LABEL] = attrs[
                semconv.GEN_AI_EVALUATION_SCORE_LABEL
            ]
        if r.error_type is not None:
            counter_attrs[semconv.ERROR_TYPE] = attrs[semconv.ERROR_TYPE]
        self.evaluations.add(1, counter_attrs)

        if r.score is not None and r.error_type is None and r.label != semconv.LABEL_EXEMPT:
            self.score.record(r.score, common)

        duration_attrs: dict[str, AttributeValue] = {
            semconv.GEN_AI_EVALUATION_NAME: record.evaluator_name
        }
        if r.error_type is not None:
            duration_attrs[semconv.ERROR_TYPE] = attrs[semconv.ERROR_TYPE]
        self.duration.record((record.end_ns - record.start_ns) / 1e9, duration_attrs)
