"""Builders for OTLP protobuf payloads used as test fixtures."""

import json
from collections.abc import Mapping
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

TRACE_ID = bytes.fromhex("0af7651916cd43dd8448eb211c80319c")
SPAN_ID = bytes.fromhex("b7ad6b7169203331")
PARENT_SPAN_ID = bytes.fromhex("00f067aa0ba902b7")


def any_value(value: Any) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, float):
        return AnyValue(double_value=value)
    if isinstance(value, str):
        return AnyValue(string_value=value)
    if isinstance(value, bytes):
        return AnyValue(bytes_value=value)
    if isinstance(value, Mapping):
        out = AnyValue()
        out.kvlist_value.values.extend(key_values(value))
        return out
    if isinstance(value, list):
        out = AnyValue()
        out.array_value.values.extend(any_value(v) for v in value)
        return out
    raise TypeError(type(value))


def key_values(attrs: Mapping[str, Any]) -> list[KeyValue]:
    return [KeyValue(key=k, value=any_value(v)) for k, v in attrs.items()]


def make_span(
    attributes: Mapping[str, Any],
    *,
    trace_id: bytes = TRACE_ID,
    span_id: bytes = SPAN_ID,
    parent_span_id: bytes = PARENT_SPAN_ID,
    flags: int = 0x301,  # sampled + has/is-remote bits, which must be masked off
    name: str = "chat gpt-4o-mini",
) -> Span:
    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        flags=flags,
        name=name,
        kind=Span.SPAN_KIND_CLIENT,
    )
    span.attributes.extend(key_values(attributes))
    return span


def make_request(
    *spans: Span, service_name: str | None = "support-bot"
) -> ExportTraceServiceRequest:
    resource_attrs = {"service.name": service_name} if service_name else {}
    rs = ResourceSpans()
    rs.resource.attributes.extend(key_values(resource_attrs))
    rs.scope_spans.append(ScopeSpans(spans=list(spans)))
    return ExportTraceServiceRequest(resource_spans=[rs])


def text(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "parts": [{"type": "text", "content": content}]}


def semconv_attrs(
    input_messages: list[dict[str, Any]],
    output_messages: list[dict[str, Any]] | None = None,
    *,
    system: list[dict[str, Any]] | None = None,
    as_json: bool = False,
    operation: str = "chat",
) -> dict[str, Any]:
    encode = (lambda v: json.dumps(v)) if as_json else (lambda v: v)
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.response.id": "chatcmpl-123",
        "gen_ai.input.messages": encode(input_messages),
    }
    if output_messages is not None:
        attrs["gen_ai.output.messages"] = encode(output_messages)
    if system is not None:
        attrs["gen_ai.system_instructions"] = encode(system)
    return attrs


def make_export_request(
    *,
    trace_id: bytes = TRACE_ID,
    span_id: bytes = SPAN_ID,
    service_name: str | None = "support-bot",
    input_messages: list[dict[str, Any]],
    output_messages: list[dict[str, Any]] | None = None,
) -> ExportTraceServiceRequest:
    span = make_span(
        semconv_attrs(input_messages, output_messages), trace_id=trace_id, span_id=span_id
    )
    return make_request(span, service_name=service_name)


def chat_request(content: str, **kwargs: Any) -> ExportTraceServiceRequest:
    return make_export_request(
        input_messages=[text("user", content)],
        output_messages=[text("assistant", "Anotado.")],
        **kwargs,
    )
