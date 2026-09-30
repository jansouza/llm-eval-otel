"""Turn OTLP spans into :class:`GenAIInteraction` objects.

Two content formats are read, in this order, stopping at the first one found:

1. Current semconv: ``gen_ai.system_instructions``, ``gen_ai.input.messages`` and
   ``gen_ai.output.messages``, structured (array of kvlist) or as a JSON string.
2. OpenLLMetry: indexed ``gen_ai.prompt.{n}.*`` and ``gen_ai.completion.{n}.*``.
"""

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Span

from llm_eval_otel import semconv
from llm_eval_otel.evaluators.base import GenAIInteraction, Message, PartSpan

_OPENLLMETRY = re.compile(semconv.OPENLLMETRY_PATTERN)
_OPENLLMETRY_TOOL_ARGS = re.compile(
    r"^gen_ai\.(prompt|completion)\.(\d+)\.tool_calls\.(\d+)\.arguments$"
)
# OpenLLMetry's llm.request.type, mapped to gen_ai.operation.name.
_LLM_REQUEST_TYPE = "llm.request.type"
_REQUEST_TYPE_TO_OPERATION = {"chat": "chat", "completion": "text_completion"}

_SYSTEM = "system"
_ASSISTANT = "assistant"


@dataclass
class ExtractResult:
    interactions: list[GenAIInteraction] = field(default_factory=list)
    received: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


def any_value(value: AnyValue) -> Any:
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    if kind == "array_value":
        return [any_value(v) for v in value.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: any_value(kv.value) for kv in value.kvlist_value.values}
    return getattr(value, kind)


def attributes(kvs: Iterable[KeyValue]) -> dict[str, Any]:
    return {kv.key: any_value(kv.value) for kv in kvs}


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _structured(value: Any) -> list[Any] | None:
    """A semconv messages attribute, either structured or a JSON string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, list) else None


def _message(role: str, pieces: Iterable[tuple[str, str]]) -> Message | None:
    """Join ``(part type, text)`` pieces with ``"\\n"``, recording where each part sits.

    A lone text part is stored without ``parts``, which already means "all text".
    """
    texts: list[str] = []
    parts: list[PartSpan] = []
    offset = 0
    for part_type, text in pieces:
        if not text:
            continue
        if texts:
            offset += 1  # the "\n" separator
        parts.append(PartSpan(part_type, offset, offset + len(text)))
        texts.append(text)
        offset += len(text)
    if not texts:
        return None
    if len(parts) == 1 and parts[0].type == semconv.PART_TEXT:
        parts = []
    return Message(role=role, text="\n".join(texts), parts=tuple(parts))


def _part_pieces(parts: Iterable[Any]) -> Iterator[tuple[str, str]]:
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if not isinstance(part_type, str) or part_type not in semconv.PART_CONTENT_FIELDS:
            continue  # blob, file, uri, server tool calls: out of scope
        yield part_type, _to_text(part.get(semconv.PART_CONTENT_FIELDS[part_type]))


def _semconv_messages(items: list[Any]) -> list[Message]:
    messages = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "unknown")
        if message := _message(role, _part_pieces(item.get("parts") or [])):
            messages.append(message)
    return messages


def _semconv_system_instructions(value: Any) -> list[Message]:
    """``gen_ai.system_instructions`` is a list of parts; tolerate a list of messages too."""
    items = _structured(value) or []
    if any(isinstance(i, Mapping) and "parts" in i for i in items):
        return [Message(_SYSTEM, m.text, m.parts) for m in _semconv_messages(items)]
    message = _message(_SYSTEM, _part_pieces(items))
    return [message] if message else []


def new_in_turn(messages: list[Message]) -> list[Message]:
    """Messages after the last assistant message; all of them when there is none."""
    last_assistant = max(
        (index for index, m in enumerate(messages) if m.role == _ASSISTANT), default=-1
    )
    return messages[last_assistant + 1 :]


def _split_system(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    """System messages go to every provider call, so they are always evaluated."""
    system = [m for m in messages if m.role == _SYSTEM]
    rest = [m for m in messages if m.role != _SYSTEM]
    return system, rest


@dataclass
class _Content:
    system: list[Message]
    inputs: list[Message]
    outputs: list[Message]
    # Per-output finish reasons found in the messages themselves; used only when the
    # span has no gen_ai.response.finish_reasons.
    finish_reasons: tuple[str, ...] = ()


def _from_semconv(attrs: Mapping[str, Any]) -> _Content:
    system = _semconv_system_instructions(attrs.get(semconv.GEN_AI_SYSTEM_INSTRUCTIONS))
    inline_system, inputs = _split_system(
        _semconv_messages(_structured(attrs.get(semconv.GEN_AI_INPUT_MESSAGES)) or [])
    )
    output_items = _structured(attrs.get(semconv.GEN_AI_OUTPUT_MESSAGES)) or []
    finish_reasons = tuple(
        reason
        for item in output_items
        if isinstance(item, Mapping)
        and (reason := _as_str(item.get(semconv.MESSAGE_FINISH_REASON)))
    )
    return _Content(
        system + inline_system,
        new_in_turn(inputs),
        _semconv_messages(output_items),
        finish_reasons,
    )


def _openllmetry_content(value: Any) -> str:
    """Content is a plain string, or a JSON list of OpenAI-style content parts."""
    if not isinstance(value, str):
        return _to_text(value)
    stripped = value.lstrip()
    if stripped.startswith("[{"):
        try:
            parts = json.loads(stripped)
        except ValueError:
            return value
        if isinstance(parts, list) and all(isinstance(p, Mapping) for p in parts):
            return "\n".join(str(p["text"]) for p in parts if p.get("type") == "text")
    return value


def _from_openllmetry(attrs: Mapping[str, Any]) -> _Content | None:
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for key, value in attrs.items():
        if m := _OPENLLMETRY.match(key):
            entries.setdefault((m.group(1), int(m.group(2))), {})[m.group(3)] = value
        elif m := _OPENLLMETRY_TOOL_ARGS.match(key):
            entry = entries.setdefault((m.group(1), int(m.group(2))), {})
            entry.setdefault("tool_calls", {})[int(m.group(3))] = value
    if not any("content" in e or "tool_calls" in e for e in entries.values()):
        return None

    def of_kind(kind: str) -> list[dict[str, Any]]:
        return [entry for (entry_kind, _), entry in sorted(entries.items()) if entry_kind == kind]

    def build(kind: str) -> list[Message]:
        messages = []
        for entry in of_kind(kind):
            pieces = [(semconv.PART_TEXT, _openllmetry_content(entry.get("content")))]
            pieces += [
                (semconv.PART_TOOL_CALL, _to_text(v))
                for _, v in sorted(entry.get("tool_calls", {}).items())
            ]
            if message := _message(str(entry.get("role") or "unknown"), pieces):
                messages.append(message)
        return messages

    system, prompts = _split_system(build("prompt"))
    finish_reasons = tuple(
        reason
        for entry in of_kind("completion")
        if (reason := _as_str(entry.get(semconv.MESSAGE_FINISH_REASON)))
    )
    return _Content(system, new_in_turn(prompts), build("completion"), finish_reasons)


def _operation_name(attrs: Mapping[str, Any]) -> str | None:
    operation = _as_str(attrs.get(semconv.GEN_AI_OPERATION_NAME))
    if operation is not None:
        return operation
    request_type = _as_str(attrs.get(_LLM_REQUEST_TYPE))
    return _REQUEST_TYPE_TO_OPERATION.get(request_type or "")


def _output_type(attrs: Mapping[str, Any]) -> str | None:
    """``gen_ai.output.type``, or ``json`` when OpenLLMetry recorded a JSON output format."""
    if output_type := _as_str(attrs.get(semconv.GEN_AI_OUTPUT_TYPE)):
        return output_type
    requested = _as_str(attrs.get(semconv.OPENLLMETRY_STRUCTURED_OUTPUT_SCHEMA))
    if requested is None:
        return None
    try:
        parsed = json.loads(requested)
    except ValueError:
        return None
    # A JSON schema, or an OpenAI response_format such as {"type": "json_object"}.
    if isinstance(parsed, Mapping) and parsed.get("type") == semconv.OUTPUT_TYPE_TEXT:
        return semconv.OUTPUT_TYPE_TEXT
    return semconv.OUTPUT_TYPE_JSON


def _finish_reasons(attrs: Mapping[str, Any], from_messages: tuple[str, ...]) -> tuple[str, ...]:
    value = attrs.get(semconv.GEN_AI_RESPONSE_FINISH_REASONS)
    if isinstance(value, str):  # some instrumentations record a single string
        value = [value]
    if isinstance(value, list) and (reasons := tuple(r for r in value if _as_str(r))):
        return reasons
    return from_messages


def _association_properties(attrs: Mapping[str, Any]) -> dict[str, str]:
    """Scalar ``traceloop.association.properties.*`` values, keyed without the prefix."""
    prefix = semconv.TRACELOOP_ASSOCIATION_PREFIX
    return {
        key[len(prefix) :]: value if isinstance(value, str) else json.dumps(value)
        for key, value in attrs.items()
        if key.startswith(prefix)
        and len(key) > len(prefix)
        and isinstance(value, str | bool | int | float)
    }


def extract_span(span: Span, service_name: str | None) -> GenAIInteraction | str:
    """Return the interaction, or the skip reason."""
    attrs = attributes(span.attributes)
    operation = _operation_name(attrs)
    if operation is not None and operation not in semconv.INFERENCE_OPERATIONS:
        return semconv.SKIP_NOT_INFERENCE

    has_semconv = any(
        k in attrs
        for k in (
            semconv.GEN_AI_SYSTEM_INSTRUCTIONS,
            semconv.GEN_AI_INPUT_MESSAGES,
            semconv.GEN_AI_OUTPUT_MESSAGES,
        )
    )
    content = _from_semconv(attrs) if has_semconv else _from_openllmetry(attrs)
    if content is None:
        # No operation name and no OpenLLMetry content: not a GenAI inference span.
        return semconv.SKIP_NO_CONTENT if operation else semconv.SKIP_NOT_INFERENCE
    if not (content.system or content.inputs or content.outputs):
        return semconv.SKIP_NO_CONTENT

    return GenAIInteraction(
        trace_id=bytes(span.trace_id),
        span_id=bytes(span.span_id),
        parent_span_id=bytes(span.parent_span_id) or None,
        trace_flags=span.flags & 0xFF,
        service_name=service_name,
        operation_name=operation or "chat",
        provider_name=_as_str(attrs.get(semconv.GEN_AI_PROVIDER_NAME))
        or _as_str(attrs.get(semconv.GEN_AI_SYSTEM)),
        request_model=_as_str(attrs.get(semconv.GEN_AI_REQUEST_MODEL)),
        response_id=_as_str(attrs.get(semconv.GEN_AI_RESPONSE_ID)),
        system_instructions=content.system,
        input_messages=content.inputs,
        output_messages=content.outputs,
        association_properties=_association_properties(attrs),
        output_type=_output_type(attrs),
        finish_reasons=_finish_reasons(attrs, content.finish_reasons),
    )


def extract(request: ExportTraceServiceRequest) -> ExtractResult:
    result = ExtractResult()
    for resource_spans in request.resource_spans:
        resource_attrs = attributes(resource_spans.resource.attributes)
        service_name = _as_str(resource_attrs.get(semconv.SERVICE_NAME))
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                result.received += 1
                outcome = extract_span(span, service_name)
                if isinstance(outcome, str):
                    result.skipped[outcome] += 1
                else:
                    result.interactions.append(outcome)
    return result
