import json

from otlp import PARENT_SPAN_ID, SPAN_ID, TRACE_ID, make_request, make_span, semconv_attrs, text

from llm_eval_otel.evaluators.base import Message, PartSpan
from llm_eval_otel.extract.genai import extract

CPF = "529.982.247-25"


def only(request):  # type: ignore[no-untyped-def]
    result = extract(request)
    assert len(result.interactions) == 1, result.skipped
    return result.interactions[0]


def test_ids_match_byte_for_byte_after_protobuf_roundtrip() -> None:
    request = make_request(make_span(semconv_attrs([text("user", "oi")])))
    wire = type(request).FromString(request.SerializeToString())
    i = only(wire)
    assert i.trace_id == TRACE_ID
    assert i.span_id == SPAN_ID
    assert i.parent_span_id == PARENT_SPAN_ID
    assert i.trace_flags == 0x01  # upper bits of span.flags masked off


def test_root_span_has_no_parent() -> None:
    i = only(make_request(make_span(semconv_attrs([text("user", "oi")]), parent_span_id=b"")))
    assert i.parent_span_id is None


def test_reads_metadata_and_resource_service_name() -> None:
    i = only(make_request(make_span(semconv_attrs([text("user", "oi")])), service_name="bank"))
    assert i.service_name == "bank"
    assert i.operation_name == "chat"
    assert i.provider_name == "openai"
    assert i.request_model == "gpt-4o-mini"
    assert i.response_id == "chatcmpl-123"


def test_legacy_gen_ai_system_is_provider_fallback() -> None:
    attrs = semconv_attrs([text("user", "oi")])
    del attrs["gen_ai.provider.name"]
    attrs["gen_ai.system"] = "anthropic"
    assert only(make_request(make_span(attrs))).provider_name == "anthropic"


def test_structured_and_json_string_are_equivalent() -> None:
    messages = [text("user", "Qual meu saldo?")]
    outputs = [text("assistant", "R$ 10")]
    system = [{"type": "text", "content": "Você é um assistente."}]
    structured = only(make_request(make_span(semconv_attrs(messages, outputs, system=system))))
    as_json = only(
        make_request(make_span(semconv_attrs(messages, outputs, system=system, as_json=True)))
    )
    assert structured == as_json
    assert structured.system_instructions == [Message("system", "Você é um assistente.")]
    assert structured.input_messages == [Message("user", "Qual meu saldo?")]
    assert structured.output_messages == [Message("assistant", "R$ 10")]


def test_reads_text_reasoning_tool_call_and_tool_response_parts() -> None:
    inputs = [
        text("user", "consulte"),
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "1", "name": "db", "arguments": {"q": "x"}}],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "1", "response": "resultado"}],
        },
    ]
    outputs = [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "pensando"},
                {"type": "text", "content": "pronto"},
                {"type": "tool_call", "id": "2", "name": "f", "arguments": '{"a": 1}'},
                {"type": "blob", "modality": "image", "content": "aGVsbG8="},
                {"type": "uri", "modality": "image", "uri": "https://x/y.png"},
                {"type": "server_tool_call", "name": "web_search"},
            ],
        }
    ]
    i = only(make_request(make_span(semconv_attrs(inputs, outputs))))
    # Only what came after the last assistant message: the tool result.
    assert i.input_messages == [
        Message("tool", "resultado", (PartSpan("tool_call_response", 0, 9),))
    ]
    [output] = i.output_messages
    assert output.text == 'pensando\npronto\n{"a": 1}'
    assert output.parts == (
        PartSpan("reasoning", 0, 8),
        PartSpan("text", 9, 15),
        PartSpan("tool_call", 16, 24),
    )
    assert output.text_of("text") == "pronto"
    assert output.text_of("text", "tool_call") == 'pronto\n{"a": 1}'
    assert output.text_of("tool_call_response") == ""


def test_tool_call_arguments_object_is_serialized() -> None:
    outputs = [
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "name": "f", "arguments": {"key": "AKIA"}}],
        }
    ]
    i = only(make_request(make_span(semconv_attrs([text("user", "x")], outputs))))
    assert json.loads(i.output_messages[0].text) == {"key": "AKIA"}


def test_reads_association_properties() -> None:
    attrs = semconv_attrs([text("user", "oi")])
    attrs.update(
        {
            "traceloop.association.properties.scenario": "pix",
            "traceloop.association.properties.attempt": 2,
            "traceloop.association.properties.tags": ["a", "b"],  # not a scalar: dropped
            "traceloop.association.properties.": "no key",
            "traceloop.workflow.name": "not an association property",
        }
    )
    i = only(make_request(make_span(attrs)))
    assert i.association_properties == {"scenario": "pix", "attempt": "2"}


def test_multiple_choices_are_all_evaluated() -> None:
    outputs = [text("assistant", "a"), text("assistant", "b")]
    i = only(make_request(make_span(semconv_attrs([text("user", "x")], outputs))))
    assert [m.text for m in i.output_messages] == ["a", "b"]


def three_turns() -> list[dict[str, object]]:
    return [
        text("user", f"Meu CPF é {CPF}"),
        text("assistant", "Anotado."),
        text("user", "E meu saldo?"),
        text("assistant", "R$ 10."),
        text("user", "Obrigado"),
    ]


def test_third_turn_yields_only_new_content() -> None:
    i = only(make_request(make_span(semconv_attrs(three_turns(), [text("assistant", "Nada")]))))
    assert i.input_messages == [Message("user", "Obrigado")]


def test_first_turn_without_assistant_yields_everything() -> None:
    msgs = [text("user", "a"), text("user", "b")]
    assert len(only(make_request(make_span(semconv_attrs(msgs)))).input_messages) == 2


def test_system_messages_inside_input_are_always_kept() -> None:
    msgs = [text("system", "regras"), *three_turns()]
    i = only(make_request(make_span(semconv_attrs(msgs))))
    assert i.system_instructions == [Message("system", "regras")]
    assert i.input_messages == [Message("user", "Obrigado")]


def openllmetry_attrs(
    pairs: list[tuple[str, str]], completions: list[tuple[str, str]]
) -> dict[str, object]:
    attrs: dict[str, object] = {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.response.id": "chatcmpl-123",
        "llm.request.type": "chat",
        # Prompt template attributes are not content.
        "gen_ai.prompt.name": "support",
        "gen_ai.prompt.version": "3",
    }
    for n, (role, content) in enumerate(pairs):
        attrs[f"gen_ai.prompt.{n}.role"] = role
        attrs[f"gen_ai.prompt.{n}.content"] = content
    for n, (role, content) in enumerate(completions):
        attrs[f"gen_ai.completion.{n}.role"] = role
        attrs[f"gen_ai.completion.{n}.content"] = content
    return attrs


def test_both_formats_produce_the_same_interaction() -> None:
    system = [{"type": "text", "content": "Seja breve."}]
    current = only(
        make_request(
            make_span(semconv_attrs(three_turns(), [text("assistant", "De nada")], system=system))
        )
    )
    openllmetry = only(
        make_request(
            make_span(
                openllmetry_attrs(
                    [("system", "Seja breve.")]
                    + [(m["role"], m["parts"][0]["content"]) for m in three_turns()],  # type: ignore[index]
                    [("assistant", "De nada")],
                )
            )
        )
    )
    assert openllmetry == current


def test_openllmetry_indices_sort_numerically() -> None:
    pairs = [("user", str(n)) for n in range(12)]
    i = only(make_request(make_span(openllmetry_attrs(pairs, []))))
    assert [m.text for m in i.input_messages] == [str(n) for n in range(12)]


def test_openllmetry_content_parts_and_tool_calls() -> None:
    attrs = openllmetry_attrs(
        [("user", json.dumps([{"type": "text", "text": "oi"}, {"type": "image_url"}]))], []
    )
    attrs["gen_ai.completion.0.role"] = "assistant"
    attrs["gen_ai.completion.0.tool_calls.0.arguments"] = '{"token": "x"}'
    i = only(make_request(make_span(attrs)))
    assert i.input_messages == [Message("user", "oi")]
    assert i.output_messages == [
        Message("assistant", '{"token": "x"}', (PartSpan("tool_call", 0, 14),))
    ]


def test_skips_non_inference_and_empty_spans() -> None:
    embeddings = make_span({"gen_ai.operation.name": "embeddings"}, span_id=b"\x01" * 8)
    tool = make_span({"gen_ai.operation.name": "execute_tool"}, span_id=b"\x02" * 8)
    http = make_span({"http.request.method": "GET"}, span_id=b"\x03" * 8)
    no_content = make_span({"gen_ai.operation.name": "chat"}, span_id=b"\x04" * 8)
    empty_parts = make_span(
        semconv_attrs([{"role": "user", "parts": [{"type": "blob", "content": "x"}]}]),
        span_id=b"\x05" * 8,
    )
    template_only = make_span(
        {"gen_ai.prompt.name": "support", "gen_ai.prompt.version": "1"}, span_id=b"\x06" * 8
    )
    result = extract(make_request(embeddings, tool, http, no_content, empty_parts, template_only))
    assert result.interactions == []
    assert result.received == 6
    assert result.skipped == {"not_inference": 4, "no_content": 2}


def test_both_formats_produce_the_same_parts() -> None:
    current = only(
        make_request(
            make_span(
                semconv_attrs(
                    [text("user", "consulte")],
                    [
                        {
                            "role": "assistant",
                            "parts": [
                                {"type": "text", "content": "Vou consultar."},
                                {"type": "tool_call", "name": "db", "arguments": '{"q": 1}'},
                            ],
                        }
                    ],
                )
            )
        )
    )
    attrs = openllmetry_attrs([("user", "consulte")], [("assistant", "Vou consultar.")])
    attrs["gen_ai.completion.0.tool_calls.0.arguments"] = '{"q": 1}'
    openllmetry = only(make_request(make_span(attrs)))
    assert openllmetry == current
    assert current.output_messages[0].parts == (
        PartSpan("text", 0, 14),
        PartSpan("tool_call", 15, 23),
    )


def test_message_without_parts_is_all_text() -> None:
    message = Message("assistant", "oi")
    assert message.text_of("text") == "oi"
    assert message.text_of("text", "reasoning") == "oi"
    assert message.text_of("reasoning") == ""


def output_type_of(attrs: dict[str, object]) -> str | None:
    i = only(make_request(make_span({**semconv_attrs([text("user", "x")]), **attrs})))
    return i.output_type  # type: ignore[no-any-return]


def test_reads_output_type() -> None:
    assert output_type_of({}) is None
    assert output_type_of({"gen_ai.output.type": "json"}) == "json"


def test_openllmetry_structured_output_schema_means_json() -> None:
    schema = "gen_ai.request.structured_output_schema"
    assert output_type_of({schema: json.dumps({"type": "object", "properties": {}})}) == "json"
    assert output_type_of({schema: json.dumps({"type": "json_object"})}) == "json"
    assert output_type_of({schema: json.dumps({"type": "text"})}) == "text"
    assert output_type_of({schema: "not json"}) is None
    # The semconv attribute wins.
    assert output_type_of({schema: "{}", "gen_ai.output.type": "text"}) == "text"


def test_reads_finish_reasons_from_the_span() -> None:
    attrs = semconv_attrs([text("user", "x")], [text("assistant", "a"), text("assistant", "b")])
    attrs["gen_ai.response.finish_reasons"] = ["stop", "length"]
    assert only(make_request(make_span(attrs))).finish_reasons == ("stop", "length")
    attrs["gen_ai.response.finish_reasons"] = "content_filter"
    assert only(make_request(make_span(attrs))).finish_reasons == ("content_filter",)


def test_finish_reasons_fall_back_to_the_messages() -> None:
    outputs = [
        {**text("assistant", "a"), "finish_reason": "stop"},
        # Filtered by the provider: no content, but the reason still counts.
        {"role": "assistant", "parts": [], "finish_reason": "content_filter"},
    ]
    attrs = semconv_attrs([text("user", "x")], outputs, as_json=True)
    i = only(make_request(make_span(attrs)))
    assert i.finish_reasons == ("stop", "content_filter")
    assert [m.text for m in i.output_messages] == ["a"]

    attrs["gen_ai.response.finish_reasons"] = ["length"]
    assert only(make_request(make_span(attrs))).finish_reasons == ("length",)


def test_openllmetry_finish_reasons() -> None:
    attrs = openllmetry_attrs([("user", "x")], [("assistant", "a")])
    attrs["gen_ai.completion.0.finish_reason"] = "stop"
    attrs["gen_ai.completion.1.finish_reason"] = "content_filter"
    i = only(make_request(make_span(attrs)))
    assert i.finish_reasons == ("stop", "content_filter")
    assert [m.text for m in i.output_messages] == ["a"]


def test_context_is_the_last_text_messages_before_the_turn() -> None:
    i = only(make_request(make_span(semconv_attrs(three_turns(), [text("assistant", "x")]))))
    assert i.input_messages == [Message("user", "Obrigado")]
    assert i.context_messages == [
        Message("user", f"Meu CPF é {CPF}"),
        Message("assistant", "Anotado."),
        Message("user", "E meu saldo?"),
        Message("assistant", "R$ 10."),
    ]


def test_context_keeps_at_most_four_text_messages_each_capped() -> None:
    long = "a" * 5000
    history = [text("user", "first"), *three_turns()[:-1], text("assistant", long)]
    msgs = [text("system", "regras"), *history, text("user", "e agora?")]
    i = only(make_request(make_span(semconv_attrs(msgs))))
    assert [m.text for m in i.context_messages] == [
        "Anotado.",
        "E meu saldo?",
        "R$ 10.",
        "a" * 1000,
    ]
    assert all(m.role != "system" for m in i.context_messages)


def test_context_leaves_out_tool_calls_and_results() -> None:
    msgs = [
        text("user", "consulte o pedido"),
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "Vou consultar."},
                {"type": "tool_call", "name": "db", "arguments": {"id": 1}},
            ],
        },
        {"role": "tool", "parts": [{"type": "tool_call_response", "response": "ok"}]},
        {"role": "assistant", "parts": [{"type": "tool_call", "name": "db", "arguments": {}}]},
        text("user", "e então?"),
    ]
    i = only(make_request(make_span(semconv_attrs(msgs))))
    assert i.context_messages == [
        Message("user", "consulte o pedido"),
        Message("assistant", "Vou consultar."),
    ]


def test_first_turn_has_no_context() -> None:
    assert only(make_request(make_span(semconv_attrs([text("user", "oi")])))).context_messages == []


def test_spans_from_the_service_itself_are_skipped() -> None:
    span = make_span(semconv_attrs([text("user", "oi")]))
    own = make_request(span, service_name="llm-eval-otel")
    assert extract(own, "llm-eval-otel").skipped == {"self_telemetry": 1}
    assert len(extract(own).interactions) == 1  # only when the own name is known
    other = make_request(span, service_name="support-bot")
    assert len(extract(other, "llm-eval-otel").interactions) == 1
