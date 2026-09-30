import json

import pytest
from otlp import make_request, make_span, semconv_attrs, text

from llm_eval_otel.evaluators.base import GenAIInteraction, Message, PartSpan
from llm_eval_otel.evaluators.output_format import OutputFormatValidator
from llm_eval_otel.extract.genai import extract

VALID = json.dumps({"cliente": "Maria", "saldo": 10.5, "itens": [1, 2, 3]})


def interaction(
    *outputs: Message, output_type: str | None = "json", finish_reasons: tuple[str, ...] = ()
) -> GenAIInteraction:
    return GenAIInteraction(
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        parent_span_id=None,
        trace_flags=1,
        service_name="svc",
        operation_name="chat",
        provider_name="openai",
        request_model="gpt-4o-mini",
        response_id=None,
        system_instructions=[],
        input_messages=[Message("user", "responda em JSON")],
        output_messages=list(outputs),
        output_type=output_type,
        finish_reasons=finish_reasons,
    )


def assistant(content: str) -> Message:
    return Message("assistant", content)


async def test_valid_json_passes() -> None:
    result = await OutputFormatValidator().evaluate(interaction(assistant(VALID)))
    assert (result.score, result.label) == (1.0, "pass")
    assert result.explanation == "valid_json=1 of 1 (output)"
    assert result.attributes == {}


async def test_truncated_json_with_length_is_truncated() -> None:
    result = await OutputFormatValidator().evaluate(
        interaction(assistant(VALID[:25]), finish_reasons=("length",))
    )
    assert (result.score, result.label) == (0.0, "fail")
    assert result.attributes == {"llm_eval.output_format.error": "truncated"}
    assert result.explanation == (
        "invalid_json=1 of 1 (output): Unterminated string starting at char 21, "
        "finish_reason=length"
    )


async def test_markdown_fence_fails() -> None:
    fenced = f"```json\n{VALID}\n```"
    result = await OutputFormatValidator().evaluate(interaction(assistant(fenced)))
    assert result.label == "fail"
    assert result.attributes == {"llm_eval.output_format.error": "syntax"}
    assert result.explanation == "invalid_json=1 of 1 (output): Expecting value at char 0"


async def test_explanation_never_quotes_the_text() -> None:
    secret = '{"cpf": "529.982.247-25", "nome": "Maria" "extra"}'
    result = await OutputFormatValidator().evaluate(interaction(assistant(secret)))
    assert result.explanation == (
        "invalid_json=1 of 1 (output): Expecting ',' delimiter at char 42"
    )
    assert "529" not in (result.explanation or "")


async def test_empty_text_fails() -> None:
    result = await OutputFormatValidator().evaluate(interaction(assistant("  \n")))
    assert result.attributes == {"llm_eval.output_format.error": "empty"}
    assert result.explanation == "invalid_json=1 of 1 (output): empty"


async def test_score_is_share_of_valid_outputs() -> None:
    result = await OutputFormatValidator().evaluate(
        interaction(assistant(VALID), assistant("{oops"))
    )
    assert result.score == 0.5
    assert result.label == "fail"
    assert result.explanation is not None
    assert result.explanation.startswith("invalid_json=1 of 2 (output): ")


async def test_reasoning_before_json_does_not_interfere() -> None:
    reasoning = "The user wants JSON; I will build the object."
    message = Message(
        "assistant",
        f"{reasoning}\n{VALID}",
        (
            PartSpan("reasoning", 0, len(reasoning)),
            PartSpan("text", len(reasoning) + 1, len(reasoning) + 1 + len(VALID)),
        ),
    )
    result = await OutputFormatValidator().evaluate(interaction(message))
    assert result.label == "pass"


@pytest.mark.parametrize("output_type", [None, "text", "image"])
def test_applies_only_to_json_output(output_type: str | None) -> None:
    assert not OutputFormatValidator().applies_to(
        interaction(assistant(VALID), output_type=output_type)
    )


def test_does_not_apply_to_tool_call_only_output() -> None:
    tool_only = Message("assistant", '{"q": 1}', (PartSpan("tool_call", 0, 8),))
    assert not OutputFormatValidator().applies_to(interaction(tool_only))
    assert OutputFormatValidator().applies_to(interaction(assistant(VALID)))


def test_span_without_output_type_does_not_apply() -> None:
    span = make_span(semconv_attrs([text("user", "x")], [text("assistant", VALID)]))
    [i] = extract(make_request(span)).interactions
    assert not OutputFormatValidator().applies_to(i)
