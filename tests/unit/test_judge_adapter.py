"""The ``openai`` adapter against the fake OpenAI-compatible server in tools/.

The adapter is the same code for the OpenAI API and for local servers, so this exercises
the production path: the SDK, the three response format modes, refusals, truncation,
invalid output, missing usage and timeouts.
"""

import asyncio
import json
import threading
from collections.abc import Iterator
from typing import Any

import openai
import pytest
from fake_judge_server import FakeJudgeServer

from llm_eval_otel.config import Settings
from llm_eval_otel.evaluators.relevance import SCHEMA, SYSTEM_PROMPT
from llm_eval_otel.judge.client import (
    JudgeConfigError,
    JudgeError,
    JudgeInvalidOutput,
    JudgeRefusal,
    JudgeTruncated,
    recording,
)
from llm_eval_otel.judge.evaluator import envelope
from llm_eval_otel.judge.openai_adapter import OpenAIJudge, ResponseFormat

MODEL = "gpt-5-mini-2025-08-07"
CONTENT = envelope(
    {
        "context": [],
        "request": ["Qual o horário de atendimento?"],
        "response": ["O atendimento é das 9h às 18h, de segunda a sexta."],
    }
)


def start(**options: Any) -> FakeJudgeServer:
    server = FakeJudgeServer(("127.0.0.1", 0), **options)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def server() -> Iterator[FakeJudgeServer]:
    server = start()
    yield server
    server.shutdown()


def adapter(server: FakeJudgeServer, **options: Any) -> OpenAIJudge:
    return OpenAIJudge(model=MODEL, base_url=server.base_url, api_key="test", **options)


@pytest.mark.parametrize("mode", ["json_schema", "json_object", "none"])
async def test_three_response_format_modes(server: FakeJudgeServer, mode: ResponseFormat) -> None:
    judge = adapter(server, response_format=mode)
    with recording() as calls:
        response = await judge.judge(SYSTEM_PROMPT, CONTENT, SCHEMA)
    assert response.output == {
        "reason": "fake judge: the response shares 1 content words with the request",
        "score": 5,
    }
    assert response.model == MODEL and response.finish_reason == "stop"

    [request] = server.requests
    system, user = request["messages"]
    assert user == {"role": "user", "content": CONTENT}
    if mode == "json_schema":
        assert request["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "evaluation", "schema": SCHEMA, "strict": True},
        }
        assert system == {"role": "system", "content": SYSTEM_PROMPT}
    else:
        # Not enforced by the server: the schema goes in the prompt, after the fixed part.
        assert system["content"].startswith(SYSTEM_PROMPT)
        assert json.dumps(SCHEMA) in system["content"]
        if mode == "json_object":
            assert request["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in request

    [call] = calls
    assert (call.provider_name, call.request_model, call.response_model) == (
        "openai",
        MODEL,
        MODEL,
    )
    assert (call.server_address, call.server_port) == ("127.0.0.1", server.server_address[1])
    assert call.input_tokens and call.output_tokens and call.finish_reason == "stop"
    assert call.error_type is None and call.end_ns >= call.start_ns


async def test_server_without_strict_mode(server: FakeJudgeServer) -> None:
    strictless = start(reject_json_schema=True)
    try:
        with recording() as calls, pytest.raises(openai.BadRequestError):
            await adapter(strictless).judge(SYSTEM_PROMPT, CONTENT, SCHEMA)
        assert calls[0].error_type == "BadRequestError"
        response = await adapter(strictless, response_format="json_object").judge(
            SYSTEM_PROMPT, CONTENT, SCHEMA
        )
        assert response.output["score"] == 5
    finally:
        strictless.shutdown()


@pytest.mark.parametrize(
    ("marker", "error"),
    [
        ("FAKE_JUDGE:refuse", JudgeRefusal),
        ("FAKE_JUDGE:content_filter", JudgeRefusal),
        ("FAKE_JUDGE:length", JudgeTruncated),
        ("FAKE_JUDGE:invalid", JudgeInvalidOutput),
    ],
)
async def test_unusable_answers_raise_and_still_record_usage(
    server: FakeJudgeServer, marker: str, error: type[JudgeError]
) -> None:
    with recording() as calls, pytest.raises(error) as raised:
        await adapter(server).judge(SYSTEM_PROMPT, f"{CONTENT}\n{marker}", SCHEMA)
    assert str(raised.value) == ""  # no message: it could quote the judge
    [call] = calls
    assert call.error_type == error.error_type
    assert call.input_tokens and call.input_tokens > 0  # the call still cost tokens


async def test_missing_usage_is_recorded_as_unknown(server: FakeJudgeServer) -> None:
    with recording() as calls:
        response = await adapter(server).judge(
            SYSTEM_PROMPT, f"{CONTENT}\nFAKE_JUDGE:no_usage", SCHEMA
        )
    assert not response.usage_reported
    assert (calls[0].input_tokens, calls[0].output_tokens, calls[0].tokens_used) == (
        None,
        None,
        None,
    )


async def test_cached_prefix_is_reported(server: FakeJudgeServer) -> None:
    judge = adapter(server)
    first = await judge.judge(SYSTEM_PROMPT, CONTENT, SCHEMA)
    second = await judge.judge(SYSTEM_PROMPT, CONTENT, SCHEMA)
    assert first.cache_read_tokens == 0 and second.cache_read_tokens > 0


async def test_optional_parameters_only_when_configured(server: FakeJudgeServer) -> None:
    await adapter(server).judge(SYSTEM_PROMPT, CONTENT, SCHEMA)
    await adapter(server, temperature=0.0, reasoning_effort="low").judge(
        SYSTEM_PROMPT, CONTENT, SCHEMA
    )
    plain, tuned = server.requests
    assert "temperature" not in plain and "reasoning_effort" not in plain
    assert (tuned["temperature"], tuned["reasoning_effort"]) == (0.0, "low")
    assert plain["max_completion_tokens"] == 1024


async def test_timeout_cancels_the_call_and_records_it() -> None:
    slow = start(delay_s=2)
    try:
        with recording() as calls, pytest.raises(TimeoutError):
            await asyncio.wait_for(adapter(slow).judge(SYSTEM_PROMPT, CONTENT, SCHEMA), 0.2)
        assert calls[0].error_type == "timeout"
    finally:
        slow.shutdown()


async def test_server_down_raises_the_sdk_error() -> None:
    down = OpenAIJudge(model=MODEL, base_url="http://127.0.0.1:9/v1", api_key="test")
    with recording() as calls, pytest.raises(openai.APIConnectionError):
        await down.judge(SYSTEM_PROMPT, CONTENT, SCHEMA)
    assert calls[0].error_type == "APIConnectionError"


def test_openai_api_by_default() -> None:
    judge = OpenAIJudge(model=MODEL, api_key="test")
    assert (judge.server_address, judge.server_port) == ("api.openai.com", 443)


def test_from_settings_requires_a_model() -> None:
    with pytest.raises(JudgeConfigError):
        OpenAIJudge.from_settings(Settings(), evaluator="relevance", timeout_s=30)
