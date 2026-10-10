"""The ``typesafe`` adapter against the fake System One server in tools/.

The real SDK, over HTTP: the request it sends, the answers in the service's types, usage,
the API errors as their class names, invalid output, timeouts, and the SDK's DEBUG logging
of request bodies kept out of the service's logs.
"""

import asyncio
import json
import logging
import threading
from collections.abc import Iterator
from typing import Any

import pytest
import typesafe_sdk
from fake_judge_server import FakeJudgeServer
from typesafe_sdk import AsyncTypeSafeClient
from typesafe_sdk import SystemOneResponse as SDKResponse
from typesafe_sdk._core.logging import setup_logging

from llm_eval_otel.config import Settings
from llm_eval_otel.judge.client import (
    JudgeConfigError,
    JudgeInvalidOutput,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    recording,
)
from llm_eval_otel.judge.typesafe_adapter import TypeSafeJudge

MODEL = "jev-1.13.0"
STATE = {
    "context": [],
    "request": ["Qual o horário de atendimento da loja?"],
    "response": ["A loja atende das 9h às 18h, de segunda a sexta."],
}
LEVELS = ("unrelated", "touches the topic", "partially", "with gaps", "fully")
QUESTIONS: dict[str, Question] = {
    "jev_relevance": ScoreQuestion("How well does `response` address `request`?", LEVELS),
    "jev_refusal": NoulQuestion("Does `response` refuse?", "it declines", "it answers"),
    "jev_toxicity": NoulQuestion("Is `response` offensive?"),
}


def start(**options: Any) -> FakeJudgeServer:
    server = FakeJudgeServer(("127.0.0.1", 0), **options)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def server() -> Iterator[FakeJudgeServer]:
    server = start()
    yield server
    server.shutdown()


def adapter(server: FakeJudgeServer, **options: Any) -> TypeSafeJudge:
    return TypeSafeJudge(model=MODEL, base_url=server.root_url, api_key="test", **options)


def marked(marker: str) -> dict[str, Any]:
    return {**STATE, "request": [f"{STATE['request'][0]} {marker}"]}


async def test_one_request_answers_every_question(server: FakeJudgeServer) -> None:
    with recording() as calls:
        response = await adapter(server).ask(STATE, QUESTIONS)

    [request] = server.requests
    assert (request["model"], request["state"]) == (MODEL, STATE)
    assert request["questions"] == {
        "jev_relevance": {
            "type": "score",
            "instructions": "How well does `response` address `request`?",
            "criteria": list(LEVELS),
        },
        "jev_refusal": {
            "type": "noul",
            "instructions": "Does `response` refuse?",
            "criteria": {"true": "it declines", "false": "it answers"},
        },
        "jev_toxicity": {"type": "noul", "instructions": "Is `response` offensive?"},
    }

    relevance = response.answers["jev_relevance"]
    assert isinstance(relevance, ScoreAnswer)
    assert relevance.score == pytest.approx(3.8)  # rated 5: 0.8 on the top level
    assert relevance.confidence == 0.8 and len(relevance.probabilities) == 5
    assert response.answers["jev_refusal"] == NoulAnswer(0.1)
    assert response.answers["jev_toxicity"] == NoulAnswer(0.1)
    assert response.model == MODEL and response.input_tokens and response.output_tokens == 6

    [call] = calls
    assert (call.provider_name, call.operation_name) == ("typesafe", "system_one")
    assert (call.request_model, call.response_model) == (MODEL, MODEL)
    assert (call.server_address, call.server_port) == ("127.0.0.1", server.server_address[1])
    assert call.input_tokens == response.input_tokens and call.output_tokens == 6
    assert call.finish_reason is None and call.error_type is None
    assert call.end_ns >= call.start_ns


async def test_a_missing_answer_is_invalid_output(server: FakeJudgeServer) -> None:
    with recording() as calls, pytest.raises(JudgeInvalidOutput) as raised:
        await adapter(server).ask(marked("FAKE_JUDGE:invalid"), QUESTIONS)
    assert str(raised.value) == "" and raised.value.__cause__ is None
    [call] = calls
    assert call.error_type == "judge_invalid_output"
    assert call.input_tokens  # the call still cost tokens


def sdk_response(answers: dict[str, Any]) -> SDKResponse:
    body = {"model": MODEL, "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}
    return SDKResponse.model_validate_json(json.dumps(body))  # as the SDK decodes it


SCORE_OK = {
    "type": "score",
    "score": 2.5,
    "confidence": 0.7,
    "legend": {str(n): level for n, level in enumerate(LEVELS)},
    "probabilities": {"0": 0.0, "1": 0.1, "2": 0.4, "3": 0.4, "4": 0.1},
}


@pytest.mark.parametrize(
    "answers",
    [
        {"jev_relevance": {"type": "noul", "noul": 0.5}},  # the wrong type
        {"jev_relevance": {**SCORE_OK, "score": 4.5}},  # above the top level
        {"jev_relevance": {**SCORE_OK, "confidence": 1.2}},
        {"jev_relevance": {**SCORE_OK, "probabilities": {"0": 0.5, "1": 0.5}}},  # levels missing
        {"jev_refusal": {"type": "noul", "noul": 1.5}},
        {"jev_refusal": {"type": "noul", "noul": float("nan")}},
        {"jev_refusal": {"type": "score", **{k: v for k, v in SCORE_OK.items() if k != "type"}}},
    ],
)
def test_answers_must_match_the_question(answers: dict[str, Any]) -> None:
    complete = {
        "jev_relevance": SCORE_OK,
        "jev_refusal": {"type": "noul", "noul": 0.2},
        "jev_toxicity": {"type": "noul", "noul": 0.2},
    }
    with pytest.raises(JudgeInvalidOutput):
        TypeSafeJudge._check(sdk_response({**complete, **answers}), QUESTIONS)
    assert TypeSafeJudge._check(sdk_response(complete), QUESTIONS)["jev_relevance"] == (
        ScoreAnswer(2.5, 0.7, (0.0, 0.1, 0.4, 0.4, 0.1))
    )


async def test_missing_usage_is_recorded_as_unknown(server: FakeJudgeServer) -> None:
    with recording() as calls:
        response = await adapter(server).ask(marked("FAKE_JUDGE:no_usage"), QUESTIONS)
    assert (response.input_tokens, response.output_tokens) == (None, None)
    assert calls[0].tokens_used is None


@pytest.mark.parametrize(
    ("marker", "error", "requests"),
    [
        # Retried once by the SDK, after the server's retry-after.
        ("FAKE_JEV:overloaded", typesafe_sdk.TypeSafeInternalServerError, 2),
        ("FAKE_JEV:rate_limited", typesafe_sdk.TypeSafeRateLimitError, 2),
        ("FAKE_JEV:422", typesafe_sdk.TypeSafeUnprocessableEntityError, 1),
    ],
)
async def test_api_errors_are_recorded_by_class_name(
    server: FakeJudgeServer, marker: str, error: type[Exception], requests: int
) -> None:
    with recording() as calls, pytest.raises(error):
        await adapter(server).ask(marked(marker), QUESTIONS)
    assert len(server.requests) == requests
    [call] = calls
    assert call.error_type == error.__name__ and call.tokens_used is None


async def test_timeout_cancels_the_call_and_records_it() -> None:
    slow = start(delay_s=2)
    try:
        with recording() as calls, pytest.raises(TimeoutError):
            await asyncio.wait_for(adapter(slow).ask(STATE, QUESTIONS), 0.2)
        assert calls[0].error_type == "timeout"
    finally:
        slow.shutdown()


async def test_server_down_raises_the_sdk_error() -> None:
    down = TypeSafeJudge(model=MODEL, base_url="http://127.0.0.1:9", api_key="test")
    with recording() as calls, pytest.raises(typesafe_sdk.TypeSafeAPIConnectionError):
        await down.ask(STATE, QUESTIONS)
    assert calls[0].error_type == "TypeSafeAPIConnectionError"


async def test_no_state_text_in_the_logs_even_with_debug(
    server: FakeJudgeServer, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """TYPESAFE_LOG_LEVEL=debug makes the SDK log request and response bodies."""
    secret_text = "texto-do-usuario-que-nunca-vai-para-o-log"
    state = {**STATE, "request": [secret_text]}
    caplog.set_level(logging.DEBUG)  # every handler and the root logger
    monkeypatch.setenv("TYPESAFE_LOG_LEVEL", "debug")
    sdk_logger = logging.getLogger("typesafe_sdk")
    monkeypatch.setattr(sdk_logger, "level", sdk_logger.level)  # restored after the test
    setup_logging()  # what importing the SDK does with the variable set
    assert sdk_logger.level == logging.DEBUG

    judge = adapter(server)
    await judge.ask(state, QUESTIONS)
    with pytest.raises(typesafe_sdk.TypeSafeUnprocessableEntityError):
        await judge.ask({**state, "response": ["FAKE_JEV:422"]}, QUESTIONS)
    assert secret_text not in caplog.text

    # The control: with the SDK's logger left at DEBUG, the body would be there.
    sdk_logger.setLevel(logging.DEBUG)
    await judge.ask(state, QUESTIONS)
    assert secret_text in caplog.text


async def test_fake_server_lists_models(server: FakeJudgeServer) -> None:
    async with AsyncTypeSafeClient(api_key="test", base_url=server.root_url) as client:
        models = await client.models.list()
    assert [m.name for m in models.models] == ["jev-1.13.0", "jev-latest"]


def test_typesafe_api_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    judge = TypeSafeJudge(model=MODEL, api_key="test")
    assert (judge.server_address, judge.server_port) == ("api.typesafe.ai", 443)


def test_from_settings_requires_a_model() -> None:
    with pytest.raises(JudgeConfigError, match="jev_refusal needs LLM_EVAL_JEV_JUDGE_MODEL"):
        TypeSafeJudge.from_settings(Settings(), evaluator="jev_refusal", timeout_s=5)
