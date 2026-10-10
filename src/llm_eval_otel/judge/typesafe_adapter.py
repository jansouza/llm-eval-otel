"""The ``typesafe`` adapter: TypeSafe's System One API (the Jev models) through its SDK.

The SDK's types stay in this module: evaluators see :class:`SystemOneClient` and the
service's own question and answer types.

Two things keep content in: the SDK logs request and response bodies at ``DEBUG`` (on with
``TYPESAFE_LOG_LEVEL=debug``), so its logger is held at ``WARNING`` once the client exists;
and an API error's message can quote the request (a 422 echoes the invalid input), so only
the exception's class name is ever recorded. No instrumentation library wraps the SDK.
"""

import asyncio
import logging
import math
import os
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeAPIResponseValidationError,
)
from typesafe_sdk import NoulAnswer as SDKNoulAnswer
from typesafe_sdk import ScoreAnswer as SDKScoreAnswer
from typesafe_sdk import SystemOneResponse as SDKResponse
from typesafe_sdk.constants import BASE_URL_ENV, DEFAULT_BASE_URL

from llm_eval_otel import semconv
from llm_eval_otel.config import Settings
from llm_eval_otel.judge.client import (
    Answer,
    JudgeCall,
    JudgeConfigError,
    JudgeError,
    JudgeInvalidOutput,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneResponse,
    record,
)

SDK_LOGGER = "typesafe_sdk"
_DEFAULT_PORTS = {"https": 443, "http": 80}


def quiet_sdk_logger() -> None:
    """Hold the SDK's logger at WARNING: below that it logs request and response bodies."""
    logging.getLogger(SDK_LOGGER).setLevel(logging.WARNING)


def _sdk_question(question: Question) -> Noul | Score:
    if isinstance(question, ScoreQuestion):
        return Score(instructions=question.instructions, criteria=list(question.levels))
    criteria = {"true": question.true, "false": question.false}
    return Noul(
        instructions=question.instructions,
        criteria={k: v for k, v in criteria.items() if v is not None} or None,  # type: ignore[arg-type]
    )


def _unit(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _answer(question: Question, answer: Any) -> Answer | None:
    """The answer in the service's types, or None when it doesn't fit the question."""
    if isinstance(question, NoulQuestion):
        if isinstance(answer, SDKNoulAnswer) and _unit(answer.noul):
            return NoulAnswer(answer.noul)
        return None
    if not isinstance(answer, SDKScoreAnswer):
        return None
    top = len(question.levels) - 1
    probabilities = tuple(answer.probabilities.get(level, math.nan) for level in range(top + 1))
    if (
        math.isfinite(answer.score)
        and 0.0 <= answer.score <= top
        and _unit(answer.confidence)
        and all(_unit(p) for p in probabilities)
    ):
        return ScoreAnswer(answer.score, answer.confidence, probabilities)
    return None


class TypeSafeJudge:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        timeout_s: float = 5.0,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        base_url = base_url or os.environ.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
        # The SDK already retries 429 and 529 (respecting retry-after); one retry, within the
        # evaluator's timeout.
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout_s,
            retry=RetryPolicy(max_retries=1, timeout=timeout_s),
        )
        # After the client: the SDK applies TYPESAFE_LOG_LEVEL when it is imported.
        quiet_sdk_logger()
        url = urlsplit(base_url)
        self.server_address: str | None = url.hostname or None
        self.server_port: int | None = url.port or _DEFAULT_PORTS.get(url.scheme)

    @classmethod
    def from_settings(
        cls, settings: Settings, *, evaluator: str, timeout_s: float
    ) -> "TypeSafeJudge":
        if not settings.jev_judge_model:
            raise JudgeConfigError(f"{evaluator} needs LLM_EVAL_JEV_JUDGE_MODEL")
        return cls(
            model=settings.jev_judge_model,
            base_url=settings.jev_judge_base_url,
            timeout_s=timeout_s,
        )

    def _call(
        self,
        start_ns: int,
        response: SDKResponse | None = None,
        error_type: str | None = None,
    ) -> JudgeCall:
        usage = response.usage if response is not None else None
        return JudgeCall(
            provider_name=semconv.PROVIDER_TYPESAFE,
            request_model=self.model,
            server_address=self.server_address,
            server_port=self.server_port,
            start_ns=start_ns,
            end_ns=time.time_ns(),
            response_model=response.model if response is not None else None,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            error_type=error_type,
            operation_name=semconv.OPERATION_SYSTEM_ONE,
        )

    async def ask(
        self, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> SystemOneResponse:
        start_ns = time.time_ns()
        try:
            response = await self._client.system_one(
                state=dict(state),
                questions={qid: _sdk_question(q) for qid, q in questions.items()},
            )
        except asyncio.CancelledError:
            # The runner's timeout cancels the HTTP call.
            record(self._call(start_ns, error_type=semconv.ERROR_TIMEOUT))
            raise
        except TypeSafeAPIResponseValidationError:
            record(self._call(start_ns, error_type=JudgeInvalidOutput.error_type))
            # Not chained: the SDK's error carries the response body.
            raise JudgeInvalidOutput() from None
        except Exception as exc:
            record(self._call(start_ns, error_type=type(exc).__name__))
            raise
        try:
            answers = self._check(response, questions)
        except JudgeError as exc:
            record(self._call(start_ns, response, error_type=exc.error_type))
            raise
        record(self._call(start_ns, response))
        usage = response.usage
        return SystemOneResponse(answers, response.model, usage.input_tokens, usage.output_tokens)

    @staticmethod
    def _check(response: SDKResponse, questions: Mapping[str, Question]) -> dict[str, Answer]:
        """Every id asked must come back with the type asked and values in range."""
        answers: dict[str, Answer] = {}
        for qid, question in questions.items():
            answer = _answer(question, response.answers.get(qid))
            if answer is None:
                raise JudgeInvalidOutput()
            answers[qid] = answer
        return answers
