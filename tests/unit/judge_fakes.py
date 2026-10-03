"""A scripted, deterministic JudgeClient, and interactions for judge tests."""

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from llm_eval_otel.config import Settings
from llm_eval_otel.evaluators.base import GenAIInteraction, Message
from llm_eval_otel.evaluators.relevance import RelevanceJudge
from llm_eval_otel.judge.client import JudgeCall, JudgeError, JudgeResponse, record

MODEL = "fake-judge-1"


@dataclass
class FakeJudgeClient:
    """Answers with ``score`` and ``reason``; records every content it was sent."""

    score: int = 5
    reason: str = "the response answers the question"
    delay_s: float = 0.0
    raise_error: type[JudgeError] | None = None
    tokens: tuple[int, int] = (120, 30)  # (input, output); (0, 0) = no usage reported
    received: list[str] = field(default_factory=list)

    async def judge(self, system: str, content: str, schema: Mapping[str, Any]) -> JudgeResponse:
        start_ns = time.time_ns()
        self.received.append(content)
        await asyncio.sleep(self.delay_s)
        response = JudgeResponse(
            output={"reason": self.reason, "score": self.score},
            model=MODEL,
            input_tokens=self.tokens[0],
            output_tokens=self.tokens[1],
            cache_read_tokens=0,
            finish_reason="stop",
        )
        error = self.raise_error(response) if self.raise_error else None
        usage = response.usage_reported
        record(
            JudgeCall(
                provider_name="openai",
                request_model=MODEL,
                server_address="fake-judge",
                server_port=8080,
                start_ns=start_ns,
                end_ns=time.time_ns(),
                response_model=MODEL,
                input_tokens=self.tokens[0] if usage else None,
                output_tokens=self.tokens[1] if usage else None,
                finish_reason="stop",
                error_type=error.error_type if error else None,
            )
        )
        if error is not None:
            raise error
        return response


def relevance(client: FakeJudgeClient | None = None, **settings: Any) -> RelevanceJudge:
    return RelevanceJudge(client or FakeJudgeClient(), Settings(**settings))


def chat(
    question: str = "Qual o horário de atendimento?",
    answer: str = "De segunda a sexta, das 9h às 18h.",
    *,
    trace_id: bytes = b"\x01" * 16,
    service: str | None = "support-bot",
    context: list[Message] | None = None,
) -> GenAIInteraction:
    return GenAIInteraction(
        trace_id=trace_id,
        span_id=b"\x02" * 8,
        parent_span_id=None,
        trace_flags=1,
        service_name=service,
        operation_name="chat",
        provider_name="openai",
        request_model="gpt-4o-mini",
        response_id=None,
        system_instructions=[],
        input_messages=[Message("user", question)],
        output_messages=[Message("assistant", answer)],
        context_messages=context or [],
    )
