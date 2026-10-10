"""The ``openai`` judge adapter: the official SDK over the Chat Completions API.

Chat Completions is the API OpenAI-compatible servers implement, so the same code talks to
the OpenAI API (``LLM_EVAL_LLM_JUDGE_BASE_URL`` unset) and to a judge the adopter hosts (vLLM,
Ollama, or a gateway such as LiteLLM in front of other providers).

No instrumentation library wraps the SDK: they can record message content when an
environment variable is set. The adapter records each call as a :class:`JudgeCall` instead.
"""

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from llm_eval_otel import semconv
from llm_eval_otel.config import Settings
from llm_eval_otel.judge.client import (
    JudgeCall,
    JudgeConfigError,
    JudgeError,
    JudgeInvalidOutput,
    JudgeRefusal,
    JudgeResponse,
    JudgeTruncated,
    record,
)
from llm_eval_otel.judge.schema import is_valid

ResponseFormat = Literal["json_schema", "json_object", "none"]

SCHEMA_NAME = "evaluation"
_DEFAULT_PORTS = {"https": 443, "http": 80}


def describe_schema(schema: Mapping[str, Any]) -> str:
    """Appended to the system prompt when the server does not enforce the schema."""
    return (
        "\n\nReply with only a JSON object, with no text before or after it, that matches "
        "this JSON schema:\n" + json.dumps(schema, ensure_ascii=False)
    )


def parse_object(text: str | None) -> Any:
    """The JSON object in the reply, tolerating Markdown fences or prose around it."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except ValueError:
        return None


class OpenAIJudge:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        response_format: ResponseFormat = "json_schema",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 1024,
        timeout_s: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.response_format = response_format
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        # The SDK already retries 429 and 5xx; one retry fits inside the evaluator's timeout.
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=1, timeout=timeout_s
        )
        url = self._client.base_url
        self.server_address: str | None = url.host or None
        self.server_port: int | None = url.port or _DEFAULT_PORTS.get(url.scheme)

    @classmethod
    def from_settings(
        cls, settings: Settings, *, evaluator: str, timeout_s: float
    ) -> "OpenAIJudge":
        if not settings.llm_judge_model:
            raise JudgeConfigError(f"{evaluator} needs LLM_EVAL_LLM_JUDGE_MODEL")
        return cls(
            model=settings.llm_judge_model,
            base_url=settings.llm_judge_base_url,
            response_format=settings.llm_judge_response_format,
            temperature=settings.llm_judge_temperature,
            reasoning_effort=settings.llm_judge_reasoning_effort,
            max_output_tokens=settings.llm_judge_max_output_tokens,
            timeout_s=timeout_s,
        )

    def _request(self, system: str, content: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        # The fixed judge prompt goes first, as a stable prefix the provider can cache.
        if self.response_format != "json_schema":
            system += describe_schema(schema)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.response_format == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME, "schema": dict(schema), "strict": True},
            }
        elif self.response_format == "json_object":
            request["response_format"] = {"type": "json_object"}
        # Servers and models disagree on these two: send them only when configured.
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        return request

    def _call(
        self,
        start_ns: int,
        response: JudgeResponse | None = None,
        error_type: str | None = None,
    ) -> JudgeCall:
        usage = response is not None and response.usage_reported
        return JudgeCall(
            provider_name=semconv.PROVIDER_OPENAI,
            request_model=self.model,
            server_address=self.server_address,
            server_port=self.server_port,
            start_ns=start_ns,
            end_ns=time.time_ns(),
            response_model=response.model if response else None,
            input_tokens=response.input_tokens if response and usage else None,
            output_tokens=response.output_tokens if response and usage else None,
            cache_read_tokens=response.cache_read_tokens if response and usage else None,
            finish_reason=response.finish_reason if response else None,
            error_type=error_type,
        )

    async def judge(self, system: str, content: str, schema: Mapping[str, Any]) -> JudgeResponse:
        start_ns = time.time_ns()
        try:
            completion: ChatCompletion = await self._client.chat.completions.create(
                **self._request(system, content, schema)
            )
        except asyncio.CancelledError:
            # The runner's timeout cancels the HTTP call.
            record(self._call(start_ns, error_type=semconv.ERROR_TIMEOUT))
            raise
        except Exception as exc:
            record(self._call(start_ns, error_type=type(exc).__name__))
            raise
        try:
            response = self._check(completion, schema)
        except JudgeError as exc:
            record(self._call(start_ns, exc.response, error_type=exc.error_type))
            raise
        record(self._call(start_ns, response))
        return response

    @staticmethod
    def _check(completion: ChatCompletion, schema: Mapping[str, Any]) -> JudgeResponse:
        usage = completion.usage
        details = usage.prompt_tokens_details if usage else None
        choice = completion.choices[0] if completion.choices else None
        finish_reason = choice.finish_reason if choice else ""
        output = parse_object(choice.message.content) if choice else None
        response = JudgeResponse(
            output=output if isinstance(output, dict) else {},
            model=completion.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_read_tokens=(details.cached_tokens or 0) if details else 0,
            finish_reason=finish_reason,
        )
        if choice is None:
            raise JudgeInvalidOutput(response)
        if choice.message.refusal or finish_reason == semconv.FINISH_REASON_CONTENT_FILTER:
            raise JudgeRefusal(response)
        if finish_reason == semconv.FINISH_REASON_LENGTH:
            raise JudgeTruncated(response)
        if not isinstance(output, dict) or not is_valid(output, schema):
            raise JudgeInvalidOutput(response)
        return response
