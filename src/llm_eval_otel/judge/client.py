"""The judge client contract, the judge's errors, and a record of each call.

Evaluators talk to a :class:`JudgeClient` and know nothing about the provider, the SDK or
OpenTelemetry. Each call the client makes is recorded as a :class:`JudgeCall` (model,
endpoint, tokens, finish reason, timing; never content). The runner collects them with
:func:`recording`, the emitter turns them into the judge's ``chat`` spans and GenAI
client metrics, and the judge lane settles the token budget with them.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol


@dataclass(frozen=True)
class JudgeResponse:
    output: Mapping[str, Any]  # already validated against the schema
    model: str  # the model that answered, not the one requested
    input_tokens: int  # 0 when the server did not report usage
    output_tokens: int
    cache_read_tokens: int
    finish_reason: str

    @property
    def usage_reported(self) -> bool:
        return self.input_tokens > 0 or self.output_tokens > 0


class JudgeClient(Protocol):
    async def judge(
        self, system: str, content: str, schema: Mapping[str, Any]
    ) -> JudgeResponse: ...


class JudgeError(Exception):
    """The judge answered, but not with a usable evaluation.

    Carries no message: it would come from the judge and could quote the content.
    """

    error_type: ClassVar[str] = "judge_error"

    def __init__(self, response: JudgeResponse | None = None) -> None:
        super().__init__()
        self.response = response


class JudgeRefusal(JudgeError):
    error_type = "judge_refusal"


class JudgeTruncated(JudgeError):
    error_type = "judge_truncated"


class JudgeInvalidOutput(JudgeError):
    error_type = "judge_invalid_output"


class JudgeConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class JudgeCall:
    """One request to the judge, as the telemetry needs it. Never any content."""

    provider_name: str
    request_model: str
    server_address: str | None
    server_port: int | None
    start_ns: int
    end_ns: int
    response_model: str | None = None
    input_tokens: int | None = None  # None when the server did not report usage
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    finish_reason: str | None = None
    error_type: str | None = None

    @property
    def tokens_used(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


_calls: ContextVar[list[JudgeCall] | None] = ContextVar("llm_eval_judge_calls", default=None)


@contextmanager
def recording() -> Iterator[list[JudgeCall]]:
    """Collect the judge calls made inside the block, including ones that failed."""
    calls: list[JudgeCall] = []
    token = _calls.set(calls)
    try:
        yield calls
    finally:
        _calls.reset(token)


def record(call: JudgeCall) -> None:
    calls = _calls.get()
    if calls is not None:
        calls.append(call)
