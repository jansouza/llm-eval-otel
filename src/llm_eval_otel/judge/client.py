"""The judge client contracts, the judge's errors, and a record of each call.

Evaluators talk to a :class:`JudgeClient` (a generative judge: prompt in, JSON out) or a
:class:`SystemOneClient` (a decision model: a state and typed questions in, typed answers
out) and know nothing about the provider, the SDK or OpenTelemetry. Each call a client makes
is recorded as a :class:`JudgeCall` (model, endpoint, tokens, finish reason, timing; never
content). The runner collects them with :func:`recording`, the emitter turns them into the
judge's ``chat``/``system_one`` spans and GenAI client metrics, and the lanes settle the
token budget with them.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from llm_eval_otel import semconv


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


@dataclass(frozen=True)
class NoulQuestion:
    """A yes/no question; the answer is the probability of "yes"."""

    instructions: str
    true: str | None = None  # what counts as yes
    false: str | None = None  # what counts as no


@dataclass(frozen=True)
class ScoreQuestion:
    """A rating on an ordered rubric; the answer's score is an index into ``levels``."""

    instructions: str
    levels: tuple[str, ...]  # lowest first; the score of levels[n] is n


Question = NoulQuestion | ScoreQuestion


@dataclass(frozen=True)
class NoulAnswer:
    probability: float  # of "yes", 0 to 1


@dataclass(frozen=True)
class ScoreAnswer:
    score: float  # expected level, 0 to len(levels) - 1; may fall between levels
    confidence: float  # 0 to 1
    probabilities: tuple[float, ...]  # one per level


Answer = NoulAnswer | ScoreAnswer


@dataclass(frozen=True)
class SystemOneResponse:
    answers: Mapping[str, Answer]  # every question id asked, already checked
    model: str  # the model that answered, not the one requested
    input_tokens: int | None  # None when the server did not report usage
    output_tokens: int | None


class SystemOneClient(Protocol):
    async def ask(
        self, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> SystemOneResponse: ...


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
    operation_name: str = semconv.OPERATION_CHAT

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
