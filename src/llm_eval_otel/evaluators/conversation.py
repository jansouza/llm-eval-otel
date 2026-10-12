"""What judges read of a turn: the user's new messages, the response's text, the context.

Shared by ``relevance`` and the Jev checks, so both judges see the same conversation.
"""

from collections.abc import Callable
from typing import Any

from llm_eval_otel import semconv
from llm_eval_otel.evaluators.base import GenAIInteraction, Message


def user_texts(messages: list[Message]) -> list[str]:
    return [
        text
        for m in messages
        if m.role == semconv.ROLE_USER and (text := m.text_of(semconv.PART_TEXT).strip())
    ]


def response_texts(messages: list[Message]) -> list[str]:
    """Text parts only: no reasoning, no tool calls."""
    return [text for m in messages if (text := m.text_of(semconv.PART_TEXT).strip())]


def calls_tools(messages: list[Message]) -> bool:
    return any(p.type == semconv.PART_TOOL_CALL for m in messages for p in m.parts)


def conversation(interaction: GenAIInteraction, text: Callable[[str], str]) -> dict[str, list[Any]]:
    """``context``, ``request`` and ``response``, every text passed through ``text`` (masking).

    The system instructions are left out: they repeat on every call and judging them is not
    the point.
    """
    return {
        "context": [{"role": m.role, "text": text(m.text)} for m in interaction.context_messages],
        "request": [text(t) for t in user_texts(interaction.input_messages)],
        "response": [text(t) for t in response_texts(interaction.output_messages)],
    }
