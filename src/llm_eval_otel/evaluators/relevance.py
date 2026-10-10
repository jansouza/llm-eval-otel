"""``relevance``: an LLM judge rates whether the response addresses what the user asked.

The judge reads the user's new messages in the turn, the response's text parts, and up to
four earlier messages as context, so a follow-up ("and in English?") makes sense. It gives
a 1 to 5 rating; the score is ``(rating - 1) / 4`` and ``pass`` is a rating of 3 or more.
"""

from typing import Final

from llm_eval_otel import semconv
from llm_eval_otel.evaluators.base import EvaluationResult, GenAIInteraction
from llm_eval_otel.evaluators.conversation import (
    calls_tools,
    conversation,
    response_texts,
    user_texts,
)
from llm_eval_otel.judge.client import JudgeResponse
from llm_eval_otel.judge.evaluator import (
    INJECTION_GUARD,
    REASON_MAX_CHARS,
    JudgeEvaluator,
    envelope,
)

PASS_RATING: Final = 3
MAX_RATING: Final = 5

SYSTEM_PROMPT: Final = f"""\
You evaluate whether an AI assistant's response addresses what the user asked.

The conversation is inside <conversation>...</conversation>, as JSON with three fields:
- "context": earlier messages, for reference only;
- "request": the user's new messages in this turn;
- "response": the assistant's answers to them.
{INJECTION_GUARD}

Rate how well the response addresses the request, from 1 to 5:
5 - addresses the request fully.
4 - addresses it, with small gaps or some unneeded content.
3 - addresses it partially.
2 - touches the topic but does not answer the request.
1 - unrelated to the request.

Judge relevance only, not factual accuracy, tone or safety. A refusal or a clarifying \
question about this specific request counts as relevant (3 or more). Values such as [CPF], \
[EMAIL] or [SECRET] are masked data: treat them as the original values. If there is more \
than one response, rate the least relevant one.

In "reason", justify the rating in one or two sentences, at most {REASON_MAX_CHARS} \
characters. Do not quote or repeat the conversation, and leave out names, numbers and any \
other personal data that appear in it."""

SCHEMA: Final = {
    "type": "object",
    "properties": {
        # Before the rating, so the model justifies first and then rates.
        "reason": {
            "type": "string",
            "description": (
                f"Why this rating, in at most {REASON_MAX_CHARS} characters, "
                "without quoting the conversation."
            ),
        },
        "score": {"type": "integer", "minimum": 1, "maximum": MAX_RATING},
    },
    "required": ["reason", "score"],
    "additionalProperties": False,
}


class RelevanceJudge(JudgeEvaluator):
    name = "relevance"
    timeout_s = 30.0
    sample_rate = 0.05
    max_chars: int | None = 16_000
    system_prompt = SYSTEM_PROMPT
    schema = SCHEMA

    def applies_to(self, interaction: GenAIInteraction) -> bool:
        """A user message in the turn and a text answer that ends it.

        A response with a tool call is an agent step, not the answer: its text is narration
        ("let me look that up") and the newest user message is often the orchestrator's
        ("step 2 of 6"), so the judge would fail every step of a working agent.
        """
        return (
            bool(user_texts(interaction.input_messages))
            and bool(response_texts(interaction.output_messages))
            and not calls_tools(interaction.output_messages)
        )

    def content(self, interaction: GenAIInteraction) -> str:
        return envelope(conversation(interaction, self.text))

    def verdict(self, response: JudgeResponse) -> EvaluationResult:
        rating = int(response.output["score"])
        passed = rating >= PASS_RATING
        return EvaluationResult(
            score=(rating - 1) / (MAX_RATING - 1),
            label=semconv.LABEL_PASS if passed else semconv.LABEL_FAIL,
            explanation=self.explanation(response, f"score={rating}/{MAX_RATING}"),
            attributes={
                semconv.LLM_EVAL_JUDGE_MODEL: response.model,
                semconv.LLM_EVAL_JUDGE_RAW_SCORE: rating,
            },
        )
