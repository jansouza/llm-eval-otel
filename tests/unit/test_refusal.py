import pytest

from llm_eval_otel.evaluators.base import GenAIInteraction, Message, PartSpan
from llm_eval_otel.evaluators.refusal import RefusalDetector, refusal_language


def interaction(*outputs: Message, finish_reasons: tuple[str, ...] = ()) -> GenAIInteraction:
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
        input_messages=[Message("user", "pedido")],
        output_messages=list(outputs),
        finish_reasons=finish_reasons,
    )


def assistant(text: str) -> Message:
    return Message("assistant", text)


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Desculpe, mas não posso ajudar com isso.", "pt"),
        ("Infelizmente não consigo atender a esse pedido.", "pt"),
        ("Não posso fornecer essa informação.", "pt"),
        ("NÃO POSSO TE AJUDAR COM ESSE ASSUNTO", "pt"),
        ("Sinto muito, não estou autorizado a compartilhar dados de clientes.", "pt"),
        ("I'm sorry, but I can't help with that.", "en"),
        ("I can\N{RIGHT SINGLE QUOTATION MARK}t assist with that request.", "en"),
        ("I cannot provide instructions for that.", "en"),
        ("I'm unable to comply with this request.", "en"),
        ("I won't be able to help you with that.", "en"),
        ("I must respectfully decline this request.", "en"),
        ("Lo siento, pero no puedo ayudarte con eso.", "es"),
        ("No puedo proporcionar esa información.", "es"),
        ("No me es posible atender esta solicitud.", "es"),
    ],
)
def test_detects_refusal_phrases(text: str, language: str) -> None:
    assert refusal_language(text) == language


@pytest.mark.parametrize(
    "text",
    [
        "Não posso deixar de mencionar que o prazo é amanhã.",
        "Não posso garantir, mas o pedido deve chegar amanhã.",
        "I can't wait to help you plan the trip!",
        "I can't stress this enough: back up your data.",
        "No puedo creer lo rápido que fue. Aquí está tu resumen.",
        "Claro! Segue o resumo do seu pedido.",
        # A refusal phrase deep inside a normal answer
        "Aqui está o relatório completo. "
        + "Os números subiram. " * 20
        + "Não posso ajudar com previsões, mas os dados estão acima.",
    ],
)
def test_ignores_near_misses(text: str) -> None:
    assert refusal_language(text) is None


async def test_refusal_is_fail_with_source_and_language() -> None:
    result = await RefusalDetector().evaluate(
        interaction(assistant("Desculpe, não posso ajudar com isso."))
    )
    assert (result.score, result.label) == (0.0, "fail")
    assert result.explanation == "refusal=1 (output), source=phrase, lang=pt"
    assert result.attributes == {
        "llm_eval.refusal.source": "phrase",
        "llm_eval.refusal.language": "pt",
    }


async def test_normal_answer_passes() -> None:
    result = await RefusalDetector().evaluate(interaction(assistant("Seu saldo é R$ 10.")))
    assert (result.score, result.label, result.explanation) == (1.0, "pass", "no refusal")


async def test_content_filter_without_text_is_a_refusal() -> None:
    i = interaction(finish_reasons=("content_filter",))
    detector = RefusalDetector()
    assert detector.applies_to(i)
    result = await detector.evaluate(i)
    assert result.label == "fail"
    assert result.explanation == "refusal=1 (output), source=finish_reason"
    assert result.attributes == {"llm_eval.refusal.source": "finish_reason"}


async def test_every_output_is_checked() -> None:
    result = await RefusalDetector().evaluate(
        interaction(
            assistant("Claro, aqui está."),
            assistant("I can't help with that."),
            assistant("No puedo ayudar con eso."),
        )
    )
    assert result.explanation == "refusal=2 (output), source=phrase, lang=en"


async def test_reads_only_text_parts() -> None:
    reasoning = "I can't help with that directly, so I will look it up."
    text = "Encontrei o pedido."
    message = Message(
        "assistant",
        f"{reasoning}\n{text}",
        (
            PartSpan("reasoning", 0, len(reasoning)),
            PartSpan("text", len(reasoning) + 1, len(reasoning) + 1 + len(text)),
        ),
    )
    result = await RefusalDetector().evaluate(interaction(message))
    assert result.label == "pass"


def test_applies_only_with_output_text_or_finish_reasons() -> None:
    detector = RefusalDetector()
    assert not detector.applies_to(interaction())
    tool_only = Message("assistant", '{"q": 1}', (PartSpan("tool_call", 0, 8),))
    assert not detector.applies_to(interaction(tool_only))
    assert detector.applies_to(interaction(tool_only, finish_reasons=("tool_call",)))
    assert detector.applies_to(interaction(assistant("oi")))
