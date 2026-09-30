import pytest

from llm_eval_otel.emit.sanitize import REDACTED, sanitize
from llm_eval_otel.evaluators.base import GenAIInteraction, Message
from llm_eval_otel.evaluators.pii import PII_TYPES, PIIDetector, card_brand, find_pii


def interaction(user: str = "", assistant: str = "", system: str = "") -> GenAIInteraction:
    return GenAIInteraction(
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        parent_span_id=None,
        trace_flags=1,
        service_name="svc",
        operation_name="chat",
        provider_name=None,
        request_model=None,
        response_id=None,
        system_instructions=[Message("system", system)] if system else [],
        input_messages=[Message("user", user)] if user else [],
        output_messages=[Message("assistant", assistant)] if assistant else [],
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # CPF
        ("Meu CPF é 529.982.247-25", ["cpf"]),
        ("cpf 111.444.777-35 ok", ["cpf"]),
        ("CPF: 52998224725", ["cpf"]),
        ("meu cpf, se precisar, é o número 52998224725", ["cpf"]),
        ("Documento 529.982.247-25 anexado", ["cpf"]),  # formatted needs no keyword
        # E-mail
        ("escreva para fulano.silva+ti@example.com.br", ["email"]),
        # Credit cards: brand prefix, length and Luhn
        ("cartão 4111 1111 1111 1111", ["credit_card"]),  # Visa
        ("5555-5555-5555-4444", ["credit_card"]),  # Mastercard
        ("2223003122003222", ["credit_card"]),  # Mastercard 2-series
        ("amex 378282246310005", ["credit_card"]),
        ("elo 6362970000457013", ["credit_card"]),
        ("hipercard 6062825624254001", ["credit_card"]),
        # Several at once
        ("CPF 529.982.247-25, email a@b.co", ["cpf", "email"]),
        # CNPJ, numeric and alphanumeric; formatted needs no keyword
        ("CNPJ 11.222.333/0001-81", ["cnpj"]),
        ("empresa 11.222.333/0001-81 cadastrada", ["cnpj"]),
        ("CNPJ: 11222333000181", ["cnpj"]),
        ("o cnpj da loja é 12ABC34501DE35", ["cnpj"]),
        ("fornecedor 12.ABC.345/01DE-35", ["cnpj"]),
        # Phones: landline and mobile, several formats; bare digits need a keyword
        ("ligue (11) 98765-4321 ou +55 11 98765-4321", ["phone", "phone"]),
        ("telefone 11987654321", ["phone"]),
        ("fixo (21) 3456-7890", ["phone"]),
        ("fale no 11 3456 7890", ["phone"]),
        ("+5561987654321", ["phone"]),
        ("(48)99876-5432", ["phone"]),
        ("WhatsApp: 5511987654321", ["phone"]),
        ("celular 85 99876 5432", ["phone"]),
        # PIX random key: a UUID v4 with "pix" nearby
        ("chave pix 123e4567-e89b-42d3-a456-426614174000", ["pix_key"]),
        ("PIX: 123E4567-E89B-42D3-A456-426614174000", ["pix_key"]),
        # One stretch, one type: the card wins over the phone it contains
        ("5555-5555-5555-4444", ["credit_card"]),
    ],
)
def test_detects(text: str, expected: list[str]) -> None:
    assert sorted(f.kind for f in find_pii(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "CPF 529.982.247-26",  # wrong check digit
        "CPF 111.111.111-11",  # all digits equal
        "CPF: 11111111111",
        "número 52998224725 sem a palavra",  # 11 digits, no "CPF" nearby
        "CPF está na ficha. " + "x" * 40 + " 52998224725",  # keyword too far
        "1234567812345670",  # passes Luhn, no known brand
        "4111111111111112",  # Visa prefix, fails Luhn
        "411111111111111",  # Visa prefix, 15 digits: wrong length
        "Pedido 20260929123456 confirmado",  # order number
        "Pedido nº 98765432101234",
        "protocolo 2026-0001-3344-5566",
        # CNPJ
        "CNPJ 11.222.333/0001-82",  # wrong check digit
        "CNPJ 12.ABC.345/01DE-36",
        "CNPJ 00.000.000/0000-00",  # all characters equal
        "código 11222333000181 do produto",  # 14 digits, no "CNPJ" nearby
        "CNPJ na nota. " + "x" * 40 + " 11222333000181",  # keyword too far
        # Phones
        "ligue (20) 98765-4321",  # DDD 20 does not exist
        "11987654321",  # bare digits without a keyword
        "5555-5555-5555-4445",  # a card number that fails Luhn is not a phone
        "(11) 8765-4321",  # landlines start with 2 to 5
        "Pedido 11 98765-4321-7",  # longer digit run
        # PIX: a bare UUID is not PII
        "request id 123e4567-e89b-42d3-a456-426614174000",
        "pix " + "x" * 40 + " 123e4567-e89b-42d3-a456-426614174000",  # keyword too far
        "chave pix 123e4567-e89b-12d3-a456-426614174000",  # UUID v1
        "versão 1.2.3, build 20260929",
        "Olá, tudo bem? Quero trocar meu plano.",
    ],
)
def test_ignores(text: str) -> None:
    assert find_pii(text) == []


def test_card_brand_needs_matching_length() -> None:
    assert card_brand("4111111111111111") == "visa"
    assert card_brand("378282246310005") == "amex"
    assert card_brand("3782822463100051") is None


async def test_fail_reports_types_counts_and_location_only() -> None:
    result = await PIIDetector().evaluate(
        interaction(
            user="Meu CPF é 529.982.247-25",
            assistant="Confirmado para a@b.co e c@d.co",
        )
    )
    assert result.score == 0.0
    assert result.label == "fail"
    assert result.explanation == "cpf=1 (input), email=2 (output)"
    assert result.attributes == {"llm_eval.pii.types": ["cpf", "email"]}


async def test_system_location() -> None:
    result = await PIIDetector().evaluate(interaction(system="Contato: suporte@example.com"))
    assert result.explanation == "email=1 (system)"


async def test_clean_text_passes() -> None:
    result = await PIIDetector().evaluate(interaction(user="Oi", assistant="Olá!"))
    assert (result.score, result.label) == (1.0, "pass")


async def test_new_types_in_result() -> None:
    result = await PIIDetector().evaluate(
        interaction(
            user="CNPJ 11.222.333/0001-81, chave pix 123e4567-e89b-42d3-a456-426614174000",
            assistant="Ligaremos para (11) 98765-4321.",
        )
    )
    assert result.label == "fail"
    assert result.explanation == "cnpj=1 (input), pix_key=1 (input), phone=1 (output)"
    assert result.attributes == {"llm_eval.pii.types": ["cnpj", "phone", "pix_key"]}


async def test_pii_types_setting_turns_off_phone_but_not_the_sanitizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EVAL_PII_TYPES", "cpf,cnpj,email,credit_card")
    detector = PIIDetector()
    assert detector.types == {"cpf", "cnpj", "email", "credit_card"}
    result = await detector.evaluate(interaction(user="meu celular (11) 98765-4321"))
    assert (result.score, result.label) == (1.0, "pass")
    result = await detector.evaluate(interaction(user="CPF 529.982.247-25, tel (11) 98765-4321"))
    assert result.explanation == "cpf=1 (input)"
    assert sanitize({"a": "(11) 98765-4321"}) == ({"a": REDACTED}, 1)


def test_pii_types_default_to_all() -> None:
    assert PIIDetector().types == set(PII_TYPES)


def test_unknown_pii_type_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="rg"):
        PIIDetector(types=["cpf", "rg"])
