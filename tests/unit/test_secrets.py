import pytest

from llm_eval_otel.evaluators.base import GenAIInteraction, Message
from llm_eval_otel.evaluators.secrets import SecretDetector, find_secrets, shannon_entropy

# Fake credentials, shaped like the real ones.
AWS = "AKIAIOSFODNN7EXAMPLE"
GITHUB = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
GITHUB_PAT = "github_pat_11ABCDEFG0123456789_abcdefghijKLMNOP"
OPENAI = "sk-proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv"
ANTHROPIC = "sk-ant-api03-Zy1xWv4uTs7rQp0oNm3lKj6iHg"
JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
    ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
)
GENERIC = "Zx9Qm2Lp7Vt4Rk8Wn3Hs6Bd1"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"aws_access_key_id={AWS}", ["aws_access_key"]),
        (f"use o token {GITHUB}", ["github_token"]),
        (f"token {GITHUB_PAT}", ["github_token"]),
        (f"OPENAI_API_KEY {OPENAI}", ["llm_api_key"]),
        (f"chave {ANTHROPIC}", ["llm_api_key"]),
        (f"Authorization: Bearer {JWT}", ["jwt"]),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...", ["private_key"]),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", ["private_key"]),
        ("-----BEGIN PRIVATE KEY-----", ["private_key"]),
        ("DATABASE_URL=postgres://app:S3cr3tPass@db:5432/app", ["connection_string"]),
        ("mongodb+srv://admin:hunter22@cluster0.example.net/db", ["connection_string"]),
        (f'api_key = "{GENERIC}"', ["generic_secret"]),
        (f"senha: {GENERIC}", ["generic_secret"]),
        (f'{{"client_secret": "{GENERIC}"}}', ["generic_secret"]),
    ],
)
def test_detects(text: str, expected: list[str]) -> None:
    assert sorted(f.kind for f in find_secrets(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "request id 550e8400-e29b-41d4-a716-446655440000",  # UUID
        "fix in commit 9fceb02d0ae598e95dc970b74767f19372d61af8",  # commit hash
        "sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhf"
        "DwAChwGA60e6kgAAAABJRU5ErkJggg==",  # base64 image
        "postgres://app:***@db:5432/app",  # placeholders
        "postgres://app:<password>@db:5432/app",
        "postgres://app:${DB_PASSWORD}@db:5432/app",
        "https://example.com/docs",
        "password = aaaaaaaaaaaaaaaaaaaaaaaa",  # low entropy
        "password = ${DB_PASSWORD_FROM_THE_VAULT}",
        "the key point of this task-force-assessment-document is",
        "eyJmb28iOiJiYXIifQ.eyJzdWIiOiIxIn0.c2lnbmF0dXJl",  # header without alg
        "Olá, preciso de ajuda com o deploy.",
    ],
)
def test_ignores(text: str) -> None:
    assert find_secrets(text) == []


def test_prefixed_pattern_wins_over_generic() -> None:
    assert [f.kind for f in find_secrets(f"api_key={OPENAI}")] == ["llm_api_key"]


def test_entropy() -> None:
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy(GENERIC) > 3.5


async def test_explanation_never_quotes_the_secret() -> None:
    interaction = GenAIInteraction(
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        parent_span_id=None,
        trace_flags=1,
        service_name=None,
        operation_name="chat",
        provider_name=None,
        request_model=None,
        response_id=None,
        system_instructions=[],
        input_messages=[Message("user", f"minha chave {AWS} e {JWT}")],
        output_messages=[],
    )
    result = await SecretDetector().evaluate(interaction)
    assert result.label == "fail"
    assert result.score == 0.0
    assert result.explanation == "aws_access_key=1 (input), jwt=1 (input)"
    assert result.attributes == {"llm_eval.secret.types": ["aws_access_key", "jwt"]}
    for fragment in (AWS, AWS[:8], AWS[-6:], JWT[:20]):
        assert fragment not in (result.explanation or "")
