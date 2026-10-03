"""``llm-eval-judge`` against the fake OpenAI-compatible judge server; no API key needed."""

import asyncio
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_judge_server import FakeJudgeServer
from judge_fakes import FakeJudgeClient, relevance

from llm_eval_otel.cli import evaluate, main

CPF = "529.982.247-25"
JUDGE_ENV = ("LLM_EVAL_JUDGE_MODEL", "LLM_EVAL_JUDGE_BASE_URL", "OPENAI_API_KEY")


@pytest.fixture
def server() -> Iterator[FakeJudgeServer]:
    server = FakeJudgeServer(("127.0.0.1", 0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No judge settings from the environment or a .env in the working directory.

    setenv then delenv, so whatever a .env file loads is removed after the test.
    """
    monkeypatch.chdir(tmp_path)
    for name in JUDGE_ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def judge_env(monkeypatch: pytest.MonkeyPatch, server: FakeJudgeServer) -> None:
    monkeypatch.setenv("LLM_EVAL_JUDGE_MODEL", "fake-judge-1")
    monkeypatch.setenv("LLM_EVAL_JUDGE_BASE_URL", server.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def results(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.mark.usefixtures("judge_env")
def test_judges_one_interaction(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["-i", "Como troco a senha do aplicativo?", "-o", "Abra o aplicativo e troque a senha."]
    )
    assert code == 0
    [result] = results(capsys)
    assert result["evaluator"] == "relevance"
    assert (result["label"], result["score"], result["error_type"]) == ("pass", 1.0, None)
    assert result["attributes"] == {
        "llm_eval.judge.model": "fake-judge-1",
        "llm_eval.judge.raw_score": 5,
    }
    assert result["explanation"].startswith("fake judge:")
    [call] = result["judge_calls"]
    assert call["model"] == "fake-judge-1" and call["input_tokens"] > 0
    assert call["finish_reason"] == "stop"


@pytest.mark.usefixtures("judge_env")
def test_context_goes_to_the_judge(
    capsys: pytest.CaptureFixture[str], server: FakeJudgeServer
) -> None:
    main(
        [
            "-i",
            "E em inglês?",
            "-o",
            "Good morning.",
            "-c",
            "user:Como digo bom dia?",
            "-c",
            "assistant:Bom dia.",
        ]
    )
    content = server.requests[0]["messages"][1]["content"]
    assert '"context": [{"role": "user", "text": "Como digo bom dia?"}' in content


@pytest.mark.usefixtures("judge_env")
def test_jsonl_keeps_the_order_and_ids(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    lines = [
        {"id": "relevant", "input": "Qual o horário da loja?", "output": "A loja abre às 9h."},
        {
            "id": "off-topic",
            "input": "Como cancelo minha assinatura?",
            "output": (
                "Temos promoções de televisores, notebooks e celulares de várias marcas "
                "nesta semana inteira, aproveite agora mesmo."
            ),
        },
    ]
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    assert main(["--jsonl", str(path)]) == 0
    assert [(r["id"], r["label"]) for r in results(capsys)] == [
        ("relevant", "pass"),
        ("off-topic", "fail"),
    ]


def test_dry_run_shows_the_masked_content_without_calling(
    capsys: pytest.CaptureFixture[str], server: FakeJudgeServer
) -> None:
    # No model and no API key: a dry run needs neither.
    assert main(["-i", f"Meu CPF é {CPF}", "-o", "Anotado.", "--dry-run"]) == 0
    [result] = results(capsys)
    assert CPF not in result["content"] and "[CPF]" in result["content"]
    assert result["content"].startswith("<conversation>")
    assert server.requests == []


@pytest.mark.usefixtures("judge_env")
def test_judge_error_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-i", "oi FAKE_JUDGE:invalid", "-o", "olá"]) == 1
    [result] = results(capsys)
    assert result["error_type"] == "judge_invalid_output" and result["label"] is None


@pytest.mark.usefixtures("judge_env")
def test_runs_any_evaluator(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-e", "pii_detection", "-i", f"CPF {CPF}", "-o", "ok"]) == 0
    [result] = results(capsys)
    assert result["label"] == "fail" and result["explanation"] == "cpf=1 (input)"
    assert result["judge_calls"] == []


def test_prints_what_the_service_would_emit() -> None:
    """A judge reason that quotes a CPF goes through the sanitizer, as in the service."""
    evaluator = relevance(FakeJudgeClient(reason=f"the answer repeats the CPF {CPF}"))
    result = asyncio.run(evaluate(evaluator, {"input": "oi", "output": "olá"}, 0, dry_run=False))
    assert result["explanation"] == "[REDACTED]"
    assert result["attributes"]["llm_eval.judge.raw_score"] == 5


def test_reads_dotenv(
    capsys: pytest.CaptureFixture[str], server: FakeJudgeServer, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        f"LLM_EVAL_JUDGE_MODEL=from-dotenv\nLLM_EVAL_JUDGE_BASE_URL={server.base_url}\nOPENAI_API_KEY=test\n"
    )
    assert main(["-i", "Qual o horário?", "-o", "Das 9h às 18h."]) == 0
    assert server.requests[0]["model"] == "from-dotenv"


def test_environment_wins_over_dotenv(
    monkeypatch: pytest.MonkeyPatch, server: FakeJudgeServer, tmp_path: Path
) -> None:
    env = tmp_path / "judge.env"
    env.write_text(
        f"LLM_EVAL_JUDGE_MODEL=from-dotenv\nLLM_EVAL_JUDGE_BASE_URL={server.base_url}\nOPENAI_API_KEY=test\n"
    )
    monkeypatch.setenv("LLM_EVAL_JUDGE_MODEL", "from-environment")
    assert main(["--env-file", str(env), "-i", "Qual o horário?", "-o", "Das 9h."]) == 0
    assert server.requests[0]["model"] == "from-environment"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["-i", "oi", "-o", "olá"], "relevance needs LLM_EVAL_JUDGE_MODEL"),
        (["-i", "oi"], "give --input and --output, or --jsonl"),
        (["-i", "oi", "-o", "olá", "-c", "system:x", "--dry-run"], "--context takes ROLE:TEXT"),
        (["-e", "nope", "-i", "oi", "-o", "olá"], "'nope' is not registered"),
        (["-e", "pii_detection", "-i", "oi", "-o", "olá", "--dry-run"], "needs a judge evaluator"),
        (["--env-file", "missing.env", "-i", "oi", "-o", "olá"], "missing.env not found"),
    ],
)
def test_usage_and_configuration_errors_exit_2(
    capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and message in captured.err
