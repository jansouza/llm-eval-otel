import logging
import os
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import OtelMemory

from llm_eval_otel import main
from llm_eval_otel.config import OTEL_DEFAULTS, Settings
from llm_eval_otel.engine.service import Service
from llm_eval_otel.evaluators.pii import PIIDetector
from llm_eval_otel.main import build_app, log_startup

TOKEN = "s3cr3t-t0ken-value"


@pytest.fixture
def service_logger_level() -> Iterator[None]:
    """run() sets the service logger's level from the settings; put it back."""
    logger = logging.getLogger("llm_eval_otel")
    level = logger.level
    yield
    logger.setLevel(level)


async def test_lifespan_starts_and_drains_the_service(
    otel_memory: OtelMemory, caplog: pytest.LogCaptureFixture
) -> None:
    service = Service(Settings(http_port=4321), otel_memory.telemetry, [PIIDetector()])
    app = build_app(service)
    async with app.router.lifespan_context(app):
        assert service.accepting
    assert not service.accepting
    assert "accepting OTLP/HTTP on port 4321" in caplog.messages
    assert "shutdown complete" in caplog.messages


@pytest.mark.usefixtures("service_logger_level")
def test_run_wires_settings_into_uvicorn(
    monkeypatch: pytest.MonkeyPatch, otel_memory: OtelMemory
) -> None:
    for name in OTEL_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_EVAL_HTTP_PORT", "4555")
    monkeypatch.setenv("LLM_EVAL_DRAIN_TIMEOUT_S", "7.5")
    monkeypatch.setattr(main.sdk, "build_otlp", lambda: otel_memory.telemetry)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(main.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))

    main.run()

    [kwargs] = calls
    assert kwargs["port"] == 4555
    assert kwargs["timeout_graceful_shutdown"] == 7
    assert kwargs["ssl_certfile"] is None and kwargs["ssl_keyfile"] is None
    assert kwargs["access_log"] is False
    # The service's OTEL defaults are applied before the SDK reads them.
    assert {name: os.environ[name] for name in OTEL_DEFAULTS} == OTEL_DEFAULTS
    for noisy in ("openai", "httpx", "httpx2", "typesafe_sdk"):
        assert logging.getLogger(noisy).level == logging.WARNING


def test_startup_log_lists_evaluators_and_config_without_secrets(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4319")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", f"authorization=Bearer {TOKEN}")
    settings = Settings(auth_token=TOKEN, evaluators=["pii_detection", "refusal"], workers=8)
    with caplog.at_level(logging.INFO, logger="llm_eval_otel"):
        log_startup(settings)
    lines = caplog.messages
    assert lines[0] == (
        "evaluators available: jev_prompt_injection, jev_refusal, jev_relevance, jev_toxicity, "
        "output_format, pii_detection, refusal, relevance, secret_detection, system_prompt_leak"
    )
    assert lines[1] == "evaluators enabled: pii_detection, refusal"
    assert lines[2].startswith("config: http_port=4318, evaluators=['pii_detection', 'refusal']")
    assert "workers=8" in lines[2]
    assert "auth_token=set" in lines[2]
    assert "OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4319" in lines[3]
    assert all(TOKEN not in line for line in lines)


def test_describe_masks_secrets() -> None:
    assert Settings().describe()["auth_token"] == "unset"
    assert Settings(auth_token=TOKEN).describe()["auth_token"] == "set"
    assert Settings().describe()["pii_types"] == [
        "cpf",
        "cnpj",
        "email",
        "credit_card",
        "phone",
        "pix_key",
    ]
