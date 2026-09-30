import logging

import pytest

from llm_eval_otel.config import Settings
from llm_eval_otel.main import log_startup

TOKEN = "s3cr3t-t0ken-value"


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
        "evaluators available: output_format, pii_detection, refusal, secret_detection, "
        "system_prompt_leak"
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
