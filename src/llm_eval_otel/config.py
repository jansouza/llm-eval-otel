"""Service settings, read from ``LLM_EVAL_*`` environment variables.

``OTEL_*`` variables are read by the OpenTelemetry SDK itself; the defaults the
service needs are applied in :func:`apply_otel_defaults`.
"""

import json
import os
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

OTEL_DEFAULTS = {
    "OTEL_SERVICE_NAME": "llm-eval-otel",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector:4319",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_EVAL_", extra="ignore")

    http_port: int = 4318
    evaluators: Annotated[list[str], NoDecode] = ["pii_detection", "secret_detection"]
    sample_rates: Annotated[dict[str, float], NoDecode] = {}
    exceptions: Annotated[dict[str, list[str]], NoDecode] = {}
    # Association property keys kept off the metrics: per-request values (one series each)
    association_exclude: Annotated[list[str], NoDecode] = ["correlation_id"]
    # Types pii_detection reports; the sanitizer always redacts every type.
    pii_types: Annotated[list[str], NoDecode] = [
        "cpf",
        "cnpj",
        "email",
        "credit_card",
        "phone",
        "pix_key",
    ]
    workers: int = 4
    queue_max: int = 10_000
    timeout_s: float = 5.0
    max_request_bytes: int = 16 * 1024 * 1024
    emit_spans: bool = True
    dedup_ttl_s: float = 600.0
    auth_token: str | None = None
    tls_cert_file: str | None = None
    tls_key_file: str | None = None
    drain_timeout_s: float = 30.0

    @field_validator("evaluators", "association_exclude", "pii_types", mode="before")
    @classmethod
    def _split_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        return value

    @field_validator("sample_rates", mode="before")
    @classmethod
    def _parse_sample_rates(cls, value: object) -> object:
        """Accept ``relevance=0.05,other=0.5`` or a JSON object."""
        if not isinstance(value, str):
            return value
        value = value.strip()
        if not value:
            return {}
        if value.startswith("{"):
            return json.loads(value)
        rates: dict[str, float] = {}
        for item in value.split(","):
            name, _, rate = item.partition("=")
            rates[name.strip()] = float(rate)
        return rates

    @field_validator("sample_rates")
    @classmethod
    def _check_rates(cls, value: dict[str, float]) -> dict[str, float]:
        for name, rate in value.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"sample rate for {name} must be between 0 and 1")
        return value

    @field_validator("exceptions", mode="before")
    @classmethod
    def _parse_exceptions(cls, value: object) -> object:
        """Accept a JSON object; an empty value means no exceptions."""
        if not isinstance(value, str):
            return value
        if not value.strip():
            return {}
        return json.loads(value)

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert_file and self.tls_key_file)

    def describe(self) -> dict[str, object]:
        """Every setting, for the startup log. Secrets show only whether they are set."""
        return {
            name: ("set" if value else "unset") if name in SECRET_SETTINGS else value
            for name, value in self.model_dump().items()
        }


# Settings never logged by value. A new secret setting must be added here.
SECRET_SETTINGS = frozenset({"auth_token"})


def apply_otel_defaults() -> None:
    for key, value in OTEL_DEFAULTS.items():
        os.environ.setdefault(key, value)
