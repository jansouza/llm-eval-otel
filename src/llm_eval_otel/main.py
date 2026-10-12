"""Entry point: SDK providers, evaluators, workers and the HTTP server."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from llm_eval_otel.config import OTEL_DEFAULTS, Settings, apply_otel_defaults
from llm_eval_otel.emit import sdk
from llm_eval_otel.engine.service import Service
from llm_eval_otel.evaluators import registry
from llm_eval_otel.ingest.http import create_app
from llm_eval_otel.version import __version__

log = logging.getLogger("llm_eval_otel")


def build_app(service: Service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service.start()
        log.info("accepting OTLP/HTTP on port %d", service.settings.http_port)
        try:
            yield
        finally:
            log.info("shutting down: draining %d queued interactions", service.queue.size)
            await service.shutdown()
            log.info("shutdown complete")

    return create_app(service, lifespan=lifespan)


def log_startup(settings: Settings) -> None:
    """Available evaluators and the effective configuration, without secret values."""
    log.info("evaluators available: %s", ", ".join(sorted(registry.available())))
    log.info("evaluators enabled: %s", ", ".join(settings.evaluators))
    log.info(
        "config: %s", ", ".join(f"{name}={value}" for name, value in settings.describe().items())
    )
    # Only the variables the service sets defaults for: others, such as
    # OTEL_EXPORTER_OTLP_HEADERS, may carry credentials.
    log.info("otel: %s", ", ".join(f"{name}={os.environ.get(name)}" for name in OTEL_DEFAULTS))


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # The judges' SDKs log every retry, and their HTTP client (httpx2) every request, at
    # INFO: a line per judge call. The summary and the failing/recovered lines already say
    # how the judges are doing. typesafe_sdk also logs request bodies at DEBUG; its adapter
    # sets WARNING again once the client exists, since TYPESAFE_LOG_LEVEL applies on import.
    for noisy in ("openai", "httpx", "httpx2", "typesafe_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log.info("starting llm-eval-otel %s", __version__)
    apply_otel_defaults()
    settings = Settings()
    # Only the service's loggers: DEBUG on the root would also turn on httpx and openai.
    log.setLevel(settings.log_level)
    log_startup(settings)
    evaluators = registry.load(settings.evaluators)
    service = Service(settings, sdk.build_otlp(), evaluators)
    uvicorn.run(
        build_app(service),
        host="0.0.0.0",
        port=settings.http_port,
        ssl_certfile=settings.tls_cert_file if settings.tls_enabled else None,
        ssl_keyfile=settings.tls_key_file if settings.tls_enabled else None,
        # uvicorn's own access log never sees bodies; keep it off to stay quiet under load.
        access_log=False,
        timeout_graceful_shutdown=int(settings.drain_timeout_s),
    )


if __name__ == "__main__":
    run()
