"""OTLP/HTTP receiver: ``POST /v1/traces`` with protobuf, plus health endpoints.

Status codes follow the OTLP/HTTP spec so the Collector's exporter retries the
right cases: 429 (queue full, with Retry-After) is retried; 400 is not.
"""

import hmac
import logging
import zlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI, Request, Response
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from llm_eval_otel.engine.queue import QueueFull
from llm_eval_otel.engine.service import Service

log = logging.getLogger(__name__)

PROTOBUF = "application/x-protobuf"
RETRY_AFTER_S = "5"


class PayloadTooLarge(Exception):
    pass


class BadEncoding(Exception):
    pass


async def read_body(chunks: AsyncIterator[bytes], limit: int) -> bytes:
    body = bytearray()
    async for chunk in chunks:
        body += chunk
        if len(body) > limit:
            raise PayloadTooLarge
    return bytes(body)


def gunzip(data: bytes, limit: int) -> bytes:
    """Decompress with a ceiling, so a small gzip bomb cannot exhaust memory."""
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = decompressor.decompress(data, limit + 1)
    except zlib.error:
        raise BadEncoding from None
    if len(out) > limit or decompressor.unconsumed_tail:
        raise PayloadTooLarge
    if not decompressor.eof:
        raise BadEncoding
    return out


def _text(status: int, message: str, headers: dict[str, str] | None = None) -> Response:
    return Response(message, status_code=status, media_type="text/plain", headers=headers)


def _encoding(request: Request) -> str:
    return request.headers.get("content-encoding", "identity").strip().lower()


def check_request(
    request: Request, expected_auth: str | None, accepting: bool
) -> tuple[int, str, dict[str, str] | None] | None:
    """Why an export is refused before its body is read (status, message, headers), or None.

    The checks run in this order: an unauthenticated peer gets 401 even during shutdown.
    """
    if expected_auth is not None and not hmac.compare_digest(
        request.headers.get("authorization", "").encode(), expected_auth.encode()
    ):
        return 401, "unauthorized", None
    if not accepting:
        return 503, "shutting down", {"Retry-After": RETRY_AFTER_S}
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != PROTOBUF:
        return 415, f"only {PROTOBUF} is supported", None
    if _encoding(request) not in ("identity", "gzip"):
        return 415, "only gzip content-encoding is supported", None
    return None


async def decode_export(request: Request, limit: int) -> ExportTraceServiceRequest:
    """Read, gunzip and parse the body; raises PayloadTooLarge, BadEncoding or DecodeError."""
    body = await read_body(request.stream(), limit)
    if _encoding(request) == "gzip":
        body = gunzip(body, limit)
    return ExportTraceServiceRequest.FromString(body)


def create_app(
    service: Service,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
) -> FastAPI:
    app = FastAPI(title="llm-eval-otel", docs_url=None, redoc_url=None, lifespan=lifespan)
    settings = service.settings
    expected_auth = f"Bearer {settings.auth_token}" if settings.auth_token else None

    def _reject(
        request: Request, status: int, message: str, headers: dict[str, str] | None = None
    ) -> Response:
        """A refused export: DEBUG with the peer and the reason, counted in the summary."""
        peer = request.client.host if request.client else "unknown"
        log.debug("rejected export from %s: %d %s", peer, status, message)
        service.record_rejected(status)
        return _text(status, message, headers)

    @app.post("/v1/traces")
    async def export_traces(request: Request) -> Response:
        if (refusal := check_request(request, expected_auth, service.accepting)) is not None:
            return _reject(request, *refusal)

        try:
            export = await decode_export(request, settings.max_request_bytes)
        except PayloadTooLarge:
            return _reject(request, 413, "request too large")
        except (BadEncoding, DecodeError):
            service.record_invalid_payload()
            return _reject(request, 400, "invalid payload")

        try:
            service.ingest(export)
        except QueueFull:
            return _reject(request, 429, "queue full", {"Retry-After": RETRY_AFTER_S})
        return Response(ExportTraceServiceResponse().SerializeToString(), media_type=PROTOBUF)

    @app.get("/healthz")
    async def healthz() -> Response:
        return _text(200, "ok")

    @app.get("/readyz")
    async def readyz() -> Response:
        if service.ready:
            return _text(200, "ready")
        return _text(503, "not ready")

    return app
