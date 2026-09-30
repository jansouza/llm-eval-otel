import gzip
from collections.abc import AsyncIterator

import httpx
import pytest
from conftest import OtelMemory, ServiceFactory
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
from otlp import chat_request

from llm_eval_otel.engine.service import Service
from llm_eval_otel.ingest.http import create_app

PROTOBUF = {"content-type": "application/x-protobuf"}


def client_for(service: Service) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(service))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def client(service: Service) -> AsyncIterator[httpx.AsyncClient]:
    async with client_for(service) as c:
        yield c


async def test_accepts_plain_protobuf(
    client: httpx.AsyncClient, service: Service, otel_memory: OtelMemory
) -> None:
    body = chat_request("CPF 529.982.247-25").SerializeToString()
    response = await client.post("/v1/traces", content=body, headers=PROTOBUF)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-protobuf"
    ExportTraceServiceResponse.FromString(response.content)
    await service.drain()
    [event] = otel_memory.events("gen_ai.evaluation.result", name="pii_detection")
    assert (event.log_record.attributes or {})["gen_ai.evaluation.score.label"] == "fail"


async def test_accepts_gzip(
    client: httpx.AsyncClient, service: Service, otel_memory: OtelMemory
) -> None:
    body = gzip.compress(chat_request("oi").SerializeToString())
    response = await client.post(
        "/v1/traces", content=body, headers={**PROTOBUF, "content-encoding": "gzip"}
    )
    assert response.status_code == 200
    await service.drain()
    assert len(otel_memory.events("gen_ai.evaluation.result")) == 2


async def test_invalid_payload_is_400_and_counted(
    client: httpx.AsyncClient, otel_memory: OtelMemory
) -> None:
    response = await client.post(
        "/v1/traces", content=b"\xff\xff\xff not protobuf", headers=PROTOBUF
    )
    assert response.status_code == 400
    response = await client.post(
        "/v1/traces", content=b"not gzip", headers={**PROTOBUF, "content-encoding": "gzip"}
    )
    assert response.status_code == 400
    assert (
        otel_memory.counter("llm_eval.spans.skipped", {"llm_eval.skip.reason": "invalid_payload"})
        == 2
    )


async def test_truncated_gzip_is_400(client: httpx.AsyncClient) -> None:
    body = gzip.compress(chat_request("oi").SerializeToString())[:-10]
    response = await client.post(
        "/v1/traces", content=body, headers={**PROTOBUF, "content-encoding": "gzip"}
    )
    assert response.status_code == 400


async def test_request_size_limit_applies_after_decompression(make_service: ServiceFactory) -> None:
    service = make_service(max_request_bytes=1_000)
    async with client_for(service) as client:
        body = chat_request("a" * 5_000).SerializeToString()
        assert (await client.post("/v1/traces", content=body, headers=PROTOBUF)).status_code == 413
        bomb = gzip.compress(body)
        assert len(bomb) < 1_000
        response = await client.post(
            "/v1/traces", content=bomb, headers={**PROTOBUF, "content-encoding": "gzip"}
        )
        assert response.status_code == 413


async def test_unsupported_media_type(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/traces", content=b"{}", headers={"content-type": "application/json"}
    )
    assert response.status_code == 415
    response = await client.post(
        "/v1/traces", content=b"", headers={**PROTOBUF, "content-encoding": "br"}
    )
    assert response.status_code == 415


async def test_full_queue_is_429_with_retry_after(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    service = make_service(queue_max=2, workers=0)  # nobody drains the queue
    async with client_for(service) as client:
        for n in range(2):
            body = chat_request("oi", span_id=bytes([n + 1]) * 8).SerializeToString()
            assert (
                await client.post("/v1/traces", content=body, headers=PROTOBUF)
            ).status_code == 200
        body = chat_request("oi", span_id=b"\x09" * 8).SerializeToString()
        response = await client.post("/v1/traces", content=body, headers=PROTOBUF)
        assert response.status_code == 429
        assert response.headers["retry-after"] == "5"
        assert (await client.get("/readyz")).status_code == 503


async def test_auth_token(make_service: ServiceFactory) -> None:
    service = make_service(auth_token="s3cret")
    body = chat_request("oi").SerializeToString()
    async with client_for(service) as client:
        assert (await client.post("/v1/traces", content=body, headers=PROTOBUF)).status_code == 401
        wrong = {**PROTOBUF, "authorization": "Bearer nope"}
        assert (await client.post("/v1/traces", content=body, headers=wrong)).status_code == 401
        right = {**PROTOBUF, "authorization": "Bearer s3cret"}
        assert (await client.post("/v1/traces", content=body, headers=right)).status_code == 200


async def test_health_and_readiness(client: httpx.AsyncClient, service: Service) -> None:
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/readyz")).status_code == 200
    service.accepting = False
    assert (await client.get("/readyz")).status_code == 503
    body = chat_request("oi").SerializeToString()
    assert (await client.post("/v1/traces", content=body, headers=PROTOBUF)).status_code == 503
