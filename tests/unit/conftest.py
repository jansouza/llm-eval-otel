import logging
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import pytest
from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_eval_otel.config import Settings
from llm_eval_otel.emit.sdk import Telemetry, new_tracer_provider
from llm_eval_otel.engine.service import Service
from llm_eval_otel.evaluators.base import Evaluator
from llm_eval_otel.evaluators.pii import PIIDetector
from llm_eval_otel.evaluators.secrets import SecretDetector


class OtelMemory:
    """The service's output, captured by the SDK's in-memory exporters."""

    def __init__(self, caplog: pytest.LogCaptureFixture) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.log_exporter = InMemoryLogRecordExporter()
        self.metric_reader = InMemoryMetricReader()
        self.caplog = caplog
        res = Resource.create({"service.name": "llm-eval-otel-test"})
        tracer_provider = new_tracer_provider(res)
        tracer_provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))
        logger_provider = LoggerProvider(resource=res)
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(self.log_exporter))
        meter_provider = MeterProvider(resource=res, metric_readers=[self.metric_reader])
        self.telemetry = Telemetry(tracer_provider, meter_provider, logger_provider)
        self._metrics_json: list[str] = []

    def events(self, event_name: str, name: str | None = None) -> list[ReadableLogRecord]:
        return [
            r
            for r in self.log_exporter.get_finished_logs()
            if r.log_record.event_name == event_name
            and (
                name is None
                or (r.log_record.attributes or {}).get("gen_ai.evaluation.name") == name
            )
        ]

    def spans(self, name: str) -> list[ReadableSpan]:
        return [s for s in self.span_exporter.get_finished_spans() if s.name == name]

    def _points(self, metric_name: str) -> list[Any]:
        data = self.metric_reader.get_metrics_data()
        if data is None:
            return []
        self._metrics_json.append(data.to_json())
        return [
            point
            for rm in data.resource_metrics
            for sm in rm.scope_metrics
            for metric in sm.metrics
            if metric.name == metric_name
            for point in metric.data.data_points
        ]

    def counter(self, metric_name: str, attrs: Mapping[str, Any] | None = None) -> int:
        attrs = attrs or {}
        return sum(
            p.value
            for p in self._points(metric_name)
            if all(p.attributes.get(k) == v for k, v in attrs.items())
        )

    def histogram_count(self, metric_name: str, attrs: Mapping[str, Any] | None = None) -> int:
        attrs = attrs or {}
        return sum(
            p.count
            for p in self._points(metric_name)
            if all(p.attributes.get(k) == v for k, v in attrs.items())
        )

    def serialize_all(self) -> str:
        """Every event, span, metric and service log line, as text."""
        self._points("")  # snapshot metrics
        parts = [r.to_json() for r in self.log_exporter.get_finished_logs()]
        parts += [s.to_json() for s in self.span_exporter.get_finished_spans()]
        parts += self._metrics_json
        parts.append(self.caplog.text)
        return "\n".join(parts)


@pytest.fixture
def otel_memory(caplog: pytest.LogCaptureFixture) -> OtelMemory:
    caplog.set_level(logging.DEBUG)
    return OtelMemory(caplog)


@pytest.fixture
def settings() -> Settings:
    return Settings(workers=2, queue_max=100, drain_timeout_s=5)


ServiceFactory = Callable[..., Service]


@pytest.fixture
async def make_service(
    otel_memory: OtelMemory, settings: Settings
) -> AsyncIterator[ServiceFactory]:
    started: list[Service] = []

    def factory(evaluators: list[Evaluator] | None = None, **overrides: Any) -> Service:
        service = Service(
            settings.model_copy(update=overrides),
            otel_memory.telemetry,
            evaluators if evaluators is not None else [PIIDetector(), SecretDetector()],
        )
        service.start()
        started.append(service)
        return service

    yield factory
    for service in started:
        await service.stop_workers()


@pytest.fixture
async def service(make_service: ServiceFactory) -> Service:
    return make_service()
