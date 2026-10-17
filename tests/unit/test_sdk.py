"""The service's SDK providers: built with the given exporters, OTLP/HTTP in production."""

from typing import Any

import pytest
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_eval_otel.emit import sdk
from llm_eval_otel.version import __version__


def test_build_sends_everything_to_the_given_exporters() -> None:
    spans, logs, metrics = (
        InMemorySpanExporter(),
        InMemoryLogRecordExporter(),
        InMemoryMetricReader(),
    )
    telemetry = sdk.build(spans, logs, metrics)
    telemetry.tracer_provider.get_tracer("test").start_span("evaluate x").end()
    telemetry.logger_provider.get_logger("test").emit(event_name="gen_ai.evaluation.result")
    telemetry.meter_provider.get_meter("test").create_counter("llm_eval.test").add(1)
    telemetry.force_flush()

    [span] = spans.get_finished_spans()
    assert span.resource.attributes["service.version"] == __version__
    [log] = logs.get_finished_logs()
    assert log.log_record.event_name == "gen_ai.evaluation.result"
    data = metrics.get_metrics_data()
    assert data is not None
    [metric] = data.resource_metrics[0].scope_metrics[0].metrics
    assert metric.name == "llm_eval.test"
    telemetry.shutdown()


def test_build_otlp_uses_the_otlp_http_exporters(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[Any, ...]] = []
    monkeypatch.setattr(sdk, "build", lambda *args: built.append(args))
    sdk.build_otlp()
    [(span_exporter, log_exporter, reader)] = built
    assert isinstance(span_exporter, OTLPSpanExporter)
    assert isinstance(log_exporter, OTLPLogExporter)
    assert isinstance(reader, PeriodicExportingMetricReader)
    assert isinstance(reader._exporter, OTLPMetricExporter)
    reader.shutdown()  # never registered with a provider: nothing to export
