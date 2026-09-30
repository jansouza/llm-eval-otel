"""OpenTelemetry SDK providers for the service's own output.

Providers are built here and injected into the emitter, never set globally, so
tests can swap the OTLP exporters for in-memory ones.

In opentelemetry-sdk 1.45 the Logs API still lives in ``opentelemetry._logs``
(underscore module, no stability guarantee); versions are pinned in pyproject.toml.
"""

from dataclasses import dataclass

from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from llm_eval_otel.version import __version__


@dataclass
class Telemetry:
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    logger_provider: LoggerProvider

    def force_flush(self, timeout_ms: int = 30_000) -> None:
        self.tracer_provider.force_flush(timeout_ms)
        self.logger_provider.force_flush(timeout_ms)
        self.meter_provider.force_flush(timeout_ms)

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.logger_provider.shutdown()
        self.meter_provider.shutdown()


def resource() -> Resource:
    """``service.name`` from OTEL_SERVICE_NAME; ``service.version`` versions the rules."""
    return Resource.create({SERVICE_VERSION: __version__})


def new_tracer_provider(res: Resource) -> TracerProvider:
    """Always sample the child spans.

    The evaluated span was already sampled, or it would not have been exported. Its
    OTLP ``flags`` often lack the sampled bit (several SDKs leave them 0), and the
    default parent-based sampler would then drop every child span.
    """
    return TracerProvider(resource=res, sampler=ALWAYS_ON)


def build(
    span_exporter: SpanExporter,
    log_exporter: LogRecordExporter,
    metric_reader: MetricReader,
    res: Resource | None = None,
) -> Telemetry:
    res = res or resource()
    tracer_provider = new_tracer_provider(res)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    logger_provider = LoggerProvider(resource=res)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    meter_provider = MeterProvider(resource=res, metric_readers=[metric_reader])
    return Telemetry(tracer_provider, meter_provider, logger_provider)


def build_otlp() -> Telemetry:
    """OTLP/HTTP exporters configured by the standard OTEL_EXPORTER_OTLP_* variables."""
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return build(
        OTLPSpanExporter(),
        OTLPLogExporter(),
        PeriodicExportingMetricReader(OTLPMetricExporter()),
    )
