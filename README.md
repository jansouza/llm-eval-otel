# llm-eval-otel

An evaluator for GenAI telemetry that runs outside the application's request path. It receives
GenAI inference spans from the OpenTelemetry Collector, checks prompts and responses, and sends
the results back to the Collector as standard OTel telemetry linked to the original trace.

The application does not change and does not wait for it. The service only flags: it never
blocks or alters a response.

```
app ──OTLP──▶ Collector ──traces/backend──────────────────────────▶ backend
                  │
                  └─traces/genai (inference spans only)──▶ llm-eval-otel
                                                              │
backend ◀── otlp/eval :4319 (never routed back) ◀─────────────┘ event + span + metrics
```

For every evaluated span, and for each evaluator, it emits:

- A `gen_ai.evaluation.result` event (a log record) with the span's TraceID and SpanID.
  Severity is `INFO` for `pass`/`exempt`, `WARN` for `fail` and `ERROR` when the evaluator failed.
- An `evaluate {name}` child span of the evaluated span, so trace UIs show the result inline.
  You can turn it off with `LLM_EVAL_EMIT_SPANS=false`.
- Two metrics: `llm_eval.evaluations` (counter) and `llm_eval.evaluation.score` (histogram).

Five evaluators ship with the service. All are local heuristics with no model. The first two are
on by default. The other three are opt-in through `LLM_EVAL_EVALUATORS`:

| Evaluator | Default | Detects |
| --- | --- | --- |
| `pii_detection` | on | CPF and CNPJ, numeric or alphanumeric (check digits validated), e-mail, credit cards (brand prefix, length and Luhn), Brazilian phone numbers (valid DDD), PIX random keys (a UUID v4 with "pix" nearby) |
| `secret_detection` | on | AWS access keys, GitHub tokens, LLM API keys (`sk-…`), JWTs, private keys, connection strings with a password, high-entropy values assigned to `key`/`token`/`secret`/`password`/`senha` |
| `refusal` | opt-in | Responses in which the model declines the request, in Portuguese, English or Spanish, and provider refusals (`finish_reason=content_filter`) |
| `system_prompt_leak` | opt-in | Responses that copy stretches of the system instructions (word 8-gram overlap) |
| `output_format` | opt-in | Invalid JSON when the client asked for JSON (`gen_ai.output.type=json`) |

No raw sensitive value leaves the service. Explanations carry only types, counts, numbers and
where they appeared (`cpf=1 (input), email=2 (output)`). A sanitizer also re-scans every string
attribute before it reaches the SDK, and the service's own logs never include message content.

**Phone numbers are detected since 0.2.0.** Support chatbots often handle phone numbers, so
`pii_detection` may report more `fail` results after an upgrade. To turn off one type without
exempting the whole evaluator, set `LLM_EVAL_PII_TYPES`. For example,
`LLM_EVAL_PII_TYPES=cpf,cnpj,email,credit_card,pix_key` leaves out `phone`.

## Quick start

You need Docker with the compose plugin. No LLM API key is required.

```sh
docker compose -f deploy/docker-compose.yaml up --build
```

The stack has four containers:

- `span-generator`: sends synthetic GenAI spans every 30 s. The cases are clean text, PII, a
  credential, a credential in a tool flow, a three-turn conversation, an exempt service, the
  OpenLLMetry format, CNPJ/phone/PIX, a refusal, a leak of the system instructions, valid and
  truncated JSON output, and a non-GenAI span. The demo enables all five evaluators.
- `otel-collector`: runs the config in [deploy/otel-collector-config.yaml](deploy/otel-collector-config.yaml).
- `llm-eval-otel`: this service.
- `backend`: [`grafana/otel-lgtm`](https://github.com/grafana/docker-otel-lgtm), which bundles
  Grafana, Tempo, Loki and Prometheus.

Open Grafana at <http://localhost:3000>:

- **Tempo:** search for any trace from `support-bot`. The `evaluate pii_detection` and
  `evaluate secret_detection` spans sit under the chat span.
- **Loki:** `{service_name="llm-eval-otel"} | gen_ai_evaluation_score_label="fail"` lists the
  flagged events, each carrying `trace_id` and `span_id`.
- **Prometheus:** `sum by (gen_ai_evaluation_name, gen_ai_evaluation_score_label) (llm_eval_evaluations_total)`.

## Prerequisites for adopting it

The service can only evaluate what reaches it:

1. **Content capture must be on.** Instrumentations record `gen_ai.input.messages` and
   `gen_ai.output.messages` only when content capture is enabled, because both attributes are
   opt-in in the semantic conventions. Without capture, the service receives spans with nothing
   to evaluate. Content sent only as log events is not read in this version.
2. **Application sampling limits coverage.** The service sees only sampled spans. At 10% sampling
   in the app, 90% of interactions go unevaluated, and that includes the security heuristics.
3. **Your backend needs to show linked log records.** The evaluation event is a log record linked
   to the trace. If your trace UI does not show linked logs, rely on the child span, which is on
   by default.
4. **Two evaluators need more than messages.** `system_prompt_leak` needs the system
   instructions on the span, in `gen_ai.system_instructions` or as a `system` message, with at
   least 30 words. `output_format` needs `gen_ai.output.type`. With OpenLLMetry, it reads
   `gen_ai.request.structured_output_schema` instead, which OpenLLMetry records when the client
   asks for structured output. When a span lacks this data, the evaluator does not apply and
   emits nothing.

## Using it with a real application

Replace the generator with any app instrumented for GenAI and point it at the Collector's
`otlp` receiver (`:4317` gRPC or `:4318` HTTP). For example, with the OpenAI SDK and
`opentelemetry-instrumentation-openai-v2`:

```sh
pip install openai opentelemetry-distro opentelemetry-exporter-otlp opentelemetry-instrumentation-openai-v2
export OTEL_SERVICE_NAME=my-chatbot
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
# Record message content on spans (attribute names follow the latest GenAI semconv).
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
opentelemetry-instrument python my_chatbot.py
```

The content-capture variables and their accepted values differ between instrumentation
libraries and versions, so check your instrumentation's documentation. The service reads two
content formats:

- **Current semconv:** `gen_ai.system_instructions`, `gen_ai.input.messages` and
  `gen_ai.output.messages`, either structured or as JSON strings.
- **OpenLLMetry:** `gen_ai.prompt.{n}.*` and `gen_ai.completion.{n}.*`.

To add the ports to the compose demo, publish `4317`/`4318` on `otel-collector` and set
`GENERATOR_EVERY` high, or remove `span-generator`.

## What gets evaluated in a span

- **System instructions:** always evaluated, since they go to the provider on every call.
- **Input messages:** only the ones new in this turn, meaning everything after the last
  `assistant` message, including tool results. Chat spans resend the whole history, so this rule
  flags a CPF typed in turn 1 on turn 1's span only.
- **Output messages:** all of them, including every choice when `n > 1`.
- **Message parts:** `text` and `reasoning` (`content`), `tool_call` (`arguments`) and
  `tool_call_response` (`response`). Blob, file, URI and server-side tool calls are skipped.
  `pii_detection` and `secret_detection` scan every part. The other evaluators read only the
  output parts the user or a tool receives:

  | Evaluator | Reads | Applies when |
  | --- | --- | --- |
  | `refusal` | output `text`, first 300 characters of each message; `gen_ai.response.finish_reasons` | there is output text or a finish reason |
  | `system_prompt_leak` | system instructions; output `text` and `tool_call` | the instructions have 30+ words and there is output text or a tool call |
  | `output_format` | output `text` | `gen_ai.output.type` is `json` and there is output text |

- **Finish reasons:** `gen_ai.response.finish_reasons`, or, when it is absent, the deprecated
  `finish_reason` of each output message (OpenLLMetry still writes it) or
  `gen_ai.completion.{n}.finish_reason`.

Only spans with `gen_ai.operation.name` of `chat`, `text_completion` or `generate_content` are
evaluated, plus OpenLLMetry spans that carry content. Anything else is counted in
`llm_eval.spans.skipped` with its reason.

### Reading the results

- **`pii_detection`, `secret_detection`:** score `0.0` and `fail` when anything is found,
  `1.0` and `pass` otherwise. When one stretch of text matches two PII types, it counts once, in
  this order: CPF, CNPJ, card, phone.
- **`refusal`:** `fail` means the model refused, not that it misbehaved: refusing an abusive
  request is correct. Read it as a refusal rate per model and service. A phrase counts only
  with a refusal verb and an object ("não posso ajudar com", "I can't assist with") in the
  first 300 characters. Partial refusals and other languages are not detected.
  Explanation: `refusal=1 (output), source=phrase, lang=pt`.
- **`system_prompt_leak`:** score is `1 - coverage`. The label is `fail` when 20 or more words
  are copied in a row, or coverage is above 0.15, so a long copied run can fail with a high
  score. Both thresholds are initial values. Paraphrases and translations are not detected. If
  a service's instructions hold text the model must repeat (an FAQ, standard replies), exempt it
  in `LLM_EVAL_EXCEPTIONS`. Explanation: `coverage=0.42, longest_run=37 words (output)`.
- **`output_format`:** score is the share of outputs that parse as JSON. Markdown fences are
  not stripped, because a fence means JSON mode was not honored. Only syntax is checked: the
  semconv has no attribute with the requested schema. The explanation gives the parser's
  message and position, never the text: `invalid_json=1 of 2 (output): Expecting ','
  delimiter at char 132`, plus `finish_reason=length` when the output was cut off.

## Configuration

`OTEL_*` variables are the SDK's standard ones. `LLM_EVAL_*` variables belong to the service.

| Variable | Default | Effect |
| --- | --- | --- |
| `OTEL_SERVICE_NAME` | `llm-eval-otel` | `service.name` of everything the service emits |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4319` | Collector receiver reserved for evaluator output |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Export protocol |
| `LLM_EVAL_HTTP_PORT` | `4318` | OTLP/HTTP receiver and health endpoints |
| `LLM_EVAL_EVALUATORS` | `pii_detection,secret_detection` | Enabled evaluators, comma-separated; also available: `refusal`, `system_prompt_leak`, `output_format` |
| `LLM_EVAL_PII_TYPES` | `cpf,cnpj,email,credit_card,phone,pix_key` | Types `pii_detection` reports; the sanitizer always redacts all of them |
| `LLM_EVAL_SAMPLE_RATES` | empty | Per-evaluator sample rate override, e.g. `relevance=0.05` |
| `LLM_EVAL_EXCEPTIONS` | empty | JSON `service → [evaluators]` exempted per service |
| `LLM_EVAL_ASSOCIATION_EXCLUDE` | `correlation_id` | Association property keys kept off the metrics, comma-separated |
| `LLM_EVAL_WORKERS` | `4` | Asyncio workers draining the queue; they help evaluators that wait on I/O |
| `LLM_EVAL_QUEUE_MAX` | `10000` | Queued interactions before answering 429 |
| `LLM_EVAL_TIMEOUT_S` | `5` | Default per-evaluator timeout |
| `LLM_EVAL_MAX_REQUEST_BYTES` | `16777216` | Maximum request size, after decompression |
| `LLM_EVAL_EMIT_SPANS` | `true` | Emit the child span |
| `LLM_EVAL_DEDUP_TTL_S` | `600` | Deduplication window for resent spans |
| `LLM_EVAL_AUTH_TOKEN` | empty | When set, requires `Authorization: Bearer <token>` |
| `LLM_EVAL_TLS_CERT_FILE`, `LLM_EVAL_TLS_KEY_FILE` | empty | Enable TLS in uvicorn when both are set |

**Exempt services.** Some services legitimately handle sensitive data, such as a bank chatbot
that receives the customer's own CPF. List them in `LLM_EVAL_EXCEPTIONS`:

```sh
LLM_EVAL_EXCEPTIONS='{"bank-chatbot": ["pii_detection"], "devops-assistant": ["secret_detection"]}'
```

For those services the evaluator still runs, but the result is labeled `exempt` and has no
score. The explanation still says what was found (`exempt service; cpf=2 (input)`), so you can
see how much sensitive data an exempt service sends, without raising alerts. Service names must
match exactly, and a span without `service.name` is never exempt.

**Sampling.** Heuristics run on every span. Expensive evaluators can run on a fraction of
traces. The decision follows the OTel `ProbabilitySampler` rule on the TraceID, so it is
identical across replicas and retries. Don't add a `probabilistic_sampler` to the Collector's
`traces/genai` pipeline, because that would also reduce what the heuristics see.

**Auth and TLS with the Collector.** If you set `LLM_EVAL_AUTH_TOKEN`, add the header to the
Collector exporter:

```yaml
exporters:
  otlp_http/evaluator:
    endpoint: https://llm-eval-otel:4318
    headers: { Authorization: "Bearer ${env:LLM_EVAL_AUTH_TOKEN}" }
```

## Running in production

- **Backpressure.** The service returns 200 as soon as spans are queued. A full queue returns 429
  with `Retry-After`, and the Collector's exporter retries (keep `retry_on_failure` and
  `sending_queue` on). Invalid payloads return 400 and are not retried.
- **Losses on crash.** The queue lives in memory, so a crash loses what was queued. The original
  spans still reach your backend unchanged.
- **Shutdown.** On SIGTERM the service stops accepting data, drains the queue for up to 30 s, then
  flushes the SDK.
- **Scaling.** Scale with replicas: regex work is bound by the GIL, so more workers do not add
  throughput. Deduplication is per instance, so a Collector retry that lands on a different
  replica can be evaluated twice. That only happens on retries. The Collector's `loadbalancing`
  exporter (`routing_key: traceID`) would pin each trace to one replica, but as of Collector
  0.161.0 it only speaks OTLP/gRPC, which this version does not accept.
- **Container.** The image runs as a non-root user (uid 10001) and works with a read-only root
  filesystem. It exposes `GET /healthz` (liveness) and `GET /readyz`, which fails while the queue
  is above 90% or during shutdown.

## Writing an evaluator

Evaluators know nothing about OpenTelemetry. They receive an extracted `GenAIInteraction` and
return an `EvaluationResult`, and the service handles the telemetry.

```python
from llm_eval_otel.evaluators.base import (
    EvaluationResult, EvaluatorKind, GenAIInteraction,
)

class Relevance:
    name = "relevance"                 # becomes gen_ai.evaluation.name
    kind = EvaluatorKind.LLM_JUDGE     # heuristics run in a thread; others on the event loop
    timeout_s = 20.0                   # 0 = use LLM_EVAL_TIMEOUT_S
    sample_rate = 0.05                 # about 5% of traces
    max_chars = 8_000                  # the runner truncates and marks llm_eval.content.truncated

    def applies_to(self, interaction: GenAIInteraction) -> bool:
        return bool(interaction.output_messages)

    async def evaluate(self, interaction: GenAIInteraction) -> EvaluationResult:
        score = await ask_the_judge(interaction)  # normalize to 0..1, higher is better
        return EvaluationResult(
            score=score,
            label="pass" if score >= 0.6 else "fail",
            explanation="answer addresses the question",  # never quote the content
            attributes={"llm_eval.relevance.threshold": 0.6},  # llm_eval.* keys only
        )
```

Register it in your own package's `pyproject.toml` and enable it by name. You don't need to
change this repository:

```toml
[project.entry-points."llm_eval.evaluators"]
relevance = "my_package.relevance:Relevance"
```

```sh
LLM_EVAL_EVALUATORS=pii_detection,secret_detection,relevance
```

The runner turns exceptions and timeouts into results that carry `error.type` (the exception's
class name, never its message), and one failure does not stop the other evaluators. The
sanitizer also covers your evaluator: if an explanation quotes PII or a credential, that value
becomes `[REDACTED]`.

## Telemetry reference

GenAI semantic conventions are still in Development status. The service follows
[semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) at
commit `e57c543`, and every name lives in
[src/llm_eval_otel/semconv.py](src/llm_eval_otel/semconv.py). Names that the semconv does not
define use the `llm_eval.*` prefix, so they cannot collide with future `gen_ai.*` definitions.

| Attribute on the event and span | Example |
| --- | --- |
| `gen_ai.evaluation.name` | `pii_detection` |
| `gen_ai.evaluation.score.value` | `0.0` (1.0 = pass; absent when exempt or on error) |
| `gen_ai.evaluation.score.label` | `pass`, `fail` or `exempt` |
| `gen_ai.evaluation.explanation` | `cpf=1 (input), email=1 (output)` |
| `gen_ai.response.id`, `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model` | copied from the evaluated span |
| `error.type` | `timeout` (only on failure) |
| `llm_eval.source.service.name` | `service.name` of the app that produced the span |
| `traceloop.association.properties.*` | copied from the evaluated span (OpenLLMetry), e.g. `scenario` |
| `llm_eval.evaluation.type` | `heuristic`, `model` or `llm_judge` |
| `llm_eval.pii.types`, `llm_eval.secret.types` | `["cpf", "email"]` |
| `llm_eval.refusal.source` | `phrase` or `finish_reason` |
| `llm_eval.refusal.language` | `pt`, `en` or `es` (only with `source=phrase`) |
| `llm_eval.prompt_leak.coverage` | `0.42`: share of the instructions' 8-grams found in the output |
| `llm_eval.prompt_leak.longest_run` | `37`: longest run of copied words |
| `llm_eval.output_format.error` | `syntax`, `empty` or `truncated` (syntax error with `finish_reason=length`) |
| `llm_eval.content.truncated` | `true` when the evaluator's `max_chars` cut the text |

| Metric | Type | Attributes |
| --- | --- | --- |
| `llm_eval.evaluations` | Counter | evaluation name, label, `error.type`, source service, provider, model, association properties |
| `llm_eval.evaluation.score` | Histogram (0.1 … 1.0) | evaluation name, source service, provider, model, association properties; excludes exempt and errors |
| `llm_eval.evaluation.duration` | Histogram, `s` | evaluation name, `error.type` |
| `llm_eval.spans.received` | Counter | none |
| `llm_eval.spans.skipped` | Counter | `llm_eval.skip.reason`: `not_inference`, `no_content`, `duplicate`, `invalid_payload` |
| `llm_eval.queue.size` | UpDownCounter | none |
| `llm_eval.sanitizer.redactions` | Counter | evaluation name |

Metric attributes never include TraceID, SpanID, response IDs or free text, which keeps
cardinality low. Association properties go on the metrics with the same names as on the
application's own OpenLLMetry metrics, so both can be filtered by the same key. Keys whose
value changes per request (a correlation or session ID) create one series per request: list
them in `LLM_EVAL_ASSOCIATION_EXCLUDE`. The event and the span keep every property.
The sanitizer also covers the properties, so a value that is PII shows up as `[REDACTED]`. `service.version` on the resource identifies the version of the detection
rules.

## Performance

Measured with `uv run python tools/load_test.py --spans 3000 --text-kb 10` on 2026-09-30,
version 0.2.0. The machine had 8 cores, and the test ran one process with 4 workers. Each span
carried 10 KB of text with some PII and credentials mixed in. Each evaluator got 10 KB of what
it reads; `system_prompt_leak` compared 10 KB of instructions with 10 KB of output. The
throughput run used the real HTTP server with the default evaluators and counted exported
events at a local fake Collector.

| Measure | Target | Measured |
| --- | --- | --- |
| `pii_detection` p99, 10 KB | ≤ 5 ms | 2.04 ms (p50 1.55 ms) |
| `secret_detection` p99, 10 KB | ≤ 5 ms | 3.74 ms (p50 1.90 ms) |
| `refusal` p99, 10 KB | ≤ 5 ms | 0.29 ms (p50 0.18 ms) |
| `system_prompt_leak` p99, 10 KB + 10 KB | ≤ 5 ms | 2.31 ms (p50 1.83 ms) |
| `output_format` p99, 10 KB | ≤ 5 ms | 0.20 ms (p50 0.14 ms) |
| Throughput per process, `pii_detection` + `secret_detection` | ≥ 100 spans/s | 194 spans/s |

Typical chat spans carry less than 10 KB of new content, so expect more throughput in practice.
Add replicas to scale.

## Development

```sh
uv sync                       # Python 3.12
uv run ruff check . && uv run ruff format --check .
uv run mypy                   # strict
uv run pytest                 # unit and API tests, in-memory exporters
uv run pytest tests/e2e -m e2e  # docker compose + the Collector's file exporter
```

The layout:

```
src/llm_eval_otel/
  main.py              # FastAPI app, workers and SDK providers
  config.py            # LLM_EVAL_* settings
  semconv.py           # every attribute, event and metric name
  ingest/http.py       # POST /v1/traces, /healthz, /readyz
  extract/genai.py     # span -> GenAIInteraction (two formats)
  engine/queue.py      # bounded queue, dedup, workers
  engine/runner.py     # sampling, timeouts, exemptions, truncation
  engine/service.py    # wires the pieces together
  evaluators/          # base (the contract), pii, secrets, refusal, prompt_leak,
                       # output_format, registry (entry points)
  version.py           # the version, also service.version
  emit/                # sdk (providers), emitter (event, span, metrics), sanitize
tools/span_generator.py  # synthetic spans for the demo
tools/load_test.py       # latency and throughput measurement
deploy/                  # Collector config and docker compose
```

### Versioning

The version lives only in `src/llm_eval_otel/version.py`; `pyproject.toml` reads it from there.
It is the `service.version` on all the service's telemetry, so it tells which detection rules
produced a result. Bump the minor version when what gets detected changes (a new type,
evaluator or threshold) and the patch version for fixes that don't change detection. Add the
change to [CHANGELOG.md](CHANGELOG.md).

To release, edit `version.py`, merge, and push a matching tag (`git tag v0.2.0 && git push
origin v0.2.0`). The release workflow fails if the tag and `version.py` differ.

## Known limits

- The original span, PII included, still goes to your backend as the app emitted it. To mask it
  there, add a `transform` processor to the backend pipeline.
- The service doesn't read content sent as log events
  (`gen_ai.client.inference.operation.details`) or the legacy span-event format.
- Agent, tool, embeddings and retrieval spans are not evaluated.
- The receiver supports OTLP/HTTP with protobuf only, not gRPC or JSON.

## License

Apache-2.0. See [LICENSE](LICENSE).
