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

For each evaluated span and evaluator, it emits a `gen_ai.evaluation.result` event linked to the
trace, an `evaluate {name}` child span (optional) and metrics. See the
[telemetry reference](docs/telemetry.md).

## Evaluators

| Evaluator | Default | Detects |
| --- | --- | --- |
| `pii_detection` | on | CPF, CNPJ, e-mail, credit cards, Brazilian phone numbers, PIX keys |
| `secret_detection` | on | Cloud and API keys, tokens, JWTs, private keys, connection strings with a password |
| `refusal` | opt-in | The model declining the request (pt, en, es) or a provider content filter |
| `system_prompt_leak` | opt-in | Responses that copy the system instructions |
| `output_format` | opt-in | Invalid JSON when the client asked for JSON |
| `relevance` | opt-in | Off-topic answers, rated by an LLM judge (OpenAI or any compatible server) on 5% of traces |
| `jev_relevance`, `jev_refusal`, `jev_toxicity`, `jev_prompt_injection` | opt-in, experimental | Off-topic answers, refusals, toxic answers and prompt injection attempts, answered by TypeSafe's Jev in one request per span, on 10% of traces |

Enable them by name in `LLM_EVAL_EVALUATORS`. `relevance` and `jev_relevance` check answer
quality rather than safety. The judge and the Jev-as-a-Judge checks need a model and a key, so
they stay opt-in (see [Configuration](#configuration) to turn the judge on, and
[docs/jev-as-a-judge.md](docs/jev-as-a-judge.md) for Jev-as-a-Judge). What each one reads, how to read its results and
how to write your own: [docs/evaluators.md](docs/evaluators.md).

**No raw sensitive value leaves the service.** Explanations carry only types, counts and
locations (`cpf=1 (input), email=2 (output)`), a sanitizer re-scans every attribute, and the
logs never include message content. The opt-in judges are the exception: `relevance` and the
`jev_*` checks send the conversation to their provider with PII and credentials masked, but
names and addresses go as they are. See
[What goes to the judge provider](docs/llm-as-a-judge.md#what-goes-to-the-judge-provider) and
[What goes to TypeSafe](docs/jev-as-a-judge.md#what-goes-to-typesafe).

## Quick start

You need Docker with the compose plugin. No LLM API key is required.

```sh
docker compose -f deploy/docker-compose.yaml up --build
```

A span generator sends synthetic GenAI spans (PII, credentials, refusals, prompt leaks, JSON,
relevant and off-topic answers) every 30 s through the Collector to the service, which runs all
ten evaluators against a deterministic fake judge. The results go to
[`grafana/otel-lgtm`](https://github.com/grafana/docker-otel-lgtm). Open Grafana at
<http://localhost:3000>:

- **Tempo:** search for a trace from `support-bot`; the `evaluate …` spans sit under the chat span.
- **Loki:** `{service_name="llm-eval-otel"} | gen_ai_evaluation_score_label="fail"`.
- **Prometheus:** `sum by (gen_ai_evaluation_name, gen_ai_evaluation_score_label) (llm_eval_evaluations_total)`.

## Prerequisites

The service can only evaluate what reaches it:

1. **Content capture must be on** in your instrumentation, so spans carry
   `gen_ai.input.messages` and `gen_ai.output.messages`. Content sent only as log events is not
   read.
2. **Application sampling limits coverage:** at 10% sampling, 90% of interactions go unevaluated.
3. **Linked logs:** if your trace UI doesn't show logs linked to a trace, rely on the child span.
4. `system_prompt_leak` needs the system instructions on the span (30+ words), and
   `output_format` needs `gen_ai.output.type`. Without them, they emit nothing.

## Using it with a real application

Point any app instrumented for GenAI at the Collector's `otlp` receiver (`:4317` gRPC or `:4318`
HTTP). For example, with the OpenAI SDK:

```sh
pip install openai opentelemetry-distro opentelemetry-exporter-otlp opentelemetry-instrumentation-openai-v2
export OTEL_SERVICE_NAME=my-chatbot
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY
opentelemetry-instrument python my_chatbot.py
```

The content-capture variables differ between instrumentation libraries, so check yours. The
service reads the current GenAI semconv (`gen_ai.input.messages` etc.) and the OpenLLMetry format
(`gen_ai.prompt.{n}.*`, `gen_ai.completion.{n}.*`).

## Configuration

The most used variables; the full list is in [docs/configuration.md](docs/configuration.md).

| Variable | Default | Effect |
| --- | --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4319` | Collector receiver reserved for evaluator output |
| `LLM_EVAL_EVALUATORS` | `pii_detection,secret_detection` | Enabled evaluators, comma-separated; add `relevance` to turn on the LLM judge |
| `LLM_EVAL_EXCEPTIONS` | empty | JSON `service → [evaluators]` exempted per service, e.g. `{"bank-chatbot": ["pii_detection"]}` |
| `LLM_EVAL_SAMPLE_RATES` | empty | Per-evaluator sample rate, e.g. `relevance=0.05` |
| `LLM_EVAL_AUTH_TOKEN` | empty | When set, requires `Authorization: Bearer <token>` |
| `LLM_EVAL_LLM_JUDGE_MODEL` | empty | Judge model ID; required when `relevance` is enabled |
| `LLM_EVAL_LLM_JUDGE_BASE_URL` | empty (OpenAI API) | Any OpenAI-compatible endpoint (vLLM, Ollama, LiteLLM) |
| `OPENAI_API_KEY` | none | The judge's API key |

To turn on the judge:

```sh
LLM_EVAL_EVALUATORS=pii_detection,secret_detection,relevance
LLM_EVAL_LLM_JUDGE_MODEL=gpt-5-mini-2025-08-07     # pin a dated version, not an alias
OPENAI_API_KEY=...
```

`uv run llm-eval-judge -i "question" -o "answer"` tries a model before you enable it. See
[docs/llm-as-a-judge.md](docs/llm-as-a-judge.md).

## Documentation

- [Evaluators](docs/evaluators.md): what is evaluated, reading the results, writing an evaluator
- [LLM-as-a-Judge](docs/llm-as-a-judge.md): `relevance`, what goes to the provider, CLI, calibration
- [Jev-as-a-Judge](docs/jev-as-a-judge.md): the `jev_*` checks, what goes to TypeSafe, cost, rate limits, calibration
- [Configuration](docs/configuration.md): every variable, exemptions, sampling, auth and TLS
- [Operations](docs/operations.md): backpressure, shutdown, scaling, logs, performance
- [Telemetry reference](docs/telemetry.md): event and span attributes, judge spans, metrics
- [Development](docs/development.md): commands, layout, versioning and releases

One process handles about 190 spans/s of 10 KB with the default evaluators, at under 4 ms p99 per
heuristic; scale with replicas.

## Known limits

- The original span, PII included, still goes to your backend as the app emitted it. To mask it
  there, add a `transform` processor to the backend pipeline.
- Content sent as log events and the legacy span-event format are not read.
- Agent, tool, embeddings and retrieval spans are not evaluated.
- The receiver supports OTLP/HTTP with protobuf only, not gRPC or JSON.
- `relevance` doesn't judge the final answer of a tool loop, and is not calibrated yet. Nor are
  the `jev_*` checks, whose quality in Portuguese is still to be measured.

## License

Apache-2.0. See [LICENSE](LICENSE).
