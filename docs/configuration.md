# Configuration

`OTEL_*` variables are the SDK's standard ones. `LLM_EVAL_*` variables belong to the service.
[.env.example](../.env.example) lists them, the service's first and then the judges'; the
service reads only the environment, while `llm-eval-judge` and `tools/benchmark.py` also read
`./.env`.

| Variable | Default | Effect |
| --- | --- | --- |
| `OTEL_SERVICE_NAME` | `llm-eval-otel` | `service.name` of everything the service emits |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4319` | Collector receiver reserved for evaluator output |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Export protocol |
| `LLM_EVAL_HTTP_PORT` | `4318` | OTLP/HTTP receiver and health endpoints |
| `LLM_EVAL_EVALUATORS` | `pii_detection,secret_detection` | Enabled evaluators, comma-separated; also available: `refusal`, `system_prompt_leak`, `output_format`, `relevance`, `jev_relevance`, `jev_refusal`, `jev_toxicity`, `jev_prompt_injection` |
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
| `LLM_EVAL_DRAIN_TIMEOUT_S` | `30` | Seconds to drain the queue and the judge lanes on shutdown |
| `LLM_EVAL_AUTH_TOKEN` | empty | When set, requires `Authorization: Bearer <token>` |
| `LLM_EVAL_TLS_CERT_FILE`, `LLM_EVAL_TLS_KEY_FILE` | empty | Enable TLS in uvicorn when both are set |
| `LLM_EVAL_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` for the service's own logs. See [Running in production](operations.md#running-in-production) |
| `LLM_EVAL_LOG_SUMMARY_INTERVAL_S` | `60` | Seconds between `INFO` summary lines; `0` turns them off |
| `LLM_EVAL_LLM_JUDGE_MODEL` | empty | Judge model ID; required when `relevance` is enabled |
| `LLM_EVAL_LLM_JUDGE_BASE_URL` | empty (OpenAI API) | Any OpenAI-compatible endpoint, such as vLLM, Ollama or LiteLLM |
| `LLM_EVAL_LLM_JUDGE_RESPONSE_FORMAT` | `json_schema` | `json_schema`, `json_object` or `none`, depending on what the server supports |
| `LLM_EVAL_LLM_JUDGE_TEMPERATURE` | empty (not sent) | `temperature` of the judge call |
| `LLM_EVAL_LLM_JUDGE_REASONING_EFFORT` | empty (not sent) | `reasoning_effort`, for reasoning models |
| `LLM_EVAL_LLM_JUDGE_MAX_OUTPUT_TOKENS` | `1024` | `max_completion_tokens` of the judge call, reasoning included |
| `LLM_EVAL_LLM_JUDGE_MAX_CONCURRENCY` | `8` | Judge calls at a time |
| `LLM_EVAL_LLM_JUDGE_QUEUE_MAX` | `1000` | Evaluations waiting in the judge lane before dropping |
| `LLM_EVAL_LLM_JUDGE_TOKENS_PER_MINUTE` | empty (no limit) | Token budget for the judge |
| `LLM_EVAL_JUDGE_REDACT` | `true` | Mask PII and credentials before sending to the judge, and to Jev |
| `LLM_EVAL_LLM_JUDGE_EXPLANATION` | `true` | Use the judge's justification as the explanation; `false` gives `score=4/5` |
| `OPENAI_API_KEY` | none | The judge's API key, read by the `openai` SDK |
| `LLM_EVAL_JEV_JUDGE_MODEL` | empty | Jev model, e.g. `jev-1.13.0`; required when a `jev_*` check is enabled |
| `LLM_EVAL_JEV_JUDGE_BASE_URL` | empty (TypeSafe's API) | Another System One endpoint; the SDK adds `/v1/systemone` |
| `LLM_EVAL_JEV_JUDGE_MAX_CONCURRENCY` | `16` | Jev requests at a time |
| `LLM_EVAL_JEV_JUDGE_QUEUE_MAX` | `1000` | Requests waiting in the `jev_judge` lane before dropping |
| `LLM_EVAL_JEV_JUDGE_TOKENS_PER_MINUTE` | empty (no limit) | Token budget for Jev |
| `TYPESAFE_API_KEY` | none | Jev's API key, read by the `typesafe-sdk` |

## Exempt services

Some services legitimately handle sensitive data, such as a bank chatbot that receives the
customer's own CPF. List them in `LLM_EVAL_EXCEPTIONS`:

```sh
LLM_EVAL_EXCEPTIONS='{"bank-chatbot": ["pii_detection"], "devops-assistant": ["secret_detection"]}'
```

For those services the evaluator still runs, but the result is labeled `exempt` and has no
score. The explanation still says what was found (`exempt service; cpf=2 (input)`), so you can
see how much sensitive data an exempt service sends, without raising alerts. Service names must
match exactly, and a span without `service.name` is never exempt. A judge is the exception: it
is not called for an exempt service (`exempt service; not evaluated`). For the `jev_*` checks,
that check's question is left out of the request, and the others still go.

## Sampling

Heuristics run on every span. Expensive evaluators can run on a fraction of traces. The
decision follows the OTel `ProbabilitySampler` rule on the TraceID, so it is identical across
replicas and retries. Don't add a `probabilistic_sampler` to the Collector's `traces/genai`
pipeline, because that would also reduce what the heuristics see.

## Auth and TLS with the Collector

If you set `LLM_EVAL_AUTH_TOKEN`, add the header to the Collector exporter:

```yaml
exporters:
  otlp_http/evaluator:
    endpoint: https://llm-eval-otel:4318
    headers: { Authorization: "Bearer ${env:LLM_EVAL_AUTH_TOKEN}" }
```
