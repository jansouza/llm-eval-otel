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

Six evaluators ship with the service: five local heuristics with no model, and one
LLM-as-a-Judge. The first two are on by default. The others are opt-in through
`LLM_EVAL_EVALUATORS`:

| Evaluator | Default | Detects |
| --- | --- | --- |
| `pii_detection` | on | CPF and CNPJ, numeric or alphanumeric (check digits validated), e-mail, credit cards (brand prefix, length and Luhn), Brazilian phone numbers (valid DDD), PIX random keys (a UUID v4 with "pix" nearby) |
| `secret_detection` | on | AWS access keys, GitHub tokens, LLM API keys (`sk-…`), JWTs, private keys, connection strings with a password, high-entropy values assigned to `key`/`token`/`secret`/`password`/`senha` |
| `refusal` | opt-in | Responses in which the model declines the request, in Portuguese, English or Spanish, and provider refusals (`finish_reason=content_filter`) |
| `system_prompt_leak` | opt-in | Responses that copy stretches of the system instructions (word 8-gram overlap) |
| `output_format` | opt-in | Invalid JSON when the client asked for JSON (`gen_ai.output.type=json`) |
| `relevance` | opt-in | Responses that don't address what the user asked, rated 1 to 5 by an LLM judge (the OpenAI API or any OpenAI-compatible server) on 5% of traces. See [LLM-as-a-Judge](#llm-as-a-judge-relevance) |

No raw sensitive value leaves the service. The heuristics' explanations carry only types,
counts, numbers and where they appeared (`cpf=1 (input), email=2 (output)`). A sanitizer also
re-scans every string attribute before it reaches the SDK, and the service's own logs never
include message content. `relevance` is the exception in two ways, both opt-in: it sends the
conversation to the judge, with PII and credentials masked first, and its explanation is the
judge's own justification, cut to 300 characters and sanitized. See
[What goes to the judge provider](#what-goes-to-the-judge-provider).

**Phone numbers are detected since 0.2.0.** Support chatbots often handle phone numbers, so
`pii_detection` may report more `fail` results after an upgrade. To turn off one type without
exempting the whole evaluator, set `LLM_EVAL_PII_TYPES`. For example,
`LLM_EVAL_PII_TYPES=cpf,cnpj,email,credit_card,pix_key` leaves out `phone`.

## Quick start

You need Docker with the compose plugin. No LLM API key is required.

```sh
docker compose -f deploy/docker-compose.yaml up --build
```

The stack has five containers:

- `span-generator`: sends synthetic GenAI spans every 30 s. The cases are clean text, PII, a
  credential, a credential in a tool flow, a three-turn conversation, an exempt service, the
  OpenLLMetry format, CNPJ/phone/PIX, a refusal, a leak of the system instructions, valid and
  truncated JSON output, a relevant and an off-topic answer, and a non-GenAI span. The demo
  enables all six evaluators, with `relevance` on every span instead of 5%.
- `otel-collector`: runs the config in [deploy/otel-collector-config.yaml](deploy/otel-collector-config.yaml).
- `llm-eval-otel`: this service.
- `fake-judge`: [tools/fake_judge_server.py](tools/fake_judge_server.py), a deterministic
  OpenAI-compatible server that stands in for the judge model. It scores by word overlap
  between the question and the answer, so it is only good for the demo and the tests. To use
  a real model, see [LLM-as-a-Judge](#llm-as-a-judge-relevance).
- `backend`: [`grafana/otel-lgtm`](https://github.com/grafana/docker-otel-lgtm), which bundles
  Grafana, Tempo, Loki and Prometheus.

Open Grafana at <http://localhost:3000>:

- **Tempo:** search for any trace from `support-bot`. The `evaluate pii_detection` and
  `evaluate secret_detection` spans sit under the chat span. Under `evaluate relevance` sits
  the judge's own `chat fake-judge-1` call, with its tokens.
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
  | `relevance` | `user` messages new in the turn (`text`); output `text`; up to 4 earlier user/assistant `text` messages as context, 1,000 characters each | there is a user message in the turn and output text, and no output tool call (an agent step) |

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
- **`relevance`:** the judge's rating from 1 to 5 is in `llm_eval.judge.raw_score`; the score
  is `(rating - 1) / 4` and `pass` is a rating of 3 or more. That threshold is an initial
  value, not yet calibrated against human labels. Only relevance is judged, not accuracy,
  tone or safety, and a refusal or clarifying question about the request counts as relevant
  (`refusal` tracks those). The explanation is the judge's justification. Read it as a rate per
  service and model: the evaluator samples traces, so its metrics count only the sample.

## LLM-as-a-Judge (`relevance`)

`relevance` asks a model whether the response addresses what the user asked. It uses the
official `openai` SDK over the Chat Completions API, which OpenAI-compatible servers also
implement, so the same code talks to the OpenAI API, to a model you host (vLLM, Ollama), or to a
gateway such as LiteLLM in front of other providers.

```sh
LLM_EVAL_EVALUATORS=pii_detection,secret_detection,relevance
LLM_EVAL_JUDGE_MODEL=gpt-5-mini-2025-08-07     # required; pin a dated version, not an alias
OPENAI_API_KEY=...                             # mount it as a secret

# Or a judge on your own network, so content never leaves it:
LLM_EVAL_JUDGE_BASE_URL=http://vllm:8000/v1    # Ollama: http://ollama:11434/v1
LLM_EVAL_JUDGE_MODEL=Qwen/Qwen3-8B
OPENAI_API_KEY=unused                          # the SDK requires the variable to exist
```

The service won't start with `relevance` enabled and no `LLM_EVAL_JUDGE_MODEL`: the model sets
both the cost and the quality, so there is no default.

- **Structured output.** By default the request uses `response_format` with a strict JSON
  schema. Servers that don't support it can use `LLM_EVAL_JUDGE_RESPONSE_FORMAT=json_object` or
  `none`; then the schema also goes in the prompt. The service validates the answer against the
  schema in every mode.
- **Optional parameters.** `temperature` and `reasoning_effort` are sent only when set
  (`LLM_EVAL_JUDGE_TEMPERATURE`, `LLM_EVAL_JUDGE_REASONING_EFFORT`), because reasoning models
  reject `temperature` and several servers reject `reasoning_effort`. Reasoning tokens count
  against `LLM_EVAL_JUDGE_MAX_OUTPUT_TOKENS` (default 1024): raise it if a reasoning model ends
  in `judge_truncated`.
- **Its own lane.** Judge calls run apart from the heuristics, in a lane with a queue of
  `LLM_EVAL_JUDGE_QUEUE_MAX` evaluations and `LLM_EVAL_JUDGE_MAX_CONCURRENCY` calls at a time.
  A slow or unreachable judge never holds the heuristics and never causes a 429. When the lane
  is full, the evaluation is dropped and counted in `llm_eval.evaluations.dropped` with
  `llm_eval.drop.reason=lane_full`: under overload, the judge sees a smaller sample. Alert on
  that counter.
- **Token budget.** `LLM_EVAL_JUDGE_TOKENS_PER_MINUTE` caps spend: each call reserves an
  estimate (characters / 4 plus the maximum output) and settles with the usage the server
  reports. Without budget left, the evaluation is dropped with `llm_eval.drop.reason=budget`.
- **Errors.** Each failure is an event with `error.type` and severity `ERROR`: `timeout`
  (30 s), `judge_refusal` (the judge refused or its filter fired), `judge_truncated`
  (`finish_reason=length`), `judge_invalid_output` (not JSON, or outside the schema), or the
  SDK's exception class, such as `APIConnectionError` or `RateLimitError`. The SDK retries 429s
  and 5xx once.
- **Exempt services.** A service exempted from `relevance` in `LLM_EVAL_EXCEPTIONS` never
  reaches the judge: the event is `exempt` with the explanation `exempt service; not
  evaluated`. Running the judge there would pay to send content out for nothing.
- **Cost.** The service does not compute money, because prices change. Multiply
  `gen_ai.client.token.usage` by your model's prices. The judge prompt is fixed and goes first,
  so providers with prefix caching (OpenAI does it automatically) bill it as cached input;
  `gen_ai.usage.cache_read.input_tokens` on the `chat` span shows whether it hit.
- **Manipulation.** The evaluated content goes to the judge as JSON inside
  `<conversation>…</conversation>`, escaped so it can't close the tag, and the prompt says
  everything inside is data, never instructions. The output is limited to the schema. Content
  can still bias the rating within the scale.

### Judging from the command line

`llm-eval-judge` runs the judge on interactions you give it, with the same configuration and
code as the service: masking, the 16,000-character cut, the timeout, and the sanitizer on what
it prints. Unlike the service, it always runs, with no sampling and no exemptions. It's for
trying a model or a server before turning `relevance` on, and for checking a single case.

```sh
cp .env.example .env          # set LLM_EVAL_JUDGE_MODEL and OPENAI_API_KEY (or a base URL)

uv run llm-eval-judge -i "Qual o horário de atendimento?" -o "Das 9h às 18h, de segunda a sexta."
uv run llm-eval-judge -i "E em inglês?" -o "Good morning." \
    -c "user:Como digo bom dia em espanhol?" -c "assistant:Buenos días."
uv run llm-eval-judge --jsonl tools/data/relevance-smoke.jsonl   # one interaction per line
uv run llm-eval-judge -i "Meu CPF é 529.982.247-25" -o "Anotado." --dry-run
```

It reads `./.env` (or `--env-file`); variables already set in the environment win. It prints
one JSON object per interaction: `label`, `score`, `explanation`, `attributes`, `error_type`,
and each judge call with model, endpoint, tokens, finish reason and time. `--dry-run` prints
exactly what would go to the judge, masked, without calling it and without needing a model or
key. `-e` runs another evaluator, heuristics included. The exit status is 0 when every
interaction was evaluated, 1 when any ended with `error_type`, and 2 on bad usage or
configuration. The image has the command too:
`docker run --rm --env-file .env --entrypoint llm-eval-judge <image> -i "…" -o "…"`.

### Judging synthetic spans in the demo stack

To try the judge through the whole path (span → Collector → service → judge → telemetry back
through the Collector), the compose override swaps the fake judge for the one in `./.env` and
the fixed cases for [tools/data/relevance-synthetic.jsonl](tools/data/relevance-synthetic.jsonl),
30 labeled interactions in Portuguese and English across eight services, some with context,
PII, a credential or an attempt to steer the judge:

```sh
docker compose -f deploy/docker-compose.yaml -f deploy/docker-compose.judge.yaml up --build
# send the set again, with the stack running
GENERATOR_SEED=2 docker compose -f deploy/docker-compose.yaml -f deploy/docker-compose.judge.yaml run --rm span-generator
```

Every span is judged (`relevance=1.0`), so each run of the set is 29 judge calls (`bank-chatbot`
is exempt). Change `GENERATOR_SEED` on every run: the seed sets the trace and span IDs, and
the service drops repeated IDs as Collector retries for 10 minutes. The `.env` variables
configure the judge; the override sets the endpoint, evaluators, exemptions and sample rates.
Each chat span carries `synthetic.id`, `synthetic.label` and `synthetic.score`, the human
labels, to compare in Tempo with the `evaluate relevance` span below it.
`span_generator.py --dataset` takes any file in the `tools/benchmark.py` format, plus optional
`service` and `system` fields.

### What goes to the judge provider

For each sampled span: the user's new messages in the turn, the response's text parts (not
reasoning, not tool calls), and up to four earlier user/assistant messages. The system
instructions are not sent. The text is cut to 16,000 characters, keeping the response first;
then the event gets `llm_eval.content.truncated=true`.

Before sending, everything `pii_detection` and `secret_detection` detect is replaced by its type:
`[CPF]`, `[CNPJ]`, `[EMAIL]`, `[CREDIT_CARD]`, `[PHONE]`, `[PIX_KEY]`, `[SECRET]`. This is on by
default (`LLM_EVAL_JUDGE_REDACT=true`) and uses every type, whatever `LLM_EVAL_PII_TYPES` says.
**Names, addresses and anything else the regexes don't detect are sent as they are.** If that
is not acceptable, point `LLM_EVAL_JUDGE_BASE_URL` at a model on your own network.

The judge's justification comes back as the explanation. The prompt asks it not to quote the
conversation; the service cuts it to 300 characters and runs the sanitizer on it, so a CPF or
key in it becomes `[REDACTED]`. A name it quotes would pass the sanitizer. To keep the judge's
free text out of your telemetry, set `LLM_EVAL_JUDGE_EXPLANATION=false`, and the explanation
becomes `score=4/5`.

### Calibrating

[tools/benchmark.py](tools/benchmark.py) runs an evaluator over a labeled JSONL set and reports
agreement with the human pass/fail label, how much the rating varies on repeated runs, latency,
tokens and the cost per thousand evaluations:

```sh
LLM_EVAL_JUDGE_MODEL=gpt-5-mini-2025-08-07 OPENAI_API_KEY=... \
uv run python tools/benchmark.py my-labeled-set.jsonl --repeat 3 \
    --price-input 0.25 --price-cached 0.025 --price-output 2.00
```

[tools/data/relevance-smoke.jsonl](tools/data/relevance-smoke.jsonl) is a 16-item smoke set
that shows the format and checks the setup. It is too small and too easy for calibration. The
target is about 200 human-labeled interactions in Portuguese and English, compared on a large
and a small OpenAI model and an open model served locally, with 80% agreement or more on
pass/fail. That calibration has not been run yet, so the model choice and the threshold are
still open.

## Configuration

`OTEL_*` variables are the SDK's standard ones. `LLM_EVAL_*` variables belong to the service.
[.env.example](.env.example) lists the judge's variables; the service reads only the
environment, while `llm-eval-judge` and `tools/benchmark.py` also read `./.env`.

| Variable | Default | Effect |
| --- | --- | --- |
| `OTEL_SERVICE_NAME` | `llm-eval-otel` | `service.name` of everything the service emits |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4319` | Collector receiver reserved for evaluator output |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Export protocol |
| `LLM_EVAL_HTTP_PORT` | `4318` | OTLP/HTTP receiver and health endpoints |
| `LLM_EVAL_EVALUATORS` | `pii_detection,secret_detection` | Enabled evaluators, comma-separated; also available: `refusal`, `system_prompt_leak`, `output_format`, `relevance` |
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
| `LLM_EVAL_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` for the service's own logs. See [Running in production](#running-in-production) |
| `LLM_EVAL_LOG_SUMMARY_INTERVAL_S` | `60` | Seconds between `INFO` summary lines; `0` turns them off |
| `LLM_EVAL_JUDGE_MODEL` | empty | Judge model ID; required when `relevance` is enabled |
| `LLM_EVAL_JUDGE_BASE_URL` | empty (OpenAI API) | Any OpenAI-compatible endpoint, such as vLLM, Ollama or LiteLLM |
| `LLM_EVAL_JUDGE_RESPONSE_FORMAT` | `json_schema` | `json_schema`, `json_object` or `none`, depending on what the server supports |
| `LLM_EVAL_JUDGE_TEMPERATURE` | empty (not sent) | `temperature` of the judge call |
| `LLM_EVAL_JUDGE_REASONING_EFFORT` | empty (not sent) | `reasoning_effort`, for reasoning models |
| `LLM_EVAL_JUDGE_MAX_OUTPUT_TOKENS` | `1024` | `max_completion_tokens` of the judge call, reasoning included |
| `LLM_EVAL_JUDGE_MAX_CONCURRENCY` | `8` | Judge calls at a time |
| `LLM_EVAL_JUDGE_QUEUE_MAX` | `1000` | Evaluations waiting in the judge lane before dropping |
| `LLM_EVAL_JUDGE_TOKENS_PER_MINUTE` | empty (no limit) | Token budget for the judge |
| `LLM_EVAL_JUDGE_REDACT` | `true` | Mask PII and credentials before sending to the judge |
| `LLM_EVAL_JUDGE_EXPLANATION` | `true` | Use the judge's justification as the explanation; `false` gives `score=4/5` |
| `OPENAI_API_KEY` | none | The judge's API key, read by the `openai` SDK |

**Exempt services.** Some services legitimately handle sensitive data, such as a bank chatbot
that receives the customer's own CPF. List them in `LLM_EVAL_EXCEPTIONS`:

```sh
LLM_EVAL_EXCEPTIONS='{"bank-chatbot": ["pii_detection"], "devops-assistant": ["secret_detection"]}'
```

For those services the evaluator still runs, but the result is labeled `exempt` and has no
score. The explanation still says what was found (`exempt service; cpf=2 (input)`), so you can
see how much sensitive data an exempt service sends, without raising alerts. Service names must
match exactly, and a span without `service.name` is never exempt. A judge is the exception: it
is not called for an exempt service (`exempt service; not evaluated`).

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
- **Shutdown.** On SIGTERM the service stops accepting data, drains the queue and then the judge
  lane, both within 30 s, then flushes the SDK. Judge evaluations still pending are counted in
  `llm_eval.evaluations.dropped` with `llm_eval.drop.reason=shutdown`.
- **The judge lane doesn't push back.** `/readyz` and the 429s look only at the main queue. A
  backed-up judge drops evaluations (`lane_full`) instead of slowing the Collector down.
- **Scaling.** Scale with replicas: regex work is bound by the GIL, so more workers do not add
  throughput. Deduplication is per instance, so a Collector retry that lands on a different
  replica can be evaluated twice. That only happens on retries. The Collector's `loadbalancing`
  exporter (`routing_key: traceID`) would pin each trace to one replica, but as of Collector
  0.161.0 it only speaks OTLP/gRPC, which this version does not accept.
- **Logs.** The service logs to stderr, and the volume at `INFO` doesn't grow with traffic.
  Besides startup, configuration and shutdown, it logs one summary per
  `LLM_EVAL_LOG_SUMMARY_INTERVAL_S` (60 s by default; also once more at shutdown):

  ```
  last 60s: received=11400 queued=11380 skipped=duplicate:20 rejected=none | evaluations:
  pii_detection=11380 (fail:312,pass:11068) relevance=569 (error:3,fail:66,pass:500)
  | judge_tokens=812345 | queue=12/10000 llm_judge=3/1000
  ```

  The summary is a `WARNING` when the interval had rejected exports, evaluation errors or
  drops. State changes get one line each when they happen: the queue filling up (429s) and
  accepting again, an evaluator failing (3 errors in a row) and recovering, the judge lane
  starting and stopping to drop (`lane_full`, `budget`). `DEBUG` adds a line per export batch,
  rejected export (peer address and status), interaction and evaluation (label or error,
  score, duration, judge calls and tokens). Lines carry counts, TraceID, SpanID and
  `service.name`, never content or explanations. `DEBUG` costs a few lines per span, so keep
  it for troubleshooting.
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

class Conciseness:
    name = "conciseness"               # becomes gen_ai.evaluation.name
    kind = EvaluatorKind.LLM_JUDGE     # heuristics run in a thread; judges in the judge lane
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
            explanation="answer is to the point",  # never quote the content
            attributes={"llm_eval.conciseness.threshold": 0.6},  # llm_eval.* keys only
        )
```

Register it in your own package's `pyproject.toml` and enable it by name. You don't need to
change this repository:

```toml
[project.entry-points."llm_eval.evaluators"]
conciseness = "my_package.conciseness:Conciseness"
```

```sh
LLM_EVAL_EVALUATORS=pii_detection,secret_detection,conciseness
```

A judge can subclass `JudgeEvaluator` from `llm_eval_otel.judge.evaluator`, as `relevance`
does in [src/llm_eval_otel/evaluators/relevance.py](src/llm_eval_otel/evaluators/relevance.py).
It then gets the `LLM_EVAL_JUDGE_*` configuration, masking before sending, the
`<conversation>` envelope, the judge spans and metrics, and the `judge_*` errors; it supplies
the prompt, the schema, the content and the verdict.

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
| `error.type` | `timeout`, `judge_refusal`, `judge_truncated`, `judge_invalid_output` or an exception class name (only on failure) |
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
| `llm_eval.judge.model` | `gpt-5-mini-2025-08-07`: the model that answered (`relevance`) |
| `llm_eval.judge.raw_score` | `4`: the judge's rating, 1 to 5 (`relevance`) |

Each call to the judge is a `chat {model}` span (kind `CLIENT`) under `evaluate relevance`,
with the GenAI semconv attributes of a client call and no content: `gen_ai.operation.name`,
`gen_ai.provider.name` (`openai`, the API used), `gen_ai.request.model`,
`gen_ai.response.model`, `server.address` and `server.port` (which tell the OpenAI API from a
local server), `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`gen_ai.usage.cache_read.input_tokens`, `gen_ai.response.finish_reasons` and `error.type`.
`gen_ai.input.messages` and `gen_ai.output.messages` are never set, and no instrumentation
library wraps the SDK, because those can record content.

| Metric | Type | Attributes |
| --- | --- | --- |
| `llm_eval.evaluations` | Counter | evaluation name and type, label, `error.type`, source service, provider, model, association properties |
| `llm_eval.evaluation.score` | Histogram (0.1 … 1.0) | evaluation name and type, source service, provider, model, association properties; excludes exempt and errors |
| `llm_eval.evaluation.duration` | Histogram, `s` | evaluation name, `error.type` |
| `llm_eval.spans.received` | Counter | none |
| `llm_eval.spans.skipped` | Counter | `llm_eval.skip.reason`: `not_inference`, `no_content`, `duplicate`, `invalid_payload`, `self_telemetry` |
| `llm_eval.queue.size` | UpDownCounter | none |
| `llm_eval.sanitizer.redactions` | Counter | evaluation name |
| `llm_eval.evaluations.dropped` | Counter | evaluation name, `llm_eval.drop.reason`: `lane_full`, `budget`, `shutdown` |
| `llm_eval.lane.size` | UpDownCounter | `llm_eval.lane`: `llm_judge` |
| `gen_ai.client.token.usage` | Histogram, `{token}` | `gen_ai.token.type` (`input`, `output`), evaluation name, provider, request and response model, server address and port |
| `gen_ai.client.operation.duration` | Histogram, `s` | evaluation name, provider, request and response model, server address and port, `error.type` |

`self_telemetry` counts spans whose `service.name` is the service's own: the evaluator's output
routed back to it by mistake. They are never evaluated.

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

With `relevance` on, measured on 2026-10-01 with version 0.3.0 on the same machine, with
`tools/load_test.py --spans 3000 --text-kb 10 --judge slow|down` (3 runs each, `relevance` at
0.05, so about 140 judge evaluations per run):

| Heuristics' throughput | Spans/s | 429s |
| --- | --- | --- |
| No judge | 192, 195, 195 | 0 |
| Judge answering after 5 s per call | 189, 191, 189 | 0 |
| Judge unreachable (connection refused) | 176, 175, 179 | 0 |

The judge never blocks the heuristics: its calls wait in their own lane, and the main queue
never filled. What it costs is CPU on the same process, about 6.5 ms per evaluation of 10 KB
(3.2 ms of it masking) when the judge answers, and about 18 ms when it is unreachable,
because the SDK tries twice and builds the error each time. With a slow judge most of that work
happens after the run's last span, so it barely shows; with the judge down it all lands within
the run, about 9% of the heuristics' throughput at a 5% sample of 10 KB spans. In the slow runs,
8 concurrent calls of 5 s could not keep up with 140 evaluations: the shutdown's 30 s drain
finished 72 and counted 67 as `shutdown` drops, as designed.

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
  cli.py               # llm-eval-judge: the judge from the command line
  config.py            # LLM_EVAL_* settings
  semconv.py           # every attribute, event and metric name
  ingest/http.py       # POST /v1/traces, /healthz, /readyz
  extract/genai.py     # span -> GenAIInteraction (two formats)
  engine/queue.py      # bounded queue, dedup, workers
  engine/runner.py     # sampling, timeouts, exemptions, truncation
  engine/lanes.py      # the judge lane and the token budget
  engine/service.py    # wires the pieces together
  evaluators/          # base (the contract), pii, secrets, refusal, prompt_leak,
                       # output_format, relevance, registry (entry points)
  judge/               # client (contract, errors, call records), openai_adapter,
                       # evaluator (shared judge code), redact, schema
  version.py           # the version, also service.version
  emit/                # sdk (providers), emitter (event, span, metrics), sanitize
tools/span_generator.py    # synthetic spans for the demo
tools/fake_judge_server.py # deterministic OpenAI-compatible judge for tests and the demo
tools/benchmark.py         # an evaluator against a labeled set: agreement, tokens, cost
tools/load_test.py         # latency and throughput measurement
deploy/                    # Collector config and docker compose
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
- `relevance` needs a user message in the turn, so it does not judge the final answer of a tool
  loop, where the turn's new input is the tool result.
- `relevance` is not calibrated yet; see [Calibrating](#calibrating). `faithfulness` (answers
  checked against retrieved documents) is planned but not in this version.

## License

Apache-2.0. See [LICENSE](LICENSE).
