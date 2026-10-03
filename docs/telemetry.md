# Telemetry reference

For every evaluated span, and for each evaluator, the service emits:

- A `gen_ai.evaluation.result` event (a log record) with the span's TraceID and SpanID.
  Severity is `INFO` for `pass`/`exempt`, `WARN` for `fail` and `ERROR` when the evaluator failed.
- An `evaluate {name}` child span of the evaluated span, so trace UIs show the result inline.
  You can turn it off with `LLM_EVAL_EMIT_SPANS=false`.
- The metrics below, including `llm_eval.evaluations` (counter) and `llm_eval.evaluation.score`
  (histogram).

GenAI semantic conventions are still in Development status. The service follows
[semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) at
commit `e57c543`, and every name lives in
[src/llm_eval_otel/semconv.py](../src/llm_eval_otel/semconv.py). Names that the semconv does not
define use the `llm_eval.*` prefix, so they cannot collide with future `gen_ai.*` definitions.

## Event and span attributes

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

## Judge spans

Each call to the judge is a `chat {model}` span (kind `CLIENT`) under `evaluate relevance`,
with the GenAI semconv attributes of a client call and no content: `gen_ai.operation.name`,
`gen_ai.provider.name` (`openai`, the API used), `gen_ai.request.model`,
`gen_ai.response.model`, `server.address` and `server.port` (which tell the OpenAI API from a
local server), `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`gen_ai.usage.cache_read.input_tokens`, `gen_ai.response.finish_reasons` and `error.type`.
`gen_ai.input.messages` and `gen_ai.output.messages` are never set, and no instrumentation
library wraps the SDK, because those can record content.

## Metrics

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
The sanitizer also covers the properties, so a value that is PII shows up as `[REDACTED]`.
`service.version` on the resource identifies the version of the detection rules.
