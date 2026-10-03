# Changelog

The version is `service.version` on everything the service emits, so it identifies the
detection rules behind each result. The minor version goes up whenever what gets detected
changes (a new type, evaluator or threshold); the patch version for fixes that don't change
detection. See [Versioning](README.md#versioning).

## [Unreleased]

## [0.3.2]

### Added

- Processing logs. At `INFO`, one summary line per `LLM_EVAL_LOG_SUMMARY_INTERVAL_S` (60 s):
  spans received, queued and skipped, rejected exports, results per evaluator and label,
  errors, drops, judge tokens and queue sizes; a `WARNING` when something went wrong. State
  changes are logged once: queue full and accepting again, an evaluator failing (3 errors in a
  row) and recovering, the judge lane dropping and no longer dropping. `LLM_EVAL_LOG_LEVEL=DEBUG`
  adds a line per batch, rejected export, interaction and evaluation. Lines carry counts, IDs,
  labels, scores and timings, never content or explanations.
- `llm_eval.evaluation.type` (`heuristic`, `model`, `llm_judge`) on the `llm_eval.evaluations`
  counter and the `llm_eval.evaluation.score` histogram, so a dashboard can select the judges
  without listing them. It follows from the evaluation name, so it adds no series.

### Changed

- `relevance` skips responses with a tool call. An agent step (ReAct and similar) narrates
  before calling a tool ("let me look that up"), and the newest user message is often the
  orchestrator's ("step 2 of 6"), so the judge failed every step of a working agent: 3 of 4
  results on a real ReAct service, while the answer the user got passed. Only the response
  that ends the turn is judged now. When an agent returns its answer as a tool call
  (`final_answer`), judge the call that writes the user-facing answer instead.
- `scripts/push-nexus.sh` reads `scripts/.env` instead of the `.env` at the repository root,
  which is now `llm-eval-judge`'s.

### Fixed

- The compose demo on a fresh checkout: Docker created `deploy/e2e-output` as root, the
  Collector (uid 10001) couldn't open its file exporter and exited, so nothing reached the
  service or Grafana. A one-shot `output-dir` service now opens up the directory first. The
  span generator no longer inherits the service's healthcheck (it showed as unhealthy).

## [0.3.0]

### Added

- `relevance`, the first LLM-as-a-Judge evaluator (opt-in). A judge rates from 1 to 5 whether
  the response addresses what the user asked; the score is `(rating - 1) / 4` and `pass` is a
  rating of 3 or more. It samples 5% of traces by default and needs `LLM_EVAL_JUDGE_MODEL`.
  The threshold is an initial value: the calibration against human labels is still to be done.
- The judge talks to the OpenAI API or to any OpenAI-compatible server (vLLM, Ollama, LiteLLM)
  through the official `openai` SDK, configured by the new `LLM_EVAL_JUDGE_*` settings. The
  key is the SDK's own `OPENAI_API_KEY`.
- PII and credentials are masked by type (`[CPF]`, `[EMAIL]`, `[SECRET]`) before anything goes
  to the judge (`LLM_EVAL_JUDGE_REDACT`, on by default).
- Judge evaluators run in their own lane, with a bounded queue and `LLM_EVAL_JUDGE_MAX_CONCURRENCY`
  concurrent calls, so a slow or unreachable judge never holds the heuristics or causes a 429.
  What the lane can't run is dropped and counted in `llm_eval.evaluations.dropped`
  (`lane_full`, `budget`, `shutdown`). `LLM_EVAL_JUDGE_TOKENS_PER_MINUTE` caps token spend.
- Each judge call is a `chat {model}` span under `evaluate relevance`, with model, endpoint,
  tokens and finish reason and no content, plus the GenAI client metrics
  `gen_ai.client.token.usage` and `gen_ai.client.operation.duration`.
- `GenAIInteraction.context_messages`: up to four earlier user/assistant text messages, so a
  judge can follow a follow-up question. Heuristics ignore it.
- Spans whose `service.name` is the service's own are skipped as `self_telemetry`, in case the
  Collector routes the evaluator's output back to it.
- `llm-eval-judge`, a command that runs the judge (or any evaluator) on interactions given on
  the command line or as JSONL, with the service's configuration and code path. It reads
  `./.env`; `.env.example` lists the variables. `--dry-run` prints the masked content that
  would be sent, without calling the judge.
- `tools/fake_judge_server.py` (a deterministic OpenAI-compatible judge for tests and the demo)
  and `tools/benchmark.py` (agreement, rating variation, latency, tokens and cost of an
  evaluator over a labeled JSONL set). `tools/load_test.py --judge slow|down`.

### Changed

- A judge for an exempt service is not called: the event is `exempt` with the explanation
  `exempt service; not evaluated`. Heuristics still run for exempt services, as before.
- On shutdown the service drains the main queue, then the judge lane, within the same 30 s.
- `max_chars` truncation now keeps the output first, then the input, then the system
  instructions, and keeps message part offsets. It only affects evaluators that set
  `max_chars`; none did before `relevance`.

## [0.2.0]

### Behavior change

- `pii_detection` now detects Brazilian phone numbers, so services that handle them (support
  chatbots) will see more `fail` results. Set `LLM_EVAL_PII_TYPES` to turn off one type
  without exempting the whole evaluator, e.g. `LLM_EVAL_PII_TYPES=cpf,cnpj,email,credit_card,pix_key`.
  The sanitizer keeps redacting every type.

### Added

- `pii_detection`: CNPJ, numeric and alphanumeric (check digits validated), phone numbers
  (valid DDD) and PIX random keys (a UUID v4 with "pix" nearby). When two types match the same
  stretch, it counts once: CPF, CNPJ, card, phone.
- `LLM_EVAL_PII_TYPES`, the types `pii_detection` reports.
- Three opt-in evaluators, enabled by name in `LLM_EVAL_EVALUATORS`:
  - `refusal`: refusal phrases in Portuguese, English and Spanish, and
    `finish_reason=content_filter`.
  - `system_prompt_leak`: output that copies stretches of the system instructions.
  - `output_format`: invalid JSON when `gen_ai.output.type` is `json`.
- The extractor reads `gen_ai.output.type` and `gen_ai.response.finish_reasons`, with
  fallbacks for OpenLLMetry, and keeps where each message part sits (`Message.parts`,
  `Message.text_of()`). The new `GenAIInteraction` and `Message` fields have defaults, so
  existing evaluators keep working.
- The startup log shows the version, the available and enabled evaluators, every `LLM_EVAL_*`
  setting (secrets only as `set`/`unset`) and the `OTEL_*` defaults the service applies.

### Changed

- The version lives only in `src/llm_eval_otel/version.py`, and the release workflow checks
  that the tag matches it.

## [0.1.0]

First release: `pii_detection` (CPF, email, credit card) and `secret_detection`, OTLP/HTTP
ingest from the Collector, and results exported as `gen_ai.evaluation.result` events, child
spans and metrics.
