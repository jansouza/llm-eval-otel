# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) and other coding agents when working with code in this repository.

## What this is

`llm-eval-otel` is an out-of-band evaluator for GenAI spans. The OTel Collector fans GenAI inference spans out to it over OTLP/HTTP; the service scans prompts and responses (PII, leaked credentials, refusals, prompt leaks, JSON validity, and relevance via an LLM judge) and exports the results back to the Collector as OTel telemetry linked to the original trace. It only flags: it never blocks or changes the application's traffic.

## Commands

Python 3.12, managed with `uv`. CI (`.github/workflows/ci.yml`) runs exactly these:

```sh
uv sync                                   # CI uses --frozen
uv run ruff check . && uv run ruff format --check .
uv run mypy                               # strict, covers src/llm_eval_otel only
uv run pytest                             # unit + API tests (testpaths = tests/unit)
uv run pytest tests/unit/test_runner.py::test_name   # a single test
uv run pytest tests/e2e -m e2e            # needs Docker compose; builds the image
```

- Demo stack (synthetic spans → Collector → service → Grafana LGTM on :3000): `docker compose -f deploy/docker-compose.yaml up --build`
- Latency/throughput benchmark: `uv run python tools/load_test.py --spans 3000 --text-kb 10` (the Performance table in `docs/operations.md` was measured with this). `--judge slow|down` adds `relevance` against the fake judge (5 s per call) or a closed port.
- Judge from the command line: `uv run llm-eval-judge -i "question" -o "answer"` (or `--jsonl`, `--dry-run` to see the masked content without calling). `src/llm_eval_otel/cli.py`; reads `./.env` without overriding the environment (`.env.example` lists the variables; the service itself never reads `.env`).
- Judge calibration: `uv run python tools/benchmark.py <labeled.jsonl>` with `LLM_EVAL_JUDGE_*` and `OPENAI_API_KEY` set. `tools/fake_judge_server.py` is a deterministic OpenAI-compatible judge (stdlib only) for tests, the demo and the load test.
- `scripts/push-nexus.sh` pushes a dev image to a local Nexus; real releases are `v*` tags, built by `release.yml` to GHCR.
- The version's single source is `src/llm_eval_otel/version.py` (`pyproject.toml` has `dynamic = ["version"]`, read by hatchling). Edit it by hand: `uv version --bump` doesn't work with a dynamic version. It becomes `service.version`, which identifies the detection rules, so any change to what gets detected bumps the minor version and gets a `CHANGELOG.md` entry. `release.yml` fails when the `v*` tag doesn't match it.

## Architecture

Request flow, one module per stage under `src/llm_eval_otel/`:

1. **`ingest/http.py`**: `POST /v1/traces`, protobuf only (no JSON, no gRPC), optional gzip with a decompression ceiling. Status codes are chosen for the Collector exporter's retry logic: 200 once queued, 429 + `Retry-After` on a full queue (retried), 400 on bad payloads (not retried), 503 during shutdown. `/readyz` fails when the queue is >90% full.
2. **`extract/genai.py`**: OTLP spans → `GenAIInteraction`. Reads the current semconv (`gen_ai.input.messages` etc., structured or JSON string) first, then falls back to OpenLLMetry indexed attributes. Only *new* input messages (after the last `assistant` message) are kept so a chat history isn't re-flagged every turn. Non-inference or content-less spans are counted as skipped with a reason.
   `context_messages` keeps up to 4 earlier user/assistant text messages (1,000 chars each) for judges. Spans whose resource `service.name` is the service's own are skipped as `self_telemetry`.
3. **`engine/queue.py`**: bounded `asyncio.Queue`, per-instance `(trace_id, span_id)` dedup LRU (Collector retries resend whole batches), N async workers.
4. **`engine/runner.py`**: per evaluator: `applies_to`, TraceID-based sampling (same rule as OTel ProbabilitySampler, so it's consistent across replicas/retries), `max_chars` truncation (output, then input, then system), timeout, exception → `error_type` result, per-service exemptions (label `exempt`, score dropped). `HEURISTIC` evaluators run via `asyncio.to_thread` to keep the event loop answering. `LLM_JUDGE` evaluators are offered to the judge lane without waiting; an exempt service never reaches the judge (`exempt service; not evaluated`). `Runner.execute` collects the judge's `JudgeCall` records via a contextvar (`judge.client.recording`).
5. **`engine/lanes.py`**: the judge lane: its own bounded queue, `LLM_EVAL_JUDGE_MAX_CONCURRENCY` workers, optional `TokenBudget`. Full lane, no budget and shutdown drop the evaluation and count it in `llm_eval.evaluations.dropped`; it never raises `QueueFull` or affects `/readyz`.
6. **`emit/emitter.py`**: per record, a `gen_ai.evaluation.result` log event and an optional `evaluate {name}` child span, both parented on a `NonRecordingSpan` built from the original trace/span IDs, plus metrics. Judge calls become `chat {model}` CLIENT spans under the `evaluate` span, plus `gen_ai.client.*` metrics. Every attribute passes through **`emit/sanitize.py`** first, which re-runs the PII/secret detectors and replaces any matching string with `[REDACTED]`.
7. **`engine/activity.py`**: counts for the periodic `INFO` summary (`LLM_EVAL_LOG_SUMMARY_INTERVAL_S`) and state-change lines (an evaluator failing/recovering). Per-span and per-evaluation lines are `DEBUG`, so `INFO` volume doesn't grow with traffic; keep new logs to that rule. `main.py` holds the `openai`/`httpx`/`httpx2` loggers at `WARNING` because they log every judge call.
8. **`engine/service.py`** wires the pieces; **`main.py`** builds the SDK providers, loads evaluators and runs uvicorn with a lifespan that drains the queue, then the lane, on shutdown.

### Evaluators

- The contract is in `evaluators/base.py`: a `Protocol` (`name`, `kind`, `timeout_s`, `sample_rate`, `max_chars`, `applies_to`, async `evaluate`). Evaluators are OTel-agnostic: they take a `GenAIInteraction` and return an `EvaluationResult`.
- Evaluators are discovered **only via the `llm_eval.evaluators` entry point group** (`evaluators/registry.py`) and enabled by name in `LLM_EVAL_EVALUATORS`. A new built-in evaluator must be added to `[project.entry-points."llm_eval.evaluators"]` in `pyproject.toml` (then `uv sync` to re-register), and its `name` must equal the entry point name.
- `pii.py` exposes `find_pii` and `secrets.py` exposes `find_secrets`; the sanitizer depends on both, so changing detection rules also changes what gets redacted from all output. `LLM_EVAL_PII_TYPES` filters only what `pii_detection` reports, never the sanitizer.
- `Message.text` joins all parts; evaluators that must ignore some parts (reasoning, tool calls) use `Message.text_of(*types)`. An empty `parts` means the whole text is one `text` part.
- The PII regexes run on every span at 100%: check new ones against the 10 KB p99 budget with `tools/load_test.py`. A leading lookahead on the first character (`(?=\d)`) before the lookbehind makes them several times cheaper.

### LLM-as-a-Judge (`judge/`)

- `judge/client.py`: the `JudgeClient` protocol, `JudgeResponse`, the `judge_*` errors (no message, since it could quote the judge) and `JudgeCall`, the content-free record of each call.
- `judge/openai_adapter.py`: the only adapter, the `openai` SDK (pinned) over Chat Completions; `LLM_EVAL_JUDGE_BASE_URL` points it at any compatible server. Strict JSON schema by default, `json_object`/`none` fallbacks; the output is always validated (`judge/schema.py`). `temperature` and `reasoning_effort` are sent only when set.
- `judge/evaluator.py`: `JudgeEvaluator`, the base for judge evaluators: masking (`judge/redact.py`, every PII type and secrets, before sending), the escaped `<conversation>` JSON envelope, `JudgeError` → `error_type`, and the explanation (reason cut to 300 chars, or a template with `LLM_EVAL_JUDGE_EXPLANATION=false`). `evaluators/relevance.py` is the only judge so far; `faithfulness` from the v0.3 plan is deferred.

## Invariants to preserve

- **No evaluated content ever leaves the service.** Explanations are built only from type/location counts (`summarize_findings`, e.g. `cpf=1 (input)`). Logs and `error.type` use the exception's class name, never its message, because messages may quote content. Judges are the documented exception: content goes to the judge provider only after `judge.redact.mask`, and the explanation is the judge's reason, cut to 300 chars and sanitized. Judge spans never carry `gen_ai.input.messages`/`gen_ai.output.messages`, and no instrumentation library may wrap the SDK.
- **All attribute, event and metric names live in `semconv.py`.** GenAI semconv is still in Development (pinned to semantic-conventions-genai commit `e57c543`). Names the semconv doesn't define use the `llm_eval.*` prefix, never `gen_ai.*`; the emitter drops any evaluator attribute not prefixed `llm_eval.`.
- **Metrics stay low-cardinality:** no trace/span IDs, response IDs or free text in metric attributes.
- SDK providers are built in `emit/sdk.py` and injected, never set globally, so tests can swap in in-memory exporters. The tracer uses `ALWAYS_ON` on purpose: incoming spans often have trace flags 0, and a parent-based sampler would drop every child span.
- OTel packages are pinned exactly; the Logs API is still the underscore module `opentelemetry._logs`.
- Configuration: `LLM_EVAL_*` env vars via pydantic-settings in `config.py`; `OTEL_*` are read by the SDK, with service defaults applied in `apply_otel_defaults()`.

## Tests

- `tests/unit/conftest.py` provides `OtelMemory` (in-memory span/log/metric exporters wrapped in a `Telemetry`) and service fixtures; `tests/unit/otlp.py` builds OTLP protobuf payloads. Use these instead of hand-rolling protobufs or exporters. pytest-asyncio runs in `auto` mode.
- Judge tests: `tests/unit/judge_fakes.py` has a scripted `FakeJudgeClient`; adapter tests run the real SDK against `tools/fake_judge_server.py` in a thread (pytest has `pythonpath = ["tools"]`). The fake server's markers (`FAKE_JUDGE:refuse`, `:content_filter`, `:length`, `:invalid`, `:no_usage`) force the edge cases. No test needs an API key.
- `tests/e2e/test_compose.py` brings up `deploy/docker-compose.yaml`, waits for one round from `tools/span_generator.py`, and reads the Collector's `file/e2e` exporter output. It asserts the expected number of inference spans and that none of the known sensitive values appear in the output. Adding a case to the span generator means updating `INFERENCE_SPANS` and the per-service label lists there. The compose demo runs `relevance` at 1.0 against the `fake-judge` container, which records every request to `judge-requests.jsonl` so the test can check nothing detectable reached the judge. `LLM_EVAL_IMAGE` tests a prebuilt image.
- In `deploy/otel-collector-config.yaml`, evaluator output comes in on a separate receiver (`otlp/eval`, :4319) whose pipelines export only to the backend, so evaluation telemetry can never loop back into the evaluator.

## Docs

`docs/spec.md` (the design spec) and `docs/plans/eval-v0-{2,3,4}-plan.md` (roadmap: more heuristics, then LLM-as-a-Judge via the OpenAI SDK and compatible servers, then local classifiers) are written in Portuguese. The README is the short user-facing entry point; the reference for evaluators, the judge, configuration, operations and telemetry is in English in `docs/{evaluators,judge,configuration,operations,telemetry,development}.md`.
