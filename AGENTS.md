# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) and other coding agents when working with code in this repository.

## What this is

`llm-eval-otel` is an out-of-band evaluator for GenAI spans. The OTel Collector fans GenAI inference spans out to it over OTLP/HTTP; the service scans prompts and responses (PII, leaked credentials, refusals, prompt leaks, JSON validity, relevance via an LLM judge, and relevance/refusal/toxicity/prompt injection via TypeSafe's Jev) and exports the results back to the Collector as OTel telemetry linked to the original trace. It only flags: it never blocks or changes the application's traffic.

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
- Latency/throughput benchmark: `uv run python tools/load_test.py --spans 3000 --text-kb 10` (the Performance table in `docs/operations.md` was measured with this). `--judge slow|down` adds `relevance` against the fake judge (5 s per call) or a closed port; `--jev fake|slow|down` adds the four `jev_*` checks the same way.
- Judge from the command line: `uv run llm-eval-judge -i "question" -o "answer"` (or `--jsonl`, `--dry-run` to see the masked content without calling; `-e jev_refusal` for a Jev-as-a-Judge check). `src/llm_eval_otel/cli.py`; reads `./.env` without overriding the environment (`.env.example` lists the variables; the service itself never reads `.env`).
- Judge calibration: `uv run python tools/benchmark.py <labeled.jsonl>` with `LLM_EVAL_LLM_JUDGE_*` and `OPENAI_API_KEY` set (or `--evaluator jev_...` with `LLM_EVAL_JEV_JUDGE_MODEL` and `TYPESAFE_API_KEY`; `--against <evaluator>` compares two on the same items). Labeled sets are in `tools/data/`. `tools/fake_judge_server.py` is a deterministic OpenAI-compatible judge and System One API (stdlib only) for tests, the demo and the load test.
- `scripts/push-nexus.sh` pushes a dev image to a local Nexus; real releases are `v*` tags, built by `release.yml` to GHCR.
- The version's single source is `src/llm_eval_otel/version.py` (`pyproject.toml` has `dynamic = ["version"]`, read by hatchling). Edit it by hand: `uv version --bump` doesn't work with a dynamic version. It becomes `service.version`, which identifies the detection rules, so any change to what gets detected bumps the minor version and gets a `CHANGELOG.md` entry. `release.yml` fails when the `v*` tag doesn't match it.

## Architecture

Request flow, one module per stage under `src/llm_eval_otel/`:

1. **`ingest/http.py`**: `POST /v1/traces`, protobuf only (no JSON, no gRPC), optional gzip with a decompression ceiling. Status codes are chosen for the Collector exporter's retry logic: 200 once queued, 429 + `Retry-After` on a full queue (retried), 400 on bad payloads (not retried), 503 during shutdown. `/readyz` fails when the queue is >90% full.
2. **`extract/genai.py`**: OTLP spans → `GenAIInteraction`. Reads the current semconv (`gen_ai.input.messages` etc., structured or JSON string) first, then falls back to OpenLLMetry indexed attributes. Only *new* input messages (after the last `assistant` message) are kept so a chat history isn't re-flagged every turn. Non-inference or content-less spans are counted as skipped with a reason.
   `context_messages` keeps up to 4 earlier user/assistant text messages (1,000 chars each) for judges. Spans whose resource `service.name` is the service's own are skipped as `self_telemetry`.
3. **`engine/queue.py`**: bounded `asyncio.Queue`, per-instance `(trace_id, span_id)` dedup LRU (Collector retries resend whole batches), N async workers.
4. **`engine/runner.py`**: per evaluator: `applies_to`, TraceID-based sampling (same rule as OTel ProbabilitySampler, so it's consistent across replicas/retries), `max_chars` truncation (output, then input, then system), timeout, exception → `error_type` result, per-service exemptions (label `exempt`, score dropped). `HEURISTIC` evaluators run via `asyncio.to_thread` to keep the event loop answering. Evaluators are offered without waiting to the lane named by their optional `lane` attribute (default `str(kind)`; no such lane → inline); an exempt service never reaches a judge (`exempt service; not evaluated`). After sampling and exemption, `BatchEvaluator`s with the same `(batch_key, max_chars)` become one `Job` (`Job.evaluators`) run by the first one's `evaluate_batch`, with the largest timeout; `Runner.execute` returns one record per evaluator, collects the `JudgeCall` records via a contextvar (`judge.client.recording`) and puts them on the first record only, so a batch's call is emitted once.
5. **`engine/lanes.py`**: the lanes: each its own bounded queue, workers and optional `TokenBudget`, settled once per job. `llm_judge` (`LLM_EVAL_LLM_JUDGE_*`) always; `jev_judge` (`LLM_EVAL_JEV_JUDGE_*`) only when an enabled evaluator asks for it, drained after `llm_judge`. Full lane, no budget and shutdown drop the job and count each evaluator in it in `llm_eval.evaluations.dropped`; they never raise `QueueFull` or affect `/readyz`.
6. **`emit/emitter.py`**: per record, a `gen_ai.evaluation.result` log event and an optional `evaluate {name}` child span, both parented on a `NonRecordingSpan` built from the original trace/span IDs, plus metrics. Judge calls become `{operation_name} {model}` CLIENT spans (`chat` or `system_one`) under the `evaluate` span, plus `gen_ai.client.*` metrics. Every attribute passes through **`emit/sanitize.py`** first, which re-runs the PII/secret detectors and replaces any matching string with `[REDACTED]`.
7. **`engine/activity.py`**: counts for the periodic `INFO` summary (`LLM_EVAL_LOG_SUMMARY_INTERVAL_S`) and state-change lines (an evaluator failing/recovering). Per-span and per-evaluation lines are `DEBUG`, so `INFO` volume doesn't grow with traffic; keep new logs to that rule. `main.py` holds the `openai`/`httpx`/`httpx2`/`typesafe_sdk` loggers at `WARNING` because they log every judge call; `typesafe_sdk` also logs request bodies at `DEBUG`, so `TypeSafeJudge` sets it to `WARNING` again after creating its client (the SDK applies `TYPESAFE_LOG_LEVEL` on import, which happens when the evaluators load).
8. **`engine/service.py`** wires the pieces; **`main.py`** builds the SDK providers, loads evaluators and runs uvicorn with a lifespan that drains the queue, then the lane, on shutdown.

### Evaluators

- The contract is in `evaluators/base.py`: a `Protocol` (`name`, `kind`, `timeout_s`, `sample_rate`, `max_chars`, `applies_to`, async `evaluate`; optional `lane`). `BatchEvaluator` adds `batch_key` and the classmethod `evaluate_batch`. Evaluators are OTel-agnostic: they take a `GenAIInteraction` and return an `EvaluationResult`.
- Evaluators are discovered **only via the `llm_eval.evaluators` entry point group** (`evaluators/registry.py`) and enabled by name in `LLM_EVAL_EVALUATORS`. A new built-in evaluator must be added to `[project.entry-points."llm_eval.evaluators"]` in `pyproject.toml` (then `uv sync` to re-register), and its `name` must equal the entry point name.
- `pii.py` exposes `find_pii` and `secrets.py` exposes `find_secrets`; the sanitizer depends on both, so changing detection rules also changes what gets redacted from all output. `LLM_EVAL_PII_TYPES` filters only what `pii_detection` reports, never the sanitizer.
- `Message.text` joins all parts; evaluators that must ignore some parts (reasoning, tool calls) use `Message.text_of(*types)`. An empty `parts` means the whole text is one `text` part. What judges read of a turn (`user_texts`, `response_texts`, `calls_tools`, the `context`/`request`/`response` dict) is in `evaluators/conversation.py`, shared by `relevance` and the `jev_*` checks.
- Rubrics, questions and thresholds of the `jev_*` checks (`evaluators/jev_checks.py`) are detection rules: changing one bumps the minor version.
- The PII regexes run on every span at 100%: check new ones against the 10 KB p99 budget with `tools/load_test.py`. A leading lookahead on the first character (`(?=\d)`) before the lookbehind makes them several times cheaper.

### LLM-as-a-Judge and Jev-as-a-Judge (`judge/`)

- `judge/client.py`: the `JudgeClient` protocol, `JudgeResponse`, the `SystemOneClient` protocol with the service's own question/answer types (`NoulQuestion`, `ScoreQuestion`, `NoulAnswer`, `ScoreAnswer`, `SystemOneResponse`), the `judge_*` errors (no message, since it could quote the judge) and `JudgeCall`, the content-free record of each call (`operation_name` is `chat` or `system_one`).
- `judge/openai_adapter.py`: the only adapter, the `openai` SDK (pinned) over Chat Completions; `LLM_EVAL_LLM_JUDGE_BASE_URL` points it at any compatible server. Strict JSON schema by default, `json_object`/`none` fallbacks; the output is always validated (`judge/schema.py`). `temperature` and `reasoning_effort` are sent only when set.
- `judge/evaluator.py`: `JudgeEvaluator`, the base for judge evaluators: masking (`judge/redact.py`, every PII type and secrets, before sending), the escaped `<conversation>` JSON envelope, `JudgeError` → `error_type`, and the explanation (reason cut to 300 chars, or a template with `LLM_EVAL_LLM_JUDGE_EXPLANATION=false`). `evaluators/relevance.py` is the only judge so far; `faithfulness` from the v0.3 plan is deferred.
- `judge/typesafe_adapter.py`: `TypeSafeJudge`, the second adapter, `typesafe-sdk` (pinned) over TypeSafe's System One API (`LLM_EVAL_JEV_JUDGE_BASE_URL` without `/v1`; the SDK adds it). SDK types never leave it. Every question id must come back with the asked type and values in range, else `JudgeInvalidOutput`; other errors are recorded by class name (a 422 body can echo the request).
- `judge/jev.py`: `JevEvaluator` (`kind`, lane and `batch_key` `jev_judge`, 5 s, 0.1, 16,000 chars), `ScoreCheck` and `NoulCheck`: the state (`conversation()`, masked), questions keyed by the evaluator's name, the batch call (one request for every check; a `JudgeError` is every check's error) and template explanations (`score=3.6/5 confidence=0.82`, `p=0.93`). The plan is `docs/plans/eval-jev-judge-plan.md`.

## Invariants to preserve

- **No evaluated content ever leaves the service.** Explanations are built only from type/location counts (`summarize_findings`, e.g. `cpf=1 (input)`). Logs and `error.type` use the exception's class name, never its message, because messages may quote content. Judges are the documented exception: content goes to the judge provider (OpenAI-compatible or TypeSafe) only after `judge.redact.mask`, and the explanation is the judge's reason, cut to 300 chars and sanitized (Jev gives none: template only). Judge spans never carry `gen_ai.input.messages`/`gen_ai.output.messages`, no instrumentation library may wrap either SDK, and the `typesafe_sdk` logger stays at `WARNING`.
- **All attribute, event and metric names live in `semconv.py`.** GenAI semconv is still in Development (pinned to semantic-conventions-genai commit `e57c543`). Names the semconv doesn't define use the `llm_eval.*` prefix, never `gen_ai.*`; the emitter drops any evaluator attribute not prefixed `llm_eval.`.
- **Metrics stay low-cardinality:** no trace/span IDs, response IDs or free text in metric attributes.
- SDK providers are built in `emit/sdk.py` and injected, never set globally, so tests can swap in in-memory exporters. The tracer uses `ALWAYS_ON` on purpose: incoming spans often have trace flags 0, and a parent-based sampler would drop every child span.
- OTel packages are pinned exactly; the Logs API is still the underscore module `opentelemetry._logs`.
- Configuration: `LLM_EVAL_*` env vars via pydantic-settings in `config.py`; `OTEL_*` are read by the SDK, with service defaults applied in `apply_otel_defaults()`.

## Tests

- `tests/unit/conftest.py` provides `OtelMemory` (in-memory span/log/metric exporters wrapped in a `Telemetry`) and service fixtures; `tests/unit/otlp.py` builds OTLP protobuf payloads. Use these instead of hand-rolling protobufs or exporters. pytest-asyncio runs in `auto` mode.
- Judge tests: `tests/unit/judge_fakes.py` has a scripted `FakeJudgeClient` and `FakeSystemOneClient` (plus `jev_checks()`); adapter tests (`test_judge_adapter.py`, `test_typesafe_adapter.py`) run the real SDK against `tools/fake_judge_server.py` in a thread (pytest has `pythonpath = ["tools"]`; `server.base_url` for openai, `server.root_url` for TypeSafe). The fake server's markers (`FAKE_JUDGE:refuse`, `:content_filter`, `:length`, `:invalid`, `:no_usage`; for System One `FAKE_JUDGE:invalid`, `:no_usage`, `FAKE_JEV:overloaded`, `:rate_limited`, `:422`) force the edge cases. To test `TYPESAFE_LOG_LEVEL=debug`, call `typesafe_sdk._core.logging.setup_logging()` with the variable set: the SDK reads it only on import. No test needs an API key.
- `tests/e2e/test_compose.py` brings up `deploy/docker-compose.yaml`, waits for one round from `tools/span_generator.py`, and reads the Collector's `file/e2e` exporter output. It asserts the expected number of inference spans and that none of the known sensitive values appear in the output. Adding a case to the span generator means updating `INFERENCE_SPANS` and the per-service label lists there. The compose demo runs `relevance` and the four `jev_*` checks at 1.0 against the `fake-judge` container (`bank-chatbot` is exempt from `relevance` and `jev_relevance`), which records every request of both APIs to `judge-requests.jsonl` so the test can check nothing detectable reached either judge. `deploy/docker-compose.jev.yaml` swaps in the real Jev from `./.env`. `LLM_EVAL_IMAGE` tests a prebuilt image.
- In `deploy/otel-collector-config.yaml`, evaluator output comes in on a separate receiver (`otlp/eval`, :4319) whose pipelines export only to the backend, so evaluation telemetry can never loop back into the evaluator.

## Docs

`docs/spec.md` (the design spec) and `docs/plans/eval-heuristics-plan.md` (more heuristics), `docs/plans/eval-llm-judge-plan.md` (LLM-as-a-Judge via the OpenAI SDK and compatible servers), `docs/plans/eval-local-classifiers-plan.md` (local classifiers) and `docs/plans/eval-jev-judge-plan.md` (Jev-as-a-Judge) are written in Portuguese. The README is the short user-facing entry point; the reference for evaluators, LLM-as-a-Judge, Jev-as-a-Judge, configuration, operations and telemetry is in English in `docs/{evaluators,llm-as-a-judge,jev-as-a-judge,configuration,operations,telemetry,development}.md`.
