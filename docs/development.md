# Development

```sh
uv sync                       # Python 3.12
uv run ruff check . && uv run ruff format --check .
uv run mypy                   # strict
uv run pytest                 # unit and API tests, in-memory exporters
uv run pytest tests/e2e -m e2e  # docker compose + the Collector's file exporter
```

## Layout

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

## Versioning

The version lives only in `src/llm_eval_otel/version.py`; `pyproject.toml` reads it from there.
It is the `service.version` on all the service's telemetry, so it tells which detection rules
produced a result. Bump the minor version when what gets detected changes (a new type,
evaluator or threshold) and the patch version for fixes that don't change detection. Add the
change to [CHANGELOG.md](../CHANGELOG.md).

To release, edit `version.py`, merge, and push a matching tag (`git tag v0.2.0 && git push
origin v0.2.0`). The release workflow fails if the tag and `version.py` differ.
