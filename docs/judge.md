# LLM-as-a-Judge (`relevance`)

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
both the cost and the quality, so there is no default. All `LLM_EVAL_JUDGE_*` variables are in
[Configuration](configuration.md).

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

## What goes to the judge provider

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

## Judging from the command line

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

## Judging synthetic spans in the demo stack

To try the judge through the whole path (span → Collector → service → judge → telemetry back
through the Collector), the compose override swaps the fake judge for the one in `./.env` and
the fixed cases for [tools/data/relevance-synthetic.jsonl](../tools/data/relevance-synthetic.jsonl),
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

## Calibrating

[tools/benchmark.py](../tools/benchmark.py) runs an evaluator over a labeled JSONL set and reports
agreement with the human pass/fail label, how much the rating varies on repeated runs, latency,
tokens and the cost per thousand evaluations:

```sh
LLM_EVAL_JUDGE_MODEL=gpt-5-mini-2025-08-07 OPENAI_API_KEY=... \
uv run python tools/benchmark.py my-labeled-set.jsonl --repeat 3 \
    --price-input 0.25 --price-cached 0.025 --price-output 2.00
```

[tools/data/relevance-smoke.jsonl](../tools/data/relevance-smoke.jsonl) is a 16-item smoke set
that shows the format and checks the setup. It is too small and too easy for calibration. The
target is about 200 human-labeled interactions in Portuguese and English, compared on a large
and a small OpenAI model and an open model served locally, with 80% agreement or more on
pass/fail. That calibration has not been run yet, so the model choice and the threshold are
still open.
