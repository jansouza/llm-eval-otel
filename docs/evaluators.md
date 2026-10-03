# Evaluators

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
| `relevance` | opt-in | Responses that don't address what the user asked, rated 1 to 5 by an LLM judge (the OpenAI API or any OpenAI-compatible server) on 5% of traces. See [LLM-as-a-Judge](judge.md) |

**Phone numbers are detected since 0.2.0.** Support chatbots often handle phone numbers, so
`pii_detection` may report more `fail` results after an upgrade. To turn off one type without
exempting the whole evaluator, set `LLM_EVAL_PII_TYPES`. For example,
`LLM_EVAL_PII_TYPES=cpf,cnpj,email,credit_card,pix_key` leaves out `phone`.

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

## Reading the results

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
does in [src/llm_eval_otel/evaluators/relevance.py](../src/llm_eval_otel/evaluators/relevance.py).
It then gets the `LLM_EVAL_JUDGE_*` configuration, masking before sending, the
`<conversation>` envelope, the judge spans and metrics, and the `judge_*` errors; it supplies
the prompt, the schema, the content and the verdict.

The runner turns exceptions and timeouts into results that carry `error.type` (the exception's
class name, never its message), and one failure does not stop the other evaluators. The
sanitizer also covers your evaluator: if an explanation quotes PII or a credential, that value
becomes `[REDACTED]`.
