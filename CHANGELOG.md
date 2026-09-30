# Changelog

The version is `service.version` on everything the service emits, so it identifies the
detection rules behind each result. The minor version goes up whenever what gets detected
changes (a new type, evaluator or threshold); the patch version for fixes that don't change
detection. See [Versioning](README.md#versioning).

## [Unreleased]

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
