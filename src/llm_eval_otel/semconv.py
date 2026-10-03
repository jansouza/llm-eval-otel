"""Every attribute, event and metric name the service reads or emits.

GenAI semantic conventions are still in Development status, so all names live here.
Reference: open-telemetry/semantic-conventions-genai at commit e57c543 (2026-09-24).
Names under ``gen_ai.*`` come from the semconv; names under ``llm_eval.*`` are this
service's own, because inventing names in the ``gen_ai`` namespace could collide with
a future definition.
"""

from typing import Final

# --- Input: GenAI span attributes -----------------------------------------------------
GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME: Final = "gen_ai.provider.name"
GEN_AI_SYSTEM: Final = "gen_ai.system"  # legacy fallback for provider name
GEN_AI_REQUEST_MODEL: Final = "gen_ai.request.model"
GEN_AI_RESPONSE_ID: Final = "gen_ai.response.id"
GEN_AI_SYSTEM_INSTRUCTIONS: Final = "gen_ai.system_instructions"
GEN_AI_INPUT_MESSAGES: Final = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES: Final = "gen_ai.output.messages"
GEN_AI_OUTPUT_TYPE: Final = "gen_ai.output.type"
GEN_AI_RESPONSE_FINISH_REASONS: Final = "gen_ai.response.finish_reasons"
# Deprecated per-message field in gen_ai.output.messages, still written by OpenLLMetry;
# read only when gen_ai.response.finish_reasons is absent.
MESSAGE_FINISH_REASON: Final = "finish_reason"

OUTPUT_TYPE_JSON: Final = "json"
OUTPUT_TYPE_TEXT: Final = "text"
FINISH_REASON_LENGTH: Final = "length"
FINISH_REASON_CONTENT_FILTER: Final = "content_filter"

# OpenLLMetry indexed attributes: gen_ai.prompt.{n}.role|content, gen_ai.completion.{n}.*
OPENLLMETRY_PATTERN: Final = r"^gen_ai\.(prompt|completion)\.(\d+)\.(role|content|finish_reason)$"
# OpenLLMetry's stand-in for gen_ai.output.type: the requested response format or schema,
# as JSON. Present (and not {"type": "text"}) means the client asked for JSON.
OPENLLMETRY_STRUCTURED_OUTPUT_SCHEMA: Final = "gen_ai.request.structured_output_schema"
# OpenLLMetry association properties, copied as-is onto the evaluation telemetry
TRACELOOP_ASSOCIATION_PREFIX: Final = "traceloop.association.properties."

INFERENCE_OPERATIONS: Final = frozenset({"chat", "text_completion", "generate_content"})

# Message part types that carry content we evaluate, and the field holding it.
PART_TEXT: Final = "text"
PART_REASONING: Final = "reasoning"
PART_TOOL_CALL: Final = "tool_call"
PART_TOOL_CALL_RESPONSE: Final = "tool_call_response"
PART_CONTENT_FIELDS: Final = {
    PART_TEXT: "content",
    PART_REASONING: "content",
    PART_TOOL_CALL: "arguments",
    PART_TOOL_CALL_RESPONSE: "response",
}

SERVICE_NAME: Final = "service.name"
ROLE_USER: Final = "user"
ROLE_ASSISTANT: Final = "assistant"

# --- Output: evaluation event and span ------------------------------------------------
EVENT_EVALUATION_RESULT: Final = "gen_ai.evaluation.result"
SPAN_NAME_PREFIX: Final = "evaluate"

GEN_AI_EVALUATION_NAME: Final = "gen_ai.evaluation.name"
GEN_AI_EVALUATION_SCORE_VALUE: Final = "gen_ai.evaluation.score.value"
GEN_AI_EVALUATION_SCORE_LABEL: Final = "gen_ai.evaluation.score.label"
GEN_AI_EVALUATION_EXPLANATION: Final = "gen_ai.evaluation.explanation"
ERROR_TYPE: Final = "error.type"
ERROR_TIMEOUT: Final = "timeout"

LLM_EVAL_SOURCE_SERVICE_NAME: Final = "llm_eval.source.service.name"
LLM_EVAL_EVALUATION_TYPE: Final = "llm_eval.evaluation.type"
LLM_EVAL_PII_TYPES: Final = "llm_eval.pii.types"
LLM_EVAL_SECRET_TYPES: Final = "llm_eval.secret.types"
LLM_EVAL_REFUSAL_SOURCE: Final = "llm_eval.refusal.source"
LLM_EVAL_REFUSAL_LANGUAGE: Final = "llm_eval.refusal.language"
LLM_EVAL_PROMPT_LEAK_COVERAGE: Final = "llm_eval.prompt_leak.coverage"
LLM_EVAL_PROMPT_LEAK_LONGEST_RUN: Final = "llm_eval.prompt_leak.longest_run"
LLM_EVAL_OUTPUT_FORMAT_ERROR: Final = "llm_eval.output_format.error"
LLM_EVAL_CONTENT_TRUNCATED: Final = "llm_eval.content.truncated"
LLM_EVAL_JUDGE_MODEL: Final = "llm_eval.judge.model"
LLM_EVAL_JUDGE_RAW_SCORE: Final = "llm_eval.judge.raw_score"
LLM_EVAL_SKIP_REASON: Final = "llm_eval.skip.reason"
LLM_EVAL_DROP_REASON: Final = "llm_eval.drop.reason"
LLM_EVAL_LANE: Final = "llm_eval.lane"

LLM_EVAL_ATTRIBUTE_PREFIX: Final = "llm_eval."

# --- Labels ---------------------------------------------------------------------------
LABEL_PASS: Final = "pass"
LABEL_FAIL: Final = "fail"
LABEL_EXEMPT: Final = "exempt"

# --- Metrics --------------------------------------------------------------------------
METRIC_EVALUATIONS: Final = "llm_eval.evaluations"
METRIC_EVALUATION_SCORE: Final = "llm_eval.evaluation.score"
METRIC_EVALUATION_DURATION: Final = "llm_eval.evaluation.duration"
METRIC_SPANS_RECEIVED: Final = "llm_eval.spans.received"
METRIC_SPANS_SKIPPED: Final = "llm_eval.spans.skipped"
METRIC_QUEUE_SIZE: Final = "llm_eval.queue.size"
METRIC_SANITIZER_REDACTIONS: Final = "llm_eval.sanitizer.redactions"
METRIC_EVALUATIONS_DROPPED: Final = "llm_eval.evaluations.dropped"
METRIC_LANE_SIZE: Final = "llm_eval.lane.size"

SCORE_BUCKETS: Final = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# --- Skip and drop reasons ------------------------------------------------------------
SKIP_NOT_INFERENCE: Final = "not_inference"
SKIP_NO_CONTENT: Final = "no_content"
SKIP_DUPLICATE: Final = "duplicate"
SKIP_INVALID_PAYLOAD: Final = "invalid_payload"
SKIP_SELF_TELEMETRY: Final = "self_telemetry"
DROP_LANE_FULL: Final = "lane_full"
DROP_BUDGET: Final = "budget"
DROP_SHUTDOWN: Final = "shutdown"

# --- Output: the judge's own calls (GenAI client span and metrics) ---------------------
# Content attributes (gen_ai.input.messages, gen_ai.output.messages) are never set here.
OPERATION_CHAT: Final = "chat"
PROVIDER_OPENAI: Final = "openai"
GEN_AI_RESPONSE_MODEL: Final = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: Final = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_TOKEN_TYPE: Final = "gen_ai.token.type"
TOKEN_TYPE_INPUT: Final = "input"
TOKEN_TYPE_OUTPUT: Final = "output"
SERVER_ADDRESS: Final = "server.address"
SERVER_PORT: Final = "server.port"
METRIC_CLIENT_TOKEN_USAGE: Final = "gen_ai.client.token.usage"
METRIC_CLIENT_OPERATION_DURATION: Final = "gen_ai.client.operation.duration"

# Advisory bucket boundaries the semconv gives for the two client metrics.
TOKEN_USAGE_BUCKETS: Final = (
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864,
)  # fmt: skip
OPERATION_DURATION_BUCKETS: Final = (
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
)  # fmt: skip
