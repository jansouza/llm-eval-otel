"""The periodic summary and the failing/recovered transitions."""

import logging

import pytest

from llm_eval_otel.engine.activity import FAILING_AFTER, Activity
from llm_eval_otel.engine.runner import EvaluationRecord
from llm_eval_otel.evaluators.base import EvaluationResult, GenAIInteraction, Message
from llm_eval_otel.judge.client import JudgeCall

INTERACTION = GenAIInteraction(
    trace_id=b"\x01" * 16,
    span_id=b"\x02" * 8,
    parent_span_id=None,
    trace_flags=1,
    service_name="svc",
    operation_name="chat",
    provider_name=None,
    request_model=None,
    response_id=None,
    system_instructions=[],
    input_messages=[Message("user", "oi")],
    output_messages=[],
)


def record(
    name: str, label: str | None = "pass", error_type: str | None = None, tokens: int = 0
) -> EvaluationRecord:
    calls = (JudgeCall("openai", "m", None, None, 0, 1, input_tokens=tokens),) if tokens else ()
    result = EvaluationResult(None, label, "explanation text", error_type=error_type)
    return EvaluationRecord(INTERACTION, name, "heuristic", result, 0, 1, judge_calls=calls)


class Clock:
    now = 100.0

    def __call__(self) -> float:
        return self.now


def test_summary_counts_and_resets(caplog: pytest.LogCaptureFixture) -> None:
    clock = Clock()
    activity = Activity(clock)
    activity.add_batch(5, 3, {"duplicate": 1, "no_content": 1})
    activity.add_records([record("pii_detection"), record("pii_detection", "fail")])
    activity.add_records([record("relevance", tokens=120)])
    clock.now += 60
    with caplog.at_level(logging.INFO, logger="llm_eval_otel"):
        activity.log_summary("queue=0/10")
        activity.log_summary("queue=0/10")
    first, second = caplog.records
    assert first.levelname == "INFO"
    assert first.message == (
        "last 60s: received=5 queued=3 skipped=duplicate:1,no_content:1 rejected=none "
        "| evaluations: pii_detection=2 (fail:1,pass:1) relevance=1 (pass:1) "
        "| judge_tokens=120 | queue=0/10"
    )
    assert "explanation text" not in caplog.text
    assert second.message.startswith("last 0s: received=0 queued=0 skipped=none")
    assert "evaluations: none" in second.message


def test_summary_is_a_warning_when_something_went_wrong(caplog: pytest.LogCaptureFixture) -> None:
    for trouble in (
        lambda a: a.add_rejected(429),
        lambda a: a.add_drop("relevance", "lane_full"),
        lambda a: a.add_records([record("relevance", None, "timeout")]),
    ):
        activity = Activity()
        trouble(activity)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="llm_eval_otel"):
            activity.log_summary("")
        assert caplog.records[0].levelname == "WARNING"
    assert "relevance=1 (error:1)" in caplog.text


def test_failing_after_errors_in_a_row_then_recovered(caplog: pytest.LogCaptureFixture) -> None:
    activity = Activity()
    with caplog.at_level(logging.INFO, logger="llm_eval_otel"):
        # Isolated errors between successes are only counted.
        activity.add_records([record("relevance", None, "timeout"), record("relevance")])
        assert caplog.messages == []
        activity.add_records([record("relevance", None, "timeout")] * (FAILING_AFTER + 2))
        activity.add_records([record("relevance"), record("relevance")])
    assert caplog.messages == [
        f"relevance is failing: {FAILING_AFTER} errors in a row, last error=timeout",
        f"relevance recovered after {FAILING_AFTER + 2} errors in a row",
    ]
    assert [r.levelname for r in caplog.records] == ["WARNING", "INFO"]
