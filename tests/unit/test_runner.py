import asyncio
import dataclasses
import logging
import random
from dataclasses import dataclass

import pytest

from llm_eval_otel.engine.runner import Runner, sampled, truncate
from llm_eval_otel.evaluators.base import (
    EvaluationResult,
    EvaluatorKind,
    GenAIInteraction,
    Message,
    PartSpan,
)
from llm_eval_otel.evaluators.pii import PIIDetector
from llm_eval_otel.evaluators.secrets import SecretDetector


def interaction(
    trace_id: bytes = b"\x01" * 16, text: str = "oi", service: str | None = "svc"
) -> GenAIInteraction:
    return GenAIInteraction(
        trace_id=trace_id,
        span_id=b"\x02" * 8,
        parent_span_id=None,
        trace_flags=1,
        service_name=service,
        operation_name="chat",
        provider_name=None,
        request_model=None,
        response_id=None,
        system_instructions=[],
        input_messages=[Message("user", text)],
        output_messages=[],
    )


@dataclass
class FakeJudge:
    name: str = "fake_judge"
    kind: EvaluatorKind = EvaluatorKind.LLM_JUDGE
    timeout_s: float = 1.0
    sample_rate: float = 1.0
    max_chars: int | None = None
    delay_s: float = 0.0
    fail_with: type[Exception] | None = None

    def applies_to(self, interaction: GenAIInteraction) -> bool:
        return True

    async def evaluate(self, interaction: GenAIInteraction) -> EvaluationResult:
        await asyncio.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with("secret content 529.982.247-25 in the message")
        return EvaluationResult(
            0.8, "pass", "fine", {"llm_eval.chars": len(interaction.input_messages[0].text)}
        )


def test_sampled_matches_probability_sampler_rule() -> None:
    assert sampled(b"\x00" * 16, 1.0)
    assert not sampled(b"\x00" * 16, 0.5)
    assert sampled(b"\x00" * 9 + b"\xff" * 7, 0.01)
    assert not sampled(b"\xff" * 16, 0.0)


async def test_sample_rate_selects_about_ten_percent_deterministically() -> None:
    rng = random.Random(42)
    trace_ids = [rng.randbytes(16) for _ in range(10_000)]
    judge = FakeJudge(sample_rate=0.1)
    runner = Runner([PIIDetector(), SecretDetector(), judge], default_timeout_s=5)

    def picks(evaluator_name: str) -> set[bytes]:
        return {
            t
            for t in trace_ids
            if evaluator_name in {e.name for e in runner.select(interaction(t))}
        }

    judged = picks("fake_judge")
    assert 900 <= len(judged) <= 1100
    assert picks("fake_judge") == judged  # same trace ids every time
    assert len(picks("pii_detection")) == 10_000
    assert len(picks("secret_detection")) == 10_000


def test_lower_rate_is_subset_of_higher_rate() -> None:
    rng = random.Random(7)
    trace_ids = [rng.randbytes(16) for _ in range(2_000)]
    at_5 = {t for t in trace_ids if sampled(t, 0.05)}
    at_20 = {t for t in trace_ids if sampled(t, 0.2)}
    assert at_5 < at_20


async def test_sample_rate_override_by_name() -> None:
    runner = Runner(
        [FakeJudge(sample_rate=1.0)], default_timeout_s=5, sample_rates={"fake_judge": 0.0}
    )
    assert runner.select(interaction()) == []


async def test_timeout_and_exception_become_error_results_without_stopping_others() -> None:
    runner = Runner(
        [
            PIIDetector(),
            FakeJudge(name="slow", timeout_s=0.05, delay_s=1),
            FakeJudge(name="broken", fail_with=ValueError),
        ],
        default_timeout_s=5,
    )
    records = {r.evaluator_name: r.result for r in await runner.run(interaction())}
    assert records["pii_detection"].label == "pass"
    assert records["slow"].error_type == "timeout"
    assert records["broken"].error_type == "ValueError"
    assert records["broken"].score is None and records["broken"].explanation is None


async def test_errors_are_debug_logged_with_ids_and_class_name_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = Runner(
        [
            FakeJudge(name="slow", timeout_s=0.05, delay_s=1),
            FakeJudge(name="broken", fail_with=ValueError),
        ],
        default_timeout_s=5,
    )
    with caplog.at_level(logging.DEBUG, logger="llm_eval_otel"):
        await runner.run(interaction())
    trace = f"trace={'01' * 16} span={'02' * 8} service=svc"
    assert f"slow {trace}: error=timeout" in caplog.text
    assert f"broken {trace}: error=ValueError" in caplog.text
    assert "529.982.247-25" not in caplog.text


async def test_debug_logs_label_and_score_without_content_or_explanation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = Runner([PIIDetector()], default_timeout_s=5)
    with caplog.at_level(logging.DEBUG, logger="llm_eval_otel"):
        await runner.run(interaction(text="CPF 529.982.247-25"))
    assert "pii_detection trace=" in caplog.text
    assert "label=fail score=" in caplog.text
    assert "529.982.247-25" not in caplog.text
    assert "cpf=1" not in caplog.text


async def test_exempt_service_keeps_findings_and_drops_score() -> None:
    runner = Runner(
        [PIIDetector(), SecretDetector()],
        default_timeout_s=5,
        exceptions={"bank-chatbot": ["pii_detection"]},
    )
    records = await runner.run(
        interaction(text="CPF 529.982.247-25 e 111.444.777-35", service="bank-chatbot")
    )
    pii = next(r.result for r in records if r.evaluator_name == "pii_detection")
    assert (pii.score, pii.label) == (None, "exempt")
    assert pii.explanation == "exempt service; cpf=2 (input)"
    secret = next(r.result for r in records if r.evaluator_name == "secret_detection")
    assert secret.label == "pass"


async def test_exempt_service_without_findings() -> None:
    runner = Runner([PIIDetector()], default_timeout_s=5, exceptions={"svc": ["pii_detection"]})
    [record] = await runner.run(interaction())
    assert record.result.explanation == "exempt service; no findings"


async def test_exemption_needs_exact_service_name() -> None:
    runner = Runner([PIIDetector()], default_timeout_s=5, exceptions={"svc": ["pii_detection"]})
    for service in ("SVC", "svc-2", None):
        [record] = await runner.run(interaction(text="CPF 529.982.247-25", service=service))
        assert record.result.label == "fail"


async def test_max_chars_truncates_and_marks_record() -> None:
    runner = Runner([FakeJudge(max_chars=5)], default_timeout_s=5)
    [record] = await runner.run(interaction(text="abcdefghij"))
    assert record.truncated
    assert record.result.attributes["llm_eval.chars"] == 5


def test_truncate_budget_spans_messages() -> None:
    i = interaction(text="abc")
    i2, truncated = truncate(i, 10)
    assert not truncated and i2 == i
    i3, truncated = truncate(i, 2)
    assert truncated and i3.input_messages == [Message("user", "ab")]


def test_truncate_keeps_the_output_first_then_input_then_system() -> None:
    i = dataclasses.replace(
        interaction(text="i" * 10),
        system_instructions=[Message("system", "s" * 10)],
        output_messages=[Message("assistant", "o" * 10)],
    )
    cut, truncated = truncate(i, 15)
    assert truncated
    assert cut.output_messages == [Message("assistant", "o" * 10)]
    assert cut.input_messages == [Message("user", "i" * 5)]
    assert cut.system_instructions == []


def test_truncate_keeps_part_offsets() -> None:
    parts = (PartSpan("reasoning", 0, 8), PartSpan("text", 9, 15), PartSpan("tool_call", 16, 24))
    i = dataclasses.replace(
        interaction(text=""),
        input_messages=[],
        output_messages=[Message("assistant", 'pensando\npronto\n{"a": 1}', parts)],
    )
    cut, _ = truncate(i, 12)
    [message] = cut.output_messages
    assert message.parts == (PartSpan("reasoning", 0, 8), PartSpan("text", 9, 12))
    assert message.text_of("text") == "pro"


async def test_exempt_judge_is_not_called() -> None:
    judge = FakeJudge(delay_s=10)
    runner = Runner([judge], default_timeout_s=5, exceptions={"svc": ["fake_judge"]})
    [record] = await runner.run(interaction())
    assert record.result.label == "exempt"
    assert record.result.explanation == "exempt service; not evaluated"
    assert record.result.score is None and record.end_ns == record.start_ns
