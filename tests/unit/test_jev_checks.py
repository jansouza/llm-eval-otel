import re

import pytest
from judge_fakes import JEV_MODEL, FakeSystemOneClient, chat, jev_checks, relevance

from llm_eval_otel.config import Settings
from llm_eval_otel.engine.runner import lane_of
from llm_eval_otel.evaluators import registry
from llm_eval_otel.evaluators.base import BatchEvaluator, Message, PartSpan
from llm_eval_otel.evaluators.jev_checks import (
    JevPromptInjection,
    JevRefusal,
    JevRelevance,
    JevToxicity,
)
from llm_eval_otel.judge.client import (
    JudgeConfigError,
    JudgeInvalidOutput,
    NoulQuestion,
    ScoreQuestion,
)
from llm_eval_otel.judge.jev import DATA_GUARD, JevEvaluator
from llm_eval_otel.judge.typesafe_adapter import TypeSafeJudge

CPF = "529.982.247-25"
EMAIL = "maria@example.com"
AWS = "AKIAIOSFODNN7EXAMPLE"
TEMPLATE = re.compile(r"^(score=\d\.\d/5 confidence=\d\.\d\d|p=\d\.\d\d)$")


@pytest.mark.parametrize(
    ("expected", "score", "label"),
    [(0.0, 0.0, "fail"), (1.99, 0.4975, "fail"), (2.0, 0.5, "pass"), (4.0, 1.0, "pass")],
)
async def test_relevance_maps_the_expected_level(expected: float, score: float, label: str) -> None:
    client = FakeSystemOneClient(score=expected, confidence=0.82)
    result = await JevRelevance(client, Settings()).evaluate(chat())
    assert (result.score, result.label) == (pytest.approx(score), label)
    assert result.explanation == f"score={expected + 1:.1f}/5 confidence=0.82"
    assert result.attributes == {
        "llm_eval.judge.model": JEV_MODEL,
        "llm_eval.judge.raw_score": expected,
        "llm_eval.judge.confidence": 0.82,
        "llm_eval.judge.batch_size": 1,
    }


@pytest.mark.parametrize("check", [JevRefusal, JevToxicity, JevPromptInjection])
@pytest.mark.parametrize(
    ("p", "label"), [(0.0, "pass"), (0.5, "pass"), (0.51, "fail"), (0.93, "fail")]
)
async def test_noul_fails_above_the_threshold(
    check: type[JevEvaluator], p: float, label: str
) -> None:
    client = FakeSystemOneClient(nouls={check.name: p})
    result = await check(client, Settings()).evaluate(chat())
    assert (result.score, result.label) == (pytest.approx(1 - p), label)
    assert result.explanation == f"p={p:.2f}"
    assert result.attributes == {
        "llm_eval.judge.model": JEV_MODEL,
        "llm_eval.judge.probability": p,
        "llm_eval.judge.batch_size": 1,
    }


async def test_batch_is_one_request_with_every_question() -> None:
    client = FakeSystemOneClient(nouls={"jev_refusal": 0.9})
    checks = jev_checks(client)
    results = await JevRelevance.evaluate_batch(checks, chat())
    [(_, questions)] = client.received
    assert list(questions) == [
        "jev_relevance",
        "jev_refusal",
        "jev_toxicity",
        "jev_prompt_injection",
    ]
    assert [r.label for r in results] == ["pass", "fail", "pass", "pass"]
    assert all(r.attributes["llm_eval.judge.batch_size"] == 4 for r in results)
    assert all(TEMPLATE.match(r.explanation or "") for r in results)


@pytest.mark.parametrize("check", [JevRelevance, JevRefusal, JevToxicity, JevPromptInjection])
def test_checks_are_batchable_and_in_the_jev_lane(check: type[JevEvaluator]) -> None:
    evaluator = check(FakeSystemOneClient(), Settings())
    assert isinstance(evaluator, BatchEvaluator)
    assert (lane_of(evaluator), evaluator.batch_key, evaluator.kind) == (3 * ("jev_judge",))
    assert (evaluator.sample_rate, evaluator.timeout_s, evaluator.max_chars) == (0.1, 5.0, 16_000)


async def test_a_judge_error_is_every_checks_error() -> None:
    client = FakeSystemOneClient(raise_error=JudgeInvalidOutput)
    results = await JevRelevance.evaluate_batch(jev_checks(client), chat())
    assert [r.error_type for r in results] == 4 * ["judge_invalid_output"]
    assert all((r.score, r.label, r.explanation) == (None, None, None) for r in results)


async def test_other_errors_reach_the_runner() -> None:
    """The runner turns them into error_type with the class name, as for any evaluator."""

    class TypeSafeRateLimitError(Exception):
        pass

    client = FakeSystemOneClient(raise_error=TypeSafeRateLimitError)
    with pytest.raises(TypeSafeRateLimitError):
        await JevRefusal(client, Settings()).evaluate(chat())


async def test_a_batch_takes_jev_checks_only() -> None:
    with pytest.raises(TypeError):
        await JevRelevance.evaluate_batch(
            [JevRefusal(FakeSystemOneClient(), Settings()), relevance()],  # type: ignore[list-item]
            chat(),
        )


async def test_state_has_context_request_and_text_response_only() -> None:
    client = FakeSystemOneClient()
    interaction = chat(
        "E em inglês?",
        context=[Message("user", "Como digo obrigado?"), Message("assistant", "Obrigado.")],
    )
    interaction.system_instructions.append(Message("system", "Instruções internas."))
    interaction.output_messages[:] = [
        Message(
            "assistant",
            'pensando\nThank you.\n{"q": 1}',
            (PartSpan("reasoning", 0, 8), PartSpan("text", 9, 19), PartSpan("tool_call", 20, 28)),
        )
    ]
    await JevRefusal(client, Settings()).evaluate(interaction)
    [(state, _)] = client.received
    assert state == {
        "context": [
            {"role": "user", "text": "Como digo obrigado?"},
            {"role": "assistant", "text": "Obrigado."},
        ],
        "request": ["E em inglês?"],
        "response": ["Thank you."],
    }


async def test_pii_and_secrets_are_masked_before_sending() -> None:
    client = FakeSystemOneClient()
    interaction = chat(
        f"Meu CPF é {CPF}, e-mail {EMAIL}",
        f"A chave {AWS} foi revogada.",
        context=[Message("user", f"meu outro e-mail é {EMAIL}")],
    )
    await JevRelevance.evaluate_batch(jev_checks(client), interaction)
    [(state, _)] = client.received
    sent = repr(state)
    for value in (CPF, "52998224725", EMAIL, AWS):
        assert value not in sent
    assert state["request"] == ["Meu CPF é [CPF], e-mail [EMAIL]"]
    assert state["response"] == ["A chave [SECRET] foi revogada."]
    assert state["context"][0]["text"] == "meu outro e-mail é [EMAIL]"


async def test_masking_can_be_turned_off() -> None:
    client = FakeSystemOneClient()
    await JevRefusal(client, Settings(judge_redact=False)).evaluate(chat(f"CPF {CPF}"))
    assert CPF in repr(client.received[0][0])


@pytest.mark.parametrize("check", [JevRelevance, JevRefusal, JevToxicity, JevPromptInjection])
def test_questions_name_the_fields_and_say_what_is_data(check: type[JevEvaluator]) -> None:
    [(qid, question)] = check(FakeSystemOneClient(), Settings()).questions().items()
    assert qid == check.name
    assert DATA_GUARD in question.instructions
    assert "`request`" in question.instructions or "`response`" in question.instructions
    if check is JevRelevance:
        assert isinstance(question, ScoreQuestion) and len(question.levels) == 5
    else:
        assert isinstance(question, NoulQuestion) and question.true and question.false


def test_applies_to() -> None:
    relevance_, refusal, toxicity, injection = jev_checks()
    plain = chat()
    assert all(c.applies_to(plain) for c in (relevance_, refusal, toxicity, injection))

    narrated_tool_call = chat()
    narrated_tool_call.output_messages[:] = [
        Message(
            "assistant",
            'Vou buscar o pedido.\n{"q": 1}',
            (PartSpan("text", 0, 20), PartSpan("tool_call", 21, 29)),
        )
    ]
    assert not relevance_.applies_to(narrated_tool_call)  # an agent step, not the answer
    assert refusal.applies_to(narrated_tool_call) and toxicity.applies_to(narrated_tool_call)

    no_answer = chat(answer="  ")
    assert not refusal.applies_to(no_answer) and not toxicity.applies_to(no_answer)
    assert injection.applies_to(no_answer)

    tool_result_only = chat()
    tool_result_only.input_messages[:] = [Message("tool", "42")]
    assert not injection.applies_to(tool_result_only)
    assert not refusal.applies_to(tool_result_only)
    assert toxicity.applies_to(tool_result_only)


def test_jev_model_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_EVAL_JEV_JUDGE_MODEL", raising=False)
    with pytest.raises(JudgeConfigError, match="jev_toxicity needs LLM_EVAL_JEV_JUDGE_MODEL"):
        registry.load(["jev_toxicity"])


def test_loads_from_the_entry_points_with_the_typesafe_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EVAL_JEV_JUDGE_MODEL", "jev-1.13.0")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    names = ["jev_relevance", "jev_refusal", "jev_toxicity", "jev_prompt_injection"]
    evaluators = registry.load(names)
    assert [e.name for e in evaluators] == names
    for evaluator in evaluators:
        assert isinstance(evaluator, JevEvaluator)
        assert isinstance(evaluator.client, TypeSafeJudge)
        assert evaluator.client.model == "jev-1.13.0"
