import sys
import textwrap
from pathlib import Path

import pytest
from conftest import OtelMemory, ServiceFactory
from otlp import chat_request

from llm_eval_otel.evaluators import registry

EXAMPLE_MODULE = """
from llm_eval_otel.evaluators.base import EvaluationResult, EvaluatorKind


class LengthEvaluator:
    name = "example_length"
    kind = EvaluatorKind.HEURISTIC
    timeout_s = 1.0
    sample_rate = 1.0
    max_chars = None

    def applies_to(self, interaction):
        return True

    async def evaluate(self, interaction):
        size = sum(len(m.text) for m in interaction.input_messages)
        return EvaluationResult(1.0, "pass", f"{size} chars", {"llm_eval.example.chars": size})
"""


@pytest.fixture
def example_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A third-party package installed on sys.path, registered only by entry point."""
    (tmp_path / "example_eval.py").write_text(textwrap.dedent(EXAMPLE_MODULE))
    dist_info = tmp_path / "example_eval-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: example-eval\nVersion: 0.1.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[llm_eval.evaluators]\nexample_length = example_eval:LengthEvaluator\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("example_eval", None)


def test_builtin_evaluators_are_registered() -> None:
    names = registry.available()
    assert names["pii_detection"] == "llm_eval_otel.evaluators.pii:PIIDetector"
    assert names["secret_detection"] == "llm_eval_otel.evaluators.secrets:SecretDetector"


def test_unknown_evaluator_fails_fast() -> None:
    with pytest.raises(registry.EvaluatorLoadError, match="not registered"):
        registry.load(["nope"])


@pytest.mark.usefixtures("example_package")
async def test_third_party_evaluator_runs_without_service_changes(
    make_service: ServiceFactory, otel_memory: OtelMemory
) -> None:
    evaluators = registry.load(["pii_detection", "example_length"])
    service = make_service(evaluators)
    service.ingest(chat_request("hello"))
    await service.drain()
    [event] = otel_memory.events("gen_ai.evaluation.result", name="example_length")
    attrs = event.log_record.attributes or {}
    assert attrs["gen_ai.evaluation.explanation"] == "5 chars"
    assert attrs["llm_eval.example.chars"] == 5
    assert len(otel_memory.events("gen_ai.evaluation.result", name="pii_detection")) == 1
