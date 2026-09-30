"""Load evaluators registered under the ``llm_eval.evaluators`` entry point group.

A third-party evaluator ships in its own package, declares the entry point, and is
enabled by name in ``LLM_EVAL_EVALUATORS``. The entry point must resolve to a
zero-argument callable (usually the class) that returns an :class:`Evaluator`.
"""

import logging
from collections.abc import Iterable
from importlib.metadata import entry_points

from llm_eval_otel.evaluators.base import Evaluator

ENTRY_POINT_GROUP = "llm_eval.evaluators"

log = logging.getLogger(__name__)


class EvaluatorLoadError(RuntimeError):
    pass


def available() -> dict[str, str]:
    """Registered evaluator names and the object each one points to."""
    return {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}


def load(names: Iterable[str]) -> list[Evaluator]:
    registered = {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}
    evaluators: list[Evaluator] = []
    for name in names:
        if name not in registered:
            raise EvaluatorLoadError(
                f"evaluator {name!r} is not registered; available: {sorted(registered)}"
            )
        factory = registered[name].load()
        evaluator = factory()
        if not isinstance(evaluator, Evaluator):
            raise EvaluatorLoadError(f"entry point {name!r} does not implement Evaluator")
        if evaluator.name != name:
            raise EvaluatorLoadError(
                f"entry point {name!r} returned an evaluator named {evaluator.name!r}"
            )
        evaluators.append(evaluator)
        log.info("loaded evaluator %s (%s)", name, evaluator.kind)
    return evaluators
