"""What the service did since the last summary, for the periodic INFO line.

Per-span detail is DEBUG: at a few hundred spans per second, an INFO line per span would bury
everything else. Instead, one line per interval counts what came in, what was evaluated and
what went wrong, and state changes (an evaluator starting to fail, then recovering) are logged
once when they happen. Counts only: no IDs, no content, no explanations.
"""

import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

from llm_eval_otel.engine.runner import EvaluationRecord

log = logging.getLogger(__name__)

# Consecutive errors before an evaluator counts as failing. A single timeout among successes
# only shows in the summary; a run of them is an outage worth its own line.
FAILING_AFTER = 3

ERROR = "error"
DROPPED_PREFIX = "dropped_"


def _counts(counter: Counter[str] | Counter[int]) -> str:
    return ",".join(f"{key}:{n}" for key, n in sorted(counter.items(), key=str)) or "none"


class Activity:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._errors_in_a_row: Counter[str] = Counter()
        self._failing: set[str] = set()
        self._reset()

    def _reset(self) -> None:
        self.since = self._clock()
        self.received = 0
        self.queued = 0
        self.skipped: Counter[str] = Counter()
        self.rejected: Counter[int] = Counter()
        # evaluator name -> label, "error" or "dropped_<reason>" -> count
        self.results: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.judge_tokens = 0

    def add_batch(self, received: int, queued: int, skipped: dict[str, int]) -> None:
        self.received += received
        self.queued += queued
        self.skipped.update(skipped)

    def add_rejected(self, status: int) -> None:
        self.rejected[status] += 1

    def add_drop(self, evaluation_name: str, reason: str) -> None:
        self.results[evaluation_name][DROPPED_PREFIX + reason] += 1

    def add_records(self, records: Iterable[EvaluationRecord]) -> None:
        for record in records:
            name = record.evaluator_name
            error_type = record.result.error_type
            self.results[name][ERROR if error_type else str(record.result.label)] += 1
            self.judge_tokens += sum(
                t for call in record.judge_calls if (t := call.tokens_used) is not None
            )
            if error_type is not None:
                self._errors_in_a_row[name] += 1
                if self._errors_in_a_row[name] == FAILING_AFTER:
                    self._failing.add(name)
                    log.warning(
                        "%s is failing: %d errors in a row, last error=%s",
                        name,
                        FAILING_AFTER,
                        error_type,
                    )
            else:
                errors = self._errors_in_a_row.pop(name, 0)
                if name in self._failing:
                    self._failing.discard(name)
                    log.info("%s recovered after %d errors in a row", name, errors)

    @property
    def troubled(self) -> bool:
        """Anything an operator should look at: rejections, errors or drops."""
        return bool(self.rejected) or any(
            key == ERROR or key.startswith(DROPPED_PREFIX)
            for counts in self.results.values()
            for key in counts
        )

    def summary(self) -> str:
        evaluations = " ".join(
            f"{name}={sum(counts.values())} ({_counts(counts)})"
            for name, counts in sorted(self.results.items())
        )
        return (
            f"last {self._clock() - self.since:.0f}s: received={self.received} "
            f"queued={self.queued} skipped={_counts(self.skipped)} "
            f"rejected={_counts(self.rejected)} | evaluations: {evaluations or 'none'} "
            f"| judge_tokens={self.judge_tokens}"
        )

    def log_summary(self, queues: str) -> None:
        """Log the interval's summary, WARNING when something went wrong, and start a new one."""
        level = logging.WARNING if self.troubled else logging.INFO
        log.log(level, "%s | %s", self.summary(), queues)
        self._reset()
