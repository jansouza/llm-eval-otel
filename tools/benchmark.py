"""Run an evaluator over a labeled JSONL set and report how it agrees with the labels.

For calibrating judges (and, later, local classifiers) against human labels. The judges are
configured as in the service, by the LLM_EVAL_LLM_JUDGE_* variables and OPENAI_API_KEY
(``relevance``) or LLM_EVAL_JEV_JUDGE_* and TYPESAFE_API_KEY (``jev_*``), from the environment or
./.env (see .env.example). To judge a single interaction, use llm-eval-judge.

    LLM_EVAL_LLM_JUDGE_MODEL=gpt-5-mini-2025-08-07 OPENAI_API_KEY=... \\
    uv run python tools/benchmark.py tools/data/relevance-smoke.jsonl \\
        --repeat 3 --price-input 0.25 --price-cached 0.025 --price-output 2.00

    # Jev against the labels and against relevance on the same items; Jev bills input only.
    uv run python tools/benchmark.py tools/data/relevance-synthetic.jsonl \\
        --evaluator jev_relevance --against relevance --price-input 0.042 --price-output 0
    uv run python tools/benchmark.py tools/data/refusal-labeled.jsonl --evaluator jev_refusal

One JSON object per line:

    {"id": "pt-01", "lang": "pt", "label": "pass", "score": 5,
     "context": [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}],
     "input": "the user's message in this turn", "output": "the response"}

``label`` is the human pass/fail; ``score`` (the human 1-5 rating) is optional. The report
gives agreement with the label on pass/fail, by language, the judge's mean confidence (Jev's
score questions), how much the raw score (a rating, an expected level or a probability)
varies when the same item runs again, latency p50/p99, tokens per evaluation, and the cost
per evaluation and per thousand when prices (per million tokens) are given. ``--against``
runs a second evaluator on the same items and adds how often the two agree, by language.
``--details`` writes one line per run, with the explanations, to read the disagreements.
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_eval_otel.cli import interaction_from_record
from llm_eval_otel.engine.runner import truncate
from llm_eval_otel.evaluators import registry
from llm_eval_otel.evaluators.base import Evaluator, GenAIInteraction
from llm_eval_otel.judge.client import JudgeCall, recording


@dataclass
class Item:
    id: str
    lang: str
    label: str
    score: int | None
    interaction: GenAIInteraction


@dataclass
class Run:
    item: Item
    evaluator: str
    label: str | None
    raw_score: float | None  # relevance's rating, Jev's expected level or probability
    confidence: float | None
    explanation: str | None
    error_type: str | None
    seconds: float
    calls: list[JudgeCall] = field(default_factory=list)


def load(path: Path) -> list[Item]:
    items = []
    for n, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        data: dict[str, Any] = json.loads(line)
        interaction = interaction_from_record(data, n)
        items.append(
            Item(data["id"], data.get("lang", "?"), data["label"], data.get("score"), interaction)
        )
    return items


async def run_one(evaluator: Evaluator, item: Item, semaphore: asyncio.Semaphore) -> Run:
    interaction = item.interaction
    if evaluator.max_chars is not None:
        interaction, _ = truncate(interaction, evaluator.max_chars)
    async with semaphore:
        start = time.perf_counter()
        with recording() as calls:
            try:
                result = await asyncio.wait_for(
                    evaluator.evaluate(interaction), evaluator.timeout_s
                )
                error_type = result.error_type
            except TimeoutError:
                result, error_type = None, "timeout"
            except Exception as exc:
                result, error_type = None, type(exc).__name__
        seconds = time.perf_counter() - start
    attrs = result.attributes if result else {}
    raw = attrs.get("llm_eval.judge.raw_score", attrs.get("llm_eval.judge.probability"))
    confidence = attrs.get("llm_eval.judge.confidence")
    return Run(
        item=item,
        evaluator=evaluator.name,
        label=result.label if result else None,
        raw_score=float(raw) if isinstance(raw, int | float) else None,
        confidence=float(confidence) if isinstance(confidence, int | float) else None,
        explanation=result.explanation if result else None,
        error_type=error_type,
        seconds=seconds,
        calls=list(calls),
    )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]


def mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def report(runs: list[Run], args: argparse.Namespace) -> None:
    ok = [r for r in runs if r.error_type is None]
    errors = Counter(r.error_type for r in runs if r.error_type is not None)
    items = {r.item.id for r in runs}
    name = runs[0].evaluator if runs else "?"
    print(f"{name}: {len(items)} items x {args.repeat} runs = {len(runs)} evaluations")
    if errors:
        print("errors: " + ", ".join(f"{name}={n}" for name, n in errors.most_common()))
    if not ok:
        return

    agree = [r.label == r.item.label for r in ok]
    print(f"agreement with the human label: {sum(agree) / len(agree):.1%} ({sum(agree)}/{len(ok)})")
    by_lang: dict[str, list[bool]] = defaultdict(list)
    for r in ok:
        by_lang[r.item.lang].append(r.label == r.item.label)
    for lang, results in sorted(by_lang.items()):
        print(f"  {lang}: {sum(results) / len(results):.1%} ({sum(results)}/{len(results)})")
    for want, got in (("pass", "fail"), ("fail", "pass")):
        n = sum(1 for r in ok if r.item.label == want and r.label == got)
        print(f"  labeled {want}, judged {got}: {n}")

    rated = [r for r in ok if r.raw_score is not None and r.item.score is not None]
    if rated and name == "relevance":  # the only raw score on the human 1-5 scale
        off = [abs((r.raw_score or 0) - (r.item.score or 0)) for r in rated]
        print(f"rating distance to the human rating: mean {statistics.fmean(off):.2f}")
    confidences = [r.confidence for r in ok if r.confidence is not None]
    if confidences:
        print(f"confidence: mean {statistics.fmean(confidences):.2f}, min {min(confidences):.2f}")
    if args.repeat > 1:
        ratings: dict[str, list[float]] = defaultdict(list)
        for r in ok:
            if r.raw_score is not None:
                ratings[r.item.id].append(r.raw_score)
        spreads = [max(v) - min(v) for v in ratings.values() if len(v) > 1]
        varied = sum(1 for s in spreads if s)
        print(
            f"raw score variation on repeat: {varied}/{len(spreads)} items changed, "
            f"max spread {max(spreads, default=0):.3g}"
        )

    seconds = [r.seconds * 1000 for r in ok]
    print(f"latency: p50 {percentile(seconds, 0.5):.0f} ms, p99 {percentile(seconds, 0.99):.0f} ms")

    calls = [c for r in runs for c in r.calls]
    reported = [c for c in calls if c.tokens_used is not None]
    per_eval = len(runs) or 1
    tokens_in = sum(c.input_tokens or 0 for c in reported) / per_eval
    tokens_out = sum(c.output_tokens or 0 for c in reported) / per_eval
    cached = sum(c.cache_read_tokens or 0 for c in reported) / per_eval
    print(
        f"tokens per evaluation: input {tokens_in:.0f} (cached {cached:.0f}), "
        f"output {tokens_out:.0f}; usage reported on {len(reported)}/{len(calls)} calls"
    )
    models = Counter(c.response_model for c in calls if c.response_model)
    if models:
        print("models: " + ", ".join(f"{m}={n}" for m, n in models.most_common()))
    if args.price_input is not None and args.price_output is not None:
        price_cached = args.price_cached if args.price_cached is not None else args.price_input
        cost = (
            (tokens_in - cached) * args.price_input
            + cached * price_cached
            + tokens_out * args.price_output
        ) / 1e6
        print(f"cost: {cost:.6f} per evaluation, {cost * 1000:.4f} per 1000 evaluations")


def agreement(first: list[Run], second: list[Run]) -> None:
    """How often two evaluators give the same pass/fail on the same item and run."""
    pairs = [
        (a, b)
        for a, b in zip(first, second, strict=True)
        if a.error_type is None and b.error_type is None
    ]
    if not pairs:
        return
    name_a, name_b = first[0].evaluator, second[0].evaluator
    same = [a.label == b.label for a, b in pairs]
    print(
        f"agreement between {name_a} and {name_b}: {sum(same) / len(same):.1%} "
        f"({sum(same)}/{len(same)})"
    )
    by_lang: dict[str, list[bool]] = defaultdict(list)
    for a, b in pairs:
        by_lang[a.item.lang].append(a.label == b.label)
    for lang, results in sorted(by_lang.items()):
        print(f"  {lang}: {sum(results) / len(results):.1%} ({sum(results)}/{len(results)})")
    for label_a, label_b in (("pass", "fail"), ("fail", "pass")):
        n = sum(1 for a, b in pairs if a.label == label_a and b.label == label_b)
        print(f"  {name_a} {label_a}, {name_b} {label_b}: {n}")


async def run_all(name: str, items: list[Item], args: argparse.Namespace) -> list[Run]:
    [evaluator] = registry.load([name])
    semaphore = asyncio.Semaphore(args.concurrency)
    return list(
        await asyncio.gather(
            *(run_one(evaluator, item, semaphore) for item in items for _ in range(args.repeat))
        )
    )


async def main_async(args: argparse.Namespace) -> None:
    items = load(args.dataset)
    if args.limit:
        items = items[: args.limit]
    runs = await run_all(args.evaluator, items, args)
    report(runs, args)
    if args.against:
        # Same items in the same order, so runs pair up one to one.
        against = await run_all(args.against, items, args)
        print()
        report(against, args)
        print()
        agreement(runs, against)
        runs += against
    if args.details:
        with args.details.open("w") as out:
            for r in runs:
                out.write(
                    json.dumps(
                        {
                            "evaluator": r.evaluator,
                            "id": r.item.id,
                            "human": r.item.label,
                            "human_score": r.item.score,
                            "label": r.label,
                            "score": r.raw_score,
                            "confidence": r.confidence,
                            "explanation": r.explanation,
                            "error_type": r.error_type,
                            "ms": round(r.seconds * 1000),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--evaluator", default="relevance")
    parser.add_argument("--against", help="a second evaluator to compare with, on the same items")
    parser.add_argument("--repeat", type=int, default=1, help="runs per item, to see variation")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N items only")
    parser.add_argument("--price-input", type=float, help="per million input tokens")
    parser.add_argument("--price-cached", type=float, help="per million cached input tokens")
    parser.add_argument("--price-output", type=float, help="per million output tokens")
    parser.add_argument("--details", type=Path, help="write one JSON line per run here")
    args = parser.parse_args()
    load_dotenv(override=False)  # ./.env, when it exists
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
