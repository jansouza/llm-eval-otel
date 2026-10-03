"""``llm-eval-judge``: run an evaluator, usually the judge, on interactions from the command line.

    llm-eval-judge -i "Qual o horário de atendimento?" -o "Das 9h às 18h, de segunda a sexta."
    llm-eval-judge -i "E em inglês?" -o "Good morning." \\
        -c "user:Como digo bom dia em espanhol?" -c "assistant:Buenos días."
    llm-eval-judge --jsonl tools/data/relevance-smoke.jsonl
    llm-eval-judge -i "Meu CPF é 529.982.247-25" -o "Anotado." --dry-run

The judge is configured as in the service: the LLM_EVAL_JUDGE_* variables and the SDK's
OPENAI_API_KEY, from the environment or from a .env file (``--env-file``, ``./.env`` by
default; variables already set win). Unlike the service, it always runs: no sampling and no
exemptions. ``max_chars``, the timeout, masking and the sanitizer apply as in the service.

Prints one JSON object per interaction. Exit status: 0 when every interaction was evaluated,
1 when any ended with ``error_type``, 2 on bad usage or configuration.
"""

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_eval_otel import semconv
from llm_eval_otel.config import Settings
from llm_eval_otel.emit.sanitize import sanitize
from llm_eval_otel.engine.runner import truncate
from llm_eval_otel.evaluators import registry
from llm_eval_otel.evaluators.base import AttributeValue, Evaluator, GenAIInteraction, Message
from llm_eval_otel.judge.client import JudgeCall, JudgeResponse, recording
from llm_eval_otel.judge.evaluator import JudgeEvaluator

EXIT_OK, EXIT_ERRORS, EXIT_USAGE = 0, 1, 2
ROLES = (semconv.ROLE_USER, semconv.ROLE_ASSISTANT)


class UsageError(Exception):
    pass


def _texts(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise UsageError(f"{field!r} must be a string or a list of strings")


def interaction_from_record(data: Mapping[str, Any], n: int = 0) -> GenAIInteraction:
    """``{"input": ..., "output": ..., "context": [{"role", "text"}]}`` as an interaction.

    ``input`` is the user's message(s) in the turn and ``output`` the response(s); both take a
    string or a list of strings. The same format as ``tools/benchmark.py``.
    """
    for field in ("input", "output"):
        if field not in data:
            raise UsageError(f"missing {field!r}")
    context = []
    for item in data.get("context") or []:
        if not isinstance(item, Mapping) or item.get("role") not in ROLES:
            raise UsageError('each context item needs "role" (user or assistant) and "text"')
        context.append(Message(item["role"], str(item.get("text", ""))))
    return GenAIInteraction(
        trace_id=n.to_bytes(16, "big"),
        span_id=n.to_bytes(8, "big"),
        parent_span_id=None,
        trace_flags=1,
        service_name="llm-eval-judge",
        operation_name=semconv.OPERATION_CHAT,
        provider_name=None,
        request_model=None,
        response_id=None,
        system_instructions=[],
        input_messages=[Message(semconv.ROLE_USER, t) for t in _texts(data["input"], "input")],
        output_messages=[
            Message(semconv.ROLE_ASSISTANT, t) for t in _texts(data["output"], "output")
        ],
        context_messages=context,
    )


def parse_context(value: str) -> dict[str, str]:
    role, sep, text = value.partition(":")
    if not sep or role not in ROLES:
        raise UsageError(f"--context takes ROLE:TEXT with ROLE user or assistant, not {value!r}")
    return {"role": role, "text": text}


def read_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.jsonl is not None:
        if args.input is not None or args.output is not None or args.context:
            raise UsageError("use either --jsonl or --input/--output/--context")
        source = sys.stdin if args.jsonl == "-" else Path(args.jsonl).open()  # noqa: SIM115
        with source:
            lines = [line for line in source if line.strip()]
        records = []
        for number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except ValueError:
                raise UsageError(f"line {number} is not JSON") from None
            if not isinstance(record, dict):
                raise UsageError(f"line {number} is not a JSON object")
            records.append(record)
        return records
    if args.input is None or args.output is None:
        raise UsageError("give --input and --output, or --jsonl")
    return [
        {
            "input": args.input,
            "output": args.output,
            "context": [parse_context(c) for c in args.context],
        }
    ]


class _Offline:
    """The judge client for --dry-run: building the content never calls it."""

    async def judge(self, system: str, content: str, schema: Mapping[str, Any]) -> JudgeResponse:
        raise RuntimeError("--dry-run never calls the judge")


def load_evaluator(name: str, dry_run: bool) -> Evaluator:
    if not dry_run:
        [evaluator] = registry.load([name])
        return evaluator
    cls = registry.factory(name)
    if not (isinstance(cls, type) and issubclass(cls, JudgeEvaluator)):
        raise UsageError(f"--dry-run needs a judge evaluator; {name!r} is not one")
    judge: JudgeEvaluator = cls(_Offline(), Settings())
    return judge  # type: ignore[return-value]  # JudgeEvaluator subclasses are Evaluators


def _call(call: JudgeCall) -> dict[str, Any]:
    return {
        "model": call.response_model or call.request_model,
        "server": f"{call.server_address}:{call.server_port}",
        "input_tokens": call.input_tokens,
        "cached_tokens": call.cache_read_tokens,
        "output_tokens": call.output_tokens,
        "finish_reason": call.finish_reason,
        "error_type": call.error_type,
        "ms": round((call.end_ns - call.start_ns) / 1e6),
    }


async def evaluate(
    evaluator: Evaluator, record: Mapping[str, Any], n: int, dry_run: bool
) -> dict[str, Any]:
    out: dict[str, Any] = {"id": record.get("id", n), "evaluator": evaluator.name}
    interaction = interaction_from_record(record, n)
    truncated = False
    if evaluator.max_chars is not None:
        interaction, truncated = truncate(interaction, evaluator.max_chars)
    if dry_run:
        assert isinstance(evaluator, JudgeEvaluator)
        out["content"] = evaluator.content(interaction)  # exactly what would be sent
        return out | {"truncated": truncated}

    # As in the runner: a timeout of 0 means LLM_EVAL_TIMEOUT_S.
    timeout_s = evaluator.timeout_s if evaluator.timeout_s > 0 else Settings().timeout_s
    start = time.perf_counter()
    with recording() as calls:
        try:
            result = await asyncio.wait_for(evaluator.evaluate(interaction), timeout_s)
            error_type = result.error_type
        except TimeoutError:
            result, error_type = None, semconv.ERROR_TIMEOUT
        except Exception as exc:  # the class name only, as in the service
            result, error_type = None, type(exc).__name__
    shown: dict[str, AttributeValue] = {}
    if result is not None:
        # What the service would emit: the explanation and attributes go through the sanitizer.
        raw: dict[str, AttributeValue] = dict(result.attributes)
        if result.explanation is not None:
            raw["explanation"] = result.explanation
        shown, _ = sanitize(raw)
    return out | {
        "label": result.label if result else None,
        "score": result.score if result else None,
        "explanation": shown.pop("explanation", None),
        "attributes": shown,
        "error_type": error_type,
        "truncated": truncated,
        "ms": round((time.perf_counter() - start) * 1000),
        "judge_calls": [_call(c) for c in calls],
    }


async def run_all(args: argparse.Namespace, records: list[dict[str, Any]]) -> int:
    evaluator = load_evaluator(args.evaluator, args.dry_run)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(n: int, record: Mapping[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await evaluate(evaluator, record, n, args.dry_run)

    results = await asyncio.gather(*(one(n, r) for n, r in enumerate(records)))
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return EXIT_ERRORS if any(r.get("error_type") for r in results) else EXIT_OK


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-eval-judge",
        description="Run an evaluator, usually the judge, on interactions from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1] if __doc__ else None,
    )
    p.add_argument("-i", "--input", help="the user's message in the turn")
    p.add_argument("-o", "--output", help="the response to evaluate")
    p.add_argument(
        "-c",
        "--context",
        action="append",
        default=[],
        metavar="ROLE:TEXT",
        help="an earlier message, user or assistant; repeat in order",
    )
    p.add_argument("--jsonl", metavar="PATH", help="one interaction per line; - for stdin")
    p.add_argument("-e", "--evaluator", default="relevance", help="default: relevance")
    p.add_argument("--env-file", type=Path, help="default: ./.env, when it exists")
    p.add_argument("--dry-run", action="store_true", help="print what would be sent, masked")
    p.add_argument("--concurrency", type=int, default=4, help="with --jsonl; default: 4")
    p.add_argument("--pretty", action="store_true", help="indent the JSON output")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    env_file = args.env_file or Path(".env")
    if args.env_file is not None and not env_file.is_file():
        print(f"error: {env_file} not found", file=sys.stderr)
        return EXIT_USAGE
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    try:
        records = read_records(args)
        return asyncio.run(run_all(args, records))
    except (UsageError, registry.EvaluatorLoadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
    except Exception as exc:  # configuration: a missing model or API key, a bad setting
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
    return EXIT_USAGE


def run() -> None:
    sys.exit(main())


if __name__ == "__main__":
    run()
