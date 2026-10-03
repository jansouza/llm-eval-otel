"""Mask PII and credentials before content goes to the judge.

Each finding is replaced by its type (``[CPF]``, ``[EMAIL]``, ``[SECRET]``), which keeps the
text's structure for the judgment. Names and addresses are not detected by the regexes and
go through unmasked.
"""

from llm_eval_otel.evaluators.pii import find_pii
from llm_eval_otel.evaluators.secrets import find_secrets

SECRET_PLACEHOLDER = "[SECRET]"


def mask(text: str) -> str:
    spans = sorted(
        [
            *((f.start, f.end, f"[{f.kind.upper()}]") for f in find_pii(text)),
            *((f.start, f.end, SECRET_PLACEHOLDER) for f in find_secrets(text)),
        ],
        key=lambda s: (s[0], -s[1]),
    )
    # Merge overlaps (a PII type and a secret on the same stretch); the first one names it.
    merged: list[tuple[int, int, str]] = []
    for start, end, label in spans:
        if merged and start < merged[-1][1]:
            first_start, first_end, first_label = merged[-1]
            merged[-1] = (first_start, max(first_end, end), first_label)
        else:
            merged.append((start, end, label))
    out: list[str] = []
    position = 0
    for start, end, label in merged:
        out += [text[position:start], label]
        position = end
    out.append(text[position:])
    return "".join(out)
