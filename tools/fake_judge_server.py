"""A deterministic judge server, for tests, the demo and the load test.

No model and no API key. Two APIs on one port:

``POST /v1/chat/completions`` (OpenAI-compatible) answers the relevance judge with
``{"reason", "score"}`` computed from the conversation the service sends:

- 5 when the response shares a content word (4+ letters) with the request;
- 4 when it shares none but is short (12 words or fewer), like "Noted." or a refusal;
- 1 when it shares none and is long: an answer about something else.

Markers in the user message force the edge cases: ``FAKE_JUDGE:refuse``,
``FAKE_JUDGE:content_filter``, ``FAKE_JUDGE:length``, ``FAKE_JUDGE:invalid`` and
``FAKE_JUDGE:no_usage``. With ``response_format`` ``json_schema`` or ``json_object`` the
reply is bare JSON; with none, it comes in a Markdown fence, as small local models often do.

``POST /v1/systemone`` (TypeSafe's System One, the Jev checks) answers each question about
the ``state``: a score question with the same rating as above, placed on its levels; a noul
question by keywords, chosen by what the question asks (a refusal, offensive content, an
injection; see ``NOUL_KEYWORDS``). ``GET /v1/models`` lists one model. Markers in the state:
``FAKE_JUDGE:invalid`` (an answer missing), ``FAKE_JUDGE:no_usage``, ``FAKE_JEV:overloaded``
(529), ``FAKE_JEV:rate_limited`` (429) and ``FAKE_JEV:422`` (a validation error that echoes
the state, as the real one may).

    python tools/fake_judge_server.py --port 8080
    python tools/fake_judge_server.py --delay 5            # a slow judge
    python tools/fake_judge_server.py --reject-json-schema # a server without strict mode
    python tools/fake_judge_server.py --record /data/judge-requests.jsonl

Standard library only, so it runs in the service image.
"""

import argparse
import json
import re
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SHORT_ANSWER_WORDS = 12
STOPWORDS = frozenset({
    "para", "como", "qual", "quais", "quero", "pode", "posso", "sobre", "isso", "essa", "esse",
    "este", "esta", "minha", "meus", "minhas", "voce", "voces",
    "what", "with", "that", "this", "from", "your", "have", "does", "please", "about",
})  # fmt: skip
_CONVERSATION = re.compile(r"<conversation>\s*(.*?)\s*</conversation>", re.DOTALL)
_WORD = re.compile(r"\w+")


def content_words(text: str) -> set[str]:
    plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return {w for w in _WORD.findall(plain) if len(w) >= 4 and w not in STOPWORDS}


def rate(user_content: str) -> tuple[int, str]:
    match = _CONVERSATION.search(user_content)
    try:
        conversation = json.loads(match.group(1)) if match else None
    except ValueError:
        conversation = None
    return rate_conversation(conversation)


def rate_conversation(conversation: Any) -> tuple[int, str]:
    if not isinstance(conversation, dict):
        return 3, "fake judge: no conversation found"
    request = " ".join(conversation.get("request") or [])
    response = " ".join(conversation.get("response") or [])
    shared = content_words(request) & content_words(response)
    if shared:
        return 5, f"fake judge: the response shares {len(shared)} content words with the request"
    if len(response.split()) <= SHORT_ANSWER_WORDS:
        return 4, "fake judge: short answer with no content word from the request"
    return 1, "fake judge: long answer with no content word from the request"


def completion(request: dict[str, Any], seen_prompts: set[str]) -> tuple[int, dict[str, Any]]:
    messages = request.get("messages") or []
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    response_format = (request.get("response_format") or {}).get("type")

    content: str | None
    refusal = None
    finish_reason = "stop"
    if "FAKE_JUDGE:refuse" in user:
        content, refusal = None, "I can't help with that."
    elif "FAKE_JUDGE:content_filter" in user:
        content, finish_reason = None, "content_filter"
    elif "FAKE_JUDGE:length" in user:
        content, finish_reason = '{"reason": "the response', "length"
    elif "FAKE_JUDGE:invalid" in user:
        content = json.dumps({"score": 9})
    else:
        score, reason = rate(user)
        content = json.dumps({"reason": reason, "score": score})
        if response_format not in ("json_schema", "json_object"):
            content = f"```json\n{content}\n```"

    body: dict[str, Any] = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model", "fake-judge"),
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
    }
    if "FAKE_JUDGE:no_usage" not in user:
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4 + 1
        # The fixed system prompt is "cached" from its second use on.
        cached = len(system) // 4 if system in seen_prompts else 0
        seen_prompts.add(system)
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(content or "") // 4 + 1,
            "total_tokens": prompt_tokens + len(content or "") // 4 + 1,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
    return 200, body


# Noul questions, by a word in their instructions: the state field to read and what says yes.
NOUL_KEYWORDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "refuse": ("response", (
        "não posso", "nao posso", "não consigo ajudar", "não é possível ajudar",
        "i can't", "i cannot", "i won't", "unable to help",
    )),
    "offensive": ("response", (
        "idiota", "imbecil", "burro", "cala a boca", "idiot", "stupid", "moron", "shut up",
    )),
    "system prompt": ("request", (
        "ignore", "esqueça suas instruções", "repita suas instruções", "system prompt",
        "suas instruções", "your instructions", "you are now", "agora você é",
    )),
}  # fmt: skip
NOUL_YES = 0.9
NOUL_NO = 0.1


def noul(question: dict[str, Any], state: Any) -> float:
    instructions = str(question.get("instructions") or "").lower()
    for topic, (field, words) in NOUL_KEYWORDS.items():
        if topic in instructions:
            texts = state.get(field) if isinstance(state, dict) else None
            text = " ".join(texts or []).lower()
            return NOUL_YES if any(word in text for word in words) else NOUL_NO
    return NOUL_NO


def score(question: dict[str, Any], state: Any) -> dict[str, Any]:
    """The 1-5 rating on the question's levels: 0.8 on that level, the rest on its neighbors."""
    levels = question.get("criteria") or [""]
    top = len(levels) - 1
    rating, _ = rate_conversation(state)
    level = round((rating - 1) / 4 * top)
    neighbors = [n for n in (level - 1, level + 1) if 0 <= n <= top]
    probabilities = {n: 0.0 for n in range(top + 1)}
    probabilities[level] = 0.8 if neighbors else 1.0
    for n in neighbors:
        probabilities[n] = 0.2 / len(neighbors)
    return {
        "type": "score",
        "score": round(sum(n * p for n, p in probabilities.items()), 6),
        "confidence": probabilities[level],
        "legend": {str(n): levels[n] for n in range(top + 1)},
        "probabilities": {str(n): p for n, p in probabilities.items()},
    }


def system_one(request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    state = request.get("state")
    questions = request.get("questions") or {}
    marked = json.dumps(state, ensure_ascii=False)
    if "FAKE_JEV:overloaded" in marked:
        return 529, {"error": {"type": "overloaded_error", "message": "Overloaded"}}
    if "FAKE_JEV:rate_limited" in marked:
        return 429, {"error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}}
    if "FAKE_JEV:422" in marked:
        # A real validation error can quote the input: the service must never log it.
        detail = [{"loc": ["body", "state"], "msg": "invalid state", "input": state}]
        return 422, {"detail": detail}
    answers: dict[str, Any] = {}
    for qid, question in questions.items():
        if question.get("type") == "score":
            answers[qid] = score(question, state)
        else:
            answers[qid] = {"type": "noul", "noul": noul(question, state)}
    if "FAKE_JUDGE:invalid" in marked and answers:
        del answers[next(iter(answers))]
    body: dict[str, Any] = {"model": request.get("model", "fake-jev"), "answers": answers}
    input_tokens = len(json.dumps(request, ensure_ascii=False)) // 4 + 1
    body["usage"] = (
        {}
        if "FAKE_JUDGE:no_usage" in marked
        else {"input_tokens": input_tokens, "output_tokens": 2 * len(answers)}
    )
    return 200, body


MODELS = {
    "models": [
        {"name": "jev-1.13.0", "description": "fake Jev", "release_date": "2026-09-15"},
        {"name": "jev-latest", "description": "fake Jev", "release_date": "2026-09-15"},
    ]
}


class FakeJudgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        delay_s: float = 0.0,
        reject_json_schema: bool = False,
        record_path: str | None = None,
    ) -> None:
        super().__init__(address, Handler)
        self.delay_s = delay_s
        self.reject_json_schema = reject_json_schema
        self.record_path = record_path
        self.requests: list[dict[str, Any]] = []  # every request body, for tests
        self.lock = threading.Lock()
        self.seen_prompts: set[str] = set()

    @property
    def root_url(self) -> str:
        """For the TypeSafe SDK, which adds /v1/... itself."""
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}"

    @property
    def base_url(self) -> str:
        """For the openai SDK."""
        return f"{self.root_url}/v1"


class Handler(BaseHTTPRequestHandler):
    server: FakeJudgeServer

    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        if status in (429, 529):
            self.send_header("retry-after-ms", "10")  # keeps the SDK's one retry quick
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, MODELS)
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        jev = self.path == "/v1/systemone"
        if not jev and not self.path.endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        request = json.loads(raw)
        with self.server.lock:
            self.server.requests.append(request)
            if self.server.record_path:
                try:
                    with open(self.server.record_path, "a") as out:
                        out.write(json.dumps(request, ensure_ascii=False) + "\n")
                except OSError as exc:  # recording is for tests; keep answering without it
                    print(f"cannot record requests: {exc}", flush=True)
                    self.server.record_path = None
            if jev:
                status, body = system_one(request)
            else:
                status, body = completion(request, self.server.seen_prompts)
        if self.server.delay_s:
            time.sleep(self.server.delay_s)
        response_format = (request.get("response_format") or {}).get("type")
        if not jev and self.server.reject_json_schema and response_format == "json_schema":
            status, body = (
                400,
                {
                    "error": {
                        "message": "response_format json_schema is not supported",
                        "type": "invalid_request_error",
                    }
                },
            )
        self._send(status, body)

    def log_message(self, *args: Any) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds before each answer")
    parser.add_argument("--reject-json-schema", action="store_true")
    parser.add_argument("--record", help="append every request body to this JSONL file")
    args = parser.parse_args()
    server = FakeJudgeServer(
        (args.host, args.port),
        delay_s=args.delay,
        reject_json_schema=args.reject_json_schema,
        record_path=args.record,
    )
    print(f"fake judge listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
