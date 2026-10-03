"""A deterministic OpenAI-compatible judge server, for tests, the demo and the load test.

No model and no API key. ``POST /v1/chat/completions`` answers the relevance judge with
``{"reason", "score"}`` computed from the conversation the service sends:

- 5 when the response shares a content word (4+ letters) with the request;
- 4 when it shares none but is short (12 words or fewer), like "Noted." or a refusal;
- 1 when it shares none and is long: an answer about something else.

Markers in the user message force the edge cases: ``FAKE_JUDGE:refuse``,
``FAKE_JUDGE:content_filter``, ``FAKE_JUDGE:length``, ``FAKE_JUDGE:invalid`` and
``FAKE_JUDGE:no_usage``. With ``response_format`` ``json_schema`` or ``json_object`` the
reply is bare JSON; with none, it comes in a Markdown fence, as small local models often do.

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
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host!s}:{port}/v1"


class Handler(BaseHTTPRequestHandler):
    server: FakeJudgeServer

    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
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
            status, body = completion(request, self.server.seen_prompts)
        if self.server.delay_s:
            time.sleep(self.server.delay_s)
        response_format = (request.get("response_format") or {}).get("type")
        if self.server.reject_json_schema and response_format == "json_schema":
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
