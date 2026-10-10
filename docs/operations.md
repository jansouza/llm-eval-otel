# Operations

## Running in production

- **Backpressure.** The service returns 200 as soon as spans are queued. A full queue returns 429
  with `Retry-After`, and the Collector's exporter retries (keep `retry_on_failure` and
  `sending_queue` on). Invalid payloads return 400 and are not retried.
- **Losses on crash.** The queue lives in memory, so a crash loses what was queued. The original
  spans still reach your backend unchanged.
- **Shutdown.** On SIGTERM the service stops accepting data, drains the queue, then the judge
  lane, then the `jev_judge` lane, all within 30 s, then flushes the SDK. Judge evaluations still
  pending are counted in `llm_eval.evaluations.dropped` with `llm_eval.drop.reason=shutdown`.
- **The lanes don't push back.** `/readyz` and the 429s look only at the main queue. A
  backed-up judge or Jev drops evaluations (`lane_full`) instead of slowing the Collector down,
  and one lane backing up doesn't affect the other.
- **Scaling.** Scale with replicas: regex work is bound by the GIL, so more workers do not add
  throughput. Deduplication is per instance, so a Collector retry that lands on a different
  replica can be evaluated twice. That only happens on retries. The Collector's `loadbalancing`
  exporter (`routing_key: traceID`) would pin each trace to one replica, but as of Collector
  0.161.0 it only speaks OTLP/gRPC, which this version does not accept.
- **Logs.** The service logs to stderr, and the volume at `INFO` doesn't grow with traffic.
  Besides startup, configuration and shutdown, it logs one summary per
  `LLM_EVAL_LOG_SUMMARY_INTERVAL_S` (60 s by default; also once more at shutdown):

  ```
  last 60s: received=11400 queued=11380 skipped=duplicate:20 rejected=none | evaluations:
  pii_detection=11380 (fail:312,pass:11068) relevance=569 (error:3,fail:66,pass:500)
  | judge_tokens=812345 | queue=12/10000 llm_judge=3/1000 jev_judge=0/1000
  ```

  `judge_tokens` adds up both judges; the `jev_judge` lane appears only when a `jev_*` check is on.

  The summary is a `WARNING` when the interval had rejected exports, evaluation errors or
  drops. State changes get one line each when they happen: the queue filling up (429s) and
  accepting again, an evaluator failing (3 errors in a row) and recovering, the judge lane
  starting and stopping to drop (`lane_full`, `budget`), and the same for the `jev_judge` lane. `DEBUG` adds a line per export batch,
  rejected export (peer address and status), interaction and evaluation (label or error,
  score, duration, judge calls and tokens). Lines carry counts, TraceID, SpanID and
  `service.name`, never content or explanations. `DEBUG` costs a few lines per span, so keep
  it for troubleshooting.
- **Container.** The image runs as a non-root user (uid 10001) and works with a read-only root
  filesystem. It exposes `GET /healthz` (liveness) and `GET /readyz`, which fails while the queue
  is above 90% or during shutdown.

## Performance

Measured with `uv run python tools/load_test.py --spans 3000 --text-kb 10` on 2026-09-30,
version 0.2.0. The machine had 8 cores, and the test ran one process with 4 workers. Each span
carried 10 KB of text with some PII and credentials mixed in. Each evaluator got 10 KB of what
it reads; `system_prompt_leak` compared 10 KB of instructions with 10 KB of output. The
throughput run used the real HTTP server with the default evaluators and counted exported
events at a local fake Collector.

| Measure | Target | Measured |
| --- | --- | --- |
| `pii_detection` p99, 10 KB | ≤ 5 ms | 2.04 ms (p50 1.55 ms) |
| `secret_detection` p99, 10 KB | ≤ 5 ms | 3.74 ms (p50 1.90 ms) |
| `refusal` p99, 10 KB | ≤ 5 ms | 0.29 ms (p50 0.18 ms) |
| `system_prompt_leak` p99, 10 KB + 10 KB | ≤ 5 ms | 2.31 ms (p50 1.83 ms) |
| `output_format` p99, 10 KB | ≤ 5 ms | 0.20 ms (p50 0.14 ms) |
| Throughput per process, `pii_detection` + `secret_detection` | ≥ 100 spans/s | 194 spans/s |

Typical chat spans carry less than 10 KB of new content, so expect more throughput in practice.
Add replicas to scale.

With `relevance` on, measured on 2026-10-01 with version 0.3.0 on the same machine, with
`tools/load_test.py --spans 3000 --text-kb 10 --judge slow|down` (3 runs each, `relevance` at
0.05, so about 140 judge evaluations per run):

| Heuristics' throughput | Spans/s | 429s |
| --- | --- | --- |
| No judge | 192, 195, 195 | 0 |
| Judge answering after 5 s per call | 189, 191, 189 | 0 |
| Judge unreachable (connection refused) | 176, 175, 179 | 0 |

The judge never blocks the heuristics: its calls wait in their own lane, and the main queue
never filled. What it costs is CPU on the same process, about 6.5 ms per evaluation of 10 KB
(3.2 ms of it masking) when the judge answers, and about 18 ms when it is unreachable,
because the SDK tries twice and builds the error each time. With a slow judge most of that work
happens after the run's last span, so it barely shows; with the judge down it all lands within
the run, about 9% of the heuristics' throughput at a 5% sample of 10 KB spans. In the slow runs,
8 concurrent calls of 5 s could not keep up with 140 evaluations: the shutdown's 30 s drain
finished 72 and counted 67 as `shutdown` drops, as designed.

With the four `jev_*` checks on, measured on 2026-10-06 with version 0.4.0 on a different
machine, with `tools/load_test.py --spans 3000 --text-kb 10 --jev fake|down` (one run each; each
check at 0.1, so about 300 requests of four questions per run, against the fake server):

| Heuristics' throughput | Spans/s | 429s | Jev drops |
| --- | --- | --- | --- |
| No judge (2,000 spans, same machine) | 198 | 0 | n/a |
| Jev answering at once | 170 | 0 | 0 |
| Jev unreachable (connection refused) | 173 | 0 | 0 |
| `--judge down --judge-rate 0.5 --jev fake --jev-rate 0.5` (2,000 spans) | 101 | 0 | 0 |

Each Jev request costs about 8 ms of CPU in the service for a 10 KB span: masking the state
once, the SDK's encoding and decoding, and four events and five spans. The last row is a stress
case, five times the default rates with the OpenAI judge down: no Jev-as-a-Judge check was dropped while
`relevance` failed on every call, which is what the separate lane is for.
