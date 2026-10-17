# Talk outline: Fast Models, Slow Systems

Saturation testing LLM serving. 15 to 20 minutes. The numbers below come from
the saved runs in `runs/` (vLLM 0.30.0 on an M4 Pro, 16 output tokens, 60 s per
run, 10 s warmup, three repeats). Rebuild them with the command under "Charts" below.

## Thesis

On this Mac, vLLM behaves like a fixed pipe of about 190 output tokens per
second. It splits that pipe across every request it admits. When load goes past
the pipe, requests do not wait in a queue. Every request in the batch slows down
at the same time.

Nothing raises an alarm while this happens. There are no errors, the server's
`waiting` gauge reads 0, the harness verdict is PASS, and a closed-loop benchmark
reports a healthy peak.

**The audience should leave with:** "Errors 0, waiting 0, PASS, and p95 E2E was
8 seconds. My closed-loop benchmark had been acting as the admission controller
I do not have in production."

## What the saved data shows

| Offered req/s | Served req/s | Good req/s (TTFT ≤ 1 s, E2E ≤ 5 s) | p95 TTFT | p95 E2E | Unfinished at end | Growth |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.5 | 6.50 | 6.50 | 0.40 s | 2.35 s | 14 | 0 |
| 11 | 11.00 | 11.00 | 0.73 s | 4.39 s | 47 | 0 |
| 12 | 11.30 | 0.00 | 1.38 s | 8.30 s | 96 | +0.44/s |
| 13 | 10.22 | 0.00 | 2.67 s | 16.13 s | 211 | +2.56/s |
| 16.5 | 10.40 | 0.00 | 15.65 s | 29.65 s | 433 | +5.56/s |

- **Closed loop peak:** 12.8 req/s at concurrency 32, with p95 E2E 2.6 s. Open loop
  at 12 req/s gives p95 E2E 8.3 s and a growing backlog.
- **Overload lowers throughput.** Served rate falls from 11.3 to about 10.2 req/s.
  Token rate stays near 190 tok/s. The work becomes unfinished requests.
- **The server does not say so.** `waiting` is 0 in every run up to 13.5 req/s.
  It rises only at 14.5 (peak 54) and 16.5 req/s (peak 171). `running` peaks at
  246 and KV cache use at 2.6%, so memory is not the limit.
- **The cost per admitted request is linear.** p50 TPOT rises from 129 ms with
  about 13 requests running to 1.2 s with about 230 running. The fitted slope is
  5.3 ms per admitted request. 190 tok/s ÷ 16 tokens ≈ 11.9 req/s, which is the
  knee. See `tpot-vs-running.png`.
- **The harness is not the bottleneck.** Send lag p99 is at most 3.9 ms against a
  budget of about 6 ms or more. No request was clipped or dropped.

## Slide arc

| # | Min | Claim | Evidence | Still missing |
| --- | ---: | --- | --- | --- |
| 1 | 1 | Same Mac, two answers: 12.8 req/s at p95 2.6 s, or 12 req/s at p95 8.3 s with a growing backlog. | The two numbers. | Closed c=32 rerun in the same session as the open sweep (it ran on 2026-09-28; open ran on 2026-10-11). |
| 2 | 0.5 | Setup: 8B 4-bit model, M4 Pro, 16 tokens, `ignore_eos`. | Text. | Launch flags beyond the command are not recorded. |
| 3 | 1.5 | Closed loop fixes how many requests are in the system, so latency cannot run away. Real users arrive open loop. | Diagram. | |
| 4 | 2 | Served rate follows offered up to 11 req/s, then falls to about 10.2. | `request-knee.png`. | |
| 5 | 2 | Goodput falls off a cliff between 11 and 12 req/s. | `request-knee.png`, three SLO lines. | |
| 6 | 1 | Nothing raised an alarm: 0 errors, PASS, waiting 0. PASS only means the harness kept time. | Table. | |
| 7 | 1.5 | Stability is a slope: 11 req/s is flat at 47, 12 levels off near 96, 16.5 climbs 5.6/s. | `outstanding-work.png`. | Five-minute runs at 11 and 12 req/s. |
| 8 | 2 | The queue is inside the batch. About 5.3 ms of TPOT per admitted request predicts the knee. | `tpot-vs-running.png`. | `max_num_seqs` from the server log. |
| 9 | 1 | Closed loop capped the batch, so it looked better. Little's Law: 32 workers ÷ 2.6 s ≈ 12.4 req/s. | Same chart, closed points. | |
| 10 | 3 | Live demo on the mock. | Terminal. | |
| 11 | 2 | A fix, stated as a prediction: cap the batch (`--max-num-seqs 32`). | Hero 4. | Not run yet. |
| 12 | 1.5 | The method in six steps. | docs/methodology-review.md. | |
| 13 | 1 | What I do not know: 16 tokens, one machine, uniform arrivals, the odd closed curve. | Text. | |
| 14 | 0.5 | Takeaways. | | |

## Charts

Run `.venv/bin/python analysis/analyze.py runs/vllm-metal-open-sweep/* runs/vllm-metal-closed-sweep/* -o runs/talk-report`.

| Chart | Slide | Use |
| --- | --- | --- |
| `request-knee.png` | 4, 5 | Served and goodput against offered rate. Closed-loop peak as a line. |
| `outstanding-work.png` | 6, 7 | Requests sent but not finished at 11, 12, 16.5 req/s, with the server `waiting` gauge dashed. |
| `tpot-vs-running.png` | 8, 9 | Per-token time against admitted requests. Closed-loop runs as diamonds. |
| `send-lag.png` | 6 (one line) | Harness kept its schedule. Keep it as a backup slide. |
| Hero 4 (not built) | 11 | Goodput before and after the batch cap. Needs the runs in the next section. |

The closed-loop diamonds in `tpot-vs-running.png` sit below the open-loop line
and use workers as the x value, because the closed sweep has no server gauges.
They also come from a different session. Say so, or rerun them first.

## Live demo (about 2.5 minutes, nothing new to build)

Before the talk: `cargo build --release`, terminal font at 30 pt.

```bash
./target/release/mock 4 0.05 8123      # prints: capacity = 4/0.05 = 80.0 req/s
```

1. Closed loop, 4 workers, 10 s. Expect about 77.5 req/s, p95 about 53 ms, PASS.
   Ask the room: "Ship it?"
   `./target/release/satlab --config examples/mock-closed.json --runs-dir runs/demo`
2. Open loop at 100 req/s, 10 s. Expect about 78 req/s served, p95 about 2.7 s,
   and still PASS.
   `./target/release/satlab --config config.json --rate 100 --duration 10 --runs-dir runs/demo`
3. Show a chart made before the talk (unfinished work, 100 req/s). Do not plot live.
   `.venv/bin/python analysis/analyze.py runs/demo/* -o runs/demo-report`
4. Bridge to vLLM. The mock is a bank with 4 tellers: overload shows as waiting
   before the first token (TTFT). vLLM is one teller serving everyone at once:
   overload shows as slower tokens (TPOT). The batch cap turns vLLM into the bank.

Keep a recording of the demo in case the live run fails.

## Next runs, in order of value for effort

Do the first three. Each needs the Metal server running and a new `--runs-dir`,
so the saved evidence stays untouched. Commands are in
[the runbook](docs/vllm-metal-runbook.md). Set `NOTES` as shown there.

1. **Same session, same flags.** Run closed c=16, 32, 48 and the open sweep
   back to back. Then three 300 s runs at 11 and at 12 req/s. This removes the
   "different days" objection and tests stability.
   Hero 1 and slide 1 depend on it.
2. **Cap the batch.** Restart with `--max-num-seqs 32` and run the open sweep at
   `--rates 9,11,12,13,16.5` with `--name vllm-metal-seqs32`. This tests the
   explanation and can fail. Predictions to write down first:
   - `waiting` becomes nonzero from about 12 req/s, so the queue is visible.
   - TPOT stays bounded. Closed c=32 gave 108 ms. The open-loop line gives about
     230 ms at 32. The run tells which one holds, and so whether prefill
     interleaving matters.
   - Served rate stays near 11 to 12 req/s. The cap does not add capacity.
   - Goodput at 12 req/s may recover, because a short wait plus bounded TPOT
     can meet the 5 s SLO. At 13 req/s and above the wait grows without bound,
     so goodput falls again. Only load shedding bounds latency in open loop.
3. **Poisson arrivals.** Uniform spacing is the best case for the server, so the
   knee should move left. Use seed 0, so each repeat draws its own schedule and
   records the seed.
   ```bash
   .venv/bin/python -c "import json;d=json.load(open('smoke.json'));d['arrival']='poisson';json.dump(d,open('/tmp/metal-poisson.json','w'))"
   .venv/bin/python scripts/sweep.py sweep --base-config /tmp/metal-poisson.json \
     --name vllm-metal-poisson --runs-dir runs/vllm-metal-poisson --rates 9,11,12 --server-notes "$NOTES"
   ```
   The Poisson code has one unit test and no integration test yet.

Small enabler, not built: a `max_tokens` config field (`src/target.rs`, now fixed
at 16) so one pair of runs can use 128 tokens.

Name these as next steps on the last slide, and do not build them now: mixed
prompt and output lengths, a mock pause that shows coordinated omission, KV
pressure (not the limit here), prefix caching, an NVIDIA run, retry storms, and
load shedding.

## Objections and one-line answers

| Objection | Answer |
| --- | --- |
| 16 tokens is not realistic. | It is a short-completion workload that stresses the scheduler. TPOT growing with batch size does not depend on length. Show the 128-token pair once it exists. |
| Metal is not production. | The limit comes from the vLLM scheduler. The method carries over. The numbers do not. |
| Uniform arrivals are optimistic. | Agreed, they are the best case. Show the Poisson runs. |
| 60 s is a transient. | Show the 300 s runs and the slope. |
| Closed and open ran on different days. | Show the same-session rerun. |
| You picked the SLO to get a cliff. | Three SLOs are plotted. The cliff is sharp because p50 is close to p95 under uniform arrivals. |
| The throughput drop is a window artifact. | Partly. The claim is narrower: overload adds no throughput and turns work into unfinished requests. |
| Your load generator is the bottleneck. | Send lag p99 is at most 3.9 ms. No request was clipped or dropped. |
| Your Little's Law check is wrong. | It uses one population: requests that finished in the window. Stable runs agree (47.1 vs 47.1 at 11 req/s). Overload shows `L` above `λ·W`. |

## Known gaps

- Every run has verdict PASS. That verdict says the harness kept time. It does
  not say the latency target was met. Goodput is the measure for that.
- The closed curve is odd and unexplained: c=8 gives 2.9 req/s, c=16 gives 7.4.
  Do not put it on a slide before it is understood.
- vLLM's `max_num_seqs` is not recorded. `running` peaking at 246 is consistent
  with the default of 256. Confirm it in the server log.
- Closed c=32 has p50 TTFT 0.94 s and p95 0.95 s, so its goodput sits on the edge
  of the 1 s SLO. Small changes flip it.
