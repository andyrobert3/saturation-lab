# Methodology review

Deep review of the benchmark method behind the talk "Fast Models, Slow Systems:
Saturation Testing LLM Serving". The work runs as 12 tasks, one at a time. Each
task ends with a step report in the work log below.

Evidence rule: every number comes from the primary run files (`config.json`,
`records.csv`, `stream_events.csv`, `summary.json`, `server_metrics.csv`), not
from `sweep.csv` or the charts. The source of each number is named.

## Card: questions before a benchmark

Status: KNOWN = answered by saved data. PARTIAL = answered for the test, not
for production. UNKNOWN = the data cannot answer it.

| # | Question | Answer for this benchmark | Evidence | Status |
|---|---|---|---|---|
| 1 | Which decision does the result change? | Teaching only. The benchmark shows the method on one Mac mini (M4 Pro, 64 GB) that serves a local LLM on Metal. No capacity or purchase decision depends on it, so the result must be correct and repeatable, not representative of production. | User, 2026-10-17 | KNOWN |
| 2 | What is the SLO: which latency, which percentile, which fraction of requests? | Three tiers for a person who sends a request and waits for the 16-token answer, met by 95% of requests: fast = TTFT ≤ 0.5 s and E2E ≤ 1 s; medium = TTFT ≤ 1 s and E2E ≤ 5 s; slow = TTFT ≤ 2 s and E2E ≤ 10 s. Basis: response-time limits of 1 s (flow of thought) and 10 s (attention) from Nielsen (1993), after Miller (1968). Confirmed by the user. | User, 2026-10-17 (Steps 1b, 1c) | KNOWN |
| 3 | Open or closed load, and why? | Both, as two personas. People in a chat send independently of replies: open loop. An agent pool with a fixed number of workers waits for each reply: closed loop. Saved data: closed 27 runs (concurrency 1–64), open 27 runs (6.5–16.5 req/s). | User, 2026-10-17; `config.json` in all 54 runs | KNOWN |
| 4 | How do requests arrive, and how bursty? | Uniform deadlines in all 27 open runs. The coefficient of variation of planned gaps is 0.00000. No Poisson or bursty run exists on vLLM yet; the mock has 12 Poisson and 12 gamma (CV 2) runs (Task 4). Real traffic (Task 5, Azure trace 2023): conversation gap CV 1.09 and index of dispersion 1.39 at 1 s (close to Poisson); code gap CV 13.15 and index of dispersion 13.2 at 1 s (on/off bursts). See "Workload". | `records.csv` `intended_ns` | PARTIAL |
| 5 | What is the shape of a request? | One fixed prompt, "Explain saturation in one sentence.": 7 tokens with the local tokenizer, which adds no BOS (the server's `prompt_tokens` is not saved). Output: exactly 16 tokens in 31,233 of 31,233 successful requests (`max_tokens` 16, `ignore_eos`, temperature 0). All requests are identical. Prefix caching is on by default in vLLM 0.30.0 (`vllm/config/cache.py:130`), but it caches only full KV blocks and never the last prompt token (`kv_cache_manager.py`). With the default 16-token block a 7-token prompt has no full block, so a cache hit is unlikely; vllm-metal can change the block size (`vllm_metal/platform.py:1005`), so this is not confirmed. New runs put a random 6-digit number before the prompt (8 tokens), so no two prompts share a prefix. Real requests (Task 5): median context 1,469 (code) and 1,020 (conversation) tokens; median output 13 and 129 tokens. See "Workload". | `config.json`; `records.csv`; `src/target.rs`; installed vLLM source; local tokenizer | PARTIAL |
| 6 | Where is the system boundary? | Client and server run on the same computer (`127.0.0.1:8000`, host `Matthews-Mac-mini.local`), so the client shares CPU and memory with the server; this is not measured. TTFT and E2E start at `actual_ns` (start of the HTTP attempt), not at `intended_ns` (the planned send). E2E ends at the last token event, not at stream close. Latency from the planned send is reported as `ttft_from_plan` and `e2e_from_plan` (Task 2). | `config.json` `target_url`, `provenance.hostname`; docs/measurement-contract.md | KNOWN |
| 7 | How do you know the system is in steady state? | It was not tested before measurement. Warmup is a fixed 10 s, runs are 60 s, 3 repeats. From the records: at 6.5–11 req/s unfinished work is flat from 10 s. At 12 req/s it rises from 61 (at 10 s) to 96 (at 60 s); the slope is +0.44/s over 10–60 s but −0.01/s over 35–60 s, so it levels off near 30 s. The 12 req/s window therefore contains a transient, and its served rate (11.3 req/s) can be partly a warmup effect. From 12.5 req/s the backlog still grows over 35–60 s. | `records.csv` `actual_ns`, `completed_ns` (table in Step 1) | PARTIAL |
| 8 | Environment | vLLM 0.30.0 (reported by the server). vllm-metal 0.30.0 (installed today). Model `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, revision not pinned. Apple M4 Pro, 64 GB, macOS 26.5.2 (read today, not at run time). Launch flags: recorded for the open sweep (`--max-model-len 8192 --gpu-memory-utilization 0.60`), not recorded for the closed sweep. `max_num_seqs`: not set in the launch command; the API-server default on this Mac is 256 (`vllm/engine/arg_utils.py:2756`; the class default 128 in `vllm/config/scheduler.py:44` does not apply), which agrees with the 246 running requests observed (A3). Model snapshot on disk: `241a666dad6cb93c8ff213d39a7f34a36bf26db4`, the only one. Cache setting, power state and harness revision (`git_sha` "unknown" in all 54 runs) are not known. The closed sweep ran on 2026-09-28 and the open sweep on 2026-10-11: different sessions, 13 days apart. Closed runs ran in ascending concurrency with repeats back to back; open runs used forward, reverse and shuffled passes. | `config.json` `target_notes`, `provenance`; `server_metrics.csv` `running`; Homebrew Cellar | PARTIAL (cache, power, the revision at run time and closed-sweep flags cannot be recovered; the user has no server log) |

## Why this model

The model is `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, revision
`241a666dad6cb93c8ff213d39a7f34a36bf26db4`: a dense decoder with 32 layers,
hidden size 4096, 32 query heads and 8 KV heads (GQA), a vocabulary of 128,256
tokens, 4-bit weights (group 64), 4.2 GiB on disk (local `config.json`).
It was chosen as a small, standard model that fits the Mac mini. The data
supports these reasons to keep it:

1. It is a fixture, not the subject. The talk is about the method; a plain,
   well-known model keeps the attention on the method.
2. It is dense, not mixture-of-experts. Each token uses all the weights, so the
   cost of a step depends only on the number of tokens in it. The prediction
   "one weight read per step at batch 1, 16.5 ms" gave 16.8 ms (E1). With
   expert routing, the service time would change from token to token.
3. The model is fast alone: 19 ms per token in pure MLX (E1). What is slow
   comes from the system around it: the queue, the batch and the serving
   layer (64 ms in MLX against 160 ms in vllm-metal at batch 8, Task 6).
4. Memory is never the limit: 4.2 GiB of weights and KV-cache use ≤ 2.6%. The
   knee comes from compute and scheduling only, one cause at a time.
5. The weights are open and the revision is pinned, so anyone can repeat the
   runs on a Mac with 16 GB or more.

Limits to state before someone asks:

1. The 4-bit kernels shape the step-time curve (32-row blocks; the jump at
   12–14 tokens in vllm-metal). An fp16 model or a CUDA server gives a
   different curve (threat 2). An optional check: E1 on an 8-bit copy of the
   model (pure MLX, about 5 minutes, no server).
2. A small model with short requests. Production chat uses larger models,
   prompts of about 1,000 tokens (Task 5) and KV-cache pressure; this test
   never reaches the KV-memory limit (threat 11).
3. An older model generation (July 2024). The queueing behaviour depends on
   the architecture class (dense GQA decoder), not on the generation.

Answer for the talk: "A plain, dense, open 8B model is the most boring choice
I could make. It runs fast alone on a Mac, so everything slow that you will see
comes from the system around it, not from the model."

## How the harness and the mock work (verified 2026-10-17)

Read from the code and checked against saved runs.

`satlab`, open loop (`src/driver.rs`, `src/target.rs`):
1. It makes the planned send times in advance: uniform (i ÷ rate), Poisson
   (exponential gaps) or gamma (gamma gaps with a set CV) (`ArrivalSchedule`).
2. It sleeps until each planned time. If it wakes late, it sends at once and
   keeps the planned time in `intended_ns`. Late sends therefore show as send
   lag; they do not move later requests.
3. If the in-flight cap is full, it records the request as `clipped` and does
   not queue it. A queue in the client would turn the open loop into a closed
   loop.
4. Each request runs in its own task. `actual_ns` is read inside that task just
   before the HTTP send, so task start delay counts as send lag.
5. The request body is fixed JSON with a random 6-digit prompt prefix.
6. Each SSE event that carries token IDs is stamped when the client parses it.
   TTFT ends at the first such event, E2E at the last one. `completed_ns` is
   after the stream closes.
7. The verdict PASS means the harness kept its schedule (late sends ≤ 1%, no
   clipped or dropped records, errors ≤ 1%). It says nothing about the SLO.

`satlab`, closed loop: N tasks; each sends, waits for the whole answer, then
sends again. `intended_ns` equals `actual_ns`, so there is no send lag to see.

The mock (`src/bin/mock.rs`):
1. Each request must take one of N slots (a tokio semaphore, which serves
   waiting tasks first in, first out). Waiting requests queue without limit.
   This is a first-come-first-served queue with N servers.
2. The response starts only after the request has a slot, so the client's TTFT
   and E2E include the wait.
3. It sends 16 tokens at slot start + D × (i+1) ÷ 16, then usage and `[DONE]`,
   then frees the slot. It ignores the prompt.
4. Measured with N = 4 and nominal D = 50 ms (`runs/mock-sweep/`):
   - E2E with no wait (70 req/s, uniform): median 51.5 ms, p1–p99 50.5–52.7 ms.
     So the service time is almost constant (Cs² ≈ 0.0001).
   - Throughput in overload (90 req/s): 78.12 req/s, so each slot is held for
     4 ÷ 78.12 = 51.2 ms. The timer wakes about 1 ms late on each request.
   - The real capacity is 78.1 req/s, not 80 req/s.

Model for the check: G/D/4, first come first served, D = 51.2 ms. The client
cannot see the wait directly. The wait of a request is its E2E minus the
no-wait E2E (51.5 ms).

Limits: the mock has no batching, so this check tests the harness and the
queueing theory, not vLLM. Client and mock share one computer.

## Threats to validity

Classes: internal (does the harness record what happened?), construct (does
the metric measure the claim?), external (does the result carry over?),
statistical (is the difference real and not chance or run order?).
Sequence: first, the threat that can change a slide claim the most.
Status: CONFIRMED = the pilot data shows the problem. OPEN = not tested yet.
RESOLVED = the pilot data shows no problem.

| # | Class | Threat | Slide claim at risk | Evidence so far | Test that removes it | Status |
|---|---|---|---|---|---|---|
| 1 | Construct, statistical | The 10 s warmup is too short near and above capacity. The batch still grows during the measurement window, so completions lag behind token output. | "Overload lowers throughput: served falls to about 10.2 req/s." "The knee is at 12 req/s, served 11.3." | Served rate in the late window (35–60 s) is about 12 req/s at every rate from 12 to 16.5 req/s (11.96, 11.76, 10.96, 10.92, 11.68, 12.80), and token rate ÷ 16 is 11.8–12.6. In the early window (10–35 s) served is 8.1–10.6 req/s. | 300 s runs at 12 and 13 req/s (8b). Choose the window from the unfinished-work slope, not a fixed 10 s. | CONFIRMED. Capacity is about 12 req/s (about 190 tok/s). Overload does not lower throughput in these runs. |
| 2 | External | The Metal backend, not the vLLM scheduler, can set the cost of a larger batch. TPOT grows almost in proportion to the batch from 1 to 8 workers, falls at 16, and 32 times the batch gives only 6.6 times the token rate (30 to 197 tok/s). | "The queue is inside the batch" as a general vLLM mechanism. "5.3 ms per admitted request." | Closed-loop TPOT and token rate (Step 1 context table). | Task 6: scheduler and model-runner source, engine log, `powermetrics`, `py-spy`. Run set 8f on CUDA. | CONFIRMED (Task 6). The step-time shape comes from the Metal stack: MLX's 4-bit matrix multiply (jump above 32) and vllm-metal GPU work at 2–12 sequences. It is not a general vLLM mechanism. |
| 3 | Statistical | The closed and open sweeps ran 13 days apart, and the closed sweep has no recorded launch flags. | "Same Mac, two answers: closed 12.8 req/s at p95 2.6 s, open 12 req/s at p95 8.3 s." | `config.json` dates and `target_notes`. | Closed and open runs in one session with recorded flags (8a). | OPEN |
| 4 | Construct, external | Uniform arrivals (Ca² = 0) and a fixed request (Cs² close to 0) give almost no queueing below capacity. This is why 12 req/s holds a fixed backlog of about 96 at capacity and why goodput drops all at once. Real arrivals vary. | "Goodput falls off a cliff between 11 and 12 req/s." The knee location. | Medium-tier attainment: 100% at 11 req/s, 0.7% at 12 req/s. | Task 4 (mock: done, harness agrees with theory). Task 5 (trace burstiness: done, see "Workload"). Run set 8c (Poisson on vLLM). | OPEN (needs 8c) |
| 5 | Construct | The TPOT-against-running fit compares run averages over a window in which the batch still grows (threat 1). The slope can be biased. | "+5.3 ms per admitted request" (`tpot-vs-running.png`). | Fit uses mean `running` over 10–60 s and p50 TPOT of all window requests. | Task 6a: rebuild engine steps from `stream_events.csv`; use the late window only. Engine stats log in 6b. | CONFIRMED (Task 6). The step time is a step function of the batch, not a line; do not use the slope. |
| 6 | Construct | "Waiting = 0" depends on vLLM's definitions and on a sample each second. A request counts as running from admission, including prefill. | "The server does not see the queue." | `server_metrics.csv`: waiting = 0 up to 13.5 req/s. | Queue-time histogram from `/metrics` (8a). Scheduler source (6a). | OPEN |
| 7 | External | `max_num_seqs` is not known. The installed default is 128, but up to 246 requests ran at once. | Where `waiting` starts. The `--max-num-seqs 32` prediction. | `vllm/config/scheduler.py:44`; `server_metrics.csv`. | Source read 2026-10-17: the API-server default is 256 on this Mac (`vllm/engine/arg_utils.py:2756`); see "Anomalies", A3. New runs pass `--max-num-seqs 256`. | RESOLVED (source) |
| 8 | Construct | Goodput counts good completions per second inside the window. The SLO says "met by 95% of requests". These are different measures. | All SLO slides. | At 12.5 req/s: slow-tier attainment 29.6% of requests, goodput 3.58 of 10.80 served req/s (33%). | Report attainment (share of requests intended in the window) as the SLO measure, with goodput as the rate. | OPEN (small analysis change) |
| 9 | Internal | Client and server share one computer. At 433 open streams the client can compete with the server for CPU. | Absolute latency at high load. | Not measured. | Client CPU in `powermetrics` (6b, 8a). Optional: client on a second computer. | RESOLVED for closed runs up to 64 streams (busiest core ≤ 38%, Task 6). OPEN for open runs with 433 streams. |
| 10 | Statistical | The closed sweep ran in ascending concurrency with repeats back to back, so time and temperature change with concurrency. The open sweep changed the order between passes. | The shape of the closed curve, including the anomaly at 8 to 16 workers. | Repeats agree closely (TPOT 151.8–152.0 ms at 8 workers), but an order effect cannot be excluded. | Shuffled closed order in 8a. | RESOLVED (Task 6: shuffled order within 6% of the pilot). |
| 11 | External | The workload is narrow: a 7-token prompt, 16 output tokens, the same prompt each time, KV use at most 2.6%. Prefill and KV memory never limit throughput. | Any claim that the numbers apply to chat or agent traffic. | `records.csv`, `server_metrics.csv`. | Task 5 (real lengths: done; real prompts are 128–184 times longer, see "Workload"). State the scope on a slide. | OPEN (scope statement) |
| 12 | Statistical | Three repeats give a weak min–max band. Rate steps near the knee are coarse (11, 12, 12.5 req/s). | The knee value. | 3 repeats per rate. | 300 s runs near the knee (8b); add 11.5 req/s if needed. | OPEN |
| 13 | Internal | Every request uses the same prompt, so the prefix cache could skip prefill. | TTFT values. | A 7-token prompt has no full 16-token block, so a hit is unlikely (Step 1b). | Random prompt prefix (decided). Prefix-cache counters in `/metrics`. | RESOLVED for new runs (Task 6: 0 prefix-cache hits). |
| 14 | Internal | The harness revision of the pilot runs is unknown. | Reproduction of the pilot. | `git_sha` "unknown" in all 54 runs. | New runs record `git describe --always --dirty` (done). | RESOLVED for new runs |
| 15 | Internal | Power and thermal state are unknown. | Absolute values; drift during a sweep. | Not recorded. | `powermetrics` thermal pressure; record the power adapter and Low Power Mode. | RESOLVED for Task 6 (pressure Nominal in all samples; AC power; `powermode` 0). OPEN for the pilot. |
| 16 | Internal | Late sends hide waiting (coordinated omission). | All open-loop latency. | Planned-send latency is at most 4.6 ms longer (0.09% at p99) (Step 2). | — | RESOLVED for the pilot |
| 17 | Internal | Network reads bunch token events, so the timestamps are wrong. | TPOT, ITL. | 0% of 42,000 gaps below 1 ms in four sample runs (6.5, 12, 16.5 req/s open; 32 workers closed). Each event carries one token. | — | RESOLVED for the pilot |
| 18 | Internal | Records are lost or clipped. | All values. | 0 clipped, 0 dropped; all 54 runs pass the contract checks (Step 2). | — | RESOLVED for the pilot |
| 19 | Internal | A client resource limit caps the load: macOS gives a shell 256 open files, so at most 246 requests are in flight and the rest fail at once with HTTP status 0. | Any overload result: served rate, backlog growth, `waiting`. | Task 8a: 13–16.5 req/s runs capped at 246 in flight, 5–27% errors. | satlab raises its open-file limit at start or stops (`ensure_file_limit`). Re-run (`8a-overload`). | RESOLVED (fix); the pilot passed with 437 in flight, so its shell had a higher limit. |
| 20 | Internal | The GPU clock dips 2–4% during a session while thermal pressure stays Nominal. | Any result at the edge of capacity (12 req/s). | Task 8: minimum 1506–1552 MHz in 9 of 27 runs, 1575–1577 MHz in the rest; the 8a-wire run at 12 req/s went into overload after a dip (A8). | Record the GPU clock in every run (`powermetrics`); state capacity with its margin. | OPEN (recorded, not controlled) |

Claims in `talk.md` that change because of this review:
- "Overload lowers throughput" is not supported (threat 1).
- "12 req/s with a growing backlog" is not correct; the backlog is steady at
  about 96 (Step 1, threat 1).
- "Consistent with the default of 256" is supported by the source after all (threat 7, A3).
- The goodput cliff is real for uniform arrivals; it is not shown for real
  arrivals (threat 4).

## Predictions

### Task 4: the mock against queueing theory (written 2026-10-17, before the runs)

Model: the mock is a G/D/4 queue. Four slots (a FIFO tokio semaphore), one fixed
service time, first come first served. The service time is the measured slot
time, D = 4 ÷ 78.12 req/s = 51.20 ms (overload throughput of the 90 req/s
runs in `runs/mock-sweep/`), not the nominal 50 ms. The nominal capacity of
80 req/s is therefore 78.1 req/s in practice.

Prediction method: `analysis/queue_theory.py predict`. Exact FCFS simulation
(start of request i = max(arrival i, start of request i−4 + D)), 300 simulated
60 s runs per point, wait of requests after the 10 s warmup. Bands are the 5th
and 95th percentiles of the per-run values. The simulator matches the exact
M/D/1 formula (test in `tests/test_open_sweep.py`). Kingman: the Allen–Cunneen
form, (Ca² + Cs²) ÷ 2 × the M/M/4 mean wait, with Cs² = 0.

| Arrivals (CV) | Rate | ρ | Mean wait ms, 5/50/95% | p95 wait ms, 5/50/95% | Kingman mean ms |
|---|---:|---:|---|---|---:|
| uniform (0) | 39 | 0.50 | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| uniform (0) | 55 | 0.70 | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| uniform (0) | 66 | 0.84 | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| uniform (0) | 74 | 0.95 | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Poisson (1) | 39 | 0.50 | 1.9 / 2.5 / 3.3 | 15.5 / 19.7 / 24.9 | 2.2 |
| Poisson (1) | 55 | 0.70 | 7.6 / 9.8 / 12.2 | 37.3 / 45.6 / 57.3 | 9.4 |
| Poisson (1) | 66 | 0.84 | 21.1 / 27.5 / 39.2 | 78.1 / 99.0 / 146.1 | 28.0 |
| Poisson (1) | 74 | 0.95 | 58.1 / 96.8 / 198.7 | 172.9 / 284.6 / 516.9 | 107.5 |
| gamma (2) | 39 | 0.50 | 19.5 / 24.8 / 32.6 | 82.5 / 102.5 / 138.5 | 8.9 |
| gamma (2) | 55 | 0.70 | 44.2 / 59.6 / 81.1 | 149.9 / 204.3 / 306.9 | 37.6 |
| gamma (2) | 66 | 0.84 | 89.7 / 132.2 / 214.1 | 261.3 / 407.9 / 724.0 | 112.1 |
| gamma (2) | 74 | 0.95 | 182.5 / 323.2 / 902.2 | 479.3 / 818.7 / 1823.6 | 429.9 |

Predictions to test:
1. Uniform arrivals: no queueing at any of the four rates. The measured wait
   (E2E minus the no-wait E2E) is only timer noise: mean ≤ 1 ms, p95 ≤ 2 ms.
2. Poisson and gamma: at least 30 of the 36 runs (all three arrival types) fall
   inside the 5–95% band for the mean wait, and the same for the p95 wait.
   By chance alone, about 3–4 runs fall outside.
3. Replay check: the same send times put through the ideal queue give a mean
   wait within 10% (or 2 ms) of the measured mean wait in each run.
4. At the same utilization, waiting increases in the sequence uniform <
   Poisson < gamma. At ρ = 0.5 the gamma mean wait is about 10 times the
   Poisson mean wait. This is the slide for Kingman's rule.
5. Kingman's approximation is close for Poisson (within about 20% of the
   simulated median at ρ ≥ 0.7). It is low for gamma with CV 2, because
   the approximation is weak for very bursty arrivals with a fixed service time.
6. Burstiness of the realized schedules: planned CV 0, 1 and 2; index of
   dispersion over 1 s windows near 0, 1 and 4. The send times keep the
   planned CV within 5%.
7. Harness risk: gamma arrivals with CV 2 put many requests at nearly the same
   instant. The send lag can then exceed its budget for more than 1% of
   requests, and the harness verdict can be FAILED. If so, the result is
   recorded, not hidden.

Results (2026-10-17, after the 36 runs; `runs/methodology_review/task4-mock/check/`):

| # | Prediction | Result | Status |
|---|---|---|---|
| 1 | Uniform: mean wait ≤ 1 ms, p95 ≤ 2 ms | Mean 0.18–0.44 ms, p95 0.78–1.00 ms in all 12 runs. This is the timer-noise floor. | HELD |
| 2 | ≥ 30 of 36 runs inside the 5–95% band | Mean wait: 32 of 36 (Poisson 10/12, gamma 10/12, uniform 12/12 within prediction 1). p95 wait: 34 of 36. Outside: Poisson 39 req/s repeats 2 and 3 slightly high (3.42 and 3.61 ms against 3.34 ms); gamma 39 req/s repeat 3 and 66 req/s repeat 3 low. The two low gamma runs had less bursty arrival samples (index of dispersion 3.40 and 2.79), and the replay of the same arrivals gives the same low values. | HELD |
| 3 | Measured mean within 10% (or 2 ms) of the replay of the same send times | 36 of 36. Measured is above the replay by 0.2–0.5 ms at low load and by 1–6% at ρ = 0.95 (largest: gamma 74 req/s repeat 1, 583.7 against 550.0 ms). A slightly longer service time under load can cause this; not tested. | HELD |
| 4 | Wait order uniform < Poisson < gamma; gamma about 10× Poisson at ρ = 0.5 | The order holds at each ρ. At ρ = 0.5 the medians are 0.25, 3.42 and 23.09 ms. The ratio is 6.8× (7.5× in the replay), not about 10×. The measured Poisson values at ρ = 0.5 are above the simulated median (noise floor and arrival samples). | PARTLY HELD |
| 5 | Kingman within about 20% for Poisson at ρ ≥ 0.7; low for gamma | Poisson: −4%, +2%, +11% at ρ 0.70, 0.84, 0.95. Gamma: low at ρ ≤ 0.84 (−64%, −37%, −15%), but high at ρ 0.95 (+33%). | PARTLY HELD |
| 6 | Planned CV 0, 1, 2; index of dispersion (1 s) near 0, 1, 4; sent CV within 5% of planned | Planned CV 0.00 / 0.99–1.07 / 1.89–2.18. Sent CV within 0.02 of planned for Poisson and gamma; uniform sent CV 0.02–0.06 (send jitter). Index of dispersion 0.00 / 0.57–1.33 / 2.79–5.16. | HELD |
| 7 | Gamma can fail the harness verdict | Did not happen. All 36 runs PASS. Late sends ≤ 0.11%, send-lag p99 ≤ 3.75 ms (budget ≥ 4.36 ms), largest lag 13.1 ms, 0 clipped, 0 dropped, 0 errors. | DID NOT HAPPEN |

### Task 6: E1–E3, the cause of the step-time jumps (written 2026-10-17, before the runs)

Before this was written, one smoke test of the E1 script ran (3 steps, batch 1
and 16 only, values discarded): step 18.9 and 96.4 ms. It shows nothing
about batch 8 or 48, which decide the predictions below.

Reference: pilot engine steps (section "Anomalies"), step time 24.8 / 42.0 /
78.5 / 160.4 / 112.2 / 113.7 / 212.4 ms at batch 1 / 2 / 4 / 8 / 16 / 32 / 48.

E1, pure MLX (`scripts/mlx_step_bench.py`, server stopped):
1. H1a holds if the MLX step time has the same jumps: step(8) ≥ 1.3 ×
   step(16), and step(48) ≥ 1.5 × step(32). If step(16) ≥ step(8), H1a is
   false and the cause is in vllm-metal, not in MLX.
2. The 4-bit linear layers alone take ≥ 70% of the full step at every batch,
   with the same jumps.
3. Step(1) is 17–25 ms (one read of 4.5 GB at 273 GB/s takes 16.5 ms).

E2, closed sweep in shuffled order (`runs/methodology_review/task6b/run.sh`):
4. H1c is false if the step times match the pilot (ascending order) within
   10% at the batch sizes that both have (1, 2, 4, 8, 16, 24, 32, 48, 64).
5. The batch size where the step time falls (between 8 and 16) is the same in
   E1 and E2, within one point of the grid (6, 8, 10, 12, 14, 16).
6. vLLM adds a fixed cost per step, not a cost per sequence: E2 step time
   minus E1 step time is about constant (within ±10 ms) across batch sizes.
   If the difference grows with the batch, per-request CPU work (H1b) is
   part of the slope.
7. `powermetrics`: GPU active residency ≥ 85% at 8 and at 32 workers
   (GPU bound). H1b holds instead if GPU residency is below 60% while the
   server CPU is busy.
8. `py-spy` (engine process): at 8 workers, ≥ 60% of samples are in frames
   that wait for MLX (`mx.eval`, `async_eval`, `.item()`, `.tolist()`).
   H1b holds instead if most samples are in Python bookkeeping (sampling,
   output processing, detokenizing).
9. `vllm:iteration_tokens_total` from `/metrics` gives the same tokens per
   step as the client rebuild (within 10%).

E3, first-token split at 1 worker (`/metrics` deltas of the 1-worker run):
10. Server TTFT mean is 170–185 ms (client 178 ms minus about 2.5 ms).
11. The largest part decides: H2a if TTFT − queue − prefill ≥ 100 ms; H2b if
    prefill ≥ 100 ms; H2c if queue ≥ 20 ms. Expected: H2b, because the client
    and the API server agree within 2.5 ms and an 8-token prompt needs little
    front-end work. This is a guess, not a derived value.

Results of E1 (2026-10-17; `runs/methodology_review/task6/e1-mlx-steps.json`,
run by the user with the server stopped; 20 timed steps per batch, median):

| Batch | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 40 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vLLM step ms (pilot) | 24.8 | 42.0 | 78.5 | 160.4 | 112.2 | 110.6 | 113.7 | — | 212.4 | 220.0 |
| Pure MLX step ms (E1) | 19.1 | 21.4 | 33.9 | 64.3 | 96.8 | 102.0 | 98.7 | 191.4 | 193.3 | 192.2 |
| vLLM minus MLX ms | 5.7 | 20.6 | 44.6 | 96.1 | 15.4 | 8.6 | 15.0 | — | 19.1 | 27.8 |

| # | Prediction | Result | Status |
|---|---|---|---|
| 1 | step(8) ≥ 1.3 × step(16), and step(48) ≥ 1.5 × step(32) in pure MLX | step(8) ÷ step(16) = 0.66: MLX is faster at 8 than at 16. step(48) ÷ step(32) = 1.96. | PARTLY HELD. MLX causes the jump above 32, not the slow batch 8. |
| 2 | 4-bit linear layers ≥ 70% of the step, same jumps | 87–109% at every batch; the linear layers alone double from 32 (90.5 ms) to 40 (178.8 ms). | HELD |
| 3 | step(1) is 17–25 ms | 19.1 ms; linear layers alone 16.8 ms (one read of the weights). | HELD |

Reading: there are two causes. The 4-bit matrix multiply in MLX is flat from
about 14 to 32 rows and doubles from 33 to 64 rows (blocks of 32 rows). That is
the jump at 48. The slow batches 2–8 are not in MLX: vllm-metal adds about
12 ms per sequence there (96 ms at batch 8), but only 9–28 ms per step from
batch 16. H1a holds for the jump above 32 only; H1b (extra work in vllm-metal
at small batches) is now the main hypothesis for batches 2–8.

Added after E1, before E2 (written 2026-10-17, 15:55 UTC). Two forms of H1b:
12. H1b-CPU (Python work for each sequence in vllm-metal): at 8 workers the
    GPU active residency is at most about 50% (MLX needs about 64 ms of GPU
    time per 160 ms step), and `py-spy` puts most engine samples in
    vllm-metal Python functions, not in MLX waits.
13. H1b-GPU (vllm-metal sends more GPU work per sequence than pure MLX, for
    example one attention or sampling kernel per sequence): at 8 workers the
    GPU active residency is ≥ 85%, and `py-spy` puts most engine samples in
    MLX waits (`mx.eval`, `async_eval`, `.item()`, `.tolist()`).
14. Profiles at 8 and at 16 workers differ in the frames that cause the extra
    cost; the same frames take a smaller share at 16.

Results of E2 and E3 (2026-10-18; `runs/methodology_review/task6b/`). Session B
(08:22–08:44 UTC, after the commit `8848d8b` of these predictions) has the
server log, `/metrics`, `powermetrics` and `py-spy`. Session A (2026-10-17,
16:00–16:16 UTC) is an extra pass in the same order with runs only; it
started 12 minutes before that commit, but after the predictions were written
(15:55 UTC). Table: `analysis/engine_steps.py` on session B, output
`task6b/report/engine-steps.json` and `engine-steps.png`.

| Workers | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 20 | 24 | 32 | 40 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engine step ms (B) | 25.6 | 42.0 | 78.6 | 118.8 | 160.4 | 212.8 | 245.9 | 144.1 | 109.2 | 109.8 | 115.4 | 120.1 | 213.4 | 220.7 | 225.6 |
| Engine step ms (A) | 24.5 | 42.1 | 78.8 | 118.6 | 159.8 | 212.7 | 246.2 | 144.3 | 110.3 | 110.0 | 115.1 | 119.4 | 214.1 | 219.4 | 226.6 |
| Pure MLX step ms (E1) | 19.1 | 21.4 | 33.9 | 54.3 | 64.3 | 78.8 | 95.0 | 99.5 | 96.8 | 98.1 | 102.0 | 98.7 | 191.4 | 193.3 | 192.2 |
| Token rate tok/s | 29 | 33 | 44 | 45 | 46 | 44 | 46 | 83 | 116 | 139 | 152 | 181 | 147 | 160 | 193 |
| GPU active % | 99.0 | 99.0 | 98.9 | 99.5 | 99.4 | 99.8 | 99.8 | 99.4 | 99.4 | 99.3 | 99.5 | 99.5 | 99.6 | 99.5 | 99.7 |
| GPU power W | 15.9 | 19.4 | 19.3 | 21.2 | 20.2 | 19.1 | 20.7 | 25.4 | 27.2 | 28.4 | 24.7 | 23.5 | 27.7 | 25.2 | 25.9 |
| Server TTFT ms | 182.1 | 372.9 | 332.7 | 451.8 | 498.9 | 626.6 | 659.8 | 673.1 | 659.2 | 755.5 | 876.0 | 1086.9 | 1340.7 | 1584.7 | 1949.5 |
| Server queue ms | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.1 | 0.1 | 0.1 | 0.1 |
| Server prefill ms | 178.4 | 293.8 | 209.2 | 317.4 | 357.1 | 482.6 | 514.0 | 527.0 | 513.0 | 609.0 | 727.0 | 937.8 | 1193.4 | 1440.2 | 1807.5 |

| # | Prediction | Result | Status |
|---|---|---|---|
| 4 | Shuffled step times match the ascending pilot within 10% (H1c false) | Within 6% at all 9 shared batch sizes (largest: 32 workers, 120.1 against 113.7 ms). Session A and B agree within 1.4%, except 4% at 1 worker. | HELD. Run order and heat do not cause the shape. |
| 5 | The batch size of the fall is the same in E1 and E2 (±1 grid point) | E2 falls between 12 and 14 workers (245.9 to 144.1 ms). E1 has no fall, but its rise stops at the same point (95.0 at 12, 99.5 at 14). | PARTLY HELD. Same transition point, different shape. |
| 6 | E2 minus E1 is about constant (±10 ms) | 6.6, 20.7, 44.7, 64.4, 96.1, 134.0, 150.8 ms from 1 to 12 workers; 12–34 ms from 16 to 64. | NOT HELD. The extra cost grows by about 12.5 ms per sequence up to 12, then almost disappears. |
| 7 | GPU active ≥ 85% at 8 and 32 workers | 99.4% and 99.5%; 98.9–99.8% at every batch size. Thermal pressure Nominal in all samples. | HELD |
| 8 | `py-spy`: ≥ 60% of engine-thread samples wait for MLX at 8 workers | 92.8% in `mx.async_eval` (`_submit_paged_forward_outputs`, `model_runner.py:709`); sampling 2.7%; Python graph building under 5%. Same at 1, 16 and 32 workers (86–91%). | HELD |
| 9 | Server tokens per step agree with the client rebuild within 10% | `iteration_tokens_total` mean 1.50 / 11.29 / 22.59 / 45.18 / 90.35 at 1 / 8 / 16 / 32 / 64 workers, against workers × 23/16 (8 prompt + 15 decode tokens over 16 steps) = 1.44 / 11.5 / 23.0 / 46.0 / 92.0: within 4%. | HELD |
| 10 | Server TTFT mean 170–185 ms at 1 worker | 182.1 ms (client median 180.6 ms). | HELD |
| 11 | The largest part of TTFT decides H2a/H2b/H2c; expected H2b | Queue 0.0 ms, prefill 178.4 ms, front end (TTFT − queue − prefill) 3.7 ms. | HELD: H2b. The 8-token prefill itself takes 178 ms. |
| 12 | H1b-CPU: GPU ≤ about 50% at 8 workers, `py-spy` in Python | GPU 99.4%; 93% of samples wait in MLX; the busiest CPU core averages 13.7%. | NOT HELD. H1b-CPU is false. |
| 13 | H1b-GPU: GPU ≥ 85% and `py-spy` waits for MLX | Both hold at 8 workers. | HELD. The extra cost is GPU work that vllm-metal sends. |
| 14 | Profiles at 8 and 16 differ in the frames of the extra cost | Same shape at 8 and 16 (92.8% and 90.6% in `async_eval`). A Python profiler cannot see inside the GPU work. | NOT HELD (not visible with `py-spy`). |

Other measured facts from session B:
- Prompt tokens per request: 8 (880 prompt tokens for 110 requests at 1
  worker). Prefix-cache hits: 0 of all queries (threat 13).
- The server log shows `max_num_seqs: 256` from the launch flag.
- The busiest CPU core averages 9.9–37.8% per run, so the client and the
  API server do not limit these closed runs (threat 9, up to 64 streams).
- Source: vllm-metal packs the whole batch into one sequence,
  `mx.array([all_token_ids])` (`vllm_metal/v1/model_runner.py:1183`), and
  attention runs in its own paged Metal kernel (`attention/impls/sdpa.py`).
  E1 used shape (batch, 1) and MLX's own attention.

### Task 6: E1b, packed batch in pure MLX (written 2026-10-18, 08:51 UTC, before the run)

Question: is the extra GPU cost at 2–12 sequences in the 4-bit matrix
multiply when the batch is packed as one sequence (H1d), or in vllm-metal's own
kernels, such as paged attention and the cache scatter (H1e)?
Script: `scripts/mlx_step_bench.py` (new columns `packed_step_ms`,
`packed_matmul_ms`, shape (1, batch)).

15. H1d holds if the packed matrix multiply grows by about one weight read per
    row up to 12: ≥ 120 ms at batch 8 and ≥ 170 ms at batch 12, and falls to
    within 15% of the (batch, 1) matrix multiply from 14 rows.
16. H1e holds if the packed matrix multiply is within 20% of the (batch, 1)
    matrix multiply at batch 8 (62 ms). Then the extra cost is in the vllm-metal
    kernels, and the next test is a GPU trace of one step.
17. If H1d holds, the packed step (one forward pass of a batch-token sequence,
    which is also the prefill of a batch-token prompt) is within 25% of the
    vLLM engine step from 2 to 64, and an 8-token prompt takes ≥ 120 ms
    while a 16-token prompt takes less. This would also explain A2.

Results of E1b (2026-10-18; `runs/methodology_review/task6/e1b-mlx-packed.json`,
run by the user with the server stopped; chart
`runs/methodology_review/task6b/report/engine-steps.png`):

| Batch | 1 | 2 | 4 | 8 | 12 | 14 | 16 | 32 | 40 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Matmul (batch, 1), ms | 17.1 | 18.8 | 33.0 | 62.9 | 100.2 | 93.3 | 90.2 | 91.0 | 179.8 | 185.3 |
| Matmul packed (1, batch), ms | 16.6 | 18.8 | 31.0 | 62.8 | 101.7 | 93.3 | 90.1 | 92.9 | 179.9 | 188.0 |
| Packed step (= prefill of a batch-token prompt), ms | 18.3 | 21.6 | 33.7 | 64.5 | 93.8 | 97.9 | 95.2 | 95.8 | 184.9 | 187.8 |
| vLLM engine step, ms (E2) | 25.6 | 42.0 | 78.6 | 160.4 | 245.9 | 144.1 | 109.2 | 120.1 | 213.4 | 225.6 |

| # | Prediction | Result | Status |
|---|---|---|---|
| 15 | H1d: packed matmul ≥ 120 ms at 8, ≥ 170 ms at 12 | 62.8 and 101.7 ms; the packed matmul is within 3% of the (batch, 1) matmul at every batch. | NOT HELD. H1d is false: the shape does not change the matrix multiply. |
| 16 | H1e: packed matmul within 20% of the (batch, 1) matmul at 8 | 62.8 against 62.9 ms. | HELD. The extra cost is in vllm-metal's own GPU work, not in MLX's matrix multiply. |
| 17 | (If H1d) packed step within 25% of the vLLM step; an 8-token prompt ≥ 120 ms and a 16-token prompt faster | Packed step 64.5 ms at 8 against 160.4 ms in vLLM; a 16-token prefill (95.2 ms) is slower than an 8-token one (64.5 ms). | NOT HELD (H1d false). |

Reading:
- Pure MLX runs the same model with the same packed shape in 64.5 ms at 8
  tokens. vllm-metal takes 160.4 ms for a decode step of 8 requests and
  178.4 ms for the prefill of one 8-token prompt. Both have about 100 ms of
  extra GPU work. The extra work depends on the number of tokens in the step
  (decode or prefill), grows up to 12 tokens, and almost stops from 14.
- The candidates are the kernels that pure MLX does not run: the paged
  attention kernel, the paged cache write (`reshape_and_cache`), the head-dim
  padding and the sampler (`vllm_metal/attention/impls/sdpa.py`,
  `vllm_metal/v1/sampling_batch.py`). Which one is not measured. A Metal GPU
  trace (Xcode Instruments, not installed) or a timing test of each kernel
  alone would identify it.

### Task 8: run sets and predictions (written 2026-10-18, 09:26 UTC, before the runs)

Basis for every number below: the pilot open sweep (`runs/vllm-metal-open-sweep/`,
values recalculated from `records.csv` on 2026-10-18), the Task 6 closed sweep
(session B, `task6b/report/engine-steps.json`), and Task 4 and 5. All runs use
the launch flags of Task 6 (`--max-num-seqs 256` unless stated), the 8-token
random-prefix prompt, 16 output tokens, and go under `runs/methodology_review/`.
Changes from the original plan, because of Tasks 4–6: open runs are 120 s with
a 30 s warmup (threat 1); the closed baseline is already done (Task 6b), so 8a
repeats only three closed points; 8b adds 13 req/s; 8d is re-predicted.

Pilot anchors (uniform, median of 3 repeats; N = unfinished requests at 59.5 s):

| Offered req/s | 6.5 | 9 | 11 | 12 | 12.5 | 13 | 14.5 | 16.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Served, 35–60 s, req/s | 6.52 | 9.00 | 11.04 | 11.96 | 11.76 | 10.96 | 11.68 | 12.80 |
| N | 14 | 20 | 47 | 96 | 152 | 209 | 311 | 437 |
| E2E p50 / p95, s | 2.27 / 2.35 | 2.27 / 2.35 | 4.27 / 4.39 | 8.04 / 8.30 | 11.3 / 12.3 | 13.3 / 16.1 | 20.0 / 22.6 | 25.1 / 29.7 |
| TPOT p50, ms | 129 | 130 | 244 | 462 | 655 | 758 | 1,103 | 1,198 |
| Medium-tier attainment | 100% | 100% | 100% | 0.7% | 0% | 0% | 0% | 0% |
| Server `waiting`, max | 0 | 0 | 0 | 0 | 0 | 0 | 54 | 171 |

Closed anchors (Task 6, session B): 16 / 32 / 64 workers make 116 / 181 /
193 tok/s (7.2 / 11.3 / 12.1 req/s); server E2E mean 2.2 / 2.8 / 5.1 s.

| Set | Runs | Time | Question |
|---|---|---|---|
| 8a | Open uniform at 6.5, 9, 11, 12, 13, 14.5, 16.5 req/s, 2 repeats, shuffled order, 120 s (30 s warmup). Closed 16, 32, 64 workers once. Server log, `/metrics` before and after each run, `powermetrics`. `tcpdump` on one 12 req/s run. | about 35 min | Same-session baseline; server checks; wire check |
| 8b | Open uniform at 11, 12, 13 req/s, one 300 s run each (60 s warmup). | 15 min | Is 12 req/s sustainable? |
| 8c | Open Poisson at 9, 11, 12 req/s, 2 repeats, 120 s (30 s warmup). | 12 min | Does realistic arrival change the SLO result? |
| 8c-gamma | Open gamma (CV 2) at 9, 11 req/s, 2 repeats, 120 s (30 s warmup). Added 2026-10-18 (user decision). | 9 min | Does burstier-than-chat traffic change it more? |
| 8d | Restart with `--max-num-seqs 32`. Open uniform at 9, 11, 12, 13, 16.5 req/s, one 120 s run each. | 12 min | Where does the queue go when the batch is capped? |
| 8e | `vllm bench serve`, Poisson, 8 random input tokens, 16 output tokens, `--ignore-eos`, 9, 11, 12 req/s, 120 s of requests each. | 8 min | Does a second client agree? |
| 8f | SKIPPED (user decision, 2026-10-18: no NVIDIA GPU). | — | Is the step shape Metal-specific? Not tested; the slide states the Metal-only scope. |

Predictions. "Uniform" means run set 8a at the same rate. A prediction fails
if its threshold is not met; the result is recorded either way.

8a, same-session uniform baseline:
- 8a.1 Capacity. Served in 30–120 s is ≥ 95% of offered at 6.5, 9 and 11
  req/s. At 13, 14.5 and 16.5 req/s served is 10.5–13.0 req/s.
- 8a.2 Overload does not lower throughput. Served at 16.5 ≥ served at 13
  minus 1.0 req/s.
- 8a.3 The SLO cliff stays between 11 and 12 req/s. Medium-tier attainment
  ≥ 95% at 11 req/s and < 50% at 12 req/s.
- 8a.4 Steady backlog below capacity. N at 115 s: 9–21 (6.5 req/s), 13–30
  (9), 30–65 (11), 60–160 (12). Slope of N over 60–120 s within ±0.3/s at
  6.5, 9, 11 and 12 req/s; ≥ +0.5/s at 13 and above.
- 8a.5 The queue hides inside the batch. At 12 req/s the server `waiting`
  stays 0 in every sample, while E2E p50 ≥ 5 s. At 16.5 req/s `waiting`
  becomes > 0 once `running` reaches 256.
- 8a.6 Little's Law. L and λ·W agree within 5% at 6.5, 9 and 11 req/s.
- 8a.7 Server check (Task 9). At every rate ≤ 12 req/s: client TTFT mean minus
  server TTFT mean is 0–15 ms; client and server TPOT means within 5%.
- 8a.8 Wire check (Task 9). On the 12 req/s run with `tcpdump`, the gaps
  between request packets on the wire differ from the planned 83.3 ms gap by
  ≤ 2 ms for 99% of requests.
- 8a.9 Session check. Closed 16, 32, 64 workers make tok/s within 10% of
  Task 6 (116, 181, 193).
- 8a.10 Closed and open give different answers. Open E2E p50 at 12 req/s is
  ≥ 1.3 × the closed E2E p50 at 64 workers (12.1 req/s, about 5.1 s).

8b, long runs:
- 8b.1 11 req/s is stable: slope of N over 150–300 s within ±0.1/s; medium
  attainment ≥ 95%.
- 8b.2 12 req/s is stable but near the limit: slope over 150–300 s within
  ±0.2/s and N between 60 and 200. If it fails, the capacity statement becomes
  "between 11 and 12 req/s".
- 8b.3 13 req/s is overload: slope over 150–300 s ≥ +0.5/s; served 10.5–13.0
  req/s.

8c, Poisson arrivals:
- 8c.1 The schedules are Poisson: CV of sent gaps 0.95–1.05; index of
  dispersion (1 s) 0.6–1.4.
- 8c.2 Low load absorbs bursts. At 9 req/s the Poisson E2E p95 is ≤ 1.3 ×
  uniform, and medium attainment ≥ 95% for both. Reason: a burst makes the
  batch larger, and a larger batch serves more tokens per second here.
- 8c.3 Near capacity, bursts break the SLO. At 11 req/s the Poisson E2E p95 is
  ≥ 1.2 × uniform, and Poisson medium attainment is < 95% while uniform is
  ≥ 95%. This is the main slide claim: the uniform test passes an SLO that
  realistic arrivals fail at the same mean rate.
- 8c.4 At 12 req/s both fail the medium tier (attainment < 50%), and the
  Poisson E2E p95 is ≥ uniform.
- 8c.5 (added 2026-10-18, 09:42 UTC, before any Task 8 run) Batching absorbs even gamma CV 2
  bursts at low load. At 9 req/s gamma medium attainment is ≥ 95%, and gamma
  E2E p95 is 1.1–2.0 × uniform.
- 8c.6 (added 2026-10-18, 09:42 UTC) Burstier is worse near capacity. At 11 req/s gamma
  medium attainment is ≤ the Poisson value, and gamma E2E p95 is ≥ the Poisson
  value.
- Not predicted: a Kingman number. The service time of a batching server
  changes with the load, so the Task 4 formula does not apply directly.

8d, `--max-num-seqs 32`:
- 8d.1 At 9 req/s the cap does nothing (pilot N about 20 < 32): served and
  E2E p50 within 10% of uniform.
- 8d.2 The queue moves to `waiting`. At 12, 13 and 16.5 req/s `running` stays
  ≤ 32, `waiting` grows (> 0 and rising over 60–120 s), and TPOT p50 stays
  ≤ 300 ms (uncapped pilot: 462–1,198 ms). TTFT carries the wait: TTFT p95 at
  16.5 req/s ≥ 5 s.
- 8d.3 The cap lowers capacity. Served at 13 and 16.5 req/s is 8.5–11.5
  req/s, below the uncapped value of 8a. Reason: E2 shows 32 workers at 181
  tok/s against 193 at 64, and in open loop each step also carries the
  prompt tokens of new requests. This replaces the A4 prediction.
- 8d.4 At 11 req/s: no prediction. 11 req/s is near the predicted capped
  capacity, so either outcome is possible.

8e, second client (`vllm bench serve`):
- 8e.1 Request throughput within 5% of offered at 9 and 11 req/s.
- 8e.2 Mean TPOT and median ITL within 15% of satlab Poisson (8c) at the same
  rate; mean TTFT within 25% (the benchmark client has no warmup window).
- 8e.3 At 12 req/s the benchmark client also shows mean E2E ≥ 3 × its value
  at 9 req/s (the same knee).

8f, CUDA (SKIPPED, no NVIDIA GPU; kept for the record): on vLLM with CUDA, closed tok/s does not fall when the
batch grows from 1 to 64 workers (each step within 5% of monotonic). If it
holds, the Metal step shape (Task 6) is not a general vLLM property.

Results of 8a (2026-10-18, 09:55–10:26 UTC; `runs/methodology_review/task8/8a/`,
`8a-closed/`, `8a-report/`):

| # | Result | Status |
|---|---|---|
| 8a.1 | Served 6.51 / 9.00 / 10.99 req/s at 6.5 / 9 / 11. The 13–16.5 runs are not valid (client limit, threat 19). | HELD for ≤ 11; 13–16.5 re-run in `8a-overload/` |
| 8a.2 | Not valid (client limit). | Re-run |
| 8a.3 | Medium attainment 100% at 11 req/s, 0% at 12 req/s. | HELD |
| 8a.4 | N at 115 s: 15 (6.5), **37 (9)**, 47 (11), 120–129 (12); slopes 0.00–0.01/s. The 13+ slopes are not valid. | NOT HELD at 9 req/s (anomaly A7); re-run 13+ |
| 8a.5 | 12 req/s: `waiting` max 0 in 123 samples per run; E2E p50 10.0 s. The 16.5 part is not valid. | HELD at 12; re-run 16.5 |
| 8a.6 | L / λ·W: 14.7 / 14.8, 37.1 / 37.1, 46.8 / 46.8. | HELD |
| 8a.7 | Client minus server TTFT mean 2.5–13.3 ms; TPOT means within 0.1% (6.5–12 req/s, 8 runs). | HELD |
| 8a.9 | Closed 16 / 32 / 64 workers: 116 / 192 / 202 tok/s against 116 / 181 / 193 (0%, +6%, +5%). | HELD |
| 8a.10 | Open E2E p50 at 12 req/s 10.0 s against closed 64 workers 5.03 s (2.0×). | HELD |

Threat 19 found in 8a. Every run at 13, 14.5 and 16.5 req/s has exactly 246
requests in flight at its maximum. The errors are `http_error` with HTTP
status 0, and each fails 0.05 ms after the send, before any reply. Cause: a
macOS shell has a soft limit of 256 open files (`launchctl limit maxfiles`);
each request in flight holds a socket. vLLM raises its own limit to 65535
(`vllm/utils/system_utils.py:285`); satlab did not. The harness verdict marked
the runs FAILED, as designed. Fix: satlab now raises its soft limit to
`max_in_flight` + 64 at start, or stops with a clear message
(`ensure_file_limit` in `src/runner.rs`). The failed runs stay in `8a/` as
evidence.

Results of the 8a re-run at 13–16.5 req/s (2026-10-18, 11:19–11:34 UTC;
`runs/methodology_review/task8/8a-overload/`, commit `f1851b7`, with the file-limit
fix). All 6 runs PASS; up to 736 requests in flight; 0 errors.

| Offered req/s | 13 | 14.5 | 16.5 |
|---|---:|---:|---:|
| Served, 30–120 s, req/s (2 runs) | 11.36, 11.48 | 11.72, 11.88 | 12.02, 12.03 |
| Served, 60–120 s, req/s | 11.87, 12.12 | 12.00, 12.17 | 11.92, 11.93 |
| Slope of N, 60–120 s, per s | 0.95, 0.87 | 2.45, 2.39 | 4.58, 4.56 |
| N at the end | 319, 309 | 492, 474 | 729, 734 |
| E2E p50, s | 22.0, 22.0 | 31.1, 30.1 | 44.0, 43.7 |
| Server `running` max / `waiting` max | 242–243 / 50–55 | 244–245 / 217–225 | 247 / 462–464 |
| First `waiting` > 0, s after start | 66.0, 65.7 | 35.2, 35.4 | 24.1, 23.2 |

| # | Result | Status |
|---|---|---|
| 8a.1 | Served 11.4–12.0 req/s at 13–16.5 (range 10.5–13.0). | HELD |
| 8a.2 | Served at 16.5 (12.03) ≥ served at 13 (11.42) − 1.0. In 60–120 s all six runs serve 11.9–12.2 req/s. | HELD. Overload does not lower throughput. |
| 8a.4 | Slopes 0.87–0.95, 2.39–2.45, 4.56–4.58 per s (≥ +0.5). They match offered minus capacity: 13 − 12 = 1, 14.5 − 12 = 2.5, 16.5 − 12 = 4.5. | HELD |
| 8a.5 | At 16.5 req/s `waiting` becomes > 0 at 23–24 s and reaches 462–464. The `running` gauge stops at 242–247, not 256. | HELD, with a note |

Note on 8a.5: `running` never shows 256. A likely cause (not tested): the gauge
is read between engine steps, after the requests that finished in the last
step have left and before new ones are admitted. At about 16 steps per
request, about 1/16 of 256 (16) finish in each step. The pilot showed the same
maximum (246).

Reading: capacity is 12 req/s (about 190 tok/s), and overload adds requests to
the backlog at offered minus 12 per second, without limit. The server shows
the queue in `waiting` only after about 245 requests are already in the batch.
At 13 req/s that is 66 s into the run, when E2E p50 is already above 20 s.

### Task 8: E7, the 9 req/s slow branch (written 2026-10-18, 10:48 UTC, before the run)

Anomaly A7: at 9 req/s, both 8a repeats settle at N = 37 within 15 s, with
TPOT 231 ms and 48.4 tokens per engine step (32.2 decode + 16.1 prompt;
`metrics_before/after.prom`). The pilot at 9 req/s settled at N = 20 with
TPOT 130 ms. 6.5 and 11 req/s agree with the pilot.

- H7a: the random prefix made the prompt 8 tokens instead of 7. At 9 req/s a
  step is then near the 32-row block limit of the matrix multiply (Task 6).
  One step over the limit costs a second block (about 213 ms instead of
  130 ms); more requests arrive during it, so the next step is large too. The
  closed-loop throughput is not monotonic (32 workers 11.3 req/s, 40 workers
  9.2 req/s), so the system can stay on this slow branch.
- H7b: a start transient or the server state of this session, not the prompt.
- Test: in the same server session, one 9 req/s uniform run with a 7-token
  prompt ("Explain saturation in a sentence" + prefix: 7 tokens for 20,000 of
  20,000 random prefixes), then one 9 req/s control run with the 8-token
  prompt. `run.sh e7`.
- E7.1 (H7a): the 7-token run has N at 115 s ≤ 25 and TPOT p50 ≤ 150 ms.
- E7.2: the 8-token control run has N at 115 s ≥ 30.
- If the 7-token run also has N ≥ 30, H7a is false. If both runs have N ≤ 25,
  the 8a result was a session effect (H7b).
- Note: predictions 8c.2 and 8d.1 assumed the fast branch at 9 req/s. They
  stay as written; their results will be read with A7.

Results of E7 (2026-10-18; `runs/methodology_review/task8/e7-7tok/`, `e7-8tok/`,
same server session, commit `f1851b7`; both runs PASS):

| 9 req/s, uniform | 7-token prompt | 8-token prompt (control) |
|---|---:|---:|
| N, 15–115 s | 20–21 | 36–38 |
| TPOT p50 | 129 ms | 235 ms |
| E2E p50 / p95 | 2.27 / 2.34 s | 4.13 / 4.25 s |
| Server tokens per step (decode + prompt) | 26.3 (18.3 + 8.0) | 48.4 (32.3 + 16.1) |
| Served; medium attainment | 9.00 req/s; 100% | 9.01 req/s; 100% |

| # | Result | Status |
|---|---|---|
| E7.1 | 7-token run: N 20, TPOT 129 ms (≤ 25 and ≤ 150 ms). | HELD |
| E7.2 | 8-token control: N 38 (≥ 30). | HELD |

Reading: H7a holds; H7b (session effect) is false, because the control in the
same session returns to N = 38. The 7-token run is the same as the pilot
(N 20, TPOT 130 ms, E2E p50 2.27 s). With 7 tokens a step carries 26 tokens,
inside one 32-row block; with 8 tokens the system settles at 48 tokens per
step, two blocks. Both are self-consistent: at 9 req/s and a 129 ms step,
1.2 requests arrive per step (8 prompt tokens); at a 235 ms step, 2.1 arrive
(16 prompt tokens). One more prompt token makes E2E 82% longer at the same
rate. A7 is resolved; it is a property of this Metal stack (Task 6) and of
the workload, not a harness fault.

Consequences for the predictions not yet run (note written 11:43 UTC, before 8c and
8d; the original predictions stay as written):
- 8c.2 compares Poisson with uniform at 9 req/s. Uniform at 9 req/s is on the
  slow branch (E2E p95 4.25 s), so the comparison is between slow-branch runs.
- 8d.1 expected the cap of 32 to do nothing at 9 req/s. With the 8-token
  prompt about 37 requests run at 9 req/s, so the cap is active. 8d.1 is now
  expected not to hold. What the cap does at 9 req/s is not predicted.

Slide claims and the prediction that tests each:

| Slide claim | Prediction |
|---|---|
| This Mac serves about 12 req/s of 8-in/16-out requests (about 190 tok/s). | 8a.1, 8b.2 |
| Overload does not lower throughput; it raises latency without limit. | 8a.2, 8b.3 |
| A uniform test passes an SLO that realistic arrivals fail at the same rate. | 8c.3 |
| A batching server absorbs bursts at low load. | 8c.2, 8c.5 |
| Burstier arrivals hurt more near capacity. | 8c.6 |
| The queue hides inside the batch: `waiting` = 0 while E2E is 8 s. | 8a.5 |
| A batch cap makes the queue visible and bounds TPOT, but costs throughput. | 8d.2, 8d.3 |
| Closed and open load give different answers on the same server. | 8a.10 |
| The numbers are right: wire, server and a second client agree. | 8a.7, 8a.8, 8e.2 |

### Task 8: 8a-wire and anomaly A8

Results of 8a-wire (2026-10-18, 11:46–11:48 UTC; `runs/methodology_review/task8/8a-wire/`
and `wire-12.pcap`, commit `f1851b7`; the run PASS). `tcpdump -i lo0 -s 200
'tcp dst port 8000'` ran during the run. `analysis/wire.py` takes the first
segment of each request (payload starts with `POST`) and matches it in order to
the plan.

| Wire check, 12 req/s uniform | p1 / p50 / p99 |
|---|---:|
| Gap between POSTs, ms (plan 83.33) | 81.65 / 83.32 / 85.15 |
| Wire time − planned time, ms | 1.36 / 2.74 / 4.52 |
| Wire time − satlab send time, ms | 0.17 / 0.68 / 1.99 |

| # | Result | Status |
|---|---|---|
| 8a.8 | 1440 POSTs on the wire for 1440 requests; 99.03% of gaps within 2 ms of 83.33 ms. | HELD, small margin |

Reading: each request reaches the wire 2.7 ms (median) after its planned time,
and the delay is nearly constant, so the wire rate is the planned rate. The
wire is 0.7 ms (median) after satlab's own send time: the request build and
the TCP connect, which satlab cannot see. The total delay at p99 (4.5 ms) is
inside the send-lag budget (8.3 ms, 10% of the gap).

Anomaly A8: the same 12 req/s run went into overload. The two 8a runs at
12 req/s served 11.79 and 11.97 req/s with N flat at 117–136. 8a-wire served
10.98 req/s; N grew from 122 at 30 s to 214 at 120 s (+1.16/s); E2E p50
12.7 s against 10.0 s. Engine steps (stream events, 3 ms grouping) and
`powermetrics` show the sequence:

| Time in run | GPU clock | Tokens per step | Step time |
|---|---|---:|---|
| 30–56 s | 1577 MHz | 110 | 572 ms (192 tok/s) |
| 56–66 s | 1577 → 1537 MHz | 110–111 | 572 → 584 ms |
| 66.9 s | 1537 MHz | 111 | 584 → 671 ms (+87 ms) |
| 76.5 s | 1532 MHz | 127 | 698 → 795 ms (+98 ms) |
| 105–120 s | 1529 MHz | 181 (median) | 1020 ms (170 tok/s) |

- H8a: 12 req/s is at the edge of capacity. On the fast branch the server
  makes 110 tokens per 572 ms step (192.3 tok/s); 12 req/s needs 192 tok/s,
  a margin of about 0.2%. A GPU clock dip of 2.5% removes the margin. The
  backlog grows by a few requests, a step goes over a 32-row block boundary
  (+87 ms, the block cost of Task 6 and A7), more requests arrive in each
  longer step, and the system moves to slower branches. With 130–210 tokens
  per step the server makes about 170–185 tok/s (also at 13 req/s, 30–60 s,
  at full clock; the reason is not known), so the backlog keeps growing. This
  is A7 at 12 req/s with a different trigger.
- H8b: `tcpdump` caused the slowdown. Against it: CPU power stays 0.4–0.7 W,
  as in the other runs, and the same clock dips (minimum 1506–1552 MHz) occur
  in runs without `tcpdump` (8a-closed, 8a-overload at 13 req/s, e7). The two
  8a runs at 12 req/s had no dip (1575–1577 MHz).
- Thermal pressure is Nominal in every sample. "Pressure Nominal" does not
  mean a constant clock (threat 20).

Notes for 8b (written 2026-10-18, 11:54 UTC, before 8b; 8b.1–8b.3 stay as
written):
- 8b.2 (12 req/s stable for 300 s) can hold only if the GPU clock stays at
  about 1577 MHz for the full run.
- 8b.4 (H8a): if N at 12 req/s starts to grow (slope over the next 30 s
  > +0.5/s), a GPU clock dip below 1560 MHz comes in the 15 s before the first
  step-time increase of more than 50 ms, and N does not return to its earlier
  level before the run ends. If N grows with no clock dip before it, the
  trigger in H8a is wrong.

Results of 8b (2026-10-18, 11:57–12:12 UTC; `runs/methodology_review/task8/8b/`,
commit `f1851b7`, the predictions committed in `dd814ee` at 11:56:55 UTC; all 3
runs PASS). Slope and served are over 150–300 s.

| Uniform, 300 s | 11 req/s | 12 req/s | 13 req/s |
|---|---:|---:|---:|
| Slope of N, per s | −0.008 | 0.000 | +0.787 |
| N at 60 / 150 / 295 s | 47 / 66 / 64 | 123 / 120 / 118 | 261 / 342 / 448 |
| Served, req/s | 10.99 | 11.98 | 12.24 |
| TPOT p50; E2E p50 / p95 | 346 ms; 6.07 / 6.29 s | 571 ms; 10.0 / 10.3 s | 1235 ms; 29.3 / 35.5 s |
| Medium attainment | 7.4% | 0% | 0% |
| GPU clock | dip 71–225 s, minimum 1542 MHz | 1576–1577 MHz | 1576–1577 MHz |

| # | Result | Status |
|---|---|---|
| 8b.1 | Slope −0.008 (within ±0.1), but medium attainment 7.4% (≥ 95% predicted). | NOT HELD |
| 8b.2 | Slope 0.000; N 118–123. The GPU clock stayed at full speed. | HELD |
| 8b.3 | Slope +0.787 (≥ +0.5); served 12.24 (10.5–13.0). | HELD |
| 8b.4 | N at 12 req/s did not grow, so the condition did not occur. | NOT TESTED |

Reading: 11 req/s moved to a slower branch, the same way as 12 req/s in
8a-wire. The run held 43 tokens per 243 ms step to 77 s (as in 8a: N 47,
E2E p95 4.36 s, 100% attainment). The GPU clock fell from 1576 MHz at 71 s to
1560 MHz at 77 s (−1%). At 77.6 s the step went from 247 to 335 ms (+88 ms,
one 32-row block) with 43 tokens. The batch then grew to 61 tokens per 345 ms
step and stayed there. The clock was back at 1577 MHz from 225 s, but the
system did not return: the slower branch is also self-consistent at
11 req/s (served 10.99), so nothing pulls it back. This is the H8a pattern
(dip, block step, no return) at 11 req/s. The minimum clock in the 15 s before
the step was 1560 MHz, at the 8b.4 threshold; 8b.4 was written for 12 req/s
only, so this is support, not a test.

At 11 req/s the fast branch carries about 43 decode + 21 prompt = 64 rows per
step, exactly two 32-row blocks; a 1% slower step is enough to go over. At
12 req/s it carries 110 + 55 = 165 rows, inside six blocks (192), and it held
for 300 s at full clock.

Notes for 8c (written 2026-10-18, 12:20 UTC, before 8c; 8c.1–8c.6 stay as
written):
- Uniform at 11 req/s now has two results: the fast branch (8a, 2 runs:
  E2E p95 4.36 s, attainment 100%) and the slow branch (8b: 6.29 s, 7.4%).
  8c.3 and 8c.6 compare with the 8a runs, which have the same 120 s length.
  The results will also give the ratio against the 8b run.
- The main slide claim ("uniform passes, Poisson fails at 11 req/s") holds
  only for uniform on the fast branch. Even uniform arrivals fail the medium
  tier at 11 req/s in 1 of 3 runs.

### Task 8: 8c, Poisson arrivals

Results of 8c (2026-10-18, 12:31–12:43 UTC; `runs/methodology_review/task8/8c/`,
commit `f1851b7`, notes committed in `409e5bd` at 12:30:36 UTC; all 6 runs
PASS). Uniform references: 8a (120 s) and, at 11 req/s, also 8b.

| Poisson, 120 s | 9 r1 | 9 r2 | 11 r1 | 11 r2 | 12 r1 | 12 r2 |
|---|---:|---:|---:|---:|---:|---:|
| Gap CV (plan = sent ± 0.001) | 1.019 | 1.039 | 1.086 | 0.988 | 0.971 | 1.002 |
| Index of dispersion, 1 s | 0.85 | 1.15 | 1.19 | 0.79 | 0.98 | 0.95 |
| Offered in 30–120 s, req/s | 9.02 | 8.91 | 11.02 | 10.60 | 11.94 | 11.60 |
| Served, req/s | 9.00 | 8.86 | 10.70 | 10.87 | 11.74 | 11.08 |
| TPOT p50, ms | 239 | 224 | 485 | 466 | 1217 | 703 |
| E2E p50 / p95, s | 4.24 / 4.93 | 3.99 / 5.40 | 8.59 / 10.32 | 8.30 / 9.70 | 21.5 / 23.2 | 12.3 / 13.7 |
| Medium attainment | 96.3% | 89.5% | 0% | 7.4% | 0% | 0% |
| Uniform E2E p95; medium attainment | 4.22 s; 100% | | 4.36 s (8b: 6.29 s); 100% (8b: 7.4%) | | 10.3–10.6 s; 0% | |

| # | Result | Status |
|---|---|---|
| 8c.1 | IDC 0.79–1.19 (0.6–1.4). CV 0.971–1.039 in 5 runs; 1.086 in 11 r1. | NOT HELD (1 of 6) |
| 8c.2 | E2E p95 1.17× and 1.28× uniform (≤ 1.3). Medium attainment 96.3% and 89.5% (≥ 95%). | NOT HELD (r2 attainment) |
| 8c.3 | E2E p95 2.37× and 2.22× the 8a uniform (1.64× and 1.54× the 8b run); attainment 0% and 7.4% against 100%. | HELD |
| 8c.4 | Attainment 0% for both; Poisson p95 13.7–23.2 s against 10.3–10.6 s. | HELD |

On 8c.1: the sender is not the cause; the sent CV equals the plan CV within
0.001. The planned gaps fit an exponential with the planned rate (one-sample
KS, p = 0.06–0.96). The 11 r1 plan has five long gaps (6.8–11.2 × the mean);
without the longest, CV is 1.057. The band was too narrow: for about 1,300
exponential gaps, 6.8% of samples fall outside 0.95–1.05 (20,000 simulated
schedules), so one of six runs outside was 35% likely. A CV of ≥ 1.086 is
rarer (0.2% per run). The schedule code is the inverse transform
(`-ln(u)/λ`, `src/driver.rs`). Use simulated bands for such checks.

Reading:
- At 11 req/s the main claim holds: uniform on the fast branch passes the
  medium tier (100%), Poisson at the same mean rate fails (0–7%), with
  E2E p95 2.2–2.4× higher. Against the slow uniform branch (8b) Poisson is
  still 1.5–1.6× worse.
- Poisson does not settle on one branch. Bursts move the step through more
  32-row blocks: at 11 req/s, 15 s windows go from 44 to 99 tokens per step
  (248–561 ms). At 9 req/s the step also falls to the one-block branch
  (18–21 tokens, 146–160 ms) in quiet periods, so the Poisson E2E p50 of r2
  (3.99 s) is below uniform (4.11 s) while its p95 is higher.
- At 12 req/s two Poisson runs differ by 1.7× in TPOT (1217 against 703 ms):
  at the edge of capacity the random schedule decides the result. 12 r1
  reached the batch limit of about 240 for 45 s.
- The GPU clock dipped in 9 r1 and 11 r1 (minimum 1550–1553 MHz) and stayed at
  1575–1577 MHz in the other four. 11 r2 fails with no dip, so the clock is
  not the cause of the 11 req/s result.

Notes for 8c-gamma (written 2026-10-18, 12:49 UTC, before 8c-gamma; 8c.5 and
8c.6 stay as written):
- 8c.5 predicts gamma medium attainment ≥ 95% at 9 req/s. Poisson at 9 req/s
  already gave 89.5% in one run, so this part is expected not to hold.
- 8c.6 compares with Poisson at 11 req/s (0% and 7.4%). Gamma attainment
  ≤ Poisson is close to certain at this floor; the E2E p95 part is the real
  test.

### Task 8: 8c-gamma, gamma CV 2 arrivals

Results of 8c-gamma (2026-10-18, 13:01–13:10 UTC;
`runs/methodology_review/task8/8c-gamma/`, commit `f1851b7`, notes committed in
`62dd327` at 13:01:27 UTC; all 4 runs PASS).

| Gamma CV 2, 120 s | 9 r1 | 9 r2 | 11 r1 | 11 r2 |
|---|---:|---:|---:|---:|
| Gap CV (plan = sent ± 0.002) | 1.89 | 2.08 | 2.00 | 1.94 |
| Offered in 30–120 s, req/s | 8.47 | **11.02** | 10.87 | 11.06 |
| TPOT p50, ms | 239 | 604 | 461 | 847 |
| E2E p50 / p95, s | 4.30 / 7.69 | 10.6 / 16.2 | 8.31 / 12.9 | 15.2 / 19.5 |
| Medium attainment | 73.4% | 13.2% | 23.4% | 3.6% |
| GPU clock minimum | 1538 MHz | 1576 MHz | 1520 MHz | 1576 MHz |

| # | Result | Status |
|---|---|---|
| 8c.5 | Attainment 73.4% and 13.2% (≥ 95%). E2E p95 1.82× uniform in r1 (1.1–2.0), 3.85× in r2. | NOT HELD |
| 8c.6 | E2E p95 12.9 and 19.5 s against Poisson 9.7 and 10.3 s: HELD. Attainment 23.4% and 3.6% against Poisson 0% and 7.4%: r1 is above both Poisson runs. | NOT HELD (attainment); p95 part held |

The note before 8c-gamma said that the attainment part of 8c.6 was close to
certain. That was wrong.

Checks on the gamma schedules:
- The sender is not the cause: sent CV equals plan CV within 0.002.
- The planned gaps fit Gamma(shape 0.25, mean 1/λ): two-sample KS against
  200,000 simulated gaps, p = 0.08–0.56.
- 9 r2 drew a rare schedule: 11.02 req/s in 30–120 s at λ = 9. In 20,000
  simulated schedules, 0.07% reach this. New unit test
  `gamma_counts_spread_across_seeds_like_a_renewal_process`
  (`src/driver.rs`): over 2,000 seeds, the satlab count in 30–120 s has
  mean 810 within 2% and SD 57 within 10%. Thus the satlab code spreads
  across seeds as theory says, and 9 r2 is a tail draw, not a fault.
- At CV 2, one 120 s run does not fix the rate: the 90 s window rate has
  p5–p95 of 8.0–10.0 req/s at λ = 9 and 9.9–12.2 at λ = 11. The SD of the
  window rate is √(λ·CV²/T); for 2% at λ = 9 a run needs T = CV²/(λ·0.02²),
  about 1,100 s at CV 2 and 280 s for Poisson.

Reading:
- The tail ordering holds at 11 req/s: E2E p95 4.4 s (uniform, fast branch),
  9.7–10.3 s (Poisson), 12.9–19.5 s (gamma CV 2).
- Attainment does not follow the same order. Gamma has long quiet periods:
  in 11 r1, 15–45 s carried 24–27 tokens per 156–159 ms step (the one-block
  branch), and the requests in that period pass. The failures come in the
  bursts. Within 120 s, where the bursts fall decides the share that passes.
- 9 r1 (offered 8.47) is the fair 9 req/s run: 73.4% attainment, p95 1.82×
  uniform. Even at 9 req/s, gamma CV 2 bursts are not absorbed.

Notes for 8e (written 2026-10-18, 13:17 UTC, before 8e; 8e.1–8e.3 stay as
written):
- Prompt length. `vllm bench serve --random-input-len 8` sends 7 random
  tokens and the server adds BOS (`num_special_tokens_to_add`,
  `vllm/benchmarks/datasets/datasets.py:604`), so the server sees 8 tokens,
  as with satlab. Decode and re-encode can change the length by a token. A7
  shows that one token changes the branch, so the check is the mean prompt
  tokens per request from `bench-<r>-before/after.prom`: it must be 8.0. If
  not, 8e.2 compares different workloads.
- 8e.2 is judged at 9 and 11 req/s. At 12 req/s the two 8c runs differ by
  1.7× in TPOT, so the 12 req/s value is only reported.

## Workload: real traffic against our test

Question: how far is our test workload (uniform or Poisson arrivals, 8 prompt
tokens, 16 output tokens) from real LLM traffic? Source: the Azure LLM
inference trace 2023, two files, analysed separately (code completion and
conversation are different services). Script: `analysis/trace.py`.

Expected (written before the analysis, 2026-10-17, 14:18 UTC; only the
header and first two rows of each file were seen):
1. Each file covers about 1 hour. Mean rate: code about 2.5 req/s,
   conversation about 5 req/s.
2. Timestamps have 100 ns resolution (7 decimals) and are sorted. Few or no
   zero gaps.
3. CV of gaps over the whole trace: 1.5–3, above Poisson (1). Inside 5-minute
   pieces it is lower (1.2–2), because part of the whole-trace CV is slow rate
   drift.
4. Index of dispersion: near 1–2 at 0.1 s, and it grows with the window to
   more than 10 at 60 s (bursty at long time scales, not Poisson).
5. ContextTokens: median about 1,500 (code) and 1,000 (conversation), long
   tail to about 8,000; CV about 1.
6. GeneratedTokens: code median about 15 (short completions), conversation
   median about 100–200. CV above 1 for code, about 1 for conversation.
7. Weak correlation between context and generated tokens (|r| < 0.2).
8. Our 8-token prompt is below the 1st percentile of both files. Our 16 output
   tokens are near the code median and far below the conversation median.

Run: `.venv/bin/python -I analysis/trace.py
runs/methodology_review/traces/azure-llm-2023/*.csv -o
runs/methodology_review/task5-trace`. All numbers below come from the CSV rows
(`AzureLLMInferenceTrace_code.csv` = code, `AzureLLMInferenceTrace_conv.csv` =
conversation) through `trace-summary.json` in the output folder. Burstiness uses
`burstiness()` from `analysis/queue_theory.py`, the same function as Task 4.

Arrivals:

| Measure | code | conversation |
|---|---:|---:|
| Data rows | 8,819 | 19,366 |
| First and last `TIMESTAMP` (2023-11-16) | 18:17:03 – 19:14:19 | 18:15:46 – 19:14:08 |
| Span | 3,435.9 s (57.3 min) | 3,501.7 s (58.4 min) |
| Mean rate | 2.57 req/s | 5.53 req/s |
| Timestamp resolution | 7 decimals written; the 7th is always 0, so 1 µs | same |
| Sorted; zero gaps; smallest gap | yes; 0; 6 µs | yes; 0; 2 µs |
| CV of gaps, whole trace | 13.15 | 1.09 |
| CV of gaps in 11 full 5-minute pieces: median (min–max) | 10.15 (0.86–19.55) | 1.03 (0.99–1.19) |
| Index of dispersion at 0.1 / 1 / 10 / 60 s | 2.54 / 13.19 / 71.6 / 164.0 | 1.06 / 1.39 / 3.86 / 16.1 |
| Requests per 1 s: mean, peak, share of empty seconds | 2.57, 67, 73% | 5.53, 16, 1.1% |
| Requests per 1 min: min, mean, peak | 0, 151, 632 | 191, 333, 507 |
| Peak minute ÷ mean minute | 4.2 | 1.52 |
| Gamma CV with the same index of dispersion at 1 s (simulated at the trace rate and span, 10 runs per CV, step 0.1) | 4.7 | 1.2 |

Request shape (our test: 8 prompt tokens, exactly 16 output tokens):

| Measure | code | conversation |
|---|---:|---:|
| ContextTokens p50 / p90 / p99 / max | 1,469 / 5,187.6 / 7,436 / 7,437 | 1,020 / 2,734.5 / 4,142 / 14,050 |
| ContextTokens mean, CV | 2,048, 0.96 | 1,155, 0.96 |
| Share of requests with ContextTokens ≤ 8 | 0.19% | 0.14% |
| GeneratedTokens p50 / p90 / p99 / max | 13 / 55 / 251 / 1,899 | 129 / 424 / 601 / 1,000 |
| GeneratedTokens mean, CV | 27.9, 2.15 | 211, 0.77 |
| Share of requests with GeneratedTokens ≤ 16 | 62.5% | 1.1% |
| Correlation context–generated, linear (log–log) | 0.001 (−0.004) | −0.112 (0.088) |

Two limits show in the lengths (`awk` counts on the same files): 417 code
requests (4.7%) have 7,400–7,437 context tokens, and 11 conversation requests
end at exactly 1,000 generated tokens. Both look like length caps; this is not
confirmed.

Expected against result:

| # | Expected | Result | Status |
|---|---|---|---|
| 1 | About 1 h; code about 2.5, conversation about 5 req/s | 57.3 and 58.4 min; 2.57 and 5.53 req/s | HELD |
| 2 | 100 ns resolution, sorted, few zero gaps | 1 µs in practice (7th decimal always 0); sorted; 0 zero gaps | PARTLY HELD |
| 3 | CV 1.5–3 (whole), 1.2–2 (5-min pieces) | Code 13.15 and 10.15. Conversation 1.09 and 1.03 | NOT HELD: code is far burstier, conversation is near Poisson |
| 4 | Dispersion 1–2 at 0.1 s, grows to more than 10 at 60 s | Code 2.54 → 164. Conversation 1.06 → 16.1 | HELD |
| 5 | Context median about 1,500 and 1,000; tail to about 8,000; CV about 1 | 1,469 and 1,020; max 7,437 and 14,050; CV 0.96 and 0.96 | HELD |
| 6 | Output median about 15 and 100–200; CV above 1 (code), about 1 (conversation) | 13 and 129; CV 2.15 and 0.77 | HELD |
| 7 | \|r\| < 0.2 | 0.001 and −0.112 | HELD |
| 8 | 8 tokens below p1; 16 tokens near the code median | 0.19% and 0.14% of requests ≤ 8; 62.5% and 1.1% ≤ 16 | HELD |

Kingman factor (Ca² + Cs²) ÷ 2, a rough measure of "how much more waiting at
the same utilization". Ca = CV of gaps. Cs is replaced by the CV of
GeneratedTokens. This is a proxy only: decode time grows with output tokens,
but real service time also depends on the prefill and on the batch. Kingman
also assumes independent gaps; the trace does not have them (the dispersion
grows with the window), and Task 4 showed that the formula is weak for very
bursty arrivals.

| Workload | Ca² | Cs² | (Ca² + Cs²) ÷ 2 | ÷ Poisson test |
|---|---:|---:|---:|---:|
| Our uniform test | 0 | ≈ 0 | 0 | 0 |
| Our Poisson test | 1 | ≈ 0 | 0.5 | 1 |
| Our gamma CV 2 test (Task 4) | 4 | ≈ 0 | 2.0 | 4 |
| Conversation, 5-min pieces / whole | 1.07 / 1.20 | 0.60 | 0.83 / 0.90 | 1.7 / 1.8 |
| Code, 5-min pieces / whole | 102.9 / 173.0 | 4.61 | 53.8 / 88.8 | 108 / 178 |

Comparison with our test:
- Conversation arrivals are close to Poisson at short time scales: CV 1.03 in
  5-minute pieces, dispersion 1.06 at 0.1 s and 1.39 at 1 s. Our Poisson test
  matches this; our uniform test (CV 0, dispersion 0) does not. The extra
  burstiness is slow drift: the per-minute rate goes from 3.2 to 8.5 req/s,
  and the dispersion is 16 at 60 s.
- Code arrivals come on and off: 73% of seconds have no request, and bursts
  reach 67 requests in one second. At 1 s the dispersion is 13.2. Our
  burstiest test (gamma CV 2) gave 2.8–5.2 at 1 s in Task 4.
- Prompts: the median real context is 128 times (conversation) and 184 times
  (code) our 8 tokens. Only 0.14–0.19% of real requests have 8 tokens or fewer.
- Output: our 16 tokens is near the code median (13), but only 1.1% of
  conversation requests are that short. Our output never varies (CV 0); real
  output has CV 0.77 (conversation) and 2.15 (code).
- Our test does not represent: long prefills (and the KV memory they use; the
  pilot used at most 2.6% of the KV cache), variable output length, rate drift
  over minutes, and on/off bursts. The trace is a sample of Azure GPU
  services, so its absolute rate says nothing about our Mac. Only its shape
  transfers.

Recommendation for Task 8: for conversation traffic, gamma CV 1.2 matches the
trace at 1 s, which is almost Poisson (Ca² 1.44 against 1), so the planned
Poisson run set (8c) already represents chat arrivals at short time scales,
and gamma CV 2 can stay as the "burstier than chat" point. Code traffic needs
gamma CV 4.7 at 1 s, and no single gamma CV fits it at all scales (a gamma
process levels off near CV² while the trace keeps growing), so do not try to
copy it with gamma. Trace replay is not needed for the talk; if a "real
traffic" slide is wanted later, the least change is one more `Arrival` variant
in `src/config.rs` and `src/driver.rs` that reads time offsets from a file and
divides them by a speed-up factor. Rate drift needs no new code: run one open
point at the peak-minute rate (1.5 × the mean for conversation) to show
"size for the peak minute, not the mean". Variable output length is worth one
optional run set, with `max_tokens` drawn per request from the code file's
GeneratedTokens (median 13, mean 28, so the knee in req/s moves less than with
conversation lengths) and `ignore_eos` kept; the change is the fixed
`"max_tokens": 16` in `src/target.rs:72`.

Charts (`runs/methodology_review/task5-trace/`): `trace-arrivals.png`
(requests per second over time for both files, and the index of dispersion
against the window size) and `trace-lengths.png` (cumulative distributions of
ContextTokens and GeneratedTokens with our 8 and 16 tokens). Both carry the
caption "Data: Azure LLM inference trace 2023 (Patel et al., Splitwise, ISCA
2024)".

Citation (required on every slide that uses the trace): Pratyush Patel, Esha
Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo
Bianchini. "Splitwise: Efficient generative LLM inference using phase
splitting", in Proceedings of the International Symposium on Computer
Architecture (ISCA 2024). ACM, Buenos Aires, Argentina, 2024.

## Anomalies and hypotheses (before Task 6b)

An anomaly is a measured value that the simple story ("a bigger batch costs a
little more per step") does not explain. Each one gets hypotheses, the result
that each hypothesis predicts, and the experiment that separates them. No
experiment below has run yet, except E1 (results under "Predictions → Task 6").

### Engine steps rebuilt from the closed-loop runs

Token events of all requests that the engine serves in one step arrive at the
client within a few milliseconds. Grouping events with gaps under 3 ms gives
the engine steps. Source: `stream_events.csv` of the repeat-1 closed runs in
`runs/vllm-metal-closed-sweep/`, window 10–60 s. TPOT and TTFT are medians of
the three repeats from `records.csv`.

| Workers | Tokens per step (median) | Step time ms (median) | TPOT p50 ms | TTFT p50 ms | Token rate tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 24.8 | 23.5 | 178 | 30 |
| 2 | 2 | 42.0 | 39.6 | 336 | 34 |
| 4 | 4 | 78.5 | 73.4 | 340 | 44 |
| 8 | 8 | 160.4 | 152.0 | 497 | 46 |
| 16 | 16 | 112.2 | 103.2 | 644 | 113 |
| 24 | 24 | 110.6 | 104.8 | 831 | 160 |
| 32 | 32 | 113.7 | 109.1 | 947 | 198 |
| 48 | 48 | 212.4 | 201.5 | 1,414 | 174 |
| 64 | 64 | 220.0 | 208.4 | 1,736 | 205 |

The engine does batch: each step carries one token for every worker. But the
step time is a step function of the batch size, not a smooth line: about
20 ms per sequence from 1 to 8, then about 112 ms flat from 16 to 32, then
about 215 ms from 48 to 64. A batch of 8 costs more than a batch of 32.

### The anomalies

| # | Anomaly | Evidence |
|---|---|---|
| A1 | Step time grows in proportion to the batch from 1 to 8, falls at 16, is flat to 32, and doubles at 48. Closed 48 workers serve fewer requests than 32 (10.6 against 12.8 req/s). | Table above. |
| A2 | One request alone waits 178 ms for its first token. One step takes 25 ms, and an 8-token prefill should take about one step. | Closed 1 worker. Server-side TTFT agrees with the client within 2.5 ms (`scripts/validate_server_metrics.py`, smoke run of 2026-10-13), so the time is inside the server. |
| A3 | Up to 246 requests run at once, but the installed scheduler default is 128. | `server_metrics.csv`; `vllm/config/scheduler.py:44`. |
| A4 | In open loop the server reports `waiting = 0` up to 13.5 req/s while unfinished work grows. The queue is not visible as a queue. | `server_metrics.csv`; Step 1 table. |
| A5 | The fast tier (E2E ≤ 1 s) is never met in open loop, even at 6.5 req/s, about half of capacity. | Step 2. At 6.5 req/s about 14 requests run at once (Little's Law), so steps are in the about-112 ms band: 17 steps × 112 ms ≈ 1.9 s. |
| A6 | Mock: measured wait is 1–6% above the ideal replay at ρ 0.95, and a few uniform requests wait about 25 ms. | Task 4. No slide claim depends on it. |

### Hypotheses, predictions and experiments

**A1. Step time against batch size.**
- H1a, kernel choice in MLX (the Metal backend). For a small batch, the 4-bit
  matrix multiply runs as a matrix-vector kernel once per sequence, so the
  4.5 GB of weights are read once per sequence. At 273 GB/s one read takes
  about 16.5 ms, so a batch of 8 costs about 8 × 16.5 = 132 ms plus overhead
  (measured 160 ms). From some batch size between 9 and 16, a tiled
  matrix-matrix kernel reads the weights once per tile of rows, so the cost is
  flat; a second tile starts above 32 rows. Predicts: the same step function
  in pure MLX, with no vLLM.
- H1b, per-request CPU work in vllm-metal or the API server (sampling,
  detokenizing, streaming). Predicts: cost in proportion to the batch at all
  sizes, and no fall at 16. The fall at 16 already argues against it as the
  main cause, but it can add to the slope.
- H1c, run order and heat. The closed sweep ran in ascending order. Predicts:
  a shuffled order gives different step times.
- Experiment E1 (decides H1a): time one decode step of the same model in
  `mlx_lm`, without vLLM, at batch 1, 2, 4, 8, 10, 12, 14, 16, 24, 32, 40, 48
  and 64. If the step function appears, the cause is in MLX, not in the vLLM
  scheduler. Needs no new software (MLX 0.32.1 and mlx_lm 0.32.0 are in the
  vllm-metal environment) and no sudo. The server must be stopped.
- Experiment E2 (decides H1b, H1c): a closed sweep at the same batch sizes in
  shuffled order, with `/metrics` before and after each run
  (`vllm:iteration_tokens_total` gives the batch size per step from the
  server side), `powermetrics` (GPU busy share and frequency: a busy GPU
  means H1a, an idle GPU with a busy CPU means H1b), and `py-spy` on the
  engine process at 8 and 32 workers (CPU share of each function).

**A2. Slow first token.**
- H2a, front end. The API server (tokenizer, request set-up, asyncio) adds
  about 150 ms before the engine sees the request.
- H2b, engine prefill. The prefill step of a new request is slow, for example
  because MLX builds a new graph for the new shape.
- H2c, schedule. A new request waits for the pipelined decode step in flight
  (vllm-metal runs a one-step-ahead decode pipeline only on pure decode steps,
  `vllm_metal/v1/model_runner.py:1310`).
- Experiment E3: `/metrics` histograms for a 1-worker run:
  `vllm:time_to_first_token_seconds`, `vllm:request_queue_time_seconds`,
  `vllm:request_prefill_time_seconds`. H2a predicts TTFT − queue − prefill
  ≈ 150 ms. H2b predicts prefill ≈ 150 ms. H2c predicts queue ≈ one step.
  `py-spy` on the API server process shows where the front-end time goes.

**A3. 246 running requests.** Resolved by reading the source. For the API
server, vLLM sets `max_num_seqs` = 256 when the device has less than 70 GiB
(`vllm/engine/arg_utils.py:2756`, then `min(256, max_num_batched_tokens
2048)` at line 2987). vllm-metal reports the unified memory (64 GiB,
`vllm_metal/platform.py:131`) and is not a CPU platform
(`platform.py:70`), so it gets 256. The class default 128 applies only when
these engine arguments are not used. New runs pass `--max-num-seqs 256`
explicitly, so no log is needed.

**A4. The queue hides inside the batch.**
- H4: with `max_num_seqs` 256 and KV use at most 2.6%, the scheduler admits
  every request at once. Extra load makes a bigger batch, so each step is
  slower (A1), and nothing waits.
- Prediction for run set 8d (`--max-num-seqs 32`): above capacity `waiting`
  becomes larger than 0; step time stays at about 112 ms and TPOT stays flat;
  TTFT grows instead. Because batch 32 is the cheapest batch per token (A1),
  served rate at overload is equal to or higher than with 256. If `waiting`
  stays 0 or TPOT still grows, H4 is wrong.
  SUPERSEDED on 2026-10-18 by Task 8 prediction 8d.3, before any 8d run: E2
  shows closed 32 workers at 181 tok/s and 64 workers at 193 tok/s, so a cap
  of 32 is not the cheapest per token, and open-loop steps also carry the
  prompt tokens of new requests.

**A5. No fast tier at half load.** A consequence of A1, not a separate cause:
above 8 requests a step costs about 112 ms, so 16 output tokens take about
1.9 s. Predicts: in E2, E2E at 2 workers ≈ 17 × 42 ms ≈ 0.7 s (fast tier
met) and at 16 workers ≈ 1.9 s. No separate experiment.

**A6. Mock residual.** Parked. It does not change a claim.

### Open discrepancy: vllm-metal is slow at 2–12 tokens per step

Status: OPEN, not investigated further (user decision, 2026-10-18).

The model itself batches well. In pure MLX (E1, E1b) a step of 8 tokens
takes 64 ms, with the batch as (batch, 1) or packed as (1, batch); the shape
does not matter. In vllm-metal the same step takes 160 ms (decode, 8 requests)
or 178 ms (prefill, one 8-token prompt). The extra GPU work grows by about
12.5 ms per token from 1 to 12 tokens (246 ms at 12), then almost stops at 14
(144 ms) and 16 (109 ms). The GPU is 99% busy, so this is GPU work, not idle
time (E2). The jump between 12 and 14 is at the same point where the pure-MLX
matrix multiply stops rising, which suggests a kernel choice that depends on
the number of rows.

Candidates, most likely first. Each is a hypothesis, not a finding:

| # | Candidate | Short hypothesis | Test that decides |
|---|---|---|---|
| C1 | Paged attention kernel (vllm-metal's own Metal kernel, `attention/impls/sdpa.py`) | For few query tokens, the kernel uses a launch layout with low GPU occupancy, so each token adds time; from about 14 tokens a different layout, or enough parallel work, fills the GPU. | Time the kernel alone at 1–64 tokens with a short context. |
| C2 | Dtype or layout before the matrix multiply (casts to `kv_cache.dtype`, `mx.contiguous` copies, head-dim padding, `sdpa.py:451–527`) | vllm-metal gives MLX a dtype or memory layout that pure MLX does not use, and MLX picks a slower kernel for fewer than about 14 rows. | Log the dtypes and shapes at the linear-layer inputs inside vllm-metal; repeat E1 with them. |
| C3 | One-step-ahead decode pipeline (`VLLM_METAL_DECODE_PIPELINE`, on by default) | The pipeline adds GPU work for each token, for example a lazy sample graph that runs again. | Server A/B: closed 8 and 16 workers with `VLLM_METAL_DECODE_PIPELINE=0`. |
| C4 | Sampler (`v1/sampling_batch.py:612`) | It casts each token's 128,256 logits to float32 and samples with a sync per step. Less likely: `py-spy` puts only 2.7–4.2% of samples there. | `py-spy` share against batch; time the sampler alone. |
| C5 | Paged KV-cache write (`reshape_and_cache`, one dispatch per layer) | A fixed cost per token per layer. Least likely: it writes only a few KB per token per layer. | Same as C1. |

Excluded: MLX's matrix multiply on the packed shape (E1b, within 3%); NAX
prefill attention (`VLLM_METAL_DISABLE_NAX`, M5 only; this Mac is M4 Pro);
CPU work (E2: GPU 99% busy, CPU almost idle); run order and heat (E2).

Talk scope: present the measured effect (same model, same GPU, 64 ms in MLX
and 160 ms in the server at 8 requests) and say that the exact kernel is not
identified. Do not name a candidate as the cause.

### Task 6 commands

E1 (no server; the agent or the user runs it, about 3 minutes):

```bash
# Stop the vLLM server first.
/opt/homebrew/Cellar/vllm-metal/0.30.0/libexec/bin/python scripts/mlx_step_bench.py \
  -o runs/methodology_review/task6/e1-mlx-steps.json
```

E2 and E3 (the user runs them, about 25 minutes, three terminals):

```bash
# Terminal 1: build, then start the server with its log saved.
cargo build --release
mkdir -p runs/methodology_review/task6b
vllm serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.60 --max-num-seqs 256 \
  2>&1 | tee runs/methodology_review/task6b/server.log

# Terminal 2: GPU, CPU and thermal samples once per second for the whole session.
sudo powermetrics --samplers cpu_power,gpu_power,thermal -i 1000 \
  -o runs/methodology_review/task6b/powermetrics.txt

# Terminal 3: the runs (15 closed runs in shuffled order, then 4 profile runs).
# It asks for the sudo password once, before the py-spy pass.
runs/methodology_review/task6b/run.sh 2>&1 | tee runs/methodology_review/task6b/run.log
```

Stop `powermetrics` (Ctrl-C) after `ALL-DONE`. Do not use the Mac for other
work during the session.

### What the user needs

- Install: `py-spy` only (`brew install py-spy`). It needs `sudo` to attach
  to a process on macOS.
- Already present: `powermetrics` (needs `sudo`), MLX and mlx_lm in the
  vllm-metal environment, `/metrics` on the server.
- Order: E1 first (no server; decides the main question in minutes), then E3
  and E2 in one server session (Task 6b).

## Data sources and citations

- **Azure LLM inference trace 2023** (Task 5). Store under
  `runs/methodology_review/traces/azure-llm-2023/`.
  - Page: <https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md>
  - Files: <https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv>
    and <https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv>
  - Columns: `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`. The page says
    "collected on November 11th 2023", but the timestamps in both files are
    2023-11-16, 18:15:46 to 19:14:19 (time zone not stated; checked in
    Task 5). This review uses the data date, 2023-11-16.
  - Required citation (put it on every slide that uses the trace): Pratyush
    Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed
    Maleki, Ricardo Bianchini. "Splitwise: Efficient generative LLM inference
    using phase splitting", in Proceedings of the International Symposium on
    Computer Architecture (ISCA 2024). ACM, Buenos Aires, Argentina, 2024.
- **Open versus closed systems**: B. Schroeder, A. Wierman, M. Harchol-Balter,
  "Open versus closed: a cautionary tale", NSDI 2006:
  <https://www.usenix.org/legacy/events/nsdi06/tech/full_papers/schroeder/schroeder_html/index.html>.
- **Azure LLM inference trace 2024** (optional, one week, 2024-05-10 to 05-19):
  <https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md>.
  Read its page for its own citation before use.

## Work log

### Step 1: Complete the card (2026-10-17)

Owner: Agent, then User.

Done: read the configuration, provenance, records and summaries of the 54
saved Metal runs (27 closed in `runs/vllm-metal-closed-sweep/`, 27 open in
`runs/vllm-metal-open-sweep/`), the request code, and the local tokenizer.
Filled the card above.

Run, measured, calculated:
- Config and provenance fields of all 54 runs: one model, one prompt,
  60 s runs, 10 s warmup, vLLM 0.30.0, `git_sha` "unknown".
  Closed: `max_in_flight` 128, `target_notes` = smoke text.
  Open: `max_in_flight` 512, `target_notes` = launch command.
- Request outcomes from `records.csv`: 11,703 (closed) + 19,530 (open) =
  31,233 requests, all `ok`, all 16 output tokens. No run has a failure reason
  in `summary.json`.
- Planned arrival gaps from `intended_ns`: coefficient of variation 0 in all
  open runs.
- Prompt length with the cached tokenizer: 7 tokens.
- Unfinished work (sent minus finished, from `actual_ns` and `completed_ns`),
  median of 3 repeats:

  | Offered req/s | Slope 10–60 s | Slope 35–60 s | At 10 s | At 60 s |
  |---:|---:|---:|---:|---:|
  | 6.5 | −0.00 | −0.01 | 14 | 14 |
  | 9 | −0.00 | −0.00 | 20 | 20 |
  | 11 | −0.01 | −0.00 | 47 | 47 |
  | 12 | +0.44 | −0.01 | 61 | 96 |
  | 12.5 | +1.56 | +0.41 | 68 | 153 |
  | 13 | +2.56 | +2.07 | 76 | 211 |
  | 13.5 | +3.38 | +2.70 | 86 | 258 |
  | 14.5 | +4.41 | +2.51 | 100 | 311 |
  | 16.5 | +5.56 | +4.33 | 129 | 433 |

- Host facts read today: Apple M4 Pro, 64 GB, macOS 26.5.2.

Result:
- Three items are UNKNOWN or only defined for teaching: the decision, the SLO,
  and the real traffic shape.
- 12 req/s is not clearly overload. Its backlog levels off near 96 requests
  after about 30 s. By Little's Law, 96 ÷ 12 req/s = 8 s, which matches the
  measured E2E of about 8 s. The `talk.md` claim "12 req/s ... with a growing
  backlog" does not agree with the late slope. The 10 s warmup is too short at
  this rate. Task 3 must rank this threat, and Task 8b (300 s runs) tests it.
- `max_num_seqs` is not known: the source default (128) disagrees with the
  observed 246 running requests. The `talk.md` statement about a default of 256
  is not supported.
- The closed sweep has no recorded launch flags, so it cannot be shown to use
  the same server settings as the open sweep.

Unknown: decision; real SLO; production client model; real arrival burstiness
and request lengths; cache setting; power state; model revision;
`max_num_seqs`; launch flags of the closed sweep.

Next.
- User: answer the UNKNOWN items (questions in the chat).
- Agent: Task 2, after the user says "next".

### Step 1b: User answers and design changes (2026-10-17)

Owner: User, then Agent.

User answers:
1. Decision: teaching only, on a Mac mini M4 Pro 64 GB with a local LLM on Metal.
2. SLO: keep it realistic for a person who sends a request and waits for the
   result; define fast, medium and slow.
3. Open and closed: teach them as two personas, people (open) and agents
   (closed).
4. No production trace. Question: what does Poisson arrival data add?
5. Cache, power and revision are not known. Record that all requests use the
   same prompt; consider prompts that differ but have the same token count.
6. No server log. Question: must the experiments start again?

Measured and calculated:
- Prefix caching (installed source): on by default; only full blocks are
  cached; a hit covers at most prompt length − 1 tokens; default block 16
  tokens; vllm-metal can change the block size.
- Prompt with a fixed-width 6-digit prefix (`000123 Explain saturation in one
  sentence.`): 8 tokens for all of 20,000 random prefixes (local tokenizer).
  Without the prefix: 7 tokens.

Result (proposals for the user to confirm):
- SLO tiers: fast, medium and slow as in card row 2. The current goodput
  limits in `analysis/analyze.py` (`SLOS`) change to these tiers in Task 2.
- Arrival variability: Poisson gives one more point on Kingman's curve
  (Ca² = 1 against Ca² = 0 for uniform). A gamma arrival with CV 2 (Ca² = 4)
  gives a third point. `satlab` has uniform and Poisson only.
  `vllm bench serve --burstiness k` gives gamma arrivals with CV = 1/√k, but it
  is a different client.
- Prompt: add a fixed-width request number before the prompt in new runs, so
  that no two prompts share a prefix and the token count stays 8.
- Data: start a new, fully recorded baseline session for the talk. The saved
  54 runs become the pilot. They still set the predictions (Task 7) and test
  the analysis (Tasks 2, 3, 6a). Reason: the closed-sweep flags,
  `max_num_seqs`, cache state, power state and harness revision cannot be
  recovered, and the closed and open sweeps ran 13 days apart.

Unknown: real traffic burstiness (no trace).

Next.
- User: confirm or change the four proposals.
- Agent: Task 2 after "next".

### Step 1c: Decisions (2026-10-17)

Owner: User.

Decisions:
1. SLO tiers fast, medium and slow (card row 2): confirmed.
2. Gamma arrivals: add to `satlab` with the least code. Good to have, not
   required.
3. Prompt: a random 6-digit number before the prompt, so that requests do not
   share a prefix.
4. New baseline session: all new runs go under `runs/methodology_review/`.
   The 54 saved runs stay where they are and are the pilot.
5. Public Azure LLM trace: later. The user downloads it and stores it under
   `runs/methodology_review/traces/` (not tracked by git).

Changes to the task list:
- Task 2 also changes `SLOS` in `analysis/analyze.py` to the three tiers.
- Before Task 4, the agent adds the gamma arrival and the random prompt prefix
  to `satlab`, so that the mock test can include a third arrival type.
- Task 5 waits for the trace.

Next.
- User: say "next" to start Task 2.
- Agent: Task 2.

### Step 2: Calculate the key numbers again from the primary records (2026-10-17)

Owner: Agent.

Done:
- `analysis/analyze.py`: `SLOS` changed to the three confirmed tiers (fast,
  medium, slow). New latency metrics `ttft_from_plan` and `e2e_from_plan` that
  start at `intended_ns` (the planned send). The table shows p95 E2E from the
  plan next to p95 E2E from the actual send.
- `docs/measurement-contract.md` and the old benchmark guide (deleted 2026-10-17): definitions updated.
- Ran the analysis on the 54 pilot runs. Output:
  `runs/methodology_review/task2-pilot/` (`sweep.csv`, charts).

Run, measured, calculated (all from `records.csv` and `stream_events.csv`;
every run passed every contract check against its `summary.json`):
- Planned send compared with actual send, largest difference over all runs:
  - Open loop (27 runs): E2E p50 3.71 ms, p95 3.00 ms, p99 3.39 ms,
    p99.9 4.58 ms; TTFT p95 3.24 ms. Largest relative change at p99: 0.093%.
    Send lag p99 ≤ 3.91 ms, p99.9 ≤ 6.55 ms.
  - Closed loop (27 runs): 0 ms at every percentile.
- Goodput per SLO tier (requests per second that meet the tier, median of
  3 repeats):

  | Load | Served | Fast | Medium | Slow |
  |---|---:|---:|---:|---:|
  | open 6.5 req/s | 6.50 | 0 | 6.50 | 6.50 |
  | open 11 req/s | 11.00 | 0 | 11.00 | 11.00 |
  | open 12 req/s | 11.30 | 0 | 0 | 11.30 |
  | open 12.5 req/s | 10.80 | 0 | 0 | 3.58 |
  | open 16.5 req/s | 10.40 | 0 | 0 | 0.50 |
  | closed 1 worker | 1.88 | 1.88 | 1.88 | 1.88 |
  | closed 2 workers | 2.16 | 2.16 | 2.16 | 2.16 |
  | closed 4 workers | 2.80 | 0 | 2.80 | 2.80 |
  | closed 32 workers | 12.80 | 0 | 12.80 | 12.80 |
  | closed 48 workers | 10.56 | 0 | 0 | 10.56 |

- Per-rate table (p95 TTFT, p95 E2E, unfinished work, Little's Law): same
  values as in Step 1 and in the printed table; see
  `runs/methodology_review/task2-pilot/sweep.csv`, columns `*_from_plan_*`
  for each run.

Result:
1. In open loop the harness kept its schedule. Latency from the planned send
   is at most 4.6 ms longer than latency from the actual send. So the saved
   open-loop latency is not distorted by late sends (coordinated omission).
2. In closed loop the difference is 0 by design: a closed loop has no
   schedule, so it cannot show the time a request would have waited before it
   was sent. This is a teaching point for the closed-loop slide.
3. The fast tier (E2E ≤ 1 s) is never met in open loop, not even at 6.5 req/s,
   about half the knee. There p95 E2E is 2.35 s, 4.3 times the single-user
   value (0.54 s at 1 closed worker). Batching slows every request long before
   capacity. "Fast" on this Mac means one or two users.
4. The medium tier holds up to 11 req/s (open) and 32 workers (closed), then
   drops to 0 in one step. The drop is all or nothing because uniform arrivals
   give almost the same latency to every request. Prediction for Task 4 and
   8c: Poisson arrivals make the drop more gradual.
5. The slow tier still holds at 12 req/s. This agrees with Step 1: 12 req/s is
   a steady state with high latency (E2E about 8 s), not a growing backlog.

Unknown: the effect of Poisson arrivals on the goodput drop.

Note: `talk.md` still uses the old SLO wording. Task 10 replaces it.

Next.
- User: say "next" to start Task 3. Download the Azure trace when convenient
  (links in "Data sources and citations").
- Agent: Task 3, the threats-to-validity review.

### Step 3: Threats-to-validity review (2026-10-17)

Owner: Agent, then User.

Done: examined `src/target.rs` (how token times are taken), `src/driver.rs`,
the saved records and the server notes. Wrote 18 threats in four classes,
in sequence of their effect on slide claims (section "Threats to validity").
Tested four threats with the pilot data.

Run, measured, calculated:
- Token timestamps: `read_sse` takes `clock.now_ns()` for each parsed event
  (`src/target.rs:204`). Gaps below 1 ms between events of one request: 0% of
  42,000 gaps in runs open 6.5, 12 and 16.5 req/s (repeat 1) and closed
  32 workers (repeat 1).
- Served rate by window, median of 3 repeats (completions with the last token
  in the window ÷ 25 s; token events of successful requests ÷ 25 s ÷ 16):

  | Offered req/s | Served 10–35 s | Served 35–60 s | Tokens 35–60 s ÷ 16 |
  |---:|---:|---:|---:|
  | 11 | 10.96 | 11.04 | 10.99 |
  | 12 | 10.56 | 11.96 | 11.97 |
  | 12.5 | 9.84 | 11.76 | 12.14 |
  | 13 | 9.64 | 10.96 | 11.79 |
  | 13.5 | 9.28 | 10.92 | 12.03 |
  | 14.5 | 8.64 | 11.68 | 11.95 |
  | 16.5 | 8.12 | 12.80 | 12.64 |

- SLO attainment (share of requests intended in the window that meet the
  tier), median of 3: 11 req/s fast 0%, medium 100%, slow 100%;
  12 req/s fast 0%, medium 0.7%, slow 100%; 12.5 req/s slow 29.6%.

Result:
- Threat 1 is confirmed. The capacity of this server is about 12 req/s
  (about 190 tok/s). Above it, the late-window served rate stays near
  capacity; the lower values come from the early window, while the batch
  still grows. The slide "overload lowers throughput" must change.
- With uniform arrivals, 12 req/s is at capacity and still steady (backlog
  about 96, E2E about 8 s). This agrees with queueing theory for arrivals
  with no variability, and it makes the Poisson test (Task 4, 8c) a strong
  slide: theory predicts that with Poisson arrivals at capacity the backlog
  does not settle; it keeps growing and the wait has no limit.
- Threats 16–18 are resolved for the pilot. Threats 2–7 are open; the next
  tasks test them.

Unknown: everything marked OPEN in the table.

Next.
- User: read the sequence of threats and correct it if you know more.
  Then say "next".
- Agent: Task 4, the queueing-theory test on the mock. Before it, add the
  gamma arrival and the random prompt prefix to `satlab` (Step 1c).
  The trace files are present in `runs/methodology_review/traces/azure-llm-2023/`
  for Task 5.

### Step 4: Test the harness against queueing theory (2026-10-17)

Owner: Agent.

Done:
- Verified how the harness and the mock work, from the code and saved runs
  (section "How the harness and the mock work").
- Added to `satlab`: gamma arrivals (`{"gamma": {"cv": 2.0}}`, `rand_distr`
  crate) and a random 6-digit prompt prefix. One unit test each. Rust tests
  (40 unit, 10 integration) and clippy pass.
- Added `analysis/queue_theory.py`: exact FCFS simulator for c servers with a
  fixed service time, Kingman (Allen–Cunneen) mean wait, burstiness (CV of
  gaps, index of dispersion of counts per second). A test checks the simulator
  against the exact M/D/1 formula.
- Wrote the predictions before the runs (section "Predictions", 04:58 UTC).
- Ran 36 mock runs: uniform, Poisson and gamma (CV 2) arrivals × 39, 55, 66,
  74 req/s (ρ 0.50, 0.70, 0.84, 0.95 at the measured 51.2 ms service time)
  × 3 repeats × 60 s. Script: `runs/methodology_review/task4-mock/run.sh`.

Run, measured, calculated:
- Runs: `runs/methodology_review/task4-mock/{uniform,poisson,gamma2}/`;
  each sweep passed the contract checks in `analyze.py`.
- Comparison: `runs/methodology_review/task4-mock/check/queue-check.json` and
  `queue-check.png`. No-wait E2E 51.17 ms (uniform 39 req/s).
- Mean wait at ρ = 0.95 (3 repeats): uniform 0.26–0.29 ms; Poisson
  73.5–100.5 ms; gamma 222–584 ms.

Result:
- The harness and the mock agree with queueing theory. Measured wait is
  inside the simulated band in 32 of 36 runs (mean) and 34 of 36 (p95), and
  within 10% of an ideal replay of the same send times in 36 of 36 runs.
  This is independent check 1 for Task 9: PASS.
- The harness delivers the requested burstiness: the sent CV stays within 0.02
  of the planned CV, and no run failed the verdict, even with gamma arrivals.
- Burstiness must be measured in each run. The same setting gives an index of
  dispersion from 2.8 to 5.2 between gamma runs, and those differences explain
  the runs outside the band.
- Slide material: same server, same mean rate, only the arrival variability
  changes. At ρ = 0.95 the mean wait is about 0.3 ms (uniform), about 80 ms
  (Poisson) and about 380 ms (gamma, CV 2).
- Kingman's formula shows the direction and the (Ca² + Cs²) ÷ 2 factor. It is
  good for Poisson and poor for very bursty arrivals with a fixed service time.

Unknown: why the measured wait is 1–6% above the replay at ρ = 0.95.

Next.
- User: say "next" to start Task 5. The trace files are in place.
- Agent: Task 5, real-traffic burstiness and request lengths from the Azure
  trace, measured with the same functions (`burstiness` in
  `analysis/queue_theory.py`).

### Step 5: Measure real traffic (2026-10-17)

Owner: Agent

Done: wrote the expected values in section "Workload: real traffic against our
test" before the analysis (14:18 UTC). Added `analysis/trace.py` (147 lines;
reuses `burstiness` from `analysis/queue_theory.py` and the chart style from
`analysis/analyze.py`). Analysed the two trace files separately. Wrote the
results, the comparison and the recommendation into that section. Corrected
the collection date in "Data sources and citations".

Run, measured, calculated:
- `.venv/bin/python -I analysis/trace.py runs/methodology_review/traces/azure-llm-2023/*.csv -o runs/methodology_review/task5-trace`.
  Output: `trace-summary.json`, `trace-arrivals.png`, `trace-lengths.png`.
- `AzureLLMInferenceTrace_code.csv`: 8,819 rows over 57.3 min, 2.57 req/s.
  CV of gaps 13.15 (whole), 10.15 (median of 5-min pieces). Index of
  dispersion 2.54 / 13.19 / 71.6 / 164.0 at 0.1 / 1 / 10 / 60 s. 73% of
  seconds empty; peak 67 req in 1 s. Context median 1,469, output median 13
  (CV 2.15). Kingman factor 54–89.
- `AzureLLMInferenceTrace_conv.csv`: 19,366 rows over 58.4 min, 5.53 req/s.
  CV of gaps 1.09 (whole), 1.03 (5-min pieces). Index of dispersion 1.06 /
  1.39 / 3.86 / 16.1. Per-minute rate 3.2–8.5 req/s. Context median 1,020,
  output median 129 (CV 0.77). Kingman factor 0.83–0.90 (Poisson test 0.5,
  uniform test 0).
- Row count: the task brief said 8,818 and 19,365. That is `wc -l` minus the
  header; the last line has no newline, so each file has one more row.
- Date: both files show 2023-11-16; the dataset page says November 11th 2023.
- `.venv/bin/python -m unittest tests.test_open_sweep`: 4 tests OK.

Result:
- Conversation traffic is close to Poisson at short time scales (gamma CV 1.2
  matches it at 1 s) and drifts slowly over minutes (peak minute 1.5 × the
  mean). Our Poisson test represents it at short scales; our uniform test does
  not.
- Code traffic is on/off and far burstier than any test so far (dispersion
  13 at 1 s against 2.8–5.2 for gamma CV 2). No single gamma CV fits it.
- Our request is far from both: real prompts are 128–184 times longer at the
  median, and real output varies (CV 0.77–2.15). Prefill, KV memory and
  output variability never act in our test. State this scope on a slide
  (threat 11).
- 6 of 8 expectations held, 1 partly, 1 not held (code CV 13, not 1.5–3;
  conversation CV 1.1, not above 1.5). The prediction underrated both how
  bursty code traffic is and how Poisson-like chat traffic is.

Unknown: the time zone of `TIMESTAMP`; why the page date differs from the
data; what makes the code bursts (one client or a batch tool, or the
sampling); whether the context pile-up near 7,437 tokens and the 1,000-token
output maximum are limits; the real service-time variability on vLLM-metal
with variable output (GeneratedTokens CV is a proxy only).

Next. Agent: Task 6a: read the vLLM scheduler and the vllm-metal model runner;
rebuild engine steps from `stream_events.csv` of the closed runs; write
hypotheses and the 6b commands.  User: say "next".

### Step 5b: Anomalies before Task 6 (2026-10-17)

Owner: Agent.

Done: rebuilt the engine steps of the closed runs from `stream_events.csv`;
read the vLLM argument defaults, the vllm-metal platform and the decode
pipeline gate; wrote the section "Anomalies and hypotheses".

Run, measured, calculated:
- Engine steps: token events grouped by gaps under 3 ms, repeat-1 closed
  runs, 10–60 s. Step time 24.8 ms (1 worker), 160.4 ms (8), 112.2 ms (16),
  113.7 ms (32), 212.4 ms (48); tokens per step = workers.
- Source: `vllm/engine/arg_utils.py:2756` and `:2987`;
  `vllm_metal/platform.py:70` and `:131`; `vllm_metal/v1/model_runner.py:1310`.
- Server metric names present in vLLM 0.30.0 (`vllm/v1/metrics/loggers.py`):
  `iteration_tokens_total`, `request_queue_time_seconds`,
  `request_prefill_time_seconds`, `request_decode_time_seconds`.

Result:
- Step time is a step function of the batch size (A1). Batch 8 costs more
  than batch 32. This explains the TPOT slope (threat 2), the closed 48-worker
  dip, and why the fast tier fails at half load (A5).
- `max_num_seqs` is 256 by source (A3); threat 7 is resolved.
- The `+5.3 ms per admitted request` slope is the wrong model; the cost is in
  steps (threat 5).

Unknown: whether the step function comes from MLX kernels or from vLLM
(E1 decides); where the 178 ms first-token time goes (E3).

Next. User: read "Anomalies and hypotheses"; install `py-spy`; say whether
the agent may run E1 with the server stopped. Agent: write E1 and the 6b
commands for E2 and E3.

### Step 6a: E1, pure MLX step times (2026-10-17)

Owner: User ran E1; Agent analysed.

Done: the user ran `scripts/mlx_step_bench.py` with the server stopped. The
agent compared it with the pilot engine steps and the Task 6 predictions.
Also: deleted `docs/arrival-queuing-theory.md` and `benchmark.md` (superseded
by this review; user decision), and fixed stale lines in the contract, the
runbook and this review.

Run, measured, calculated:
- `/opt/homebrew/Cellar/vllm-metal/0.30.0/libexec/bin/python scripts/mlx_step_bench.py -o runs/methodology_review/task6/e1-mlx-steps.json`
- Pure MLX step 19.1 / 64.3 / 96.8 / 98.7 / 191.4 / 193.3 ms at batch 1 / 8 /
  16 / 32 / 40 / 48. vLLM minus MLX: 96.1 ms at batch 8, 15.4 ms at 16.

Result: predictions 2 and 3 held, prediction 1 partly held. The jump above 32
comes from MLX (blocks of 32 rows in the 4-bit matrix multiply). The slow
batches 2–8 come from vllm-metal, not MLX. The same model is 2.5 times faster
at batch 8 in pure MLX than in the server.

Unknown: what vllm-metal does per sequence at batches 2–8 (CPU or GPU work);
where the 178 ms first-token time goes. E2 and E3 decide (predictions 4–14).

Next. User: run E2 and E3 ("Task 6 commands"); say when `ALL-DONE` appears.
Agent: Task 6c, analyse E2 and E3.

### Step 6c: E2 and E3 results (2026-10-18)

Owner: User ran E2 and E3; Agent analysed.

Done: analysed the 15 closed runs of session B with the server metrics, GPU
samples and four `py-spy` profiles, and the 15 runs of session A. Added
`analysis/engine_steps.py` (engine steps, server histogram means and GPU
samples per run, and the chart). Added the packed-batch columns to
`scripts/mlx_step_bench.py` and wrote the E1b predictions before any run.

Run, measured, calculated:
- `.venv/bin/python analysis/engine_steps.py runs/methodology_review/task6b/runs/20261018T08* --metrics runs/methodology_review/task6b/metrics --powermetrics runs/methodology_review/task6b/powermetrics.txt --mlx runs/methodology_review/task6/e1-mlx-steps.json -o runs/methodology_review/task6b/report`
- Results table and prediction status: "Predictions → Task 6".
- `py-spy` (engine busy-loop thread, 2,996–2,999 samples per profile): 86–93%
  in `mx.async_eval`.

Result:
- The GPU is busy 99% of the time at every batch size. The CPU is almost idle.
  The engine thread waits for MLX. So the extra cost at 2–12 sequences is GPU
  work that vllm-metal sends (H1b-GPU), not Python work (H1b-CPU is false).
- The step time rises by about 20 ms per sequence up to 12 workers (245.9 ms),
  then falls to 109 ms at 16. A batch of 12 makes 46 tok/s; a batch of 16
  makes 116 tok/s.
- The 178 ms first-token time at 1 worker is prefill (178.4 ms), not queue or
  front end (H2b).
- Run order, heat, prefix caching and client CPU do not explain the results.

Unknown: which GPU work vllm-metal adds at 2–12 sequences: the matrix multiply
on the packed shape (H1d) or vllm-metal's own kernels (H1e). E1b decides.

Next. User: stop the server, run E1b (one command, about 5 minutes), say when
done. Agent: compare E1b with predictions 15–17 and close Task 6.

### Step 6d: E1b, the packed batch in pure MLX (2026-10-18)

Owner: User ran E1b; Agent analysed.

Done: compared E1b with predictions 15–17; redrew the engine-step chart with
the pure-MLX lines.

Run, measured, calculated:
- `/opt/homebrew/Cellar/vllm-metal/0.30.0/libexec/bin/python scripts/mlx_step_bench.py -o runs/methodology_review/task6/e1b-mlx-packed.json`
- `analysis/engine_steps.py ... --mlx runs/methodology_review/task6/e1b-mlx-packed.json`
- Packed matmul within 3% of the (batch, 1) matmul at every batch; packed step
  64.5 ms at 8 tokens against 160.4 ms (decode) and 178.4 ms (prefill) in vLLM.

Result: H1d is false and H1e holds. MLX does the same matrix multiply for both
shapes. The extra 100 ms at 8 tokens is GPU work that only vllm-metal does.
The same cost explains both A1 (slow small batches) and A2 (slow first token).

Task 6 conclusion:
- A1: two causes. Above 32 sequences, MLX's 4-bit matrix multiply works in
  blocks of 32 rows (validated, E1). At 2–12 tokens per step, vllm-metal adds
  about 12.5 ms of GPU work per token, which almost stops from 14 (validated,
  E2 and E1b). The exact kernel is not identified.
- A2: the 178 ms first token is prefill (E3), with the same vllm-metal cost.
- A3: resolved (source and launch flag). A4 and A5: follow from A1; A4 is
  tested in run set 8d.
- For the talk: the same model on the same GPU takes 64 ms per step in its own
  library and 160 ms in the server at 8 requests. A batch of 12 makes 46 tok/s,
  a batch of 16 makes 116 tok/s. This is Metal-specific (threat 2); do not
  claim it for vLLM on CUDA.

Unknown: which vllm-metal kernel does the extra work. Candidates C1–C5 are in
"Anomalies → Open discrepancy". Not investigated further (user decision).

Next. User: decide whether to identify the kernel (needs Xcode Instruments, or
a per-kernel timing test) or to close Task 6. Agent: Task 7, predictions for
the Task 8 run sets.

### Step 7: Predictions for the Task 8 run sets (2026-10-18)

Owner: Agent wrote; User approves and commits.

Done: wrote "Predictions → Task 8": six run sets with rates, durations and
repeats; 24 predictions with thresholds; a map from each slide claim to its
prediction. Marked the old A4 prediction for `--max-num-seqs 32` as
superseded, with the reason.

Run, measured, calculated: pilot anchors recalculated from `records.csv` and
`server_metrics.csv` of the 27 pilot open runs; closed anchors from
`task6b/report/engine-steps.json`. No new runs.

Result: the predictions are ready for review. Two changed because of Task 6:
the batch cap (8d.3) now predicts lower capacity, and the Poisson effect is
predicted to be small at 9 req/s and large at 11 req/s (8c.2, 8c.3).

Unknown: how the step cost grows with the prompt tokens of new requests in
open loop; 8d.3 and 8d.4 depend on it.

Next. User: review the predictions; commit `docs/methodology-review.md` before
any Task 8 run. Agent: Task 8 preparation (`scripts/sweep.py` saves the full
`/metrics` before and after each run; attainment as the SLO measure, threat
8), then the 8a commands.

### Step 8-prep: Preparation for the Task 8 runs (2026-10-18)

Owner: Agent.

Done:
- `scripts/sweep.py`: new options `--duration`, `--warmup`, `--repeats` (1–3;
  forward, reversed, shuffled); the full `/metrics` text before and after each
  run is saved as `metrics_before.prom` and `metrics_after.prom`.
- `analysis/analyze.py`: SLO attainment per tier (`attainment_<tier>`, the share
  of requests intended in the window that meet both limits); the table shows
  the medium tier (threat 8).
- Predictions 8c.5 and 8c.6 (gamma) added at 09:42 UTC; set 8c-gamma added;
  8f marked skipped (no NVIDIA GPU).
- `runs/methodology_review/task8/`: base configs (uniform, Poisson, gamma CV 2;
  no seed, so each repeat gets a new schedule), closed configs 16/32/64, and
  `run.sh <set>`. `run.sh` stops if `/metrics` is not reachable or if the
  server log does not show the expected `max_num_seqs`.

Run, measured, calculated (mock and a local stand-in for `/metrics`; no Metal
server ran):
- One 5 s Poisson run with snapshots: `metrics_before.prom`,
  `metrics_after.prom` and `server_metrics.csv` written; PASS.
- `sweep.py --rates 10,20 --repeats 2 --duration 5 --warmup 1`: order 10, 20,
  20, 10; 4 PASS; attainment shown.
- Pilot check of attainment: 100.0% at 11 req/s and 0.7% at 12 req/s, the same
  as the Task 2 values.
- `python -m unittest tests.test_open_sweep`: OK. `cargo test`: 50 passed.

Next. User: commit, restart, then the session steps ("Task 8 run plan").
Agent: analyse each set when the user reports it.

### Step 8a: Same-session uniform baseline (2026-10-18)

Owner: User ran 8a and the 8a re-run; Agent analysed.

Done: 8a (closed 16/32/64; open uniform 6.5–16.5 req/s × 2, 120 s). Found that
satlab capped itself at 246 requests in flight (threat 19), fixed it
(`ensure_file_limit`, commit `f1851b7`), and re-ran 13–16.5 req/s.

Run, measured, calculated: `run.sh 8a`, `run.sh 8a-overload`; values in
"Results of 8a" and "Results of the 8a re-run". Contract checks pass for all
runs (`analysis/analyze.py`).

Result: 9 of 10 predictions held (8a.8, the wire check, is still to run).
8a.4 did not hold at 9 req/s: the backlog settled at 37, not 13–30 (A7). The
capacity is 12 req/s, overload grows the backlog at offered − 12 per second,
and `waiting` shows the queue only after about 245 requests are in the batch.

Unknown: why the `running` gauge stops at about 245. (A7: resolved by E7, H7a.)

Next. User: `run.sh e7`. Agent: compare E7 with E7.1 and E7.2.

### Step 8a-wire: E7 and the wire check (2026-10-18)

Owner: User ran `run.sh e7` and `run.sh 8a-wire` with `tcpdump`; Agent analysed.

Done: E7 (results under "Task 8: E7"). 8a-wire: wrote `analysis/wire.py` and
checked 8a.8. Found A8 and threat 20; wrote 8b.4 before 8b.

Result: E7.1, E7.2 and 8a.8 held. A7 is resolved (H7a). The 12 req/s run went
into overload after a 2.5% GPU clock dip (A8): the capacity margin at
12 req/s is about 0.2%.

Unknown: why 130–210 tokens per step make fewer tok/s than 110 or 240.

Next. User: commit, then `run.sh 8b`. Agent: compare with 8b.1–8b.4.

### Step 8b: Long uniform runs (2026-10-18)

Owner: User ran `run.sh 8b`; Agent analysed.

Done: 8b at 11, 12 and 13 req/s, 300 s each. Compared with 8b.1–8b.4.

Result: 8b.2 and 8b.3 held; 8b.1 did not hold; 8b.4 was not tested (12 req/s
stayed stable at full clock). 11 req/s moved to a slower branch after a 1%
GPU clock dip and stayed there after the clock recovered (E2E p50 6.07 s
against 4.25 s). Capacity: 12 req/s holds at full clock; 11 req/s can still
fail the medium tier.

Unknown: why 130–210 tokens per step make fewer tok/s than 110 or 240.

Next. User: commit, then `run.sh 8c`. Agent: compare with 8c.1–8c.4.

### Step 8c: Poisson arrivals (2026-10-18)

Owner: User ran `run.sh 8c`; Agent analysed.

Done: 8c at 9, 11 and 12 req/s × 2. Compared with 8c.1–8c.4. Checked the
Poisson schedules (sent against plan, KS test, simulated CV band).

Result: 8c.3 (the main slide claim) and 8c.4 held. 8c.1 did not hold in 1 of
6 runs (CV 1.086); the cause is a band that was too narrow, not the generator.
8c.2 did not hold: one 9 req/s run had 89.5% medium attainment. Poisson bursts
move the server through the 32-row block states, in both directions.

Unknown: none new.

Next. User: commit, then `run.sh 8c-gamma`. Agent: compare with 8c.5 and 8c.6.

### Step 8c-gamma: Gamma CV 2 arrivals (2026-10-18)

Owner: User ran `run.sh 8c-gamma`; Agent analysed.

Done: 8c-gamma at 9 and 11 req/s × 2. Compared with 8c.5 and 8c.6. Checked
the gamma schedules (sent against plan, KS test, rate spread). Added a unit
test for the spread of the schedule across seeds.

Result: 8c.5 did not hold; 8c.6 held for E2E p95 and did not hold for
attainment. Burstier arrivals give a longer tail at 11 req/s (p95: uniform
4.4 s, Poisson 10 s, gamma 13–20 s). At CV 2 a 120 s run does not fix the
offered rate (one 9 req/s run drew 11.0 req/s).

Unknown: none new.

Next. User: commit, then `run.sh 8e`. Agent: check the prompt length, then
compare with 8e.1–8e.3.
