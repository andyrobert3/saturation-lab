# Measurement contract

Part of the [methodology review](methodology-review.md). Reviewed in its Step 2 (2026-10-17).

This contract defines what the client records and what the saved report uses.
It does not identify the server's internal service time or limiting resource.

## Clock and event boundaries

Request timestamps are integer nanoseconds since the run's monotonic-clock
anchor. Latency figures in the report use milliseconds. The wall-clock anchor
aligns approximate server samples; it does not measure request duration.

```text
intended_ns -> actual_ns -> first_output_ns -> last_output_ns -> completed_ns
planned       HTTP attempt  first token event   last token event  request finished
```

| Field or count | Exact meaning | Source |
| --- | --- | --- |
| Arrival; `intended_ns` | The planned deadline for one request. Open-loop arrivals are independent of replies. | [Driver](../src/driver.rs). |
| Send; `actual_ns` | Client time immediately before the HTTP attempt starts. It is not a socket-write or server-receipt timestamp. | [Driver](../src/driver.rs), [HTTP target](../src/target.rs). |
| First output; `first_output_ns` | Client time when the parser records the first SSE event with one or more token IDs. | [HTTP target](../src/target.rs). |
| Last output; `last_output_ns` | Client time when the parser records the last token-bearing SSE event. | [HTTP target](../src/target.rs). |
| Finished; `completed_ns` | Client time after the HTTP attempt returns, including stream end, an error, or a timeout. | [Driver](../src/driver.rs). |
| Event; `emitted_ns` | Despite the field name, this is a client observation time, not the server's token-generation time. | [HTTP target](../src/target.rs). |
| Event tokens; `token_count` | Number of token IDs in one parsed SSE event. An SSE event can contain more than one token. | [HTTP target](../src/target.rs). |
| Request tokens; `output_tokens` | For a successful stream, the usage count must equal the sum of event token counts. | [HTTP target](../src/target.rs). |
| `intended`, `fired`, `clipped` | Planned arrivals, admitted HTTP tasks, and arrivals rejected by the client's in-flight cap. | [Driver](../src/driver.rs). |
| `recorded`, `dropped` | Request rows accepted by the recorder, and rows lost when its channel is full. | [Recorder](../src/recorder.rs). |

The current request body uses `/v1/completions`, `max_tokens=16`,
`ignore_eos=true`, `temperature=0`, streaming, usage, and token IDs. The parser
checks token-count agreement. The configured output length is 16 tokens.
Request completion occurs after stream end, not necessarily
at the last token. No separate `[DONE]` timestamp is saved.

From 2026-10-17, each request puts a random 6-digit number and a space before
the configured prompt (for example `482913 Explain saturation in one sentence.`).
No two prompts share a prefix, so a prefix cache cannot skip prefill. On the
Llama 3 tokenizer the prompt is then 8 tokens instead of 7, for every number.
The number is not saved. Runs before this date used the same prompt for every
request.

In open-loop uniform mode, deadline `i` is `round(i * 1e9 / rate)` nanoseconds,
starting at zero. Poisson mode draws exponential gaps. Gamma mode
(`"arrival": {"gamma": {"cv": 2.0}}`) draws gamma gaps with shape 1/cv² and
scale cv²/rate, so the mean gap is 1/rate and the coefficient of variation is
cv; cv 1 is Poisson. Both random modes record the resolved seed. The 2026-10-11 open-loop sweep uses
uniform deadlines. Its older files omit the new arrival and seed fields;
the saved intended timestamps show the deadline sequence.

When the client wakes late, it attempts overdue arrivals immediately. It can
therefore send late, including after the configured deadline. The original
intended time is retained. A full cap creates a `clipped` row with zero send
and output timestamps; it does not queue that arrival inside the client.

In closed loop, a worker finishes one request before starting the next.
`intended_ns` equals `actual_ns`. The configured `rate` is unused, and a zero
reported requested rate does not mean that the worker sent no requests.

## Windows and request populations

Let `W` be the warmup duration and `T` be the sending duration. The measurement
window is **[W, T)**: include `W`, exclude `T`. The saved short sweeps use
`W=10 seconds`, `T=60 seconds`, and a 50-second measurement window.

| Measure | Selection and calculation |
| --- | --- |
| Warmup row | `intended_ns < W`. A late warmup request remains a warmup row even if it is actually sent after `W`. |
| Latency population | Successful requests intended in [W, T). Include those whose last token arrives after `T`. |
| Error population | Requests intended in [W, T) with HTTP, timeout, or stream error status. Errors during the later drain remain in this population. |
| Served rate | Successful requests with `last_output_ns` in [W, T), divided by `T-W`. Include requests intended before `W` if their last token arrives inside this window. |
| Output token rate | Sum event tokens observed in [W, T) for requests eventually marked successful, divided by `T-W`. Include token events from warmup requests. |
| Reported sent rate | `fired / T`. This includes warmup. It counts admitted tasks over the configured duration, not server receipts or actual send times inside [W, T). |
| Actual sends in the window | Count non-clipped rows with `actual_ns` in [W, T). Divide by `T-W` when comparing rates over the same window. This differs from the saved `sent_rate_hz` definition. |
| Request token total | `records.output_tokens` sums all output tokens of successful latency-population requests, including events after `T`. It differs from `output_tokens_in_window`. |
| Drain | After the schedule ends, the driver waits for admitted requests to finish or time out. Recorder and event data continue to arrive. |

Warmup exclusion depends on the measure. The statement that warmup rows are
excluded from every statistic is too broad: served and token rates use event
times and can include work that started during warmup.

Successful-only latency does not describe failed requests. Read errors and
sample counts with the latency figures. Partial output from a request later
marked failed does not enter the successful served or token rates.

## Latency and dispatch formulas

| Measure | Formula and meaning |
| --- | --- |
| TTFT | `first_output_ns - actual_ns`: delay from the client HTTP attempt to its first token event. |
| E2E | `last_output_ns - actual_ns`: delay to the last token event. It excludes the later wait for stream completion. |
| TPOT | `(last_output_ns - first_output_ns) / (output_tokens - 1)`, for requests with more than one output token. One sample per request. |
| ITL | Gap between consecutive token-bearing SSE events of the same successful request. One sample per event gap, not necessarily per token. |
| Send lag | `actual_ns - intended_ns`, for non-clipped latency-window arrivals. Include error outcomes. |
| Schedule-relative TTFT and E2E | `first_output_ns - intended_ns` and `last_output_ns - intended_ns`. They include dispatch delay. The analysis reports them as `ttft_from_plan` and `e2e_from_plan`; in closed loop they equal TTFT and E2E, because `intended_ns` equals `actual_ns`. |

TTFT and E2E include client, network, and server effects after the HTTP attempt
starts. Their formulas alone do not split prefill, decode, or queue delay.
Token IDs record counts, not an individual generation timestamp for every token.

A dispatch is late when its lag exceeds the saved `send_lag_budget_ns`.
The runner sets the budget to the maximum of 10% of the mean arrival interval,
five times its measured timer wakeup delay, and 1 millisecond. Use the saved
budget when checking old runs.

The analyzer computes linearly interpolated percentiles from raw samples.
The summary uses HDR histograms, so its percentile values are approximate.
The two percentile methods can return different values for the same samples.

At one setting, report lines are medians of per-run metrics. Their min–max bands
show repeat variation. They are not confidence intervals or percentiles of
pooled requests. A missing latency sample must not be interpreted as zero delay.

## Unfinished work and server gauges

At time `t`, client unfinished work is:

```text
non-clipped requests with actual_ns <= t
minus requests with completed_ns <= t
```

The plot samples this count every 0.1 seconds, including the run deadline.
An error or timeout ends the client request. It does not prove that the server
has stopped all associated work.

For exact accounting over [W, T), use counts immediately before each boundary:

```text
unfinished(T-) - unfinished(W-) = actual sends[W,T) - request finishes[W,T)
```

Request finishes in this identity include errors and use `completed_ns`.
Successful last-token completions used by served rate are a different count.

The outstanding-work slope is a fitted linear trend from warmup to the deadline.
It can include a startup transient. It is not a statistical test of stability.
Inspect the full and later timeline before making a sustained-rate claim.

`server_metrics.csv` stores the backend's running and waiting gauges when
available. The sampler records wall time before the HTTP scrape, then waits
about one second after the scrape. Alignment is approximate and brief changes
can be missed. Server running requests are not a count of requests executing
on a GPU at that instant. Read the selected backend's metric definitions.

A zero sampled waiting count can occur while active work grows. Client
unfinished work, server running, and server waiting are separate observations;
they need not be equal at each sampled time.

Little's Law uses means with matching system boundaries and a suitable stable
measurement period. Client unfinished work spans actual send to full request
finish; report E2E ends at the last token. A residual using mismatched windows
or lifetimes is not, by itself, proof of bad accounting. Offered rate divided
by peak served rate is not a measurement of GPU or CPU utilization.

## Saved example: 12 requests/s, repeat 2

Sources: [configuration](../runs/vllm-metal-open-sweep/20261011T144334Z_vllm-metal-open-12-r2/config.json),
[summary](../runs/vllm-metal-open-sweep/20261011T144334Z_vllm-metal-open-12-r2/summary.json),
[requests](../runs/vllm-metal-open-sweep/20261011T144334Z_vllm-metal-open-12-r2/records.csv), and
[events](../runs/vllm-metal-open-sweep/20261011T144334Z_vllm-metal-open-12-r2/stream_events.csv).

| Count or measure | Value from saved data |
| --- | ---: |
| Planned arrivals and admitted tasks | 720 each |
| Warmup arrivals | 120 |
| Successful requests intended after warmup | 600 |
| Successful last-token completions inside [10, 60) seconds | 566 |
| Of those completions, requests intended during warmup | 62 |
| Latency-population requests with their last token at or after 60 seconds | 96 |
| Actual sends / full request finishes inside [10, 60) seconds | 600 / 566 |
| Event tokens inside the window | 9,332 |
| Served rate | 566 / 50 = 11.32 requests/s |
| Output token rate | 9,332 / 50 = 186.64 tokens/s |
| p95 E2E, raw-data calculation | 8,301.416 ms |
| Client unfinished requests at 10 / 60 seconds | 62 / 96 |

The 600 latency samples and 566 served completions answer different questions.
The request finishes account for the growth: 600 sends minus 566 finishes
equals 34 additional unfinished requests. Tokens/s need not equal 16 times
served requests/s because token events and last-token completions cross the
measurement boundaries at different times.

## Review and remaining limits

`analysis/analyze.py` checks the contract each time it loads a run
(`check_contract`). It asserts that the CSV rows reproduce `summary.json`:
planned, admitted, and clipped counts; row and status counts; token sums;
window completions and token events; served and token rates; and every
percentile within the neighbouring raw samples. It stops on a mismatch. All 54
pilot runs passed again on 2026-10-17, and so did the 36 mock runs of Task 4. This confirms the recorded numbers agree with
these definitions. It does not establish server capacity.

The saved Git revision is unknown. Runs made after 2026-10-14 record
`git describe --always --dirty`. CSV records do not retain the raw
HTTP payload or SSE text, so these records cannot independently prove what was
sent on the wire or which parser revision produced an old file. Cache, power,
and model revision details also remain incomplete. A sustained capacity claim
still requires the 300 s runs (review, run set 8b).

## Implementation sources

- [Clock and driver](../src/clock.rs), [arrival and dispatch code](../src/driver.rs).
- [HTTP and SSE parser](../src/target.rs), [recorder populations](../src/recorder.rs).
- [Summary rate calculations](../src/runner.rs), [raw-data analysis and contract checks](../analysis/analyze.py).
