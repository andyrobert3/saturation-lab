# saturation-lab

This project tests when an LLM server cannot keep up with its load. This
condition is saturation. The project also shows how to find saturation when the
server does not report it.

The repository has three parts:

1. `satlab`. This is a load generator written in Rust. It sends requests on a
   fixed schedule. It records each request and each token.
2. A mock server. Its capacity is known. You use it to check `satlab`.
3. A methodology review. It applies the method to one machine: an Apple M4 Pro
   that runs vLLM (Metal) with a 4-bit Llama 3.1 8B model.

The project started as a talk, "Fast Models, Slow Systems". The file
[`talk.md`](talk.md) is the outline of that talk.

## The main idea

A closed-loop benchmark uses a fixed number of workers. Each worker sends one
request and waits for the reply. The workers slow down when the server slows
down. For this reason, the benchmark cannot overload the server.

Real users send requests on their own schedule. This is open-loop load. An
open-loop test shows what a closed-loop test hides.

In the first test on the Mac, vLLM acted like a fixed pipe of about 190 output
tokens per second. It shared this pipe between all admitted requests. When the
load went above the limit, the requests did not wait in a visible queue. Each
request became slower at the same time.

| Offered req/s | Served req/s | Good req/s | p95 E2E | Unfinished at end |
| ---: | ---: | ---: | ---: | ---: |
| 6.5 | 6.50 | 6.50 | 2.35 s | 14 |
| 11 | 11.00 | 11.00 | 4.39 s | 47 |
| 12 | 11.30 | 0.00 | 8.30 s | 96 |
| 13 | 10.22 | 0.00 | 16.13 s | 211 |
| 16.5 | 10.40 | 0.00 | 29.65 s | 433 |

Notes on the table:

- A good request has a time to first token (TTFT) of 1 s or less. Its
  end-to-end time (E2E) is 5 s or less.
- The numbers come from a 60 s test with evenly spaced arrivals.
- During this test, there were no errors. The server `waiting` gauge showed 0.
  The harness verdict was `PASS`.
- A closed loop with 32 workers gave 12.8 req/s at a p95 E2E of 2.6 s.

The methodology review later checked these first results. It corrected some
of them. For example, the review did not confirm that throughput falls under
overload. The margin at 12 req/s is only about 0.2%. A GPU clock dip of 2 to 4%
can move the server into overload. The review tested 28 predictions in Task 8.
Of these, 20 held and 8 did not hold.

These numbers are for one Mac. They are not for production GPUs. The method
applies to other systems.

## Layout

```
src/
  main.rs       Command-line program. Loads the config, runs, prints the report.
  config.rs     Run configuration and checks.
  driver.rs     Arrival schedules (uniform, Poisson, gamma) and dispatch.
  target.rs     HTTP client and parser for streamed (SSE) replies.
  recorder.rs   Request records and summary histograms.
  runner.rs     Run steps, run folder, and PASS/FAIL verdict.
  clock.rs      Monotonic clock and timer measurement.
  bin/mock.rs   Mock server. Capacity = slots / service time.
analysis/       Python scripts for charts, checks, and queueing theory.
scripts/        Sweep runner and server checks.
examples/       Example configs.
docs/           Measurement contract, methodology review, runbook.
tests/          Rust integration tests and Python unit tests.
```

## Start

You need Rust 1.85 or newer. For the analysis, you also need Python 3 and
matplotlib.

```bash
cargo build --release
python3 -m venv .venv
.venv/bin/pip install -r analysis/requirements.txt
```

### Test with the mock

The mock has `slots` places. Each request holds one place for `service_time`
seconds. Capacity is `slots / service_time`.

Terminal 1. Start the mock. Its capacity is 4 / 0.05 = 80 req/s.

```bash
./target/release/mock 4 0.05 8123
```

Terminal 2. Run a closed loop with 4 workers.

```bash
./target/release/satlab --config examples/mock-closed.json --runs-dir runs/demo
```

Run an open loop at 100 req/s. The served rate stays near 78 req/s. The latency
increases.

```bash
./target/release/satlab --config config.json --rate 100 --duration 10 --runs-dir runs/demo
```

Make the charts.

```bash
.venv/bin/python analysis/analyze.py runs/demo/* -o runs/demo-report
```

The open-loop run still shows `PASS`. `PASS` means that the harness kept its
schedule. It does not mean that the latency target was met. To check the latency
target, read the good request rate in the report.

### Test a real server

The server must have a streaming `/v1/completions` endpoint. It must return
token IDs and usage data. vLLM does this. To set up vLLM on Apple Silicon, use
the [runbook](docs/vllm-metal-runbook.md). The file [`smoke.json`](smoke.json)
is a small test of the streaming contract.

## Configuration

A run uses one JSON file. These are the main fields.

| Field | Meaning |
| --- | --- |
| `mode` | `open` sends on a schedule. `closed` uses `concurrency` workers. |
| `rate` | Requests per second. Open mode only. |
| `arrival` | `uniform` (default), `poisson`, or `{"gamma": {"cv": 2.0}}` for bursty traffic. |
| `duration_s` | Time to send requests. |
| `warmup_s` | Start of the run. The statistics do not use it. |
| `max_in_flight` | Maximum number of open requests. See the note below. |
| `request_timeout_s` | Time limit for each request. |
| `target_notes` | Free text. Write the server launch flags here. |

If the `max_in_flight` limit is full, `satlab` records the arrival as
`clipped`. It does not put the arrival in a queue. A queue in the client would
change the open loop into a closed loop. A clipped arrival makes the verdict
fail.

You can change `--rate`, `--duration`, `--config`, and `--runs-dir` on the
command line. The exit code is 0 for `PASS`, 1 for a failed verdict, and 2 for a
setup error.

## Run records

Each run writes a folder in `runs/`. Git ignores this folder.

| File | Content |
| --- | --- |
| `config.json` | The config and the provenance (git revision, host, seed). |
| `records.csv` | One row for each request. |
| `stream_events.csv` | One row for each streamed token event. |
| `summary.json` | Rates and percentiles. |
| `server_metrics.csv` | Server gauges. The sweep script saves this file when the server has `/metrics`. |

Each request has these times:

```text
intended_ns -> actual_ns -> first_output_ns -> last_output_ns -> completed_ns
planned        sent          first token       last token        finished
```

- TTFT is the first token time minus the send time.
- E2E is the last token time minus the send time.
- Send lag is the send time minus the planned time.

The file [`docs/measurement-contract.md`](docs/measurement-contract.md) gives
exact definitions. The script `analysis/analyze.py` calculates each number in
`summary.json` again from the CSV files. It stops if a number is different.

## Sweeps

A sweep runs many rates, with repeats. The server must already be on. The
script does not start it.

```bash
# 9 rates, 3 repeats, 60 s each. It continues after an interruption.
.venv/bin/python scripts/sweep.py sweep --runs-dir runs/my-sweep --server-notes "<launch flags>"

# 3 runs of 300 s at one rate. This tests stability.
.venv/bin/python scripts/sweep.py confirm --rate 12 --runs-dir runs/my-sweep --server-notes "<launch flags>"

# Make the tables and charts again. It sends no requests.
.venv/bin/python scripts/sweep.py analyze --runs-dir runs/my-sweep
```

## Analysis scripts

| Script | Use |
| --- | --- |
| `analysis/analyze.py` | Tables and charts. Checks the measurement contract. |
| `analysis/queue_theory.py` | Compares the mock with queueing theory. |
| `analysis/engine_steps.py` | Finds engine steps in closed-loop runs. |
| `analysis/trace.py` | Arrival and length statistics of the Azure LLM trace (2023). |
| `analysis/wire.py` | Compares client times with a packet capture. |
| `scripts/validate_server_metrics.py` | Compares client results with the server `/metrics`. |
| `scripts/mlx_step_bench.py` | Measures one decode step in MLX, without vLLM. |

## Tests

```bash
cargo test
.venv/bin/python -m unittest tests.test_open_sweep
```

The integration tests start a mock server and check timing. The file
`.cargo/config.toml` makes the tests run one at a time. Parallel tests compete
for CPU and cause false timing failures.

## More to read

- [`docs/methodology-review.md`](docs/methodology-review.md): the main
  document. It has the question card, 20 threats to validity, predictions, and
  the work log.
- [`docs/measurement-contract.md`](docs/measurement-contract.md): exact
  definitions of times, windows, and formulas.
- [`docs/vllm-metal-runbook.md`](docs/vllm-metal-runbook.md): commands to
  repeat the vLLM Metal runs.
- [`talk.md`](talk.md): the talk outline, objections, and known gaps.

## Limits

- The saved run data (`runs/`) is not in the repository. You must make your own
  runs to get the numbers.
- The results are for one Apple M4 Pro, one 4-bit 8B model, 16 output tokens,
  and one short prompt. Long prompts and large models were not tested.
- The client and the server run on the same computer.
