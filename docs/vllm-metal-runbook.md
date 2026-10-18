# vLLM Metal runbook

Commands assume the repository root as the working directory. The user runs
every server and benchmark command. See [methodology-review.md](methodology-review.md) for the
plan and [measurement-contract.md](measurement-contract.md) for definitions.

The saved results used an Apple M4 Pro with 64 GB, macOS 26.5, vLLM 0.30.0,
and `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`. Metal results show where
this Mac saturates. They do not predict GPU throughput.

## 1. Install and start the server

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
vllm --version

vllm serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.60 \
  --max-num-seqs 256
```

Keep that terminal open. If memory allocation fails, use
`--gpu-memory-utilization 0.50` and record the change. Do not change another
setting inside one comparison series.

## 2. Check the streaming contract

```bash
curl -sS http://127.0.0.1:8000/version
curl -N http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
  "prompt": "Explain saturation in one sentence.",
  "max_tokens": 16, "ignore_eos": true, "stream": true,
  "stream_options": {"include_usage": true}, "return_token_ids": true, "temperature": 0.0}'
```

Expect SSE `data:` records with `choices[0].token_ids`, usage of exactly 16
completion tokens, and a final `data: [DONE]`. This checks the protocol. It does
not measure capacity.

## 3. Build and run the smoke test

```bash
cargo build --release
./target/release/satlab --config smoke.json --runs-dir runs
```

Accept the run only if the verdict is `PASS`, there are no errors, clipped
requests, or dropped rows, the late-dispatch fraction is at most 1%, and every
successful request reports 16 output tokens. A smoke run is not a knee.
To check the client against the server's own `/metrics`, run
`.venv/bin/python scripts/validate_server_metrics.py --config smoke.json --runs-dir runs/methodology_review/server-validation`.

## 4. Sweeps

Record the launch flags with every sweep. `--server-notes` is required.
satlab raises its open-file limit at start (one socket per request in flight); if the
hard limit is too low it stops with a message. Check with `ulimit -Hn`.

```bash
NOTES='launch=vllm serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --host 127.0.0.1 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.60 --max-num-seqs 256; cache=<state it>; power=<state it>; model revision=241a666dad6cb93c8ff213d39a7f34a36bf26db4'

# Closed loop: 3 repeats per concurrency. Use a new directory for each session.
for c in 16 32 48; do
  sed "s/\"concurrency\": 32/\"concurrency\": $c/; s/closed-32/closed-$c/" \
    examples/metal-closed.json > /tmp/metal-closed-$c.json
  for r in 1 2 3; do
    ./target/release/satlab --config /tmp/metal-closed-$c.json --runs-dir runs/methodology_review/closed-new
  done
done

# Open loop: 9 rates x 3 repeats, 60 s each, 10 s warmup. Resumes after an interruption.
.venv/bin/python scripts/sweep.py sweep --runs-dir runs/methodology_review/open-new --server-notes "$NOTES"

# Three 300 s runs at one rate. Charts go to a separate report directory.
.venv/bin/python scripts/sweep.py confirm --rate 12 --runs-dir runs/methodology_review/open-new --server-notes "$NOTES"

# Rebuild tables and charts without sending requests.
.venv/bin/python scripts/sweep.py analyze --runs-dir runs/methodology_review/open-new \
  --closed-dir runs/methodology_review/closed-new
```

The three passes run forward, reversed, and shuffled, so drift cannot look like
a knee. If every rate keeps pace, add `--rates 18,20,24`. If the first shortfall
is below 11 req/s, add `--rates 4,5,6`. When the server exposes `/metrics`, the
script saves one `waiting`, `running`, and `kv_cache_usage` sample per second in
`server_metrics.csv`.

The candidate limit is the first clean rate whose median served rate is below
95% of the offered rate. Check all three repeats and the unfinished-work slope
before calling it stable. Name a safe rate only against the SLO tiers (review, card row 2).

## 5. The mock target

```bash
./target/release/mock 4 0.05 8123   # nominal 80 req/s; measured 78.1 req/s (slot time 51.2 ms)
.venv/bin/python scripts/sweep.py sweep --base-config config.json --name mock-open \
  --runs-dir runs/methodology_review/mock-sweep --expected-capacity 78 --rates 60,70,80,90,100 \
  --max-in-flight 4096 --server-notes "mock 4 slots x 0.05 s"
```

The analysis warns if the plateau is more than 10% from 78 req/s. That checks
the harness against a known answer. Above capacity the backlog grows by
`rate - 78.1` requests per second. A run that reaches `--max-in-flight` is clipped
and fails the verdict, so size the cap above that backlog (default 512).

## References

- <https://docs.vllm.ai/projects/vllm-metal/en/stable/installation/>
- <https://docs.vllm.ai/en/latest/benchmarking/cli/>
