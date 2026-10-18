#!/usr/bin/env python3
"""E1: time one decode step of the model in pure MLX, without vLLM, at each batch size.

Run with the vllm-metal Python while the vLLM server is stopped:
  /opt/homebrew/Cellar/vllm-metal/0.30.0/libexec/bin/python scripts/mlx_step_bench.py \
    -o runs/methodology_review/task6/e1-mlx-steps.json

"step" is one forward pass for `batch` sequences of one token each, with a KV cache that
holds the 8-token prompt (as in the satlab runs). "matmul" runs only the model's 4-bit
linear layers (all 32 layers and the output head) on the same batch, so it isolates the
quantized matrix multiply from attention and the rest. The "packed" columns put the same
batch in one sequence, shape (1, batch), as vllm-metal does (`mx.array([all_token_ids])` in
vllm_metal/v1/model_runner.py); "packed step" is one forward pass of that sequence, which is
also the prefill of a `batch`-token prompt.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
BATCHES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 48, 64)
WARMUP = 3


def timed(fn, steps):
    times = []
    for _ in range(WARMUP + steps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times[WARMUP:]) * 1e3


def step_ms(model, batch, steps, prompt_len=8):
    cache = make_prompt_cache(model)
    y = mx.argmax(model(mx.random.randint(1000, 20000, (batch, prompt_len)), cache=cache)[:, -1:], axis=-1)
    mx.eval(y)

    def one():
        nonlocal y
        y = mx.argmax(model(y, cache=cache)[:, -1:], axis=-1)
        mx.eval(y)
    return timed(one, steps)


def packed_step_ms(model, batch, steps):
    x = mx.random.randint(1000, 20000, (1, batch))
    return timed(lambda: mx.eval(model(x)), steps)


def matmul_ms(linears, batch, steps, packed=False):
    dims = {m.weight.shape[1] * 32 // m.bits for m in linears}
    dtype = linears[0].scales.dtype
    xs = {d: mx.random.normal((1, batch, d) if packed else (batch, 1, d)).astype(dtype) for d in dims}
    mx.eval(xs)
    return timed(lambda: mx.eval([m(xs[m.weight.shape[1] * 32 // m.bits]) for m in linears]), steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    model, _ = load(MODEL)
    linears = [m for _, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)]
    rows = []
    print(f"{'batch':>5} {'step ms':>8} {'matmul ms':>9} {'packed step':>11} {'packed matmul':>13}")
    for b in map(int, args.batches.split(",")):
        row = {"batch": b, "step_ms": step_ms(model, b, args.steps), "matmul_ms": matmul_ms(linears, b, args.steps),
               "packed_step_ms": packed_step_ms(model, b, args.steps),
               "packed_matmul_ms": matmul_ms(linears, b, args.steps, packed=True)}
        rows.append(row)
        print(f"{b:5} {row['step_ms']:8.1f} {row['matmul_ms']:9.1f} {row['packed_step_ms']:11.1f} {row['packed_matmul_ms']:13.1f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": MODEL, "mlx": mx.__version__, "machine": platform.platform(),
                                    "linear_layers": len(linears), "steps": args.steps, "rows": rows}, indent=1))
    print(args.out)


if __name__ == "__main__":
    main()
