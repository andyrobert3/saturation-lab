#!/usr/bin/env python3
"""Engine steps of closed-loop runs, with server histograms and GPU samples (Task 6).

  engine_steps.py run_dir... [--metrics DIR] [--powermetrics FILE] [--mlx E1.json] [-o DIR]

All requests in one engine step get their token in the same SSE burst, so token events
closer than 3 ms are one step. --metrics reads closed-<c>-{before,after}.prom; --powermetrics
averages the 1 s samples inside each run's 11-60 s window.
"""

import argparse
import collections
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyze import read_events, read_requests, save, style, SERIES, INK2
import matplotlib.pyplot as plt  # analyze.py has already selected the Agg backend

GAP_MS = 3.0
HISTS = ("time_to_first_token", "request_queue_time", "request_prefill_time")


def steps(run_dir, warmup_s, duration_s):
    t = np.array(sorted(e[1] for e in read_events(run_dir / "stream_events.csv")
                        if warmup_s * 1e9 <= e[1] < duration_s * 1e9)) / 1e6
    starts = np.concatenate(([0], np.where(np.diff(t) > GAP_MS)[0] + 1))
    sizes = np.diff(np.concatenate((starts, [len(t)])))
    return float(np.median(sizes)), float(np.median(np.diff(t[starts]))), len(t) / (duration_s - warmup_s)


def prom(path):
    values = collections.defaultdict(float)
    for line in path.read_text().splitlines():
        m = re.match(r"(vllm:[\w:]+?)(\{[^}]*\})?\s+(\S+)$", line)
        if m and 'le="' not in (m.group(2) or ""):
            values[m.group(1)] += float(m.group(3))
    return values


def server_means_ms(metrics_dir, c):
    before, after = (prom(metrics_dir / f"closed-{c}-{w}.prom") for w in ("before", "after"))
    d = {k: after[k] - before.get(k, 0.0) for k in after}
    return {h: 1e3 * d[f"vllm:{h}_seconds_sum"] / d[f"vllm:{h}_seconds_count"] for h in HISTS}


def gpu_samples(path):
    samples, cur = [], None
    for line in path.read_text().splitlines():
        if m := re.match(r"\*\*\* Sampled system activity \((.+?)\)", line):
            cur = {"t": datetime.strptime(m.group(1), "%a %b %d %H:%M:%S %Y %z").timestamp()}
            samples.append(cur)
        elif cur is not None and (m := re.match(r"GPU HW active residency:\s+([\d.]+)%", line)):
            cur["gpu_pct"] = float(m.group(1))
        elif cur is not None and (m := re.match(r"GPU Power: (\d+) mW", line)):
            cur["gpu_w"] = int(m.group(1)) / 1e3
    return [s for s in samples if "gpu_pct" in s]


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--powermetrics", type=Path)
    parser.add_argument("--mlx", type=Path, help="scripts/mlx_step_bench.py output, for the chart")
    parser.add_argument("-o", "--out", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    gpu = gpu_samples(args.powermetrics) if args.powermetrics else []
    rows = []
    for d in args.runs:
        doc = json.loads((d / "config.json").read_text())
        cfg, anchor = doc["config"], doc["provenance"]["anchor_wall_unix_ns"] / 1e9
        ok = [r for r in read_requests(d / "records.csv") if r.status == "ok" and not r.warmup]
        k, step, tok_s = steps(d, cfg["warmup_s"], cfg["duration_s"])
        row = {"run": d.name, "workers": cfg["concurrency"], "tokens_per_step": k, "step_ms": step, "tok_s": tok_s,
               "tpot_ms": statistics.median((r.last - r.first) / (r.tokens - 1) / 1e6 for r in ok),
               "ttft_ms": statistics.median((r.first - r.actual) / 1e6 for r in ok)}
        if args.metrics and (args.metrics / f"closed-{row['workers']}-after.prom").is_file():
            row.update({f"server_{h}_ms": v for h, v in server_means_ms(args.metrics, row["workers"]).items()})
        window = [s for s in gpu if anchor + 11 <= s["t"] < anchor + cfg["duration_s"]]
        if window:
            row["gpu_pct"] = statistics.mean(s["gpu_pct"] for s in window)
            row["gpu_w"] = statistics.mean(s["gpu_w"] for s in window)
        rows.append(row)
    rows.sort(key=lambda r: r["workers"])
    keys = [k for k in rows[0] if k != "run"]
    short = lambda k: k.replace("server_", "srv_").replace("request_", "").replace("time_to_first_token", "ttft")[:12]
    print(" ".join(f"{short(k):>12}" for k in keys))
    for r in rows:
        print(" ".join(f"{r.get(k, float('nan')):12.1f}" for k in keys))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "engine-steps.json").write_text(json.dumps(rows, indent=1))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot([r["workers"] for r in rows], [r["step_ms"] for r in rows], "o-", color=SERIES[0], label="vLLM-metal engine step")
    if args.mlx:
        mlx = json.loads(args.mlx.read_text())["rows"]
        for color, key, label in ((SERIES[1], "step_ms", "pure MLX step, batch as (B, 1)"),
                                  (SERIES[2], "packed_step_ms", "pure MLX, batch packed as (1, B)")):
            if key in mlx[0]:
                ax.plot([m["batch"] for m in mlx], [m[key] for m in mlx], "s--", color=color, label=label)
    ax.set_xlabel("requests in the batch", color=INK2)
    ax.set_ylabel("time per engine step, ms", color=INK2)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    print(save(fig, args.out, "engine-steps.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
