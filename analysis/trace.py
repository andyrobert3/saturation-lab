#!/usr/bin/env python3
"""Real traffic from the Azure LLM inference trace 2023: arrival burstiness and request lengths.

  trace.py FILE.csv... -o DIR   table, trace-summary.json, trace-arrivals.png, trace-lengths.png

One CSV row is one request: TIMESTAMP, ContextTokens, GeneratedTokens. Each file is
analysed alone. Burstiness uses the same function as the mock test (Task 4).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyze import save, style, SERIES, INK2
from analysis.queue_theory import burstiness
import matplotlib.pyplot as plt  # analyze.py has already selected the Agg backend

WINDOWS_S = (0.1, 1.0, 10.0, 60.0)
PIECE_S = 300.0  # 5-minute pieces: short-scale burstiness without the slow rate drift
OURS = {"ContextTokens": 8, "GeneratedTokens": 16}  # our test request
GAMMA_CVS = np.arange(1.0, 8.01, 0.1)
CITE = "Data: Azure LLM inference trace 2023 (Patel et al., Splitwise, ISCA 2024)"


def load(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    stamps = [r["TIMESTAMP"] for r in rows]
    frac = [s.partition(".")[2] for s in stamps]
    t = np.array([s.replace(" ", "T") for s in stamps], dtype="datetime64[ns]")
    lengths = {k: np.array([int(r[k]) for r in rows]) for k in OURS}
    return t, max(map(len, frac)), max(len(x.rstrip("0")) for x in frac), lengths


def shape(x, ours):
    p50, p90, p99 = np.percentile(x, (50, 90, 99))
    return {"p50": p50, "p90": p90, "p99": p99, "max": int(x.max()), "mean": float(x.mean()),
            "cv": float(x.std() / x.mean()), "share_le_ours": float(np.mean(x <= ours))}


def gamma_cv_for_idc(idc, rate, span_s, rng, reps=10):
    """Gamma CV whose arrivals (same rate and span) give the same index of dispersion at 1 s."""
    def sim(cv):
        t = np.cumsum(rng.gamma(1 / cv**2, cv**2 / rate, int(rate * span_s * 1.5)))
        return burstiness(t[t < span_s])[1]
    sims = np.array([np.mean([sim(cv) for _ in range(reps)]) for cv in GAMMA_CVS])
    return float(GAMMA_CVS[np.argmin(np.abs(sims - idc))])


def analyse(path, rng):
    t64, written, used, lengths = load(path)
    gaps_ns = np.diff(t64).astype(np.int64)
    t = np.sort((t64 - t64.min()).astype(np.int64) / 1e9)
    span = float(t[-1])
    cv_pieces = [burstiness(t[(t >= a) & (t < a + PIECE_S)])[0] for a in np.arange(0, span - PIECE_S, PIECE_S)]
    per_s, per_min = (np.histogram(t, bins=np.arange(0, span, w))[0] for w in (1.0, 60.0))  # full windows only
    ctx, gen = lengths["ContextTokens"], lengths["GeneratedTokens"]
    s = {"file": path.name, "service": path.stem.split("_")[-1], "rows": len(t), "start": str(t64.min())[:19],
         "end": str(t64.max())[:19], "span_s": span, "mean_rate": len(t) / span, "decimals": f"{written}/{used}",
         "sorted": bool((gaps_ns >= 0).all()), "zero_gaps": int((gaps_ns == 0).sum()),
         "min_gap_us": float(gaps_ns[gaps_ns > 0].min() / 1e3), "cv_whole": burstiness(t)[0],
         "cv_5min": [float(np.median(cv_pieces)), float(min(cv_pieces)), float(max(cv_pieces))],
         "idc": {f"{w:g}": burstiness(t, w)[1] for w in WINDOWS_S},
         "per_1s": [float(per_s.mean()), int(per_s.max()), float(np.mean(per_s == 0))],
         "per_1min": [int(per_min.min()), float(per_min.mean()), int(per_min.max())],
         "rate_per_min": (per_min / 60).tolist(),
         **{k: shape(v, OURS[k]) for k, v in lengths.items()},
         "corr": [float(np.corrcoef(ctx, gen)[0, 1]), float(np.corrcoef(np.log1p(ctx), np.log1p(gen))[0, 1])]}
    cs2 = s["GeneratedTokens"]["cv"] ** 2  # proxy only: decode time grows with output tokens
    s["kingman"] = [(s["cv_whole"] ** 2 + cs2) / 2, (s["cv_5min"][0] ** 2 + cs2) / 2]
    s["gamma_cv_idc_1s"] = gamma_cv_for_idc(s["idc"]["1"], s["mean_rate"], span, rng)
    return s, t, lengths


def finish(ax, title, xlabel, ylabel, **legend):
    ax.set_title(title, color=INK2, fontsize=10, loc="left")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style(ax)
    ax.legend(frameon=False, fontsize=8, **legend)


def charts(results, out_dir):
    fig = plt.figure(figsize=(12, 5.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[3, 2], hspace=0.45, wspace=0.25)
    b, windows = fig.add_subplot(grid[:, 1]), np.logspace(-2, 2, 21)
    for row, (color, (s, t, _)) in enumerate(zip(SERIES, results)):
        a = fig.add_subplot(grid[row, 0])
        per_s = np.histogram(t, bins=np.arange(0, s["span_s"], 1.0))[0]
        a.plot(np.arange(len(per_s)) / 60, per_s, color=color, alpha=0.45, linewidth=0.5, label="1 s count")
        a.plot(np.arange(len(s["rate_per_min"])) + 0.5, s["rate_per_min"], color=color, linewidth=2,
               label="per-minute mean")
        a.axhline(s["mean_rate"], color="black", linestyle="--", linewidth=1, label=f"mean {s['mean_rate']:.2f}")
        finish(a, f"{s['service']}: requests per second ({s['per_1s'][2]:.0%} of seconds have none)",
               "minutes since first request" if row else "", "req/s", loc="upper right", ncol=3)
        b.plot(windows, [burstiness(t, w)[1] for w in windows], color=color, marker="o", markersize=3,
               label=s["service"])
    b.axhline(1, color=INK2, linestyle="--", linewidth=1, label="Poisson (1)")
    b.axhline(0, color=INK2, linestyle=":", linewidth=1.5, label="uniform (0)")
    b.set(xscale="log", yscale="symlog", ylim=(-0.1, 400))
    b.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    finish(b, "Burstiness grows with the time scale", "window (s, log scale)",
           "index of dispersion (variance ÷ mean of counts)", loc="upper left")
    fig.text(0.01, -0.02, CITE, fontsize=8, color=INK2, ha="left", va="top")
    paths = [save(fig, out_dir, "trace-arrivals.png")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key in zip(axes, OURS):
        for color, (s, _, lengths) in zip(SERIES, results):
            x = np.sort(lengths[key])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, linewidth=2,
                    label=f"{s['service']}: median {s[key]['p50']:,.0f}")
        ax.axvline(OURS[key], color="black", linewidth=1.5, label=f"our test: {OURS[key]}")
        ax.set_xscale("log")
        finish(ax, f"{key}: cumulative distribution", f"{key} per request (log scale)", "share of requests ≤ x",
               loc="lower right")
    fig.text(0.01, -0.02, CITE, fontsize=8, color=INK2, ha="left", va="top")
    return paths + [save(fig, out_dir, "trace-lengths.png")]


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    rng = np.random.default_rng(20261017)
    results = [analyse(f, rng) for f in args.files]
    summ = [r[0] for r in results]
    for key in ("service", "rows", "start", "end", "span_s", "mean_rate", "decimals", "sorted", "zero_gaps",
                "min_gap_us", "cv_whole", "cv_5min", "idc", "per_1s", "per_1min", *OURS, "corr", "kingman",
                "gamma_cv_idc_1s"):
        cells = [json.dumps(json.loads(json.dumps(s[key]), parse_float=lambda x: round(float(x), 4))) for s in summ]
        print(f"{key:16}" + "".join(f"  {c:58}" for c in cells))
    print("cv_5min = [median, min, max]; per_1s = [mean, peak, share empty]; per_1min = [min, mean, peak]; "
          "corr = [linear, log-log]; kingman = (Ca²+Cs²)/2 with Ca from [whole, 5-min median], Cs from GeneratedTokens")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "trace-summary.json").write_text(json.dumps(summ, indent=1, default=float))
    for p in charts(results, args.out):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
