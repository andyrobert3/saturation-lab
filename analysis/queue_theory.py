#!/usr/bin/env python3
"""Queueing-theory check for the mock: c servers, fixed service time, FCFS.

  queue_theory.py predict                    predicted wait bands for uniform, Poisson, gamma
  queue_theory.py check run_dir... -o DIR     measured wait against prediction, burstiness, chart
  queue_theory.py arrivals run_dir... -o DIR  measured send times: ticks, counts per second, waits

The mock is a G/D/c queue (tokio's semaphore is FIFO). With equal service times in
FCFS order, request i starts when request i-c finishes: start_i = max(a_i, start_{i-c} + D).
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyze import percentile, read_requests, save, style, SERIES, INK2
import matplotlib.pyplot as plt  # analyze.py has already selected the Agg backend

SERVERS = 4
SERVICE_S = 4 / 78.12  # measured: overload throughput 78.12 req/s (runs/mock-sweep, 90 req/s)
RATES = (39.0, 55.0, 66.0, 74.0)  # utilization 0.50, 0.70, 0.85, 0.95 at the measured service time
KINDS = {"uniform": 0.0, "poisson": 1.0, "gamma2": 2.0}  # arrival coefficient of variation
DURATION_S, WARMUP_S = 60.0, 10.0


def waits(arrivals_s, servers=SERVERS, service_s=SERVICE_S):
    """Exact FCFS waits for a c-server queue with one fixed service time."""
    start = np.empty(len(arrivals_s))
    for i, a in enumerate(arrivals_s):
        start[i] = a if i < servers else max(a, start[i - servers] + service_s)
    return start - arrivals_s


def arrivals(kind, rate, rng, duration_s=DURATION_S):
    cv = KINDS[kind]
    n = int(rate * duration_s * 1.5) + 50
    gaps = (np.full(n, 1 / rate) if cv == 0 else
            rng.exponential(1 / rate, n) if cv == 1 else
            rng.gamma(1 / cv**2, cv**2 / rate, n))
    t = np.concatenate(([0.0], np.cumsum(gaps)[:-1]))
    return t[t < duration_s]


def erlang_c(c, a):
    """Probability that an arrival waits in M/M/c with offered load a = rate x service."""
    top = a**c / math.factorial(c) * c / (c - a)
    return top / (sum(a**k / math.factorial(k) for k in range(c)) + top)


def kingman_mean_wait(rate, ca2, cs2=0.0, servers=SERVERS, service_s=SERVICE_S):
    """Allen-Cunneen (Kingman for c servers): ((Ca²+Cs²)/2) x the M/M/c mean wait."""
    a = rate * service_s
    return (ca2 + cs2) / 2 * erlang_c(servers, a) / (servers / service_s - rate)


def burstiness(times_s, window_s=1.0):
    """Coefficient of variation of gaps, and index of dispersion of counts per window."""
    gaps = np.diff(np.sort(times_s))
    counts = np.histogram(times_s, bins=np.arange(times_s.min(), times_s.max(), window_s))[0]
    return gaps.std() / gaps.mean(), counts.var() / counts.mean()


def summarize(w_ms):
    return float(np.mean(w_ms)), percentile(list(w_ms), 0.95)


def predict(reps=300, seed=20261017):
    rng = np.random.default_rng(seed)
    out = {}
    for kind, cv in KINDS.items():
        for rate in RATES:
            means, p95s = [], []
            for _ in range(reps):
                a = arrivals(kind, rate, rng)
                w = waits(a)[a >= WARMUP_S] * 1e3
                m, p = summarize(w)
                means.append(m), p95s.append(p)
            band = lambda v: [float(np.percentile(v, q)) for q in (5, 50, 95)]
            out[f"{kind}@{rate:g}"] = {"kind": kind, "rate": rate, "rho": rate * SERVICE_S / SERVERS,
                                       "mean_ms": band(means), "p95_ms": band(p95s),
                                       "kingman_mean_ms": kingman_mean_wait(rate, cv**2) * 1e3}
    return out


def load(run_dir):
    cfg = json.loads((run_dir / "config.json").read_text())["config"]
    reqs = read_requests(run_dir / "records.csv")
    arrival = cfg.get("arrival", "uniform")
    kind = arrival if isinstance(arrival, str) else f"gamma{next(iter(arrival.values()))['cv']:g}"
    return cfg, reqs, kind


def check(run_dirs, out_dir):
    pred = predict()
    runs = [(d, *load(d)) for d in run_dirs]
    base = [(r.last - r.actual) / 1e6 for d, cfg, reqs, kind in runs
            if kind == "uniform" and cfg["rate"] == min(RATES) for r in reqs if r.status == "ok" and not r.warmup]
    s0 = statistics.median(base)  # E2E with no wait: service plus client overhead
    print(f"no-wait E2E S0 = {s0:.2f} ms (uniform {min(RATES):g} req/s); service D = {SERVICE_S * 1e3:.2f} ms")
    print(f"{'run':28} {'rho':>5} {'CV plan':>7} {'CV sent':>7} {'IDC 1s':>6} {'mean wait':>9} "
          f"{'band 5-95%':>15} {'replay':>7} {'p95 wait':>8} {'band 5-95%':>15} {'replay':>7} {'Kingman':>7}")
    rows = []
    for d, cfg, reqs, kind in runs:
        sent = [r for r in reqs if r.status != "clipped"]
        actual = np.array([r.actual for r in sent]) / 1e9
        order = np.argsort(actual)
        replay = waits(actual[order])[np.argsort(order)] * 1e3  # same arrivals, ideal server
        pop = [i for i, r in enumerate(sent) if r.status == "ok" and not r.warmup]
        measured = [max(0.0, (sent[i].last - sent[i].actual) / 1e6 - s0) for i in pop]
        cv_plan = burstiness(np.array([r.intended for r in reqs]) / 1e9)[0]
        cv_sent, idc = burstiness(actual)
        p = pred[f"{kind}@{cfg['rate']:g}"]
        m, p95 = summarize(measured)
        rm, rp95 = summarize(replay[pop])
        rows.append({"run": d.name, "kind": kind, "rate": cfg["rate"], "rho": p["rho"], "cv_plan": cv_plan,
                     "cv_sent": cv_sent, "idc_1s": idc, "mean_ms": m, "p95_ms": p95, "replay_mean_ms": rm,
                     "replay_p95_ms": rp95, "pred": p})
        print(f"{cfg['name']:28} {p['rho']:5.2f} {cv_plan:7.2f} {cv_sent:7.2f} {idc:6.2f} {m:9.2f} "
              f"{p['mean_ms'][0]:7.2f}-{p['mean_ms'][2]:7.2f} {rm:7.2f} {p95:8.2f} "
              f"{p['p95_ms'][0]:7.2f}-{p['p95_ms'][2]:7.2f} {rp95:7.2f} {p['kingman_mean_ms']:7.2f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queue-check.json").write_text(json.dumps({"s0_ms": s0, "service_ms": SERVICE_S * 1e3,
                                                          "rows": rows}, indent=1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in ((axes[0], "mean_ms", "mean wait"), (axes[1], "p95_ms", "p95 wait")):
        for color, kind in zip(SERIES, KINDS):
            ps = [pred[f"{kind}@{r:g}"] for r in RATES]
            x = [p["rho"] for p in ps]
            ax.fill_between(x, [p[key][0] for p in ps], [p[key][2] for p in ps], color=color, alpha=0.15, linewidth=0)
            ax.plot(x, [p[key][1] for p in ps], color=color, label=f"theory, {kind} (CV {KINDS[kind]:g})")
            pts = [(r["rho"], r[key]) for r in rows if r["kind"] == kind]
            if pts:
                ax.scatter(*zip(*pts), color=color, edgecolor="black", zorder=3, s=28)
        ax.set_title(title, color=INK2, fontsize=11, loc="left")
        ax.set_xlabel("utilization ρ = rate × service ÷ 4", color=INK2)
        ax.set_ylabel("ms", color=INK2)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Mock: measured wait (dots) vs FCFS simulation (line, band = 5–95% of 60 s runs)", fontsize=11)
    print(save(fig, out_dir, "queue-check.png"))


def arrival_patterns(run_dirs, out_dir, s0=51.17, window=(20.0, 23.0)):
    """One row per run: send ticks in a short window, requests per 1 s, and the wait of each request.

    s0 is the no-wait E2E in ms that `check` prints (median of the uniform 39 req/s runs)."""
    runs = [load(d) for d in run_dirs]
    names = {"uniform": "Uniform", "poisson": "Poisson", "gamma2": "Gamma"}
    fig, axes = plt.subplots(len(runs), 3, figsize=(13, 2.2 * len(runs)), sharex="col", sharey="col",
                             squeeze=False, gridspec_kw={"width_ratios": (1.2, 1, 1)})
    for row, color, (cfg, reqs, kind) in zip(axes, SERIES, runs):
        t = np.sort([r.actual for r in reqs if r.status != "clipped"]) / 1e9
        m = t[t >= cfg["warmup_s"]]
        counts = np.histogram(m, bins=np.arange(cfg["warmup_s"], cfg["duration_s"] + 1))[0]
        ok = [r for r in reqs if r.status == "ok" and not r.warmup]
        w = [max(0.0, (r.last - r.actual) / 1e6 - s0) for r in ok]
        row[0].vlines(t[(t >= window[0]) & (t < window[1])], 0, 1, color=color, linewidth=1)
        row[0].set_yticks([])
        row[0].set_ylabel(f"{names.get(kind, kind)}\ngap CV {burstiness(m)[0]:.2f}\ndispersion "
                          f"{counts.var() / counts.mean():.2f}", rotation=0, ha="right", va="center", fontsize=10)
        row[1].bar(np.arange(len(counts)) + cfg["warmup_s"] + 0.5, counts, width=0.85, color=color)
        row[1].axhline(counts.mean(), color=INK2, linewidth=1.2)
        row[2].scatter([r.actual / 1e9 for r in ok], w, s=5, color=color, linewidth=0)
        row[2].text(0.98, 0.92, f"mean {np.mean(w):.2f} ms", transform=row[2].transAxes, ha="right", va="top")
        for ax in row:
            style(ax)
    for ax, title in zip(axes[0], (f"Sends, {window[0]:g}–{window[1]:g} s (one tick per request)",
                                   "Requests per 1 s window (line = mean)", f"Wait per request, ms (E2E − {s0:.2f} ms)")):
        ax.set_title(title, color=INK2, fontsize=10, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("seconds since start", color=INK2)
    fig.suptitle(f"Same server, same {runs[0][0]['rate']:g} req/s planned average: only the gaps between requests change",
                 fontsize=12, x=0.08, ha="left")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(save(fig, out_dir, "arrival-patterns.png"))


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("predict", "check", "arrivals"))
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if args.action == "predict":
        print(f"service D = {SERVICE_S * 1e3:.2f} ms, {SERVERS} servers, {DURATION_S:g} s runs, {WARMUP_S:g} s warmup")
        print(f"{'arrivals':9} {'rate':>5} {'rho':>5} {'mean wait ms 5/50/95%':>24} {'p95 wait ms 5/50/95%':>26} {'Kingman mean':>12}")
        for p in predict().values():
            print(f"{p['kind']:9} {p['rate']:5g} {p['rho']:5.2f} {'/'.join(f'{v:.1f}' for v in p['mean_ms']):>24} "
                  f"{'/'.join(f'{v:.1f}' for v in p['p95_ms']):>26} {p['kingman_mean_ms']:12.1f}")
        return 0
    if args.action == "arrivals":
        arrival_patterns(args.runs, args.out)
        return 0
    check(args.runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
