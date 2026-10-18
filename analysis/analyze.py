#!/usr/bin/env python3
"""Analyze satlab run directories into sweep.csv, a summary table, and PNG charts.

Usage: analyze.py run_dir... [-o outdir] [--expected-capacity 80]

Every run is checked against docs/measurement-contract.md while it loads.
A mismatch with summary.json stops the analysis.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

QUANTILES = (0.50, 0.95, 0.99, 0.999)
Q_LABELS = {0.50: "p50", 0.95: "p95", 0.99: "p99", 0.999: "p999"}
# *_from_plan start at intended_ns (the planned send), the others at actual_ns.
LATENCY_METRICS = ("ttft", "tpot", "itl", "e2e", "ttft_from_plan", "e2e_from_plan")
HDR_KEYS = {"ttft": "successful_ttft_ns", "tpot": "successful_tpot_ns", "itl": "itl_ns",
            "e2e": "successful_e2e_ns", "sendlag": "send_lag_ns"}
# SLO tiers for a person waiting for the answer: (TTFT ms, E2E ms). A request is good
# when it meets both. Basis: 1 s and 10 s response-time limits (Nielsen 1993).
SLOS = {"fast": (500.0, 1000.0), "medium": (1000.0, 5000.0), "slow": (2000.0, 10000.0)}
COLUMNS = [
    "run_id", "name", "mode", "concurrency",
    "requested_rate_hz", "sent_rate_hz", "served_rate_hz", "output_token_rate_hz",
    *(f"goodput_{slo}_hz" for slo in SLOS),
    *(f"attainment_{slo}" for slo in SLOS),
    *(f"{metric}_{Q_LABELS[q]}_ms" for metric in (*LATENCY_METRICS, "sendlag") for q in QUANTILES),
    *(f"{metric}_n" for metric in LATENCY_METRICS),
    "outstanding_at_warmup", "outstanding_at_end", "outstanding_slope_req_s",
    "mean_running", "little_l", "little_lw",
    "late_dispatch_fraction", "clipped", "dropped", "timeouts", "http_errors",
    "stream_errors", "pass", "zero_errors", "fail_reasons", "git_sha", "target_version",
]


class Req(NamedTuple):
    id: int
    intended: int
    actual: int
    first: int
    last: int
    completed: int
    tokens: int
    status: str
    warmup: bool


def percentile(vals, q):
    if not vals:
        return 0.0
    vals = sorted(vals)
    position = (len(vals) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(vals) - 1)
    fraction = position - lower
    return vals[lower] * (1 - fraction) + vals[upper] * fraction


def read_requests(path: Path) -> list[Req]:
    with path.open(newline="") as handle:
        return [Req(int(r["req_id"]), int(r["intended_ns"]), int(r["actual_ns"]),
                    int(r["first_output_ns"]), int(r["last_output_ns"]),
                    int(r["completed_ns"]), int(r["output_tokens"]), r["status"],
                    r["warmup"] != "0")
                for r in csv.DictReader(handle)]


def read_events(path: Path) -> list[tuple[int, int, int, bool]]:
    """(req_id, emitted_ns, token_count, request_warmup) per token-bearing SSE event."""
    with path.open(newline="") as handle:
        return [(int(r["req_id"]), int(r["emitted_ns"]), int(r["token_count"]),
                 r["request_warmup"] != "0") for r in csv.DictReader(handle)]


def outstanding_series(requests: list[Req], duration_s: float) -> list[tuple[float, int]]:
    """Client requests sent but not yet fully finished, sampled every 0.1 s.

    This includes server work, server wait, and network time. It is not the
    server's internal waiting-request count.
    """
    sent = sorted(r.actual for r in requests if r.status != "clipped")
    finished = sorted(r.completed for r in requests if r.status != "clipped" and r.completed > 0)
    sent_index = finished_index = 0
    series = []
    for tick in range(round(duration_s * 10) + 1):
        timestamp = tick * 100_000_000
        while sent_index < len(sent) and sent[sent_index] <= timestamp:
            sent_index += 1
        while finished_index < len(finished) and finished[finished_index] <= timestamp:
            finished_index += 1
        series.append((tick / 10, sent_index - finished_index))
    return series


def trend_per_second(series: list[tuple[float, int]], warmup_s: float) -> float:
    points = [(second, count) for second, count in series if second >= warmup_s]
    if len(points) < 2:
        return 0.0
    return statistics.linear_regression(*zip(*points)).slope


def server_series(run_dir: Path, anchor_wall_ns: int | None,
                  duration_s: float) -> list[tuple[float, float, float | None]]:
    """(seconds since start, waiting, running) from server_metrics.csv, if saved."""
    path = run_dir / "server_metrics.csv"
    if not path.is_file() or anchor_wall_ns is None:
        return []
    series = []
    with path.open(newline="") as handle:
        for sample in csv.DictReader(handle):
            if not sample["waiting"]:
                continue
            elapsed = (int(sample["wall_unix_ns"]) - anchor_wall_ns) / 1e9
            if 0 <= elapsed <= duration_s:
                running = float(sample["running"]) if sample["running"] else None
                series.append((elapsed, float(sample["waiting"]), running))
    return series


def check_contract(name: str, summary: dict, seen: dict, samples: dict) -> None:
    """Assert that raw CSV rows reproduce summary.json under the measurement contract."""
    records, counters = summary["records"], summary["counters"]
    exact = [
        ("intended = fired + clipped", counters["intended"], counters["fired"] + counters["clipped"]),
        ("recorded rows", records["recorded"], seen["rows"]),
        ("dropped rows", records["dropped"], 0),
        ("status counts", dict(records["by_status"]), dict(seen["by_status"])),
        ("output_tokens from requests", records["output_tokens"], seen["request_tokens"]),
        ("output_tokens from events", records["output_tokens"], seen["event_tokens"]),
        ("completed_in_window_ok", records["completed_in_window_ok"], seen["served"]),
        ("output_tokens_in_window", records["output_tokens_in_window"], seen["window_tokens"]),
    ]
    for label, want, got in exact:
        assert want == got, f"{name}: {label}: summary {want} != CSV {got}"
    for label, key, count in (("served rate", "served_rate_hz", seen["served"]),
                              ("token rate", "output_token_rate_hz", seen["window_tokens"])):
        rate = count / seen["window_s"]
        assert abs(rate - summary[key]) <= 0.01, f"{name}: {label}: summary {summary[key]} != CSV {rate:.2f}"
    for metric, key in HDR_KEYS.items():
        block, vals = records[key], sorted(samples[metric])
        assert block["count"] == len(vals), (
            f"{name}: {metric} sample count: summary {block['count']} != CSV {len(vals)}")
        for q in (0.50, 0.95, 0.99) if vals else ():
            # HDR keeps three significant digits and picks a sample by rank. Python
            # interpolates, so accept anything between the neighbouring raw samples.
            rank = int((len(vals) - 1) * q)
            low = vals[max(rank - 1, 0)] * 0.998 - 0.001
            high = vals[min(rank + 2, len(vals) - 1)] * 1.002 + 0.001
            hdr_ms = block[Q_LABELS[q]] / 1e6
            assert low <= hdr_ms <= high, (
                f"{name}: {metric} {Q_LABELS[q]}: summary {hdr_ms:.3f} ms outside CSV "
                f"samples [{low:.3f}, {high:.3f}] ms")


def load_run(run_dir: Path) -> dict:
    doc = json.loads((run_dir / "config.json").read_text())
    config, provenance = doc["config"], doc["provenance"]
    summary = json.loads((run_dir / "summary.json").read_text())
    records = summary["records"]
    requests = read_requests(run_dir / "records.csv")
    events = read_events(run_dir / "stream_events.csv")

    warmup_s, duration_s = config["warmup_s"], config["duration_s"]
    start, end = int(warmup_s * 1e9), int(duration_s * 1e9)
    window_s = duration_s - warmup_s
    outstanding = outstanding_series(requests, duration_s)
    warmup_index = min(round(warmup_s * 10), len(outstanding) - 1)
    server = server_series(run_dir, provenance.get("anchor_wall_unix_ns"), duration_s)

    # Populations follow docs/measurement-contract.md.
    ok_ids, ttft, tpot, e2e, lags = set(), [], [], [], []
    ttft_plan, e2e_plan = [], []
    served = request_tokens = 0
    good = dict.fromkeys(SLOS, 0)
    # Attainment: share of requests intended in the window that meet both limits;
    # a failed request does not meet them.
    attained, population = dict.fromkeys(SLOS, 0), 0
    for r in requests:
        if r.status == "ok":
            ok_ids.add(r.id)
            if not r.warmup:
                request_tokens += r.tokens
                e2e.append((r.last - r.actual) / 1e6)
                ttft.append((r.first - r.actual) / 1e6)
                e2e_plan.append((r.last - r.intended) / 1e6)
                ttft_plan.append((r.first - r.intended) / 1e6)
                if r.tokens > 1:
                    tpot.append((r.last - r.first) / (r.tokens - 1) / 1e6)
                for slo, (ttft_ms, e2e_ms) in SLOS.items():
                    attained[slo] += ttft[-1] <= ttft_ms and e2e[-1] <= e2e_ms
            if start <= r.last < end:
                served += 1
                for slo, (ttft_ms, e2e_ms) in SLOS.items():
                    if (r.first - r.actual) / 1e6 <= ttft_ms and (r.last - r.actual) / 1e6 <= e2e_ms:
                        good[slo] += 1
        if r.status != "clipped" and not r.warmup:
            population += 1
            lags.append((r.actual - r.intended) / 1e6)

    itl, by_request = [], {}
    event_tokens = window_tokens = 0
    for req_id, emitted, token_count, warmup in events:
        if req_id not in ok_ids:
            continue
        if start <= emitted < end:
            window_tokens += token_count
        if not warmup:
            event_tokens += token_count
            by_request.setdefault(req_id, []).append(emitted)
    for times in by_request.values():
        itl.extend((b - a) / 1e6 for a, b in zip(times, times[1:]))

    latency = {"ttft": ttft, "tpot": tpot, "itl": itl, "e2e": e2e,
               "ttft_from_plan": ttft_plan, "e2e_from_plan": e2e_plan}
    percentiles = {m: {q: percentile(v, q) for q in QUANTILES} for m, v in latency.items()}
    percentiles["sendlag"] = {q: percentile(lags, q) for q in QUANTILES}
    check_contract(run_dir.name, summary, {
        "rows": len(requests), "by_status": Counter(r.status for r in requests),
        "request_tokens": request_tokens, "event_tokens": event_tokens,
        "served": served, "window_tokens": window_tokens, "window_s": window_s,
    }, {**latency, "sendlag": lags})

    # Little's Law on one population: requests that finished inside the window.
    finished = [r for r in requests if r.status != "clipped" and start <= r.completed < end]
    sojourn_s = statistics.fmean((r.completed - r.actual) / 1e9 for r in finished) if finished else 0.0
    mean_outstanding = statistics.fmean(v for _, v in outstanding[warmup_index:])
    running = [v for t, _, v in server if t >= warmup_s and v is not None]

    in_window_errors = records["in_window_errors"]
    timeouts = records["in_window_timeouts"]
    stream_errors = records["in_window_stream_errors"]
    row = {
        "run_id": run_dir.name,
        "name": config["name"],
        "mode": config["mode"],
        "concurrency": config["concurrency"],
        "requested_rate_hz": summary["requested_rate_hz"],
        "sent_rate_hz": summary["sent_rate_hz"],
        "served_rate_hz": summary["served_rate_hz"],
        "output_token_rate_hz": summary["output_token_rate_hz"],
        "outstanding_at_warmup": outstanding[warmup_index][1],
        "outstanding_at_end": outstanding[-1][1],
        "outstanding_slope_req_s": trend_per_second(outstanding, warmup_s),
        "mean_running": statistics.fmean(running) if running else "",
        "little_l": mean_outstanding,
        "little_lw": len(finished) / window_s * sojourn_s,
        "late_dispatch_fraction": records["late_dispatch_fraction"],
        "clipped": summary["counters"]["clipped"],
        "dropped": records["dropped"],
        "timeouts": timeouts,
        "http_errors": in_window_errors - timeouts - stream_errors,
        "stream_errors": stream_errors,
        "pass": not summary["failure_reasons"],
        "zero_errors": in_window_errors == 0,
        "fail_reasons": "; ".join(summary["failure_reasons"]),
        "git_sha": provenance["git_sha"],
        "target_version": provenance["target_version"],
    }
    for slo, count in good.items():
        row[f"goodput_{slo}_hz"] = count / window_s
        row[f"attainment_{slo}"] = attained[slo] / population if population else float("nan")
    for metric, quantiles in percentiles.items():
        for q, value in quantiles.items():
            row[f"{metric}_{Q_LABELS[q]}_ms"] = value
    for metric, values in latency.items():
        row[f"{metric}_n"] = len(values)
    return {
        "dir": run_dir,
        "row": row,
        "failed": bool(summary["failure_reasons"]),
        "clean": not summary["failure_reasons"] and in_window_errors == 0,
        "percentiles": percentiles,
        "raw": {"sendlag": lags},
        "outstanding": outstanding,
        "server_waiting": [(t, waiting) for t, waiting, _ in server],
        "warmup_s": warmup_s,
        "lag_budget_ms": summary["send_lag_budget_ns"] / 1e6,
    }


def write_sweep(runs, out_dir: Path) -> Path:
    path = out_dir / "sweep.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({key: run["row"][key] for key in COLUMNS} for run in runs)
    return path


def grouped(runs, mode):
    key = "requested_rate_hz" if mode == "open" else "concurrency"
    groups = {}
    for run in runs:
        if run["row"]["mode"] == mode:
            groups.setdefault(run["row"][key], []).append(run)
    return dict(sorted(groups.items()))


def aggregate(runs_in_group, field):
    values = sorted(run["row"][field] for run in runs_in_group)
    return statistics.median(values), values[0], values[-1]


def detect_knee(open_groups):
    """Candidate knee and max served rate from clean, zero-error groups."""
    medians = [(rate, aggregate(runs, "served_rate_hz")[0])
               for rate, runs in open_groups.items()
               if all(run["clean"] for run in runs)]
    if not medians:
        return None, None
    plateau = max(median for _, median in medians)
    knee = next((rate for rate, median in medians if median < 0.95 * rate), None)
    return knee, plateau


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def new_fig(title, size=(8, 4.8)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    return fig, ax


def save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def annotate_failed(ax, x, y, group):
    notes = []
    if any(r["failed"] for r in group):
        notes.append("failed")
    for field in ("clipped", "dropped"):
        worst = max(r["row"][field] for r in group)
        if worst:
            notes.append(f"{field} {worst}")
    if notes:
        ax.plot(x, y, "x", markersize=9, color=INK, zorder=5)
        ax.annotate(", ".join(notes), xy=(x, y), xytext=(0, 10),
                    textcoords="offset points", fontsize=8, color=INK2, ha="center")


def request_knee(runs, out_dir: Path):
    """Hero 1: served and goodput against offered rate."""
    groups = grouped(runs, "open")
    if not groups:
        return None
    knee, plateau = detect_knee(groups)
    fig, ax = new_fig("Served and good requests per second vs offered rate", size=(8, 5.6))
    x = list(groups)
    served = [aggregate(g, "served_rate_hz") for g in groups.values()]
    ax.fill_between(x, [s[1] for s in served], [s[2] for s in served],
                    color=SERIES[0], alpha=0.15, linewidth=0)
    ax.plot(x, [s[0] for s in served], marker="o", linewidth=2, color=SERIES[0],
            label="served (any latency)", zorder=3)
    for (slo, (ttft_ms, e2e_ms)), width, alpha in zip(SLOS.items(), (1.0, 2.4, 1.0), (0.55, 1, 0.55)):
        good = [aggregate(g, f"goodput_{slo}_hz")[0] for g in groups.values()]
        label = (f"goodput: TTFT ≤ {ttft_ms / 1e3:g} s and E2E ≤ {e2e_ms / 1e3:g} s"
                 + (" (medium)" if slo == "medium" else ""))
        ax.plot(x, good, marker="o", markersize=4, linewidth=width, color=SERIES[1],
                alpha=alpha, label=label, zorder=4)
    for rate, group in groups.items():
        annotate_failed(ax, rate, aggregate(group, "served_rate_hz")[0], group)
    top = max(x) * 1.05
    ax.plot([0, top], [0, top], color=AXIS, linestyle="--", zorder=1, label="served = offered")
    closed = grouped(runs, "closed")
    if closed:
        peak = max(aggregate(g, "served_rate_hz")[0] for g in closed.values())
        ax.axhline(peak, color=SERIES[2], linestyle=":", linewidth=1.6, zorder=2,
                   label=f"closed-loop peak ({peak:.1f} req/s)")
    if knee is not None:
        ax.axvline(knee, color=MUTED, linestyle=":", zorder=2)
        ax.annotate(f"knee ≈ {knee:g}", xy=(knee, 0), xytext=(4, 6),
                    textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlabel("offered rate (req/s)", color=INK2)
    ax.set_ylabel("requests per second, measurement window", color=INK2)
    ax.set_xlim(min(x) * 0.95, top)
    ax.set_ylim(0, top)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    style(ax)
    return save(fig, out_dir, "request-knee.png"), knee, plateau


def outstanding_work(runs, out_dir: Path):
    """Hero 2: requests sent but not finished, for the rates around the knee."""
    groups = grouped(runs, "open")
    if not groups:
        return None
    rates = list(groups)
    knee, _ = detect_knee(groups)
    below = [r for r in rates if knee is not None and r < knee]
    chosen = list(dict.fromkeys(
        ([below[-1], knee] if below else [rates[len(rates) // 2]]) + [rates[-1]]))
    fig, ax = new_fig("Requests sent but not finished (median of repeats)", size=(9, 5))
    for index, rate in enumerate(chosen):
        group = groups[rate]
        if len({len(r["outstanding"]) for r in group}) != 1:
            raise ValueError("do not combine different run durations at one offered rate")
        ticks = list(zip(*(r["outstanding"] for r in group)))
        seconds = [pairs[0][0] for pairs in ticks]
        counts = [sorted(count for _, count in pairs) for pairs in ticks]
        low, high = [c[0] for c in counts], [c[-1] for c in counts]
        median = [statistics.median(c) for c in counts]
        color = SERIES[index]
        ax.fill_between(seconds, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(seconds, median, color=color, linewidth=2)
        ax.annotate(f"{rate:g} req/s" + (" (invalid)" if any(r["failed"] for r in group) else ""),
                    xy=(seconds[-1], median[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=10, color=color, va="center")
        if index >= len(chosen) - 2:
            waiting_by_second = {}
            for run in group:
                for t, waiting in run["server_waiting"]:
                    waiting_by_second.setdefault(round(t), []).append(waiting)
            if waiting_by_second:
                points = sorted((t, statistics.median(v)) for t, v in waiting_by_second.items())
                ax.plot(*zip(*points), color=color, linestyle="--", alpha=0.8)
    ax.plot([], [], color=MUTED, linestyle="--", label="server 'waiting' gauge (dashed)")
    ax.axvline(groups[rates[0]][0]["warmup_s"], color=MUTED, linestyle=":", label="warmup ends")
    ax.set_xlabel("seconds since run start", color=INK2)
    ax.set_ylabel("requests", color=INK2)
    ax.margins(x=0.12)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    return save(fig, out_dir, "outstanding-work.png")


def tpot_vs_running(runs, out_dir: Path):
    """Hero 3: per-token time against the number of requests admitted together."""
    open_points, closed_points = [], []
    for run in runs:
        row = run["row"]
        if run["failed"] or not row["tpot_n"]:
            continue
        if row["mode"] == "open" and row["mean_running"] != "":
            open_points.append((row["mean_running"], row["tpot_p50_ms"]))
        elif row["mode"] == "closed":
            # A closed loop admits exactly one request per worker.
            closed_points.append((row["concurrency"], row["tpot_p50_ms"]))
    if len(open_points) < 3:
        return None
    fig, ax = new_fig("Per-token time grows with requests admitted together")
    xs, ys = zip(*open_points)
    fit = statistics.linear_regression(xs, ys)
    top = max(xs) * 1.05
    ax.plot([0, top], [fit.intercept, fit.intercept + fit.slope * top], color=AXIS,
            linestyle="--", zorder=1)
    ax.scatter(xs, ys, color=SERIES[0], s=40, zorder=3, label="open-loop run")
    if closed_points:
        ax.scatter(*zip(*closed_points), color=SERIES[1], marker="D", s=40, zorder=3,
                   label="closed-loop run (x = workers)")
    ax.annotate(f"+{fit.slope:.1f} ms per admitted request\n"
                f"≈ {1000 / fit.slope:.0f} output tokens/s shared by all",
                xy=(0.04, 0.78), xycoords="axes fraction", fontsize=10, color=INK2)
    ax.set_xlabel("mean requests running on the server (after warmup)", color=INK2)
    ax.set_ylabel("p50 time per output token (ms)", color=INK2)
    ax.set_xlim(0, top)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    return save(fig, out_dir, "tpot-vs-running.png")


def send_lag(runs, out_dir: Path):
    """Rebuttal chart: the harness kept its schedule."""
    groups = grouped(runs, "open")
    if not groups:
        return None
    fig, ax = new_fig("Send lag (harness, not target)")
    for rate, group in groups.items():
        p50 = statistics.median(r["percentiles"]["sendlag"][0.50] for r in group)
        p99 = statistics.median(r["percentiles"]["sendlag"][0.99] for r in group)
        low = min(min(r["raw"]["sendlag"]) for r in group if r["raw"]["sendlag"])
        budget = group[0]["lag_budget_ms"]
        ax.vlines(rate, low, p99, color=SERIES[0], linewidth=1.2, zorder=3)
        ax.plot(rate, p50, "o", markersize=6, color=INK, zorder=4)
        ax.plot(rate, budget, "x", markersize=8, color=MUTED, zorder=2)
        late = max(r["row"]["late_dispatch_fraction"] for r in group)
        ax.annotate(f"late {late * 100:.1f}%", xy=(rate, max(p99, budget)),
                    xytext=(0, 6), textcoords="offset points", fontsize=8, color=INK2, ha="center")
        annotate_failed(ax, rate, p50, group)
    ax.scatter([], [], marker="o", color=INK, label="median p50")
    ax.plot([], [], color=SERIES[0], linewidth=1.2, label="min–p99 across repeats")
    ax.scatter([], [], marker="x", color=MUTED, label="budget (verdict line)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("offered rate (req/s)", color=INK2)
    ax.set_ylabel("send lag (ms)", color=INK2)
    ax.set_ylim(0, None)
    style(ax)
    return save(fig, out_dir, "send-lag.png")


def print_table(runs) -> None:
    """One line per load level, median of repeats. Little's Law: L should equal λ·W."""
    header = (f"{'load':>6} {'served':>7} {'good':>6} {'att %':>6} {'p95 ttft':>9} {'p95 e2e':>8} {'plan':>6} "
              f"{'unfin.':>6} {'slope/s':>8} {'L':>6} {'λ·W':>6}  note")
    for mode, label in (("open", "offered req/s"), ("closed", "workers")):
        groups = grouped(runs, mode)
        if not groups:
            continue
        print(f"\n{mode} loop, {label}; median of repeats; times in seconds")
        print(header)
        for load, group in groups.items():
            med = lambda field: aggregate(group, field)[0]
            note = "FAILED harness check" if any(r["failed"] for r in group) else ""
            print(f"{load:>6g} {med('served_rate_hz'):>7.2f} {med('goodput_medium_hz'):>6.2f} "
                  f"{100 * med('attainment_medium'):>6.1f} "
                  f"{med('ttft_p95_ms') / 1e3:>9.2f} {med('e2e_p95_ms') / 1e3:>8.2f} "
                  f"{med('e2e_from_plan_p95_ms') / 1e3:>6.2f} "
                  f"{med('outstanding_at_end'):>6.0f} {med('outstanding_slope_req_s'):>8.2f} "
                  f"{med('little_l'):>6.1f} {med('little_lw'):>6.1f}  {note}")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("-o", "--out", type=Path, default=Path("analysis"), help="output directory")
    parser.add_argument("--expected-capacity", type=float, default=None,
                        help="known capacity in req/s; warn when the plateau is more than 10%% away")
    args = parser.parse_args(argv)
    if args.expected_capacity is not None and args.expected_capacity <= 0:
        parser.error("--expected-capacity must be greater than zero")

    runs = []
    for path in args.runs:
        if not (path / "records.csv").is_file():
            print(f"analyze.py: {path} is not a run directory (no records.csv)", file=sys.stderr)
            return 2
        runs.append(load_run(path))

    args.out.mkdir(parents=True, exist_ok=True)
    print(write_sweep(runs, args.out))
    print_table(runs)
    knee_result = request_knee(runs, args.out)
    if knee_result:
        path, knee, plateau = knee_result
        if plateau is None:
            print("\nno clean open-loop rate group; no knee or capacity estimate")
        elif knee is None:
            print(f"\nno knee observed; max clean served rate ≈ {plateau:.1f} req/s")
        else:
            print(f"\ncandidate knee ≈ {knee:g} req/s, max clean served rate ≈ {plateau:.1f} req/s")
        if (knee is not None and args.expected_capacity is not None
                and not 0.9 * args.expected_capacity <= plateau <= 1.1 * args.expected_capacity):
            print(f"WARNING: plateau {plateau:.1f} deviates from expected "
                  f"capacity {args.expected_capacity:.1f} by more than 10%", file=sys.stderr)
        print(path)
    for chart in (outstanding_work, tpot_vs_running, send_lag):
        path = chart(runs, args.out)
        if path:
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
