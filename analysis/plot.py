#!/usr/bin/env python3
"""Turn run directories into the charts a streaming sweep exists to produce.

The CSV is the source of truth: every percentile on these charts is recomputed
here from records.csv, never read from summary.json — whose histograms are
bucket-approximate (~0.1%). The summary contributes only what exists nowhere
else: the send-lag budget, the failure reasons, and the counters behind the
annotations. config.json contributes the offered rate, the mode, and the warmup
boundary.

Usage: plot.py run_dir... [-o outdir]
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Light-surface chrome and the reference categorical palette, in fixed slot
# order. Three slots cover the usual sweep; the rest are assigned in order,
# never cycled, and past eight series this chart set folds them into a gray
# "other" rather than invent hues.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FOIL = "#898781"  # closed mode, warmup ghosts: context, never identity


class Run:
    """One run directory, parsed once into everything the charts need."""

    def __init__(self, path: Path):
        self.dir = path
        cfg = json.loads((path / "config.json").read_text())
        self.config = cfg["config"]
        self.summary = json.loads((path / "summary.json").read_text())
        self.name = self.config["name"]
        self.mode = self.config["mode"]
        self.rate = self.config["rate"]
        self.warmup_ns = int(self.config["warmup_s"] * 1e9)
        self.duration_ns = int(self.config["duration_s"] * 1e9)
        self.concurrency = self.config["concurrency"]
        self.clipped = self.summary["counters"]["clipped"]
        self.dropped = self.summary["records"]["dropped"]
        self.failed = bool(self.summary["failure_reasons"])

        rows = list(csv.DictReader(open(path / "records.csv")))
        self.window_lat = []
        self.warmup_lat = []
        self.ttft = []
        self.tpot = []
        self.lags = []  # actual - intended, non-clipped, in window
        completed_in_window_ok = 0
        ok_requests = set()
        for r in rows:
            status = r["status"]
            intended = int(r["intended_ns"])
            if status != "clipped" and not int(r["warmup"]):
                self.lags.append((int(r["actual_ns"]) - intended) / 1e6)
            if status == "ok":
                req_id = int(r["req_id"])
                ok_requests.add(req_id)
                first = int(r["first_output_ns"])
                last = int(r["last_output_ns"])
                output_tokens = int(r["output_tokens"])
                actual = int(r["actual_ns"])
                lat = (last - actual) / 1e6
                if int(r["warmup"]):
                    self.warmup_lat.append(lat)
                else:
                    self.window_lat.append(lat)
                    self.ttft.append((first - actual) / 1e6)
                    if output_tokens > 1:
                        self.tpot.append((last - first) / (output_tokens - 1) / 1e6)
                if self.warmup_ns <= last < self.duration_ns:
                    completed_in_window_ok += 1
        self.served_rate = completed_in_window_ok / max(
            self.config["duration_s"] - self.config["warmup_s"], 1e-9
        )
        budget_ns = self.summary["send_lag_budget_ns"]
        self.late_dispatch_fraction = (
            sum(lag_ms * 1e6 > budget_ns for lag_ms in self.lags) / max(len(self.lags), 1)
        )
        self.itl = []
        output_tokens_in_window = 0
        events_by_request = {}
        with open(path / "stream_events.csv") as events_file:
            for event in csv.DictReader(events_file):
                req_id = int(event["req_id"])
                if req_id not in ok_requests:
                    continue
                emitted = int(event["emitted_ns"])
                if self.warmup_ns <= emitted < self.duration_ns:
                    output_tokens_in_window += int(event["token_count"])
                if not int(event["request_warmup"]):
                    events_by_request.setdefault(req_id, []).append(emitted)
        for emitted_times in events_by_request.values():
            self.itl.extend(
                (later - earlier) / 1e6
                for earlier, later in zip(emitted_times, emitted_times[1:])
            )
        self.output_token_rate = output_tokens_in_window / max(
            self.config["duration_s"] - self.config["warmup_s"], 1e-9
        )

    @property
    def label(self):
        if self.mode == "closed":
            return f"closed (C={self.concurrency})"
        return f"{self.rate:g} req/s"

    def quantiles(self, vals):
        """p50 / p99 / p99.9, interpolated; empty in, zeros out."""
        if not vals:
            return 0.0, 0.0, 0.0
        q = statistics.quantiles(sorted(vals), n=1000)
        return q[499], q[989], q[998]


def percentile(vals, q):
    if not vals:
        return 0.0
    vals = sorted(vals)
    position = (len(vals) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(vals) - 1)
    fraction = position - lower
    return vals[lower] * (1 - fraction) + vals[upper] * fraction


def style(ax):
    """Recessive chrome: the data is the subject, never the box around it."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK2)


def new_fig(title):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    return fig, ax


def knee_plot(runs, out: Path):
    """Served rate against requested rate, with the y=x ideal line.

    The knee is where served stops tracking requested: points ride the
    diagonal until capacity, then plateau at it. Capacity itself is read off
    the first overloaded run's plateau and labelled, which is the honest
    reading — the knee lies somewhere between the last tracking point and the
    first clipped one, and only the plateau level is exact.
    """
    open_runs = [r for r in runs if r.mode == "open"]
    if not open_runs:
        return None
    fig, ax = new_fig("knee: served vs requested")
    x = [r.rate for r in open_runs]
    y = [r.served_rate for r in open_runs]
    ax.plot(x, y, marker="o", markersize=6, linewidth=2, color=SERIES[0], zorder=3)
    top = max(x + y) * 1.15
    ax.plot([0, top], [0, top], color=AXIS, linewidth=1, linestyle="--", zorder=1)
    ax.annotate(
        "served = requested",
        xy=(0.72, 0.72),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=9,
        rotation=45,
    )
    ax.set_xlabel("requested rate (req/s)", color=INK2)
    ax.set_ylabel("served rate, window only (req/s)", color=INK2)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)

    overloaded = [r for r in open_runs if r.clipped > 0]
    for r in open_runs:
        note = []
        if r.clipped:
            note.append(f"clipped {r.clipped}")
        if r.dropped:
            note.append(f"dropped {r.dropped}")
        if r.failed:
            note.append("run failed")
        if note:
            ax.annotate(
                "\n".join(note),
                xy=(r.rate, r.served_rate),
                xytext=(r.rate, r.served_rate * 0.55),
                fontsize=8,
                color=INK2,
                ha="center",
            )
    if overloaded:
        plateau = overloaded[0].served_rate
        ax.axhline(plateau, color=SERIES[0], linewidth=1, linestyle=":", zorder=2)
        ax.annotate(
            f"capacity ≈ {plateau:.1f} req/s",
            xy=(top, plateau),
            xytext=(top * 0.98, plateau * 1.18),
            fontsize=9,
            color=SERIES[0],
            ha="right",
        )
    style(ax)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def cdf_plot(runs, out: Path):
    """Latency distributions, one line per run, warmup ghosts behind.

    Categorical slots in fixed order (open runs sorted by rate; the closed
    foil is gray and dashed because it is the same server measured the other
    way, never a fourth series to be pooled with the rest). Warmup rows are
    drawn faded — excluded from the statistics but kept visible, because the
    discarded population is part of the experiment even when it is not part of
    the answer.
    """
    fig, ax = new_fig("successful E2E latency cdf: P(latency <= t), window rows")
    open_runs = sorted((r for r in runs if r.mode == "open"), key=lambda r: r.rate)
    closed = [r for r in runs if r.mode == "closed"]
    lines = []
    for i, r in enumerate(open_runs):
        color = SERIES[i] if i < len(SERIES) else FOIL
        if r.warmup_lat:
            xs = sorted(r.warmup_lat)
            ys = [(k + 1) / len(xs) for k in range(len(xs))]
            ax.plot(xs, ys, color=FOIL, linewidth=0.8, linestyle=":", zorder=1)
        xs = sorted(r.window_lat)
        ys = [(k + 1) / len(xs) for k in range(len(xs))]
        (line,) = ax.plot(xs, ys, color=color, linewidth=1.8, zorder=3)
        lines.append((line, r.label))
    for r in closed:
        xs = sorted(r.window_lat)
        ys = [(k + 1) / len(xs) for k in range(len(xs))]
        (line,) = ax.plot(xs, ys, color=FOIL, linewidth=1.8, linestyle="--", zorder=3)
        lines.append((line, r.label))

    # Direct labels at the line ends; a legend appears too, so identity is
    # never carried by color alone.
    for line, label in lines:
        x, y = line.get_xdata()[-1], 1.0
        ax.annotate(label, xy=(x, y), xytext=(4, -2), textcoords="offset points", fontsize=8, color=INK2)
    ax.legend([l for l, _ in lines], [lab for _, lab in lines], fontsize=8, frameon=False, loc="lower right")
    ax.set_xlabel("successful E2E, last output − actual send (ms)", color=INK2)
    ax.set_ylabel("cumulative fraction", color=INK2)
    ax.set_ylim(0, 1.02)
    style(ax)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def token_knee_plot(runs, out: Path):
    """Successful output tokens emitted inside the measurement window."""
    open_runs = sorted((r for r in runs if r.mode == "open"), key=lambda r: r.rate)
    if not open_runs:
        return None
    fig, ax = new_fig("output-token throughput")
    ax.plot(
        [r.rate for r in open_runs],
        [r.output_token_rate for r in open_runs],
        marker="o",
        linewidth=2,
        color=SERIES[0],
    )
    ax.set_xlabel("requested rate (req/s)", color=INK2)
    ax.set_ylabel("successful output tokens emitted in window (tok/s)", color=INK2)
    style(ax)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def streaming_percentiles_plot(runs, out: Path):
    """p50/p95/p99 client-observed streaming latency across open-loop rates."""
    open_runs = sorted((r for r in runs if r.mode == "open"), key=lambda r: r.rate)
    if not open_runs:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, attribute) in zip(
        axes.flat,
        [("TTFT", "ttft"), ("TPOT", "tpot"), ("ITL", "itl"), ("E2E", "window_lat")],
    ):
        for q, color, label in [
            (0.50, SERIES[0], "p50"),
            (0.95, SERIES[3], "p95"),
            (0.99, SERIES[7], "p99"),
        ]:
            ax.plot(
                [r.rate for r in open_runs],
                [percentile(getattr(r, attribute), q) for r in open_runs],
                marker="o",
                color=color,
                label=label,
            )
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("requested rate (req/s)", color=INK2)
        ax.set_ylabel("milliseconds", color=INK2)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("streaming latency percentiles", color=INK, fontsize=12, x=0.08, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def lag_plot(runs, out: Path):
    """What the harness itself did late: per run, min–p99 line, p50 dot, and
    the budget the verdict was judged against. Open runs only — closed mode has
    no separate arrival time, so it has no lag to draw."""
    open_runs = [r for r in runs if r.mode == "open"]
    if not open_runs:
        return None
    fig, ax = new_fig("send lag (harness, not target)")
    for i, r in enumerate(sorted(open_runs, key=lambda r: r.rate)):
        color = SERIES[i] if i < len(SERIES) else FOIL
        p50 = percentile(r.lags, 0.50)
        p99 = percentile(r.lags, 0.99)
        lo = min(r.lags) if r.lags else 0
        ax.vlines(r.rate, lo, p99, color=color, linewidth=1.2, zorder=3)
        ax.plot(r.rate, p50, "o", markersize=6, color=INK, zorder=4)
        budget = r.summary["send_lag_budget_ns"] / 1e6
        ax.plot(r.rate, budget, "x", markersize=8, color=MUTED, zorder=2)
        ax.annotate(
            f"late {r.late_dispatch_fraction * 100:.1f}%",
            xy=(r.rate, max(p99, budget)),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
            ha="center",
        )
        if r.failed or r.dropped:
            note = []
            if r.failed:
                note.append("failed")
            if r.dropped:
                note.append(f"dropped {r.dropped}")
            ax.annotate(
                ", ".join(note), xy=(r.rate, max(p99, budget)), xytext=(0, 18),
                textcoords="offset points", fontsize=8, color=INK2, ha="center",
            )
    # One legend for the mark vocabulary, not one entry per run.
    ax.scatter([], [], marker="o", color=INK, label="p50")
    ax.plot([], [], color=AXIS, linewidth=1.2, label="min–p99")
    ax.scatter([], [], marker="x", color=MUTED, label="budget (verdict line)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("requested rate (req/s)", color=INK2)
    ax.set_ylabel("send lag (ms)", color=INK2)
    ax.set_ylim(0, None)
    style(ax)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run directories")
    ap.add_argument("-o", "--out", type=Path, default=Path("analysis"), help="output directory")
    args = ap.parse_args(argv)

    runs = []
    for p in args.runs:
        if not (p / "records.csv").is_file():
            print(f"plot.py: {p} is not a run directory (no records.csv)", file=sys.stderr)
            return 2
        runs.append(Run(p))

    args.out.mkdir(parents=True, exist_ok=True)
    for f, plot in [
        ("knee.png", knee_plot),
        ("token-knee.png", token_knee_plot),
        ("latency-cdf.png", cdf_plot),
        ("streaming-percentiles.png", streaming_percentiles_plot),
        ("send-lag.png", lag_plot),
    ]:
        path = plot(runs, args.out / f)
        if path:
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
