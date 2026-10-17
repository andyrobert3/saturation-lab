#!/usr/bin/env python3
"""Validate satlab's client-side numbers against vLLM's server-side /metrics.

One satlab run wrapped with Prometheus /metrics snapshots. The client reports
p50/p95/p99 in summary.json; the server reports cumulative histograms on
/metrics. This script compares them on the same traffic and writes the result
as server_validation.json (verdict table) and server_metrics_delta.json
(histogram/counter deltas) in the run directory.

vLLM's default histogram buckets are coarse (e.g. TPOT 10-25 ms, TTFT
100-250 ms at this workload), so histogram_quantile interpolation is only a
uniform-spread guess inside one bucket. Quantile rows therefore check that the
client value falls inside the bucket containing the server quantile, expanded
by the band. The decisive checks use exact means: the histogram's sum/count
against the client mean recomputed from records.csv with satlab's formulas.

Exit 0 when every PASS/WARN row holds; exit 1 on any FAIL; exit 2 when the
target's /metrics endpoint is unreachable.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.request import urlopen

REPO = Path(__file__).resolve().parent.parent
SATLAB = REPO / "target" / "release" / "satlab"

# vLLM metric names. Histogram values are seconds; counters are tokens/requests.
HISTOGRAMS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
    "tpot": "vllm:request_time_per_output_token_seconds",
    "itl": "vllm:inter_token_latency_seconds",
}
COUNTER_TOKENS = "vllm:generation_tokens_total"
COUNTER_SUCCESS = "vllm:request_success_total"

# TPOT tolerance bands widen with the quantile: HDR interpolation noise grows
# in the tail, and the server tail is sparser than the client tail.
TPOT_BANDS = {"p50": 0.05, "p95": 0.10, "p99": 0.20}
TTFT_E2E_MAX_OFFSET_MS = 15.0


def parse_prom(text: str, model: str | None = None) -> dict:
    """Parse Prometheus text into counters, success-by-reason, and histograms.

    Values are summed across engines; histogram series are summed per `le`
    bucket. Series carrying a `model_name` label not equal to `model` are
    skipped, so multi-model deployments do not contaminate the comparison.
    """
    counters: dict[str, float] = {}
    success_by_reason: dict[str, float] = {}
    histograms: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_labels, _, value_text = line.rpartition(" ")
        try:
            value = float(value_text)
        except ValueError:
            continue
        labels: dict[str, str] = {}
        if "{" in name_labels:
            name, _, label_text = name_labels.partition("{")
            for pair in label_text.rstrip("}").split(","):
                if "=" not in pair:
                    continue
                key, _, val = pair.partition("=")
                labels[key.strip()] = val.strip().strip('"')
        else:
            name = name_labels
        if model is not None and labels.get("model_name", model) != model:
            continue
        if name.endswith("_bucket"):
            base = name[: -len("_bucket")]
            le = float(labels.get("le", "+Inf"))
            entry = histograms.setdefault(base, {"buckets": {}, "sum": 0.0, "count": 0.0})
            entry["buckets"][le] = entry["buckets"].get(le, 0.0) + value
        elif name.endswith("_sum"):
            entry = histograms.setdefault(name[: -len("_sum")],
                                          {"buckets": {}, "sum": 0.0, "count": 0.0})
            entry["sum"] += value
        elif name.endswith("_count"):
            entry = histograms.setdefault(name[: -len("_count")],
                                          {"buckets": {}, "sum": 0.0, "count": 0.0})
            entry["count"] += value
        elif name == COUNTER_SUCCESS:
            reason = labels.get("finished_reason", "")
            success_by_reason[reason] = success_by_reason.get(reason, 0.0) + value
        else:
            counters[name] = counters.get(name, 0.0) + value
    return {"counters": counters, "success_by_reason": success_by_reason,
            "histograms": histograms}


def hist_quantile(buckets, count, q):
    """Prometheus histogram_quantile linear interpolation. `buckets` is sorted
    [(le, cumulative_count), ...]; returns None when there are no samples."""
    if count <= 0 or not buckets:
        return None
    rank = q * count
    prev_le, prev_cum = 0.0, 0
    for le, cum in buckets:  # sorted by le
        if cum >= rank:
            if le == float("inf"):
                return prev_le
            if cum == prev_cum:
                prev_le = le
                continue
            return prev_le + (le - prev_le) * (rank - prev_cum) / (cum - prev_cum)
        prev_le, prev_cum = le, cum
    return prev_le


def quantile_interval(buckets, count, q):
    """(lo, hi) of the bucket containing the true q-quantile.

    The interpolation above is only a uniform-spread guess inside this
    interval; vLLM's default buckets are coarse enough that the guess can be
    tens of percent off the true quantile. Verdicts use the interval."""
    if count <= 0 or not buckets:
        return None, None
    rank = q * count
    prev_le, prev_cum = 0.0, 0
    for le, cum in buckets:  # sorted by le
        if cum >= rank:
            return (le, le) if cum == rank else (prev_le, le)
        prev_le, prev_cum = le, cum
    return prev_le, float("inf")


def delta_histogram(before, after):
    """(sorted buckets, count, sum) of `after - before` for one histogram."""
    if before is None or after is None:
        return None, None, None
    buckets = {}
    for le, cum in after["buckets"].items():
        buckets[le] = cum - before["buckets"].get(le, 0.0)
    for le, cum in before["buckets"].items():
        if le not in buckets:
            buckets[le] = -cum
    ordered = sorted(buckets.items(), key=lambda item: item[0])
    return ordered, after["count"] - before["count"], after["sum"] - before["sum"]


def client_latency_means(run_dir: Path) -> dict:
    """Exact client means (seconds) over the summary's latency population:
    successful, non-warmup requests, using satlab's own formulas."""
    sums = {"ttft": 0.0, "e2e": 0.0, "tpot": 0.0}
    counts = {"ttft": 0, "e2e": 0, "tpot": 0}
    with (run_dir / "records.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok" or row["warmup"] != "0":
                continue
            actual = int(row["actual_ns"])
            first = int(row["first_output_ns"])
            last = int(row["last_output_ns"])
            tokens = int(row["output_tokens"])
            sums["ttft"] += (first - actual) / 1e9
            sums["e2e"] += (last - actual) / 1e9
            counts["ttft"] += 1
            counts["e2e"] += 1
            if tokens > 1:
                sums["tpot"] += (last - first) / (tokens - 1) / 1e9
                counts["tpot"] += 1
    return {key: (sums[key] / counts[key] if counts[key] else None) for key in sums}, counts


def fetch_text(url: str, timeout: float = 5.0) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def poll_metrics(url: str, stop: threading.Event, poll_s: float, samples: list) -> None:
    """Append (wall_unix_ns, text) samples until `stop` is set."""
    while not stop.is_set():
        timestamp = time.time_ns()
        try:
            samples.append((timestamp, fetch_text(url)))
        except OSError:
            pass
        stop.wait(poll_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="satlab config JSON")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--poll-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=2.0)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    target = config["target_url"].rstrip("/")
    metrics_url = target + "/metrics"
    try:
        baseline_text = fetch_text(metrics_url)
    except OSError as error:
        print(f"server /metrics unavailable at {metrics_url}: {error}", file=sys.stderr)
        return 2

    samples: list[tuple[int, str]] = [(time.time_ns(), baseline_text)]
    stop = threading.Event()
    worker = threading.Thread(target=poll_metrics,
                              args=(metrics_url, stop, args.poll_s, samples), daemon=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(config, handle)
        config_path = Path(handle.name)
    try:
        worker.start()
        proc = subprocess.run(
            [str(SATLAB), "--config", str(config_path), "--runs-dir", str(args.runs_dir)],
            capture_output=True, text=True, cwd=REPO,
        )
    finally:
        stop.set()
        worker.join(timeout=args.poll_s + 2)
        config_path.unlink(missing_ok=True)
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    match = re.search(r"^run: (.+?) \(", proc.stdout, re.MULTILINE)
    if not match or proc.returncode not in (0, 1):  # exit 1 is a failed harness verdict
        print(f"satlab failed before completing (exit {proc.returncode})", file=sys.stderr)
        return 1
    run_dir = Path(match.group(1))
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    if not (run_dir / "summary.json").exists():
        print(f"no summary.json at {run_dir}", file=sys.stderr)
        return 1

    time.sleep(args.settle_s)
    try:
        samples.append((time.time_ns(), fetch_text(metrics_url)))
    except OSError as error:
        print(f"end snapshot fetch failed: {error}; using last polled sample", file=sys.stderr)

    model = config.get("model")
    parsed = [(wall_ns, parse_prom(text, model)) for wall_ns, text in samples]

    def gen_tokens(snapshot: dict) -> float:
        return snapshot["counters"].get(COUNTER_TOKENS, 0.0)

    baseline_wall, baseline = parsed[0]
    run_start = next(((wall, snap) for wall, snap in parsed if gen_tokens(snap) > gen_tokens(baseline)),
                     (baseline_wall, baseline))
    run_start_wall = run_start[0]
    warmup_s = float(config.get("warmup_s", 0.0))
    if warmup_s > 0:
        candidates = [(wall, snap) for wall, snap in parsed
                      if wall - run_start_wall >= warmup_s * 1e9]
        warmup = candidates[0] if candidates else run_start
    else:
        warmup = (baseline_wall, baseline)
    end_wall, end = parsed[-1]

    summary = json.loads((run_dir / "summary.json").read_text())
    records = summary["records"]

    deltas = {"counters": {}, "success_by_reason": {}, "histograms": {}}
    deltas["counters"][COUNTER_TOKENS] = gen_tokens(end) - gen_tokens(warmup[1])
    for reason in set(warmup[1]["success_by_reason"]) | set(end["success_by_reason"]):
        deltas["success_by_reason"][reason] = (end["success_by_reason"].get(reason, 0.0)
                                               - warmup[1]["success_by_reason"].get(reason, 0.0))
    for key, name in HISTOGRAMS.items():
        buckets, count, total = delta_histogram(warmup[1]["histograms"].get(name),
                                                end["histograms"].get(name))
        if buckets is not None:
            deltas["histograms"][key] = {
                "name": name, "buckets": [[le, cum] for le, cum in buckets],
                "count": count, "sum": total,
            }

    rows: list[dict] = []
    statuses: list[str] = []

    def add(row: str, client, server, band: str, verdict: str, note: str = "") -> None:
        rows.append({"row": row, "client": client, "server": server,
                     "band": band, "verdict": verdict, "note": note})
        statuses.append(verdict)

    def ms(ns: int | float) -> str:
        return f"{ns / 1e6:.2f} ms"

    # TPOT: identical formula on both sides (decode_time / (gen_tokens - 1)),
    # so the network offset cancels in the subtraction. vLLM's default
    # buckets are coarse; a quantile PASS means the client value falls inside
    # the bucket containing the server quantile, expanded by the band.
    client_means, _ = client_latency_means(run_dir)
    tpot = deltas["histograms"].get("tpot")
    tpot_ns = records["successful_tpot_ns"]
    if tpot is not None and tpot["count"] > 0:
        for q, band in TPOT_BANDS.items():
            qval = float(q[1:]) / 100
            client_ms = tpot_ns[q] / 1e6
            interp_s = hist_quantile(tpot["buckets"], tpot["count"], qval)
            lo_s, hi_s = quantile_interval(tpot["buckets"], tpot["count"], qval)
            if interp_s is None:
                add(f"TPOT {q}", ms(tpot_ns[q]), "missing", f"±{band:.0%}",
                    "WARN", "server histogram has no samples at this quantile")
                continue
            lo_ms, hi_ms = lo_s * 1000, hi_s * 1000
            inside = lo_ms * (1 - band) <= client_ms and (
                hi_s == float("inf") or client_ms <= hi_ms * (1 + band))
            note = ("" if inside else
                    f"server {q} lies in bucket [{lo_ms:.1f}–{hi_ms:.1f}] ms")
            add(f"TPOT {q}", ms(tpot_ns[q]),
                f"{interp_s * 1000:.2f} ms", f"±{band:.0%}",
                "PASS" if inside else "FAIL", "" if inside else note)
        server_tpot_mean = tpot["sum"] / tpot["count"]
        client_tpot_mean = client_means["tpot"]
        if client_tpot_mean is not None:
            rel = abs(client_tpot_mean - server_tpot_mean) / server_tpot_mean
            ok = rel <= 0.05
            add("TPOT mean", f"{client_tpot_mean * 1000:.2f} ms",
                f"{server_tpot_mean * 1000:.2f} ms (n={tpot['count']:g})", "±5%",
                "PASS" if ok else "FAIL",
                "" if ok else f"relative diff {rel:.1%} outside band")
    else:
        add("TPOT p50/p95/p99", ms(tpot_ns["p50"]), "missing", "±5%..±20%",
            "WARN", "request_time_per_output_token_seconds absent from /metrics")

    # TTFT/E2E: client = server + roughly one RTT + client scheduling. The
    # offset is constant, not proportional to load. The decisive check uses
    # exact means (histogram sum/count); the p50 rows keep the plan's
    # quantile shape but degrade to WARN when the server bucket is wider
    # than the 15 ms band.
    offset_note = "offset must stay flat across rates"
    for key in ("ttft", "e2e"):
        hist = deltas["histograms"].get(key)
        ns = records[f"successful_{key}_ns"]
        label = key.upper()
        if hist is not None and hist["count"] > 0:
            server_mean_ms = hist["sum"] / hist["count"] * 1000
            client_mean_ms = client_means[key] * 1000
            offset_ms = client_mean_ms - server_mean_ms
            ok = 0.0 <= offset_ms <= TTFT_E2E_MAX_OFFSET_MS
            add(f"{label} mean offset", f"{client_mean_ms:.2f} ms",
                f"{server_mean_ms:.2f} ms (n={hist['count']:g})",
                f"0..{TTFT_E2E_MAX_OFFSET_MS:g} ms", "PASS" if ok else "FAIL",
                offset_note if ok else f"offset {offset_ms:.2f} ms outside band")
            interp_s = hist_quantile(hist["buckets"], hist["count"], 0.5)
            lo_s, hi_s = quantile_interval(hist["buckets"], hist["count"], 0.5)
            client_ms = ns["p50"] / 1e6
            lo_ms, hi_ms = lo_s * 1000, hi_s * 1000
            low_offset = client_ms - hi_ms  # client - server upper bound
            high_offset = client_ms - lo_ms  # client - server lower bound
            provable = hi_s != float("inf")
            if low_offset >= 0.0 and high_offset <= TTFT_E2E_MAX_OFFSET_MS:
                verdict, note = "PASS", offset_note
            elif provable and (high_offset < 0.0 or low_offset > TTFT_E2E_MAX_OFFSET_MS):
                verdict = "FAIL"
                note = (f"offset {low_offset:.2f}..{high_offset:.2f} ms outside band")
            else:
                verdict = "WARN"
                note = (f"server p50 bucket [{lo_ms:.1f}–{hi_ms:.1f}] ms is wider "
                        f"than the 15 ms band; see the mean offset row")
            add(f"{label} p50 offset", ms(ns["p50"]),
                f"{interp_s * 1000:.2f} ms", f"0..{TTFT_E2E_MAX_OFFSET_MS:g} ms",
                verdict, note)
        else:
            add(f"{label} p50 offset", ms(ns["p50"]), "missing",
                f"0..{TTFT_E2E_MAX_OFFSET_MS:g} ms", "WARN",
                f"{HISTOGRAMS[key]} absent from /metrics")

    # Token counters: same requests, same 16-token outputs, few percent apart.
    server_tokens = deltas["counters"][COUNTER_TOKENS]
    client_tokens = records["output_tokens_in_window"]
    token_ok = server_tokens > 0 and abs(server_tokens - client_tokens) / server_tokens <= 0.05
    add("Output tokens", str(client_tokens),
        f"{server_tokens:g}", "≤ 5% relative diff",
        "PASS" if token_ok else "FAIL",
        "" if token_ok else f"server {server_tokens:g} vs client {client_tokens}")

    # Success counter: every satlab request sets ignore_eos, so all successes
    # finish with reason "length". Anything else is a workload mismatch.
    server_ok = deltas["success_by_reason"].get("length", 0.0)
    client_ok = records["in_window_ok"]
    other_reasons = {reason: count for reason, count in deltas["success_by_reason"].items()
                     if reason != "length" and count > 0}
    success_ok = server_ok > 0 and abs(server_ok - client_ok) / server_ok <= 0.10
    add("Success count", str(client_ok), f"{server_ok:g}", "≤ 10% relative diff",
        "PASS" if success_ok else "FAIL",
        "" if success_ok else f"server {server_ok:g} vs client {client_ok}")
    if other_reasons:
        add("Finish reasons", "", ", ".join(f"{reason}={count:g}" for reason, count in
                                            other_reasons.items()), "all = length",
            "WARN", "non-length finish reasons appeared; check the workload before "
                    "trusting this run's latency numbers")

    # Population: histogram _count vs client ok count. Warmup-snapshot timing
    # makes this soft, so it warns and never fails.
    ttft_hist = deltas["histograms"].get("ttft")
    if ttft_hist is not None:
        population_note = "" if client_ok else "no successful client requests"
        if client_ok:
            server_count = ttft_hist["count"]
            rel = abs(server_count - client_ok) / client_ok
            population_note = ("" if rel <= 0.10 else
                               f"relative diff {rel:.1%}; warmup snapshot timing")
        add("Population (TTFT count)", str(client_ok),
            f"{ttft_hist['count']:g}", "≤ 10% relative diff", "WARN", population_note)

    # Weak checks, printed but never part of the verdict.
    add("ITL", "—", "—", "—", "WARN", "excluded: chunk granularity, TCP coalescing")
    add("Outstanding vs waiting", "—", "—", "—", "WARN",
        "excluded: client includes network transit")

    overall = "FAIL" if "FAIL" in statuses else "PASS"
    (run_dir / "server_metrics_delta.json").write_text(json.dumps(
        {"baseline_wall_unix_ns": baseline_wall, "run_start_wall_unix_ns": run_start_wall,
         "warmup_wall_unix_ns": warmup[0], "end_wall_unix_ns": end_wall, **deltas},
        indent=2) + "\n")
    validation = {
        "config": {"name": config.get("name"), "target_url": target,
                   "rate": config.get("rate"), "duration_s": config.get("duration_s"),
                   "warmup_s": warmup_s},
        "run_dir": str(run_dir), "overall": overall, "rows": rows,
        "excluded": ["itl", "outstanding_vs_waiting"],
    }
    (run_dir / "server_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    print(f"\nserver-metric validation: {overall}\n")
    widths = [max(len(row[key]) for row in rows + [{"row": "Row", "client": "Client",
                                                    "server": "Server", "band": "Band",
                                                    "verdict": "Verdict", "note": ""}])
              for key in ("row", "client", "server", "band", "verdict", "note")]
    header = ["Row", "Client", "Server", "Band", "Verdict", "Note"]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        line = [row["row"], row["client"], row["server"], row["band"], row["verdict"],
                row["note"]]
        print("  ".join(value.ljust(w) for value, w in zip(line, widths)))
    print(f"\nwrote {run_dir / 'server_validation.json'} and "
          f"{run_dir / 'server_metrics_delta.json'}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
