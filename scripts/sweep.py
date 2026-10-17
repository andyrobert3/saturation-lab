#!/usr/bin/env python3
"""Run an open-loop rate sweep against a running server and analyze it.

The server (vLLM or the bundled mock) must already be running. This script never starts it.

  sweep.py sweep   --server-notes "<launch flags, cache>"   9 rates x 3 repeats, 60 s each
  sweep.py confirm --rate R --server-notes "..."            3 repeats of 300 s at one rate
  sweep.py analyze                                          rebuild charts from saved runs

The three passes run forward, reversed, and shuffled so drift cannot look like a knee.
The mock: `target/release/mock 4 0.05 8123`, then add
`--base-config config.json --name mock-open --runs-dir runs/mock-sweep --expected-capacity 80
--max-in-flight 4096`.
"""

import argparse
import csv
import json
import math
import random
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
ANALYZE = REPO / "analysis" / "analyze.py"
RATES = (6.5, 9.0, 11.0, 12.0, 12.5, 13.0, 13.5, 14.5, 16.5)
METRICS = {
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_requests_running": "running",
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
}
METRIC_LINE = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)")


def schedule(rates: tuple[float, ...], repeats_rate: float | None = None) -> list[tuple[int, float]]:
    if repeats_rate is not None:  # confirm: one rate, three repeats
        return [(rep, repeats_rate) for rep in range(1, 4)]
    shuffled = list(rates)
    random.Random(20261011).shuffle(shuffled)
    return ([(1, rate) for rate in rates]
            + [(2, rate) for rate in reversed(rates)]
            + [(3, rate) for rate in shuffled])


def run_name(prefix: str, phase: str, rep: int, rate: float) -> str:
    return f"{prefix}-confirm-{rate:g}-r{rep}" if phase == "confirm" else f"{prefix}-{rate:g}-r{rep}"


def make_config(base: dict, name: str, phase: str, rate: float, server_notes: str,
                max_in_flight: int) -> dict:
    cfg = dict(base)
    cfg.update({
        "name": name,
        "mode": "open",
        "rate": rate,
        "duration_s": 300.0 if phase == "confirm" else 60.0,
        "warmup_s": 10.0,
        "max_in_flight": max_in_flight,
        "request_timeout_s": 120.0,
        "target_notes": server_notes,
    })
    return cfg


def get_text(url: str) -> str:
    with urlopen(url, timeout=2) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_metrics(body: str) -> dict[str, float | None]:
    values: dict[str, list[float]] = {key: [] for key in METRICS.values()}
    for line in body.splitlines():
        match = METRIC_LINE.match(line)
        if not match or match.group(1) not in METRICS:
            continue
        try:
            values[METRICS[match.group(1)]].append(float(match.group(2)))
        except ValueError:
            continue
    return {
        key: (max(found) if key == "kv_cache_usage" else sum(found)) if found else None
        for key, found in values.items()
    }


def sample_metrics(url: str, stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        timestamp = time.time_ns()
        try:
            samples.append({"wall_unix_ns": timestamp, **parse_metrics(get_text(url))})
        except (OSError, ValueError):
            pass
        stop.wait(1.0)


def existing_runs(runs_dir: Path) -> dict[str, Path]:
    found = {}
    for path in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not path.is_dir() or not (path / "config.json").exists():
            continue
        name = json.loads((path / "config.json").read_text())["config"]["name"]
        if name in found:
            raise RuntimeError(f"duplicate run name {name}: {found[name]} and {path}")
        found[name] = path
    return found


def run_one(cfg: dict, runs_dir: Path, metrics_url: str | None) -> Path:
    samples: list[dict] = []
    stop = threading.Event()
    worker = None
    if metrics_url:
        worker = threading.Thread(target=sample_metrics,
                                  args=(metrics_url, stop, samples), daemon=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(cfg, handle)
        config_path = Path(handle.name)
    try:
        if worker:
            worker.start()
        proc = subprocess.run(
            [str(SATLAB), "--config", str(config_path), "--runs-dir", str(runs_dir)],
            capture_output=True, text=True, cwd=REPO,
        )
    finally:
        stop.set()
        if worker:
            worker.join(timeout=3)
        config_path.unlink(missing_ok=True)
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    match = re.search(r"^run: (.+?) \(", proc.stdout, re.MULTILINE)
    if not match or proc.returncode not in (0, 1):  # exit 1 is a failed harness verdict
        raise RuntimeError(f"satlab failed before completing {cfg['name']} (exit {proc.returncode})")
    run_dir = Path(match.group(1))
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    if not (run_dir / "summary.json").exists():
        raise RuntimeError(f"no summary.json for {cfg['name']} at {run_dir}")
    if samples:
        with (run_dir / "server_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["wall_unix_ns", *METRICS.values()])
            writer.writeheader()
            writer.writerows(samples)
    return run_dir


def analyze(runs_dir: Path, out_dir: Path, closed_dir: Path | None, phase: str,
            expected_capacity: float | None) -> None:
    run_dirs = []
    for path in sorted(runs_dir.iterdir()):
        if path.is_dir() and (path / "summary.json").exists():
            name = json.loads((path / "config.json").read_text())["config"]["name"]
            if ("-confirm-" in name) == (phase == "confirm"):
                run_dirs.append(path)
    if not run_dirs:
        raise RuntimeError(f"no completed {phase} runs in {runs_dir}")
    if closed_dir:
        run_dirs += [p for p in sorted(closed_dir.iterdir()) if (p / "summary.json").exists()]
    cmd = [sys.executable, str(ANALYZE), *map(str, run_dirs), "--out", str(out_dir)]
    if expected_capacity:
        cmd += ["--expected-capacity", str(expected_capacity)]
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("sweep", "confirm", "analyze"))
    parser.add_argument("--rate", type=float, help="rate to confirm with three 5-minute runs")
    parser.add_argument("--rates", help="comma-separated open rates; default is the nine-rate sweep")
    parser.add_argument("--base-config", type=Path, default=REPO / "smoke.json")
    parser.add_argument("--name", default="vllm-metal-open", help="run name prefix")
    parser.add_argument("--max-in-flight", type=int, default=512,
                        help="client cap; a run that hits it is clipped and fails the verdict")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--closed-dir", type=Path,
                        help="closed-loop runs to add to the report, e.g. runs/vllm-metal-closed-sweep")
    parser.add_argument("--expected-capacity", type=float, help="known capacity, req/s (the mock)")
    parser.add_argument("--server-notes", help="exact server launch flags and cache setting")
    args = parser.parse_args()
    try:
        rates = tuple(sorted({float(v) for v in args.rates.split(",")})) if args.rates else RATES
    except ValueError:
        parser.error("--rates must be comma-separated numbers")
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        parser.error("--rates must contain positive finite numbers")
    if args.action == "confirm" and (
            args.rate is None or not math.isfinite(args.rate) or args.rate <= 0):
        parser.error("confirm needs --rate > 0")
    if args.action in ("sweep", "confirm") and not args.server_notes:
        parser.error("--server-notes must record launch flags and cache setting")

    phase = "confirm" if args.action == "confirm" else "sweep"
    runs_dir = (args.runs_dir or REPO / "runs" / (
        "vllm-metal-open-confirm" if phase == "confirm" else "vllm-metal-open-sweep")).resolve()
    out_dir = (args.out or runs_dir.parent / (
        f"{runs_dir.name}-{phase}-report" if phase == "confirm" else f"{runs_dir.name}-report")).resolve()
    closed_dir = args.closed_dir.resolve() if args.closed_dir and phase == "sweep" else None
    if args.action == "analyze":
        analyze(runs_dir, out_dir, closed_dir, phase, args.expected_capacity)
        return 0
    if not SATLAB.is_file():
        parser.error(f"build the release binary first: cargo build --release ({SATLAB})")

    base = json.loads(args.base_config.read_text())
    target = base["target_url"].rstrip("/")
    try:
        print(f"server version: {get_text(target + '/version').strip()}")
    except OSError as error:
        parser.error(f"cannot reach {target}/version: {error}")
    metrics_url = target + "/metrics"
    try:
        if parse_metrics(get_text(metrics_url))["waiting"] is None:
            metrics_url = None
    except OSError:
        metrics_url = None
    print("server waiting metric: " + ("available" if metrics_url else "unavailable"))

    runs_dir.mkdir(parents=True, exist_ok=True)
    found = existing_runs(runs_dir)
    for rep, rate in schedule(rates, args.rate if phase == "confirm" else None):
        cfg = make_config(base, run_name(args.name, phase, rep, rate), phase, rate,
                          args.server_notes, args.max_in_flight)
        name = cfg["name"]
        if name in found:
            saved = json.loads((found[name] / "config.json").read_text())["config"]
            if not (found[name] / "summary.json").exists() or any(
                    saved.get(key) != value for key, value in cfg.items()):
                raise RuntimeError(f"{found[name]} is incomplete or has different settings; "
                                   "use a new --runs-dir")
            print(f"keeping existing run: {found[name]}")
        else:
            print(f"starting {name}: {rate:g} req/s", flush=True)
            found[name] = run_one(cfg, runs_dir, metrics_url)
    analyze(runs_dir, out_dir, closed_dir, phase, args.expected_capacity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
