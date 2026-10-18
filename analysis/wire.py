#!/usr/bin/env python3
"""Wire check (prediction 8a.8): request send times seen by tcpdump against satlab's plan.

  wire.py capture.pcap run_dir

The capture is `tcpdump -i lo0 -s 200 -w capture.pcap 'tcp dst port 8000'` (link type NULL,
IPv4). The first segment of each request starts with "POST"; its timestamp is the wire send time.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyze import read_requests


def post_times(path: Path) -> list[float]:
    data = path.read_bytes()
    magic = data[:4]
    if magic not in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        raise ValueError("expected a little-endian pcap file")
    frac = 1e-9 if magic == b"\x4d\x3c\xb2\xa1" else 1e-6
    if struct.unpack_from("<I", data, 20)[0] != 0:
        raise ValueError("expected link type NULL (tcpdump -i lo0)")
    times, off = [], 24
    while off + 16 <= len(data):
        sec, sub, incl, _ = struct.unpack_from("<IIII", data, off)
        pkt = data[off + 16:off + 16 + incl]
        off += 16 + incl
        ip = pkt[4:]  # 4-byte loopback family header
        if len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 6:
            continue
        tcp = ip[(ip[0] & 0xF) * 4:]
        if len(tcp) >= 20 and tcp[(tcp[12] >> 4) * 4:].startswith(b"POST"):
            times.append(sec + sub * frac)
    return times


def main(argv):
    pcap, run_dir = map(Path, argv)
    doc = json.loads((run_dir / "config.json").read_text())
    anchor_s = doc["provenance"]["anchor_wall_unix_ns"] / 1e9
    plan_gap_ms = 1e3 / doc["config"]["rate"]
    sent = sorted((r for r in read_requests(run_dir / "records.csv") if r.status != "clipped"),
                  key=lambda r: r.intended)
    wire = np.array(post_times(pcap))
    if len(wire) != len(sent):
        raise SystemExit(f"{len(wire)} POST segments on the wire, {len(sent)} requests sent")
    gaps_ms = np.diff(wire) * 1e3
    within = np.mean(np.abs(gaps_ms - plan_gap_ms) <= 2.0)
    lag_ms = (wire - anchor_s - np.array([r.intended for r in sent]) / 1e9) * 1e3
    rec_ms = (wire - anchor_s - np.array([r.actual for r in sent]) / 1e9) * 1e3
    q = lambda a: " / ".join(f"{np.percentile(a, p):.2f}" for p in (1, 50, 99))
    print(f"requests: {len(wire)} on the wire, {len(sent)} sent")
    print(f"gap ms p1/p50/p99: {q(gaps_ms)} (plan {plan_gap_ms:.2f})")
    print(f"gaps within 2 ms of plan: {within:.2%}")
    print(f"wire minus plan ms p1/p50/p99: {q(lag_ms)}")
    print(f"wire minus satlab send time ms p1/p50/p99: {q(rec_ms)}")
    print("8a.8", "HELD" if within >= 0.99 else "NOT HELD")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
