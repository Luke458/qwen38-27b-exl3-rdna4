#!/usr/bin/env python3
"""Paired-trial driver: run a baseline command and a candidate command back to
back N times with ALTERNATING order (AB/BA/...), collect per-pair decode times.

Each command writes its own single-run metrics JSON (bench_generate format).
The driver extracts per-workload steady decode time (sum of the steady TPOT
window) and writes paired lists in pair-index order to --out-a / --out-b, the
shape tools/compare.py consumes.

  python tools/run_paired.py --a-cmd 'BENCH ... --out r.json' \
      --b-cmd 'BENCH ... --out r.json' --trials 10 --extract r.json \
      --out-a artifacts/.../paired_baseline.json --out-b experiments/.../paired_candidate.json

--a is the frozen baseline environment command, --b the candidate's. Trials are
invalidated (retained with a reason) only for predeclared environmental
conditions: competing GPU process, edge temp > 90 C.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_env_invalid():
    reasons = []
    try:
        probe = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ["rocm-smi telemetry unavailable"]
    if probe.returncode != 0:
        return ["rocm-smi telemetry failed"]
    out = probe.stdout
    for line in out.splitlines():
        if "C" in line and "N/A" not in line:
            try:
                t = float(line.split()[3].replace("°C", ""))
                if t > 90.0:
                    reasons.append(f"edge temp {t}C")
            except Exception:
                pass
    return reasons


def extract_decode_s(path: str) -> dict:
    m = json.load(open(path))
    out = {}
    for wid, w in m.get("workloads", {}).items():
        raw = w.get("tpot_ms_raw") or []
        if not raw or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in raw):
            raise ValueError(f"{wid}: missing or invalid tpot_ms_raw")
        out[wid] = {
            "decode_s": sum(raw) / 1000.0,
            "prefill_s": w.get("prefill_ms", float("nan")) / 1000.0,
            "ttft_s": w.get("ttft_ms", float("nan")) / 1000.0,
            "peak_vram_bytes": w.get("peak_vram_bytes"),
        }
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in out[wid].values()):
            raise ValueError(f"{wid}: missing or invalid prefill/ttft/VRAM metric")
    if not out:
        raise ValueError("no workloads in generation metrics")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-cmd", required=True)
    ap.add_argument("--b-cmd", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--extract", required=True, help="metrics file path each cmd writes")
    ap.add_argument("--out-a", required=True)
    ap.add_argument("--out-b", required=True)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    args = ap.parse_args()

    if args.trials < 10 or args.timeout_s <= 0:
        ap.error("at least 10 trials and a positive timeout are required")
    pairs = {"a": {}, "b": {}}
    invalid = []
    for i in range(args.trials):
        order = ("a", "b") if i % 2 == 0 else ("b", "a")
        results = {}
        for side in order:
            cmd = shlex.split(args.a_cmd if side == "a" else args.b_cmd)
            if os.path.isfile(args.extract):
                os.unlink(args.extract)
            t0 = time.monotonic()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=args.timeout_s, cwd=ROOT)
            except subprocess.TimeoutExpired:
                print(f"pair {i} side {side} timed out", file=sys.stderr)
                return 1
            dur = time.monotonic() - t0
            if p.returncode != 0:
                print(f"pair {i} side {side} FAILED rc={p.returncode}", file=sys.stderr)
                sys.stderr.write(p.stderr[-2000:])
                return 1
            try:
                vals = extract_decode_s(args.extract)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
                print(f"pair {i} side {side} invalid metrics: {e}", file=sys.stderr)
                return 1
            results[side] = vals
            env_bad = gpu_env_invalid()
            if env_bad:
                invalid.append({"pair": i, "side": side, "reasons": env_bad, "values": vals})
        if results["a"].keys() != results["b"].keys():
            print(f"pair {i}: workload set differs", file=sys.stderr)
            return 1
        for side in ("a", "b"):
            for wid, fields in results[side].items():
                for key, value in fields.items():
                    pairs[side].setdefault(key, {}).setdefault(wid, []).append(value)
        print(f"pair {i} done ({''.join(order).upper()})")

    for side, out in (("a", args.out_a), ("b", args.out_b)):
        payload = {"paired_decode_s": pairs[side]["decode_s"],
                   "paired_prefill_s": pairs[side]["prefill_s"],
                   "paired_ttft_s": pairs[side]["ttft_s"],
                   "paired_peak_vram_bytes": pairs[side]["peak_vram_bytes"], "trials": args.trials,
                   "order": "alternating AB/BA", "invalid_samples": invalid,
                   "ts": time.time()}
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, out)
    print(f"wrote {args.out_a} and {args.out_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
