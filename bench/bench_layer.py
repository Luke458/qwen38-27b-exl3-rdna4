#!/usr/bin/env python3
"""Projection microbenchmark through the production exl3_gemm dispatcher.

Measures the M=1 EXL3 projection pipeline (input Hadamard + dot + output Hadamard,
i.e. everything the decode step pays) with GPU events on the executing stream,
for every eligible v1 shape plus one fallback shape. Emits raw per-iteration
times (retained), median/mean, and a selected-shape projection proxy using
the manifest's static per-step invocation counts. This is not model TPOT.

Usage:
  python bench/bench_layer.py --out bench_layer_metrics.json \
      [--model DIR] [--iters 100] [--warmup 25] [--shapes 5120x5120x3,...]

Runs under whichever extension build is active (baseline .venv or a candidate
PYTHONPATH) -- pairing/order alternation across builds is done by the caller
(tools/run_paired.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

DEFAULT_MODEL = os.path.expanduser("~/models/qwen3.8-27b-exl3-11.5gb")


def load_tensor(path, name):
    from safetensors import safe_open
    with safe_open(path, framework="np") as f:
        return f.get_tensor(name)


def probe_bandwidth_gb_s(iters: int = 30, mb: int = 512) -> float:
    """Sustainable VRAM copy bandwidth (read+write counted), for the roofline."""
    n = mb * 1024 * 1024 // 2
    src = torch.randn(n, dtype=torch.half, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        e0.record()
        dst.copy_(src)
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    us = float(np.median(ts)) * 1000.0
    return (2.0 * n * 2) / us / 1000.0   # GB/s (bytes read + written)


def bench_call(A, B, C, suh_t, A_had, svh_t, iters, warmup):
    for _ in range(warmup):
        ext.exl3_gemm(A, B, C, suh_t, A_had, svh_t, -1, False, True, 0)
    torch.cuda.synchronize()
    times = []
    # GPU events on the executing stream; one event pair per iteration (the
    # op is ~10-100 us; batching would hide per-call variance)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        ext.exl3_gemm(A, B, C, suh_t, A_had, svh_t, -1, False, True, 0)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--manifest", default=os.path.join(ROOT, "configs/shape_manifest.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--shapes", default=None, help="KxNxBits,... subset")
    ap.add_argument("--rotating", type=int, default=1,
                    help="trellis copies rotated across iterations (cache-correct streaming; "
                         "set so rotating*weight_bytes >> 64 MiB Infinity Cache)")
    ap.add_argument("--peak-gbps", type=float, default=None,
                    help="override measured VRAM bandwidth for the roofline")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest if os.path.isabs(args.manifest) else os.path.join(ROOT, args.manifest)))
    want = set(args.shapes.split(",")) if args.shapes else None

    # Model text projections only: vision and MTP are not used by this direct
    # text decode workload. Group by shape and pick one real representative.
    groups = {}
    for m in manifest["modules"]:
        if not m["eligible_v1"] or not m["name"].startswith("model.language_model.layers."):
            continue
        key = f"{m['K_in']}x{m['N']}x{m['bits']}"
        g = groups.setdefault(key, {"modules": [], "K_in": m["K_in"], "N": m["N"], "bits": m["bits"]})
        g["modules"].append(m["name"])

    # One modest non-eligible text shape to time the fallback path too.
    for m in manifest["modules"]:
        key = f"{m['K_in']}x{m['N']}x{m['bits']}"
        if (not m["eligible_v1"] and m["name"].startswith("model.language_model.layers.")
                and m["N"] <= 10240 and (want is None or key in want)):
            groups.setdefault(key, {"modules": [m["name"]], "K_in": m["K_in"], "N": m["N"], "bits": m["bits"], "fallback": True})
            break

    index = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))
    results = {}
    rng = np.random.default_rng(20260923)
    peak_gb_s = args.peak_gbps or probe_bandwidth_gb_s()
    print(f"sustainable VRAM bandwidth (copy probe): {peak_gb_s:.1f} GB/s")

    for key, g in sorted(groups.items()):
        if want is not None and key not in want:
            continue
        k_in, n, bits = g["K_in"], g["N"], g["bits"]
        name = g["modules"][0]
        tpath = os.path.join(args.model, index["weight_map"][f"{name}.trellis"])
        trellis = load_tensor(tpath, f"{name}.trellis")
        suh = load_tensor(tpath, f"{name}.suh")
        svh = load_tensor(tpath, f"{name}.svh")
        assert trellis.shape == (k_in // 16, n // 16, 16 * bits), (name, trellis.shape)

        a = (rng.standard_normal(k_in) * 0.2).astype(np.float16)
        A = torch.from_numpy(a.copy()).unsqueeze(0).cuda()
        B_list = [torch.from_numpy(np.ascontiguousarray(trellis)).to(torch.int16).cuda()
                  for _ in range(max(1, args.rotating))]
        B = B_list[0]
        suh_t = torch.from_numpy(np.ascontiguousarray(suh)).cuda()
        svh_t = torch.from_numpy(np.ascontiguousarray(svh)).cuda()
        A_had = torch.empty(1, k_in, dtype=torch.half, device="cuda")
        C = torch.empty(1, n, dtype=torch.half, device="cuda")

        if args.rotating > 1:
            # cache-correct streaming: rotate trellis copies so the working set
            # exceeds the 64 MiB Infinity Cache; rank candidates on this mode,
            # not on warm-cache latencies (real decode streams with no reuse)
            for _ in range(args.warmup):
                ext.exl3_gemm(A, B_list[0], C, suh_t, A_had, svh_t, -1, False, True, 0)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []
            for i in range(args.iters):
                b = B_list[i % len(B_list)]
                start.record()
                ext.exl3_gemm(A, b, C, suh_t, A_had, svh_t, -1, False, True, 0)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
        else:
            times = bench_call(A, B, C, suh_t, A_had, svh_t, args.iters, args.warmup)
        arr = np.array(times)
        invocations = sum(m["invocations_per_decode_step"] for m in manifest["modules"]
                          if m["name"].startswith("model.language_model.layers.")
                          and f"{m['K_in']}x{m['N']}x{m['bits']}" == key)
        weight_bytes = k_in * n * bits // 8
        med_us = float(np.median(arr)) * 1000.0
        roof_us = weight_bytes / (peak_gb_s * 1000.0)  # bytes / (GB/s) -> us
        results[key] = {
            "K_in": k_in, "N": n, "bits": bits, "eligible_v1": not g.get("fallback", False),
            "representative": name, "invocations_per_decode_step": invocations,
            "raw_ms": times,
            "median_ms": float(np.median(arr)),
            "mean_ms": float(arr.mean()),
            "min_ms": float(arr.min()),
            "weight_bytes": weight_bytes,
            "rotating_copies": max(1, args.rotating),
            "peak_gb_s": peak_gb_s,
            "roofline_us": roof_us,
            "efficiency": roof_us / med_us if med_us > 0 else None,
            "effective_gb_s": weight_bytes / med_us / 1000.0 if med_us > 0 else None,
        }
        print(f"{key:22s} median {med_us:8.2f} us  eff {results[key]['efficiency']:.2f} "
              f"({results[key]['effective_gb_s']:.0f} GB/s of {peak_gb_s:.0f})  ({invocations} calls/decode-step)")

    weighted_us = sum(v["median_ms"] * v["invocations_per_decode_step"] * 1000
                      for v in results.values())
    metrics = {
        "shapes": results,
        "selected_shape_projection_us_proxy": weighted_us,
        "proxy_scope": "selected text-layer projection shapes only; not runtime TPOT or all-model coverage",
        "int8_mode": os.environ.get("EXL3_ROCM_GFX1201_INT8", "0"),
        "iters": args.iters,
        "warmup": args.warmup,
        "ext_path": ext.__file__,
        "torch": torch.__version__,
        "ts": time.time(),
    }
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=1)
    os.replace(tmp, out)
    print(f"selected-shape projection proxy: {weighted_us:.1f} us/decode-step -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
