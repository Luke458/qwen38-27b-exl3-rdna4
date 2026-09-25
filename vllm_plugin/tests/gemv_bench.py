#!/usr/bin/env python3
"""M=1 EXL3 GEMV scoreboard on the real Qwen3.8-27B shapes (DRAM-resident weights).

Each case rotates through enough trellis copies (>= 512 MiB) that the 64 MiB Infinity
Cache cannot hold them, and times a burst of back-to-back calls with events. Reports
per-call µs, weight-stream GB/s and G weights/s, plus a per-token total weighted by how
often each shape occurs in one decode step.

  python gemv_bench.py [--json out.json]        (exllamav3_ext on PYTHONPATH)
"""
import argparse
import json

import torch

import exllamav3_ext as E

# (name, k, n, K, calls per decode token)  -- from the checkpoint layout / 0017 trace
CASES = [
    ("gate|up K3", 5120, 17408, 3, 70), ("gate|up K2", 5120, 17408, 2, 56), ("gate|up K1", 5120, 17408, 1, 2),
    ("down K3", 17408, 5120, 3, 64),
    ("gdn qkv K4", 5120, 10240, 4, 12), ("gdn qkv K2", 5120, 10240, 2, 35),
    ("gdn z K4", 5120, 6144, 4, 48), ("gdn out K4", 6144, 5120, 4, 48),
    ("attn q K2", 5120, 12288, 2, 11), ("attn q K4", 5120, 12288, 4, 4),
    ("attn kv K5", 5120, 1024, 5, 18), ("attn kv K6", 5120, 1024, 6, 8), ("attn kv K4", 5120, 1024, 4, 6),
    ("attn o K3", 6144, 5120, 3, 4), ("attn o K5", 6144, 5120, 5, 4),
    ("lm_head K4", 5120, 248320, 4, 1),
]
PAIR_CASES = [("gate+up mgemm K3", 5120, 17408, 3, 35), ("gate+up mgemm K2", 5120, 17408, 2, 28)]


def copies_for(nbytes):
    return max(2, (512 << 20) // nbytes + 1)


def trellis(k, n, K, g):
    return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                         generator=g).to(torch.int16)


def time_calls(fn, ncopies, iters=40):
    for i in range(ncopies):
        fn(i)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(iters):
        fn(i % ncopies)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--m", type=int, default=1, help="rows (M>1 exercises the multi-row GEMM path)")
    ap.add_argument("--no-pairs", action="store_true")
    args = ap.parse_args()
    M = args.m
    g = torch.Generator(device="cuda").manual_seed(3)
    res = []
    per_token_us = 0.0
    for name, k, n, K, calls in CASES:
        nb = k * n * K // 8
        nc = copies_for(nb)
        ts = [trellis(k, n, K, g) for _ in range(nc)]
        suh = torch.ones(k, device="cuda", dtype=torch.float16)
        svh = torch.full((n,), 0.01, device="cuda", dtype=torch.float16)
        x = torch.randn((M, k), device="cuda", dtype=torch.float16)
        xh = torch.empty_like(x)
        y = torch.empty((M, n), device="cuda", dtype=torch.float16)
        us = time_calls(lambda i: E.exl3_gemm(x, ts[i], y, suh, xh, svh, -1, False, True, 0), nc)
        del ts
        r = {"case": name, "us": us, "GBps": nb / us / 1e3, "Gwps": k * n / us / 1e3, "calls": calls}
        res.append(r)
        if not name.startswith("gate|up") or M != 1:
            per_token_us += us * calls
        print(f"{name:18s} {us:8.1f} us  {r['GBps']:6.0f} GB/s  {r['Gwps']:6.0f} Gw/s", flush=True)
    for name, k, n, K, calls in ([] if args.no_pairs or M != 1 else PAIR_CASES):
        nb = 2 * k * n * K // 8
        nc = copies_for(nb)
        pairs = [(trellis(k, n, K, g), trellis(k, n, K, g)) for _ in range(nc)]
        ptr_t = [torch.tensor([a.data_ptr(), b.data_ptr()], dtype=torch.long, device="cuda") for a, b in pairs]
        suh = [torch.ones(k, device="cuda", dtype=torch.float16) for _ in range(2)]
        svh = [torch.full((n,), 0.01, device="cuda", dtype=torch.float16) for _ in range(2)]
        ptr_suh = torch.tensor([s.data_ptr() for s in suh], dtype=torch.long, device="cuda")
        ptr_svh = torch.tensor([s.data_ptr() for s in svh], dtype=torch.long, device="cuda")
        x = torch.randn((1, 1, k), device="cuda", dtype=torch.float16)
        xh = torch.empty((2, 1, k), device="cuda", dtype=torch.float16)
        y = torch.empty((2, 1, n), device="cuda", dtype=torch.float16)
        us = time_calls(lambda i: E.exl3_mgemm(x, ptr_t[i], y, ptr_suh, xh, ptr_svh, None, None, K, -1,
                                              False, True, -1, -1, 0, 1, None, None), nc)
        del pairs
        r = {"case": name, "us": us, "GBps": nb / us / 1e3, "Gwps": 2 * k * n / us / 1e3, "calls": calls}
        res.append(r)
        per_token_us += us * calls
        print(f"{name:18s} {us:8.1f} us  {r['GBps']:6.0f} GB/s  {r['Gwps']:6.0f} Gw/s", flush=True)
    if M == 1:
        # gate|up K1 has no mgemm pair entry; count it as two singles
        per_token_us += [r["us"] for r in res if r["case"] == "gate|up K1"][0] * 2
    print(f"weighted GEMV time per decode step, M={M}: {per_token_us / 1e3:.2f} ms")
    if args.json:
        json.dump({"cases": res, "per_token_ms": per_token_us / 1e3}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
