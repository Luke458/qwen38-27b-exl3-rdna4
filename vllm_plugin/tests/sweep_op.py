#!/usr/bin/env python3
"""Fault-isolation sweep of torch.ops.exl3rocm.linear over every distinct (k, n, K) in a
checkpoint and a list of row counts. Prints (flushed) before each call so a GPU fault
identifies the case. Random trellis words (every bit pattern is a valid code).

  python sweep_op.py /model [M ...]
"""
import sys

import torch

sys.path.insert(0, "/plugin")
from exl3rocm.plugin import _read_checkpoint_layout  # noqa: E402
from exl3rocm import ops  # noqa: E402,F401

model = sys.argv[1]
Ms = [int(m) for m in sys.argv[2:]] or [1, 2, 4, 8, 16, 512, 2048]
mods, _, _ = _read_checkpoint_layout(model)
shapes = sorted({v for v in mods.values()}, key=lambda s: (s[1] * s[2]))
print(f"{len(shapes)} distinct (K, k, n) shapes: {shapes}", flush=True)
g = torch.Generator(device="cuda").manual_seed(1)
for K, k, n in shapes:
    tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                       generator=g).to(torch.int16)
    suh = torch.ones(k, device="cuda", dtype=torch.float16)
    svh = torch.full((n,), 0.01, device="cuda", dtype=torch.float16)
    ops.reserve_weight_buffer("cuda:0", k * min(n, ops.RECON_SLICE_N))
    for M in Ms:
        print(f"K={K} k={k} n={n} M={M} ...", end=" ", flush=True)
        x = torch.randn((M, k), device="cuda", dtype=torch.float16)
        y = torch.ops.exl3rocm.linear(x, tr, suh, svh, K, False, True)
        torch.cuda.synchronize()
        print(f"ok finite={bool(torch.isfinite(y).all())}", flush=True)
print("SWEEP PASS", flush=True)
