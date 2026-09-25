#!/usr/bin/env python3
"""exl3_gemv_mr: (1) every row bitwise == the m=1 GEMV of that row, all checkpoint shapes;
(2) DRAM-resident timing at m = 1, 2, 3, 4, 8 vs the m=1 exl3_gemm path.

  python mr_check.py /model
"""
import sys

import torch

import exllamav3_ext as E

sys.path.insert(0, "/plugin")
from exl3rocm.plugin import _read_checkpoint_layout  # noqa: E402

mods, _, _ = _read_checkpoint_layout(sys.argv[1])
shapes = sorted(set(mods.values()), key=lambda s: s[1] * s[2])
g = torch.Generator(device="cuda").manual_seed(11)
bad = 0
for K, k, n in shapes:
    tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                       generator=g).to(torch.int16)
    suh = (torch.randn(k, device="cuda", generator=g).sign() * 1.03).half()
    svh = (torch.rand(n, device="cuda", generator=g) * 0.02 + 0.005).half()
    for m in (2, 3, 4, 8):
        x = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=g)
        y = torch.empty((m, n), device="cuda", dtype=torch.float16)
        E.exl3_gemv_mr(x, tr, y, suh, torch.empty_like(x), svh, K, False, True)
        ref = torch.empty((m, n), device="cuda", dtype=torch.float16)
        for r in range(m):
            xr = x[r:r + 1].contiguous()
            yr = torch.empty((1, n), device="cuda", dtype=torch.float16)
            E.exl3_gemm(xr, tr, yr, suh, torch.empty_like(xr), svh, -1, False, True, 0)
            ref[r] = yr[0]
        same = torch.equal(y.view(torch.int16), ref.view(torch.int16))
        if not same:
            bad += 1
            print(f"MISMATCH K={K} k={k} n={n} m={m}: max abs {(y.float() - ref.float()).abs().max().item():.3g}")
print(f"bitwise check: {'PASS' if bad == 0 else f'{bad} FAIL'} ({len(shapes) * 4} cases)", flush=True)

# timing, DRAM-resident (rotate copies)
CASES = [(5120, 17408, 3), (17408, 5120, 3), (5120, 10240, 4), (6144, 5120, 4), (5120, 1024, 5), (5120, 248320, 4)]
for k, n, K in CASES:
    nb = k * n * K // 8
    nc = max(2, (512 << 20) // nb + 1)
    ts = [torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                        generator=g).to(torch.int16) for _ in range(nc)]
    suh = torch.ones(k, device="cuda", dtype=torch.float16)
    svh = torch.full((n,), 0.01, device="cuda", dtype=torch.float16)
    res = []
    for m in (1, 2, 3, 4, 8):
        x = torch.randn((m, k), device="cuda", dtype=torch.float16)
        xh = torch.empty_like(x)
        y = torch.empty((m, n), device="cuda", dtype=torch.float16)
        if m == 1:
            fn = lambda i: E.exl3_gemm(x, ts[i], y, suh, xh, svh, -1, False, True, 0)
        else:
            fn = lambda i: E.exl3_gemv_mr(x, ts[i], y, suh, xh, svh, K, False, True)
        for i in range(nc):
            fn(i)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        it = 30
        e0.record()
        for i in range(it):
            fn(i % nc)
        e1.record()
        torch.cuda.synchronize()
        res.append(e0.elapsed_time(e1) * 1e3 / it)
    print(f"k={k:5d} n={n:6d} K={K}: " + "  ".join(f"m{m}={u:7.1f}us" for m, u in zip((1, 2, 3, 4, 8), res))
          + f"   m4/m1={res[3] / res[0]:.2f}", flush=True)

# plugin op, fused mixed-bit groups (strided output slices): M=4 == four M=1 calls, bitwise
from exl3rocm import ops  # noqa: E402,F401
k = 5120
spec = [(10240, 2), (6144, 4)]
tr = [torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                    generator=g).to(torch.int16) for n, K in spec]
su = [(torch.randn(k, device="cuda", generator=g).sign()).half() for _ in spec]
sv = [(torch.rand(n, device="cuda", generator=g) * 0.02 + 0.005).half() for n, _ in spec]
Ks = [K for _, K in spec]
x = torch.randn((4, k), device="cuda", dtype=torch.float16, generator=g)
y4 = torch.ops.exl3rocm.linear_groups(x, tr, su, sv, Ks, False, True)
y1 = torch.cat([torch.ops.exl3rocm.linear_groups(x[r:r + 1].contiguous(), tr, su, sv, Ks, False, True)
                for r in range(4)])
print("plugin fused mixed-bit M=4 vs 4x M=1:", "PASS" if torch.equal(y4.view(torch.int16), y1.view(torch.int16)) else "FAIL")
