#!/usr/bin/env python3
"""exl3_gemv_mr_grouped: bitwise vs the m=1 path each module uses, and timing vs ungrouped."""
import torch

import exllamav3_ext as E

g = torch.Generator(device="cuda").manual_seed(21)


def mk(k, n, K):
    tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                       generator=g).to(torch.int16)
    su = (torch.randn(k, device="cuda", generator=g).sign() * 1.02).half()
    sv = (torch.rand(n, device="cuda", generator=g) * 0.02 + 0.005).half()
    return tr, su, sv


ok = True
# 1) gate/up style: m=1 path is exl3_mgemm (bszm=2)
for K in (1, 2, 3):
    k, n = 5120, 17408
    mats = [mk(k, n, K) for _ in range(2)]
    ptr = lambda i: torch.tensor([m[i].data_ptr() for m in mats], dtype=torch.long, device="cuda")
    for m in (2, 4, 8):
        x = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=g)
        y = torch.empty((m, 2 * n), device="cuda", dtype=torch.float16)
        E.exl3_gemv_mr_grouped(x, [a[0] for a in mats], y, [a[1] for a in mats],
                               torch.empty((2, m, k), device="cuda", dtype=torch.float16), [a[2] for a in mats],
                               K, False, True, 2)
        ref = torch.empty_like(y)
        for r in range(m):
            yr = torch.empty((2, 1, n), device="cuda", dtype=torch.float16)
            E.exl3_mgemm(x[r:r + 1].contiguous().view(1, 1, k), ptr(0), yr, ptr(1),
                         torch.empty((2, 1, k), device="cuda", dtype=torch.float16), ptr(2),
                         None, None, K, -1, False, True, -1, -1, 0, 1, None, None)
            ref[r] = yr.view(-1)
        same = torch.equal(y.view(torch.int16), ref.view(torch.int16))
        ok &= same
        if not same:
            print(f"MISMATCH gate/up K={K} m={m} maxabs {(y.float() - ref.float()).abs().max().item():.3g}")
# 2) k/v style: m=1 path is two single GEMVs (bszm=1)
for K in (4, 5, 6):
    k, n = 5120, 1024
    mats = [mk(k, n, K) for _ in range(2)]
    for m in (2, 4):
        x = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=g)
        y = torch.empty((m, 2 * n), device="cuda", dtype=torch.float16)
        E.exl3_gemv_mr_grouped(x, [a[0] for a in mats], y, [a[1] for a in mats],
                               torch.empty((2, m, k), device="cuda", dtype=torch.float16), [a[2] for a in mats],
                               K, False, True, 1)
        ref = torch.empty_like(y)
        for r in range(m):
            for i, (tr, su, sv) in enumerate(mats):
                yr = torch.empty((1, n), device="cuda", dtype=torch.float16)
                xr = x[r:r + 1].contiguous()
                E.exl3_gemm(xr, tr, yr, su, torch.empty_like(xr), sv, -1, False, True, 0)
                ref[r, i * n:(i + 1) * n] = yr[0]
        same = torch.equal(y.view(torch.int16), ref.view(torch.int16))
        ok &= same
        if not same:
            print(f"MISMATCH kv K={K} m={m} maxabs {(y.float() - ref.float()).abs().max().item():.3g}")
print("grouped bitwise:", "PASS" if ok else "FAIL", flush=True)

# timing: gate/up K3 at m=4, DRAM-resident
k, n, K = 5120, 17408, 3
nc = 8
sets = [[mk(k, n, K) for _ in range(2)] for _ in range(nc)]
x = torch.randn((4, k), device="cuda", dtype=torch.float16)
y = torch.empty((4, 2 * n), device="cuda", dtype=torch.float16)
xh = torch.empty((2, 4, k), device="cuda", dtype=torch.float16)


def grouped(i):
    s = sets[i]
    E.exl3_gemv_mr_grouped(x, [a[0] for a in s], y, [a[1] for a in s], xh, [a[2] for a in s], K, False, True, 2)


def separate(i):
    s = sets[i]
    for j, (tr, su, sv) in enumerate(s):
        E.exl3_gemv_mr(x, tr, y[:, j * n:(j + 1) * n], su, xh[0], sv, K, False, True)


for name, fn in (("separate", separate), ("grouped", grouped)):
    for i in range(nc):
        fn(i)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(40):
        fn(i % nc)
    e1.record()
    torch.cuda.synchronize()
    print(f"gate+up K3 m=4 {name}: {e0.elapsed_time(e1) / 40 * 1e3:.1f} us")
