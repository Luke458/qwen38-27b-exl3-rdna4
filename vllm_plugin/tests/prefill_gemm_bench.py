#!/usr/bin/env python3
"""Prefill GEMM on reconstructed EXL3 weights: fork hgemm_recon (fp16 accumulate) vs torch.matmul
(hipBLASLt, fp32 accumulate). Speed at M rows and relative error vs an fp32 reference."""
import torch

import exllamav3_ext as E

SHAPES = [(5120, 17408, 3), (17408, 5120, 3), (5120, 10240, 4), (6144, 5120, 4), (5120, 248320, 4)]
g = torch.Generator(device="cuda").manual_seed(4)


def timeit(fn, it=10):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it


for M in (512, 2048):
    for k, n, K in SHAPES:
        n_eff = min(n, 32768)
        tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                           generator=g).to(torch.int16)
        w = torch.empty((k, n_eff), device="cuda", dtype=torch.float16)
        x = torch.randn((M, k), device="cuda", dtype=torch.float16, generator=g) * 0.5
        y = torch.empty((M, n_eff), device="cuda", dtype=torch.float16)
        rec = lambda: E.reconstruct_slice(w, tr, K, False, True, 0) if n_eff < n else E.reconstruct(w, tr, K, False, True)
        rec()
        t_rec = timeit(rec)
        t_h = timeit(lambda: E.hgemm_recon(x, w, y))
        yh = y.clone()
        t_t = timeit(lambda: torch.matmul(x, w, out=y))
        yt = y.clone()
        ref = x.float() @ w.float()
        err = lambda a: ((a.float() - ref).norm() / ref.norm()).item()
        print(f"M={M:5d} k={k:5d} n={n_eff:6d} K={K}: reconstruct {t_rec:6.2f} ms | hgemm_recon {t_h:6.2f} ms "
              f"relerr {err(yh):.2e} | torch.matmul {t_t:6.2f} ms relerr {err(yt):.2e}", flush=True)
        del tr, w, y
