#!/usr/bin/env python3
"""Bitwise cross-check of two exllamav3_ext builds on identical inputs.

  # 1) reference build (host venv, control binary on PYTHONPATH): write inputs + outputs
  python xcheck_ext.py ref  cases.pt
  # 2) candidate build (e.g. the serving image): recompute and compare bit-for-bit
  python xcheck_ext.py check cases.pt

Covers the kernels the vLLM plugin calls: exl3_gemm at M=1 (split-K GEMV) and M=4
(gfx12 WMMA GEMM), reconstruct / reconstruct_slice, had_r_128 and hgemm_recon, on
real Qwen3.8-27B shapes and every bitrate the checkpoint uses. Trellis words are
random int16 (every bit pattern is a valid EXL3 code).
"""
import sys

import torch

import exllamav3_ext as E

SHAPES = [  # (k, n, K)
    (5120, 17408, 2), (5120, 17408, 3), (17408, 5120, 3), (5120, 10240, 4),
    (5120, 6144, 2), (6144, 5120, 4), (5120, 12288, 5), (5120, 1024, 6), (5120, 1024, 1),
]
MUL1 = True


def run(case):
    x, tr, suh, svh, K = case["x"].cuda(), case["trellis"].cuda(), case["suh"].cuda(), case["svh"].cuda(), case["K"]
    k, n = x.shape[1], svh.shape[0]
    out = {}
    for M in (1, 4):
        xm = x[:M].contiguous()
        y = torch.empty((M, n), dtype=torch.float16, device="cuda")
        E.exl3_gemm(xm, tr, y, suh, torch.empty_like(xm), svh, -1, False, MUL1, 0)
        out[f"gemm_m{M}"] = y.cpu()
    w = torch.empty((k, n), dtype=torch.float16, device="cuda")
    E.reconstruct(w, tr, K, False, MUL1)
    out["reconstruct"] = w.cpu()
    ws = torch.empty((k, 1024), dtype=torch.float16, device="cuda")
    E.reconstruct_slice(ws, tr, K, False, MUL1, n - 1024)
    out["reconstruct_slice"] = ws.cpu()
    xm = x[:8].contiguous()
    xh = torch.empty_like(xm)
    E.had_r_128(xm, xh, suh, None, 1.0)
    out["had_in"] = xh.cpu()
    y = torch.empty((8, n), dtype=torch.float16, device="cuda")
    E.hgemm_recon(xh, w, y)
    E.had_r_128(y, y, None, svh, 1.0)
    out["recon_path_m8"] = y.cpu()
    torch.cuda.synchronize()
    return out


def main():
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "ref":
        g = torch.Generator().manual_seed(20260925)
        cases = []
        for k, n, K in SHAPES:
            cases.append({
                "k": k, "n": n, "K": K,
                "x": torch.randn((8, k), generator=g).half(),
                "trellis": torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32).to(torch.int16),
                "suh": (torch.randn(k, generator=g).sign() * (1 + 0.1 * torch.rand(k, generator=g))).half(),
                "svh": (torch.randn(n, generator=g).sign() * (0.01 + 0.01 * torch.rand(n, generator=g))).half(),
            })
        for c in cases:
            c["out"] = run(c)
        torch.save(cases, path)
        print(f"wrote {len(cases)} cases to {path}")
        return 0
    cases = torch.load(path)
    bad = 0
    for c in cases:
        got = run(c)
        for name, ref in c["out"].items():
            g = got[name]
            same = torch.equal(g.view(torch.int16), ref.view(torch.int16))
            finite = bool(torch.isfinite(ref.float()).all())
            if not same:
                bad += 1
                diff = (g.float() - ref.float()).abs().max().item()
                print(f"MISMATCH k={c['k']} n={c['n']} K={c['K']} {name}: max abs diff {diff:.3g}")
            elif not finite:
                print(f"note: non-finite values in reference k={c['k']} K={c['K']} {name}")
    total = sum(len(c["out"]) for c in cases)
    print(f"{total - bad}/{total} outputs bit-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
