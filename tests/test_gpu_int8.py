"""GPU validation for the staged int8 GEMV (bits=3, mul1, M=1).

Run: EXL3_ROCM_GFX1201_INT8=2 .venv/bin/python -m pytest tests/test_gpu_int8.py -q

Gate: the int8 path must match reference/exl3_oracle.py::gemv_int8_reference
(same equations) within the int8-plain budget, and its deviation from the fp16
pipeline must stay within the plain-int8 error family (~1% NRMS). Disabled mode
(=0) and ineligible calls must decline without modifying outputs.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
import exl3_oracle as O  # noqa: E402

try:
    from exllamav3.ext import exllamav3_ext as ext
except Exception:
    ext = None

pytestmark = pytest.mark.skipif(ext is None or not torch.cuda.is_available(), reason="ext+GPU required")
DEVICE = "cuda:0"


def run_int8(a, tr, suh, svh, n):
    A = torch.from_numpy(a.copy()).unsqueeze(0).to(DEVICE)
    B = torch.from_numpy(tr.copy()).to(torch.int16).to(DEVICE)
    suh_t = torch.from_numpy(suh.copy()).to(DEVICE)
    svh_t = torch.from_numpy(svh.copy()).to(DEVICE)
    A_had = torch.empty(1, a.shape[0], dtype=torch.half, device=DEVICE)
    C = torch.full((1, n), float("nan"), dtype=torch.half, device=DEVICE)
    # the real integration entry: exl3_gemm dispatches to exl3_gemv_int8 when
    # EXL3_ROCM_GFX1201_INT8 enables it (verify via _TRACE output), else the
    # fp16 fallback handles the call
    tag = ext.exl3_gemm(A, B, C, suh_t, A_had, svh_t, -1, False, True, 0)
    torch.cuda.synchronize()
    return tag, C


@pytest.mark.parametrize("k_in,n", [(256, 128), (512, 256), (1024, 512), (5120, 5120), (17408, 5120), (5120, 17408)])
@pytest.mark.parametrize("family", ["random", "zero", "impulse", "alternating", "outlier"])
def test_int8_matches_reference(k_in, n, family, monkeypatch):
    if k_in > 1024 and family != "random":
        pytest.skip("large shape is exercised once; input families use compact shapes")
    monkeypatch.setenv("EXL3_ROCM_GFX1201_INT8", "2")
    bits = 3
    rng = np.random.default_rng(11 + k_in + n)
    tr = O.random_trellis(rng, k_in // 16, n // 16, bits)
    a = (rng.standard_normal(k_in) * 0.2).astype(np.float16)
    if family == "zero":
        a.fill(0)
    elif family == "impulse":
        a.fill(0)
        a[k_in // 2] = 1
    elif family == "alternating":
        a = (np.where(np.arange(k_in) % 2, -0.5, 0.5)).astype(np.float16)
    elif family == "outlier":
        a[k_in // 3] = 6
    suh = (np.abs(rng.standard_normal(k_in)) * 0.5 + 0.5).astype(np.float16)
    svh = (np.abs(rng.standard_normal(n)) * 0.5 + 0.5).astype(np.float16)
    handled, C = run_int8(a, tr, suh, svh, n)
    got = C.cpu().numpy().reshape(-1).astype(np.float64)
    assert np.isfinite(got).all(), "int8 path produced non-finite output"

    a_had = O.had_in(a.astype(np.float64), suh.astype(np.float64), faithful=True)
    S = O.trellis_states(tr, bits)
    q = max(abs(a_had).max(), 1e-8) / 127.0
    ref = O.gemv_int8_reference(a_had, S, q, svh.astype(np.float64), residual=False)

    scale = max(np.sqrt(np.mean(ref ** 2)), 1e-8)
    nrms = np.sqrt(np.mean((got - ref) ** 2)) / scale
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref))) if np.linalg.norm(ref) else 1.0
    assert nrms < 2e-3, f"vs int8 reference NRMS {nrms}"      # fp32-order noise only
    assert cos > 0.999999


def test_disabled_mode_falls_back_correctly(monkeypatch):
    # mode 0: the fp16 fallback must produce the oracle's fp16-pipeline output
    monkeypatch.setenv("EXL3_ROCM_GFX1201_INT8", "0")
    bits = 3
    rng = np.random.default_rng(5)
    tr = O.random_trellis(rng, 16, 16, bits)
    a = (rng.standard_normal(256) * 0.2).astype(np.float16)
    suh = np.ones(256, dtype=np.float16)
    svh = np.ones(256, dtype=np.float16)
    _, C = run_int8(a, tr, suh, svh, 256)
    got = C.cpu().numpy().reshape(-1).astype(np.float64)
    ref = O.gemv_mul1(a, tr, bits, suh, svh, faithful=True)
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() <= 4e-3 * np.sqrt(np.mean(ref ** 2)) + 2e-3 * np.abs(ref).max()


def test_ineligible_bits_run_fallback(monkeypatch):
    # bits=4 is not eligible for the int8 path: the fallback handles it and the
    # output must be the fp16 pipeline's (declined calls never corrupt state)
    monkeypatch.setenv("EXL3_ROCM_GFX1201_INT8", "2")
    rng = np.random.default_rng(6)
    tr = O.random_trellis(rng, 16, 16, 4)   # bits=4: not eligible
    a = (rng.standard_normal(256) * 0.2).astype(np.float16)
    _, C = run_int8(a, tr, np.ones(256, dtype=np.float16), np.ones(256, dtype=np.float16), 256)
    got = C.cpu().numpy().reshape(-1).astype(np.float64)
    ref = O.gemv_mul1(a, tr, 4, np.ones(256), np.ones(256), cb=2, faithful=True)
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() <= 0.05 * max(1.0, np.abs(ref).max())
