"""GPU parity: pinned ROCm fork vs the independent oracle (P1 gate).

Run: .venv/bin/python -m pytest tests/test_gpu_parity.py -q
Requires the built extension and a gfx1201 GPU. Emits measured baseline
deviation stats to artifacts/baseline/gpu_parity_stats.json for tolerance
derivation (tools/freeze_policy.py).

Exactness (bit-exact) is required for decoder values and state windows.
Float pipeline results are checked against the analytic policy bounds.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
import exl3_oracle as O  # noqa: E402

torch.manual_seed(20260923)
DEVICE = "cuda:0"
STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "baseline", "gpu_parity_stats.json")

ext = None
try:
    from exllamav3.ext import exllamav3_ext as ext  # noqa: F811
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    ext is None or not torch.cuda.is_available(),
    reason="built extension + GPU required",
)

_analytic = {"max_abs_err": 0.02, "normalized_rms_err": 0.002, "cosine_similarity_min": 0.99999}


def _stats(got: np.ndarray, ref: np.ndarray) -> dict:
    got = got.astype(np.float64)
    ref = ref.astype(np.float64)
    diff = got - ref
    scale = np.sqrt(np.mean(ref ** 2)) or 1.0
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) or 1.0))
    # fp16 ULP multiples at each reference magnitude (0 -> floor): diagnostic
    ulp = np.maximum(np.abs(ref) * 2.0 ** -10, 2.0 ** -14)
    # combined absolute-plus-relative per-element bound (the gate):
    #   |diff| <= abs_part * RMS(ref) + rel_part * |ref|
    abs_part, rel_part = 4e-3, 2e-3
    bound = abs_part * scale + rel_part * np.abs(ref)
    return {
        "max_abs_err": float(np.abs(diff).max()),
        "normalized_max_abs_err": float(np.abs(diff).max() / scale),
        "normalized_rms_err": float(np.sqrt(np.mean(diff ** 2)) / scale),
        "cosine_similarity": cos,
        "max_rel_err": float((np.abs(diff) / (np.abs(ref) + 1e-4)).max()),
        "max_ulp16_multiples": float((np.abs(diff) / ulp).max()),
        "abs_rel_bound_ratio": float((np.abs(diff) / bound).max()),
    }


# Same-arithmetic analytic bounds (scale-aware); see configs/acceptance_policy.json
_analytic = {
    "normalized_max_abs_err": 3e-3,
    "normalized_rms_err": 2e-4,
    "cosine_similarity_min": 0.999999,
    "abs_rel_bound_ratio": 1.0,
}


def _record(name: str, stats: dict):
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    all_stats = {}
    if os.path.isfile(STATS_PATH):
        all_stats = json.load(open(STATS_PATH))
    all_stats[name] = stats
    tmp = STATS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(all_stats, f, indent=1)
    os.replace(tmp, STATS_PATH)


def make_fixtures(k_in, n, bits, seed, signs=False):
    rng = np.random.default_rng(seed)
    tr = O.random_trellis(rng, k_in // 16, n // 16, bits)
    if signs:
        suh = np.sign(rng.standard_normal(k_in)).astype(np.float16)
        svh = np.sign(rng.standard_normal(n)).astype(np.float16)
    else:
        suh = (np.abs(rng.standard_normal(k_in)) * 0.5 + 0.5).astype(np.float16)
        svh = (np.abs(rng.standard_normal(n)) * 0.5 + 0.5).astype(np.float16)
    a = (rng.standard_normal(k_in) * 0.3).astype(np.float16)
    return a, tr, suh, svh


# ---------------------------------------------------------------------------
# Exactness: codebook decode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cb,mul1,mcg", [(2, True, False), (1, False, True), (0, False, False)])
def test_decode_bitexact(cb, mul1, mcg):
    codes = torch.randint(0, 65536, (16, 256), dtype=torch.int32, device=DEVICE).to(torch.int16)
    decoded = torch.empty(16, 256, dtype=torch.float, device=DEVICE)
    ext.decode(codes, decoded, mcg, mul1)
    got = decoded.cpu().numpy().astype(np.float32)
    ref = O.decode_codebook(codes.cpu().numpy().astype(np.uint16), cb)
    # oracle returns exact fp16 values as float64; ext returns fp32 copies
    np.testing.assert_array_equal(got.astype(np.float64), ref), f"cb={cb} decode mismatch"


# ---------------------------------------------------------------------------
# Exactness: state windows + reconstruction (decoder/state/bitfield parity)
# ---------------------------------------------------------------------------
def test_reconstruct_bitexact_mul1():
    k_in, n, bits = 256, 128, 3
    _, tr, _, _ = make_fixtures(k_in, n, bits, 1)
    t = torch.from_numpy(tr.copy()).to(torch.int16).to(DEVICE)
    w_hat = torch.empty(k_in, n, dtype=torch.half, device=DEVICE)
    ext.reconstruct(w_hat, t, bits, False, True)
    got = w_hat.cpu().numpy().astype(np.float32)
    ref = O.decode_trellis(tr, bits, cb=2)
    # both sides are exact fp16 values
    np.testing.assert_array_equal(got.astype(np.float64), ref)
    _record("reconstruct_bitexact", {"exactness_failures": 0})


def test_state_windows_via_reconstruct_all_bits():
    """Window layout agrees with the extension for bits 1..8 (mul1 + cb0)."""
    for bits in range(1, 9):
        k_in, n = 128, 128
        _, tr, _, _ = make_fixtures(k_in, n, bits, 100 + bits)
        t = torch.from_numpy(tr.copy()).to(torch.int16).to(DEVICE)
        for cb, mcg, mul1 in ((2, False, True), (0, False, False)):
            w_hat = torch.empty(k_in, n, dtype=torch.half, device=DEVICE)
            ext.reconstruct(w_hat, t, bits, mcg, mul1)
            got = w_hat.cpu().numpy().astype(np.float64)
            ref = O.decode_trellis(tr, bits, cb=cb)
            assert (got == ref).all(), f"bits={bits} cb={cb} window/decode mismatch"


# ---------------------------------------------------------------------------
# Same-arithmetic float pipeline: exl3_gemv vs oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [
    (256, 128, 3), (512, 256, 3), (1024, 512, 3),
    (5120, 5120, 3), (17408, 5120, 3), (5120, 17408, 3),  # real checkpoint shapes
    (5120, 5120, 2), (5120, 5120, 4),                     # non-target bitrates (fallback path)
])
def test_gemv_parity(shape):
    k_in, n, bits = shape
    a, tr, suh, svh = make_fixtures(k_in, n, bits, 7 + k_in + n, signs=(k_in % 2 == 0))
    A = torch.from_numpy(a.copy()).to(DEVICE)
    B = torch.from_numpy(tr.copy()).to(torch.int16).to(DEVICE)
    suh_t = torch.from_numpy(suh.copy()).to(DEVICE)
    svh_t = torch.from_numpy(svh.copy()).to(DEVICE)
    A_had = torch.empty_like(A)
    C = torch.empty(1, n, dtype=torch.half, device=DEVICE)
    mul1, mcg = True, False
    ext.exl3_gemv(A.unsqueeze(0), B, C, suh_t, A_had, svh_t, mcg, mul1)
    torch.cuda.synchronize()
    got = C.cpu().numpy().reshape(-1).astype(np.float32)

    ref = O.gemv_mul1(
        a.astype(np.float64), tr, bits,
        suh.astype(np.float64), svh.astype(np.float64), cb=2,
    )
    # oracle is FP64; the kernel is fp16 storage + fp32 accumulate. Compare the
    # oracle against the fp16-rounded reference semantics it implements; the
    # tolerance family covers accumulation-order rounding.
    st = _stats(got, ref)
    st["shape"] = [k_in, n, bits]
    _record(f"gemv_{k_in}x{n}_b{bits}", st)
    assert st["normalized_max_abs_err"] <= _analytic["normalized_max_abs_err"], st
    assert st["normalized_rms_err"] <= _analytic["normalized_rms_err"], st
    assert st["cosine_similarity"] >= _analytic["cosine_similarity_min"], st
    assert st["max_ulp16_multiples"] >= 0.0  # recorded diagnostic
    assert st["abs_rel_bound_ratio"] <= _analytic["abs_rel_bound_ratio"], st


def test_gemv_determinism_repeat():
    a, tr, suh, svh = make_fixtures(512, 256, 3, 55)
    A = torch.from_numpy(a.copy()).to(DEVICE)
    B = torch.from_numpy(tr.copy()).to(torch.int16).to(DEVICE)
    suh_t = torch.from_numpy(suh.copy()).to(DEVICE)
    svh_t = torch.from_numpy(svh.copy()).to(DEVICE)
    outs = []
    for _ in range(3):
        A_had = torch.empty_like(A)
        C = torch.empty(1, 256, dtype=torch.half, device=DEVICE)
        ext.exl3_gemv(A.unsqueeze(0), B, C, suh_t, A_had, svh_t, False, True)
        torch.cuda.synchronize()
        outs.append(C.clone().cpu())
    assert all(torch.equal(outs[0], o) for o in outs[1:]), "repeat calls not bit-identical"


def test_different_streams():
    a, tr, suh, svh = make_fixtures(512, 256, 3, 56)
    A = torch.from_numpy(a.copy()).to(DEVICE)
    B = torch.from_numpy(tr.copy()).to(torch.int16).to(DEVICE)
    suh_t = torch.from_numpy(suh.copy()).to(DEVICE)
    svh_t = torch.from_numpy(svh.copy()).to(DEVICE)
    ref_out = None
    for _ in range(2):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            A_had = torch.empty_like(A)
            C = torch.empty(1, 256, dtype=torch.half, device=DEVICE)
            ext.exl3_gemv(A.unsqueeze(0), B, C, suh_t, A_had, svh_t, False, True)
        s.synchronize()
        out = C.cpu()
        if ref_out is None:
            ref_out = out
        else:
            assert torch.equal(ref_out, out), "stream-dependent output"


def test_zero_input():
    k_in, n = 256, 128
    _, tr, suh, svh = make_fixtures(k_in, n, 3, 3)
    A = torch.zeros(1, k_in, dtype=torch.half, device=DEVICE)
    B = torch.from_numpy(tr.copy()).to(torch.int16).to(DEVICE)
    suh_t = torch.from_numpy(suh.copy()).to(DEVICE)
    svh_t = torch.from_numpy(svh.copy()).to(DEVICE)
    A_had = torch.empty(1, k_in, dtype=torch.half, device=DEVICE)
    C = torch.empty(1, n, dtype=torch.half, device=DEVICE)
    ext.exl3_gemv(A, B, C, suh_t, A_had, svh_t, False, True)
    torch.cuda.synchronize()
    assert torch.isfinite(C).all()
    assert C.abs().max().item() <= 1e-3


# ---------------------------------------------------------------------------
# Dispatch contract: ineligible calls must decline / raise WITHOUT modifying C
# ---------------------------------------------------------------------------
def test_ineligible_declines_without_modifying_outputs():
    k_in, n = 256, 128
    a, tr, suh, svh = make_fixtures(k_in, n, 3, 9)
    B = torch.from_numpy(tr.copy()).to(torch.int16).to(DEVICE)
    suh_t = torch.from_numpy(suh.copy()).to(DEVICE)
    svh_t = torch.from_numpy(svh.copy()).to(DEVICE)

    # m = 2 is ineligible for the RDNA GEMV binding (TORCH_CHECK size_m == 1 path)
    A = torch.from_numpy(np.stack([a, a]).copy()).to(DEVICE)
    A_had = torch.empty(2, k_in, dtype=torch.half, device=DEVICE)
    C = torch.full((2, n), float("nan"), dtype=torch.half, device=DEVICE)
    declined = False
    try:
        ext.exl3_gemv(A, B, C, suh_t, A_had, svh_t, False, True)
    except Exception:
        declined = True
    torch.cuda.synchronize()
    if declined:
        assert torch.isnan(C).all(), "declined call must not modify outputs"
    else:
        # if a future kernel handles m>1 it must produce finite output
        assert torch.isfinite(C).all()
    _record("ineligible_dispatch", {"declined": declined})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
