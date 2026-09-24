"""Candidate 0008: output-transform fusion (EXL3_GEMV_FUSED_HAD=1).

Claim: bitwise-identical outputs vs the three-launch path (same dot body, same
__float2half store, same had epilogue; only the third launch disappears).
Gate = torch.equal across kernel forms, dtypes, codebooks and streams.

Run: PYTHONPATH=experiments/0008-fused-had/binary .venv/bin/python \
       -m pytest tests/test_fused_had.py -q
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

try:
    from exllamav3.ext import exllamav3_ext as ext
except Exception:
    ext = None

pytestmark = pytest.mark.skipif(ext is None or not torch.cuda.is_available(), reason="ext+GPU required")
DEVICE = "cuda:0"


def _loaded_binary():
    import importlib.util
    spec = importlib.util.find_spec("exllamav3_ext")
    if not (spec and spec.origin):
        return None, b""
    return spec.origin, open(spec.origin, "rb").read()


_SO_PATH, _SO_DATA = _loaded_binary()
_HAS_FUSED = b"dot_had_kernel" in _SO_DATA
_SO_SHA = hashlib.sha256(_SO_DATA).hexdigest() if _SO_DATA else None


def _frozen_baseline_sha():
    """SHA of the deliberately pre-fusion frozen baseline binary, from the local
    campaign record (not shipped in the public repo; may legitimately be absent)."""
    p = Path(__file__).resolve().parents[1] / "artifacts" / "baseline" / "records.json"
    try:
        return json.loads(p.read_text()).get("binary_sha256")
    except Exception:
        return None


requires_fused = pytest.mark.skipif(
    not _HAS_FUSED,
    reason="loaded binary lacks the fused kernels (baseline build); "
           "fusion tests need an experimental/candidate binary")


def run(a, tr, suh, svh, n, lds="0", fused="0", c_fp32=False, stream=None):
    A = torch.from_numpy(a.copy()).unsqueeze(0).to(DEVICE)
    B = torch.from_numpy(np.ascontiguousarray(tr)).to(torch.int16).to(DEVICE)
    suh_t = torch.from_numpy(np.ascontiguousarray(suh)).to(DEVICE)
    svh_t = torch.from_numpy(np.ascontiguousarray(svh)).to(DEVICE)
    A_had = torch.empty(1, a.shape[0], dtype=torch.half, device=DEVICE)
    C = torch.empty(1, n, dtype=torch.float if c_fp32 else torch.half, device=DEVICE)
    saved = {k: os.environ.get(k) for k in ("EXL3_GEMV_LDS", "EXL3_GEMV_FUSED_HAD")}
    try:
        os.environ["EXL3_GEMV_LDS"] = lds
        os.environ["EXL3_GEMV_FUSED_HAD"] = fused

        def call():
            ext.exl3_gemv(A, B, C, suh_t, A_had, svh_t, False, True)

        if stream is not None:
            with torch.cuda.stream(stream):
                call()
            stream.synchronize()
        else:
            call()
            torch.cuda.synchronize()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return C.clone()


def fixtures(k_in, n, bits=3, seed=0):
    rng = np.random.default_rng(seed)
    tr = rng.integers(0, 65536, size=(k_in // 16, n // 16, 16 * bits), dtype=np.uint16)
    a = (rng.standard_normal(k_in) * 0.2).astype(np.float16)
    suh = (np.abs(rng.standard_normal(k_in)) * 0.5 + 0.5).astype(np.float16)
    svh = (np.abs(rng.standard_normal(n)) * 0.5 + 0.5).astype(np.float16)
    return a, tr, suh, svh


def test_binary_contains_fused_kernels():
    """Two-sided provenance guard against trivially-passing bitwise tests.

    The fused-vs-unfused equalities below carry information ONLY on a binary
    that actually contains the fused kernels. On any other binary both env
    values take the same path and every equality passes for the wrong reason --
    e.g. a stale/partial object set (build_kernel.py rebuilds only the int8
    TU!). So: fused symbols must be present in the loaded .so, and their absence
    is allowed only for the deliberately pre-fusion frozen baseline binary
    (verified by hash where a local baseline record exists)."""
    assert _SO_PATH, "exllamav3_ext not importable"
    baseline_sha = _frozen_baseline_sha()
    if baseline_sha and _SO_SHA == baseline_sha:
        # The frozen baseline predates the fusion work entirely: if fused
        # symbols appeared under the baseline hash, the record is compromised.
        assert not _HAS_FUSED, (
            f"{_SO_PATH} matches the frozen baseline hash but contains fused "
            "kernels -- baseline binary/record mismatch, re-capture the baseline")
        pytest.skip("frozen baseline binary (pre-fusion by design); "
                    "fusion tests need an experimental/candidate binary")
    if not _HAS_FUSED:
        if not baseline_sha:
            pytest.skip("cannot establish binary provenance "
                        "(no artifacts/baseline/records.json)")
        raise AssertionError(
            f"{_SO_PATH} lacks the fused kernels and is NOT the frozen baseline "
            "-- rebuild ALL affected TUs (full uv install or extended "
            "build_kernel), then re-run")


@pytest.mark.parametrize("k_in,n", [
    (512, 256),        # split-K form, tiny
    (5120, 5120),      # real shape (gate/up-like)
    (17408, 5120),     # real shape (down, splitk 16 warps)
    (5120, 17408),     # real shape (gate/up, splitk 4 warps)
])
@requires_fused
def test_fused_bitwise_splitk(k_in, n):
    a, tr, suh, svh = fixtures(k_in, n, seed=11 + k_in)
    base = run(a, tr, suh, svh, n, fused="0")
    fused = run(a, tr, suh, svh, n, fused="1")
    assert torch.equal(fused, base), f"splitk fused != baseline for {k_in}x{n}"


@requires_fused
def test_fused_bitwise_singlewarp_form():
    # n_tiles = 32896/16 = 2056 > 2048 -> single-warp form
    a, tr, suh, svh = fixtures(256, 32896, seed=5)
    base = run(a, tr, suh, svh, 32896, fused="0")
    fused = run(a, tr, suh, svh, 32896, fused="1")
    assert torch.equal(fused, base), "single-warp fused != baseline"


@requires_fused
def test_fused_bitwise_fp32_out():
    a, tr, suh, svh = fixtures(512, 256, seed=7)
    base = run(a, tr, suh, svh, 256, fused="0", c_fp32=True)
    fused = run(a, tr, suh, svh, 256, fused="1", c_fp32=True)
    assert torch.equal(fused, base), "c_fp32 fused != baseline"


@requires_fused
def test_fused_bitwise_lds_core():
    a, tr, suh, svh = fixtures(512, 256, seed=8)
    base = run(a, tr, suh, svh, 256, lds="1", fused="0")
    fused = run(a, tr, suh, svh, 256, lds="1", fused="1")
    assert torch.equal(fused, base), "lds-core fused != baseline"


@requires_fused
def test_fused_repeat_and_streams():
    a, tr, suh, svh = fixtures(512, 256, seed=9)
    ref = run(a, tr, suh, svh, 256, fused="1")
    for _ in range(3):
        again = run(a, tr, suh, svh, 256, fused="1")
        assert torch.equal(again, ref), "fused repeat not bit-identical"
    s = torch.cuda.Stream()
    other = run(a, tr, suh, svh, 256, fused="1", stream=s)
    assert torch.equal(other, ref), "fused stream-dependent"
    # and interleaved with unfused on another stream: counters must not collide
    s2 = torch.cuda.Stream()
    u = run(a, tr, suh, svh, 256, fused="0", stream=s2)
    f = run(a, tr, suh, svh, 256, fused="1", stream=s)
    base = run(a, tr, suh, svh, 256, fused="0")
    assert torch.equal(u, base) and torch.equal(f, ref)


@requires_fused
def test_fused_cb0_bitwise():
    a, tr, suh, svh = fixtures(512, 256, seed=10)
    A = torch.from_numpy(a.copy()).unsqueeze(0).to(DEVICE)
    B = torch.from_numpy(np.ascontiguousarray(tr)).to(torch.int16).to(DEVICE)
    suh_t = torch.from_numpy(np.ascontiguousarray(suh)).to(DEVICE)
    svh_t = torch.from_numpy(np.ascontiguousarray(svh)).to(DEVICE)
    A_had = torch.empty(1, 512, dtype=torch.half, device=DEVICE)
    outs = []
    saved = os.environ.get("EXL3_GEMV_FUSED_HAD")
    try:
        for fused in ("0", "1"):
            os.environ["EXL3_GEMV_FUSED_HAD"] = fused
            C = torch.empty(1, 256, dtype=torch.half, device=DEVICE)
            ext.exl3_gemv(A, B, C, suh_t, A_had, svh_t, True, False)   # cb0 (mcg=False,mul1=False)
            torch.cuda.synchronize()
            outs.append(C.clone())
    finally:
        if saved is None:
            os.environ.pop("EXL3_GEMV_FUSED_HAD", None)
        else:
            os.environ["EXL3_GEMV_FUSED_HAD"] = saved
    assert torch.equal(outs[0], outs[1]), "cb0 fused != baseline"
