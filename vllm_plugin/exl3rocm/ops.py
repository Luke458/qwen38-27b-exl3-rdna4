"""
EXL3 linear on ROCm as an opaque torch custom op, so torch.compile and vLLM graph
capture treat it as one node. Kernels come from the exllamav3 ROCm fork's extension
(`exllamav3_ext`, built against the serving image's torch).

One call computes one checkpoint tensor ("group"): y = had(had(x * suh) @ W_inner) * svh.
Routing mirrors the fork's validated gfx1201 serving path:
  rows == 1          -> exl3_gemm (split-K GEMV)
  rows  > 1          -> reconstruct fp16 W_inner + hgemm, with standalone Hadamards
(The fork forces multi-row calls through reconstruction on gfx1201; routing small M
to the 0011 WMMA GEMM is a separate, measured step.)
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_ext = None
GEMM_MAX_ROWS = int(os.environ.get("EXL3_GEMM_MAX_ROWS", "1"))
# widest non-lm_head layer (17408): no extra slicing for model layers, and the shared
# reconstruction buffer is 178 MB instead of 335 MB (lm_head slices at 17408)
RECON_SLICE_N = int(os.environ.get("EXL3_RECON_SLICE_N", "17408"))
# rows 2..MR_MAX use the multi-row GEMV when the extension has it (patch 0002)
MR_MAX = int(os.environ.get("EXL3_MR_MAX", "16"))


def _has_mr(E):
    return hasattr(E, "exl3_gemv_mr")


# EXL3_DEBUG_MEM=N: log torch allocator stats every N embedding calls (eager steps only)
_DEBUG_MEM = int(os.environ.get("EXL3_DEBUG_MEM", "0"))
_DEBUG_MEM_STATE = {"n": 0}

# timing-only debug switch: quantized linears return zeros without running kernels
_DEBUG_SKIP = os.environ.get("EXL3_DEBUG_SKIP_LINEAR") == "1"


def ext():
    global _ext
    if _ext is None:
        import exllamav3_ext  # noqa: F401  (standalone torch extension module)
        _ext = exllamav3_ext
    return _ext


_wbuf: dict = {}


def reserve_weight_buffer(device, numel):
    """Grow the shared reconstruction buffer at load time, so graph capture never allocates it."""
    device = torch.device(device)
    buf = _wbuf.get(device)
    if buf is None or buf.numel() < numel:
        _wbuf[device] = torch.empty(numel, dtype=torch.float16, device=device)


def _weight_buffer(device, numel):
    reserve_weight_buffer(device, numel)
    return _wbuf[torch.device(device)][:numel]


@torch.library.custom_op("exl3rocm::linear", mutates_args=())
def exl3_linear(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
                K: int, mcg: bool, mul1: bool) -> torch.Tensor:
    E = ext()
    k = x.shape[-1]
    n = svh.shape[0]
    x2 = x.reshape(-1, k)
    if x2.dtype != torch.float16:
        x2 = x2.to(torch.float16)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    y = torch.empty((M, n), dtype=torch.float16, device=x.device)
    if M == 0:
        return y.view(*x.shape[:-1], n)
    if M <= GEMM_MAX_ROWS:
        xh = torch.empty_like(x2)
        E.exl3_gemm(x2, trellis, y, suh, xh, svh, -1, mcg, mul1, 0)
    elif M <= MR_MAX and _has_mr(E) and E.exl3_gemv_mr_supported(M, k, n, K, mcg, mul1):
        # decode-once multi-row GEMV: each row bitwise == the m=1 GEMV (spec-decode verify)
        E.exl3_gemv_mr(x2, trellis, y, suh, torch.empty_like(x2), svh, K, mcg, mul1)
    else:
        xh = torch.empty_like(x2)
        E.had_r_128(x2, xh, suh, None, 1.0)
        for n0 in range(0, n, RECON_SLICE_N):
            n1 = min(n0 + RECON_SLICE_N, n)
            w = _weight_buffer(x.device, k * (n1 - n0)).view(k, n1 - n0)
            if n0 == 0 and n1 == n:
                E.reconstruct(w, trellis, K, mcg, mul1)
            else:
                E.reconstruct_slice(w, trellis, K, mcg, mul1, n0)
            E.hgemm_recon(xh, w, y if (n0, n1) == (0, n) else y[:, n0:n1])
        E.had_r_128(y, y, None, svh, 1.0)
    return y.view(*x.shape[:-1], n)


@exl3_linear.register_fake
def _(x, trellis, suh, svh, K, mcg, mul1):
    return x.new_empty((*x.shape[:-1], svh.shape[0]), dtype=torch.float16)


@torch.library.custom_op("exl3rocm::linear_groups", mutates_args=())
def exl3_linear_groups(x: torch.Tensor, trellis: list[torch.Tensor], suh: list[torch.Tensor],
                       svh: list[torch.Tensor], K: list[int], mcg: bool, mul1: bool,
                       ptrs_trellis: Optional[torch.Tensor] = None, ptrs_suh: Optional[torch.Tensor] = None,
                       ptrs_svh: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A fused vLLM module made of several checkpoint tensors (e.g. in_proj_qkv + in_proj_z),
    each with its own K / suh / svh. The concatenation happens inside this opaque op on
    purpose: an Inductor-traced torch.cat of per-group outputs, fused with a later slice of
    the concatenated tensor, read the slice's source buffer with the concatenated row stride
    and ran off its end (GPU page fault, experiments/0017).

    ptrs_* = int64 device tensors of per-group data pointers, given when all groups
    share K and width (gate/up): at M=1 one grouped exl3_mgemm computes every group, and
    its (groups, 1, n) output is already the concatenated (1, groups*n) row."""
    k = x.shape[-1]
    widths = [s.shape[0] for s in svh]
    if _DEBUG_SKIP:
        return torch.zeros((*x.shape[:-1], sum(widths)), dtype=torch.float16, device=x.device)
    x2 = x.reshape(-1, k)
    if x2.dtype != torch.float16:
        x2 = x2.to(torch.float16)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]
    if len(trellis) == 1:
        return exl3_linear(x, trellis[0], suh[0], svh[0], K[0], mcg, mul1)
    E = ext()
    ng = len(trellis)
    if M == 1 and ptrs_trellis is not None:
        y = torch.empty((ng, 1, widths[0]), dtype=torch.float16, device=x.device)
        xh = torch.empty((ng, 1, k), dtype=torch.float16, device=x.device)
        E.exl3_mgemm(x2.view(1, 1, k), ptrs_trellis, y, ptrs_suh, xh, ptrs_svh,
                     None, None, K[0], -1, mcg, mul1, -1, -1, 0, 1, None, None)
        return y.view(*x.shape[:-1], ng * widths[0])
    y = torch.empty((M, sum(widths)), dtype=torch.float16, device=x.device)
    if M == 1:
        # a column slice of a single row is contiguous: write each group in place
        xh = torch.empty_like(x2)
        n0 = 0
        for t, su, sv, w in zip(trellis, suh, svh, widths):
            E.exl3_gemm(x2, t, y[:, n0:n0 + w], su, xh, sv, -1, mcg, mul1, 0)
            n0 += w
        return y.view(*x.shape[:-1], n0)
    if M <= MR_MAX and _has_mr(E) and all(E.exl3_gemv_mr_supported(M, k, w, kk, mcg, mul1)
                                           for w, kk in zip(widths, K)):
        # multi-row GEMV writes each group straight into its (row-strided) column slice; runs of
        # consecutive groups with equal (K, width) share one grouped launch (gate/up, k/v). The
        # split-K wave count follows the module's m=1 path so rows stay bitwise equal to it:
        # grouped mgemm (bszm = group count) when ptrs are given, single-matrix GEMVs otherwise.
        grouped = hasattr(E, "exl3_gemv_mr_grouped")
        bszm = ng if ptrs_trellis is not None else 1
        xh = torch.empty((min(ng, 4), M, k), dtype=torch.float16, device=x.device)
        i, n0 = 0, 0
        while i < ng:
            j = i + 1
            while grouped and j < ng and j - i < 4 and K[j] == K[i] and widths[j] == widths[i]:
                j += 1
            if j - i > 1:
                E.exl3_gemv_mr_grouped(x2, list(trellis[i:j]), y[:, n0:], list(suh[i:j]), xh,
                                       list(svh[i:j]), K[i], mcg, mul1, bszm)
            else:
                E.exl3_gemv_mr(x2, trellis[i], y[:, n0:n0 + widths[i]], suh[i], xh[0], svh[i], K[i], mcg, mul1)
            n0 += sum(widths[i:j])
            i = j
        return y.view(*x.shape[:-1], n0)
    n0 = 0
    for t, su, sv, kk, w in zip(trellis, suh, svh, K, widths):
        y[:, n0:n0 + w].copy_(exl3_linear(x2, t, su, sv, kk, mcg, mul1))
        n0 += w
    return y.view(*x.shape[:-1], n0)


@exl3_linear_groups.register_fake
def _(x, trellis, suh, svh, K, mcg, mul1, ptrs_trellis=None, ptrs_suh=None, ptrs_svh=None):
    return x.new_empty((*x.shape[:-1], sum(s.shape[0] for s in svh)), dtype=torch.float16)


@torch.library.custom_op("exl3rocm::fp8_embedding", mutates_args=())
def fp8_embedding(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Row gather from a float8_e4m3fn table, cast to fp16. Opaque on purpose: Inductor
    lowered the inline view/index/cast by materializing a full 1.19 GiB copy of the table
    (OOM when the MTP drafter compiled; experiments/0019)."""
    if _DEBUG_MEM:
        _DEBUG_MEM_STATE["n"] += 1
        if _DEBUG_MEM_STATE["n"] % _DEBUG_MEM == 0 and not torch.cuda.is_current_stream_capturing():
            import sys
            print(f"[exl3mem] step {_DEBUG_MEM_STATE['n']} tokens {ids.numel()} alloc "
                  f"{torch.cuda.memory_allocated() >> 20} MiB reserved {torch.cuda.memory_reserved() >> 20} MiB "
                  f"max_alloc {torch.cuda.max_memory_allocated() >> 20} MiB", file=sys.stderr, flush=True)
    flat = ids.reshape(-1)
    rows = weight.view(torch.uint8).index_select(0, flat)
    return rows.view(torch.float8_e4m3fn).to(torch.float16).view(*ids.shape, weight.shape[1])


@fp8_embedding.register_fake
def _(weight, ids):
    return weight.new_empty((*ids.shape, weight.shape[1]), dtype=torch.float16)
