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

import torch

_ext = None
GEMM_MAX_ROWS = int(os.environ.get("EXL3_GEMM_MAX_ROWS", "1"))
RECON_SLICE_N = int(os.environ.get("EXL3_RECON_SLICE_N", "32768"))


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
                       svh: list[torch.Tensor], K: list[int], mcg: bool, mul1: bool) -> torch.Tensor:
    """A fused vLLM module made of several checkpoint tensors (e.g. in_proj_qkv + in_proj_z),
    each with its own K / suh / svh. The concatenation happens inside this opaque op on
    purpose: an Inductor-traced torch.cat of per-group outputs, fused with a later slice of
    the concatenated tensor, read the slice's source buffer with the concatenated row stride
    and ran off its end (GPU page fault, experiments/0017)."""
    if len(trellis) == 1:
        return exl3_linear(x, trellis[0], suh[0], svh[0], K[0], mcg, mul1)
    k = x.shape[-1]
    widths = [s.shape[0] for s in svh]
    x2 = x.reshape(-1, k)
    y = torch.empty((x2.shape[0], sum(widths)), dtype=torch.float16, device=x.device)
    n0 = 0
    for t, su, sv, kk, w in zip(trellis, suh, svh, K, widths):
        y[:, n0:n0 + w].copy_(exl3_linear(x2, t, su, sv, kk, mcg, mul1))
        n0 += w
    return y.view(*x.shape[:-1], n0)


@exl3_linear_groups.register_fake
def _(x, trellis, suh, svh, K, mcg, mul1):
    return x.new_empty((*x.shape[:-1], sum(s.shape[0] for s in svh)), dtype=torch.float16)
