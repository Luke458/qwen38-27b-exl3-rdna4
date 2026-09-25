"""Gather + dequantize per-token-head int8/fp8 KV cache blocks into a persistent fp16 buffer.

Used by the long-prefill attention path (plugin._install_pth_prefill_dequant). One Triton program
per (selected block, slot, kv head) row: out[j, s, h, :] = src[blk[j], s, h, :hs] * scale[blk[j], s, h].
The output buffer is persistent and grows in 1.25x steps, so repeated prefill chunks with growing
context do not ratchet the caching allocator (a fresh torch allocation per chunk did: +650 MiB peak
at a 50k-token prompt, experiments/0022).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pth_dequant_kernel(src, scale, blk, out,
                        s_blk, s_slot, s_head,          # src strides (elements)
                        c_blk, c_slot, c_head,          # scale strides (elements)
                        o_blk, o_slot, o_head,          # out strides (elements)
                        HS: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0)
    slot = tl.program_id(1)
    head = tl.program_id(2)
    b = tl.load(blk + j).to(tl.int64)
    d = tl.arange(0, BLOCK)
    m = d < HS
    x = tl.load(src + b * s_blk + slot * s_slot + head * s_head + d, mask=m, other=0).to(tl.float32)
    sc = tl.load(scale + b * c_blk + slot * c_slot + head * c_head)
    y = (x * sc).to(out.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(out + j * o_blk + slot * o_slot + head * o_head + d, y, mask=m)


_bufs: dict = {}


def _buffer(tag, device, nb, bsz, nkv, hs, dtype):
    key = (tag, device, bsz, nkv, hs, dtype)
    buf = _bufs.get(key)
    if buf is None or buf.shape[0] < nb:
        grow = max(nb, int((buf.shape[0] if buf is not None else 0) * 1.25) + 1)
        grow = (grow + 3) // 4 * 4
        _bufs.pop(key, None)
        if buf is not None:
            # hand the old segment back to the driver before allocating the larger one; otherwise
            # every growth step leaves a cached hole too small for the next (+660 MiB reserved
            # over allocated at a 50k-token prompt, experiments/0022)
            del buf
            if not torch.cuda.is_current_stream_capturing():
                torch.cuda.empty_cache()
        buf = torch.empty((grow, bsz, nkv, hs), dtype=dtype, device=device)
        _bufs[key] = buf
    return buf[:nb]


def gather_dequant(src: torch.Tensor, scale: torch.Tensor, blocks: torch.Tensor, hs: int,
                   dtype: torch.dtype, tag: str) -> torch.Tensor:
    """src: [num_blocks, block_size, nkv, >=hs] int8/fp8 view (any strides, unit last stride);
    scale: [num_blocks, block_size, nkv] fp32 view; blocks: [n] block ids.
    Returns [n, block_size, nkv, hs] fp16 (a view of a persistent buffer)."""
    n = blocks.numel()
    _, bsz, nkv, _ = src.shape
    out = _buffer(tag, src.device, n, bsz, nkv, hs, dtype)
    assert src.stride(-1) == 1
    _pth_dequant_kernel[(n, bsz, nkv)](
        src, scale, blocks, out,
        src.stride(0), src.stride(1), src.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HS=hs, BLOCK=triton.next_power_of_2(hs))
    return out
