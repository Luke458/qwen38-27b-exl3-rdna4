"""Gather + dequantize per-token-head int8/fp8 (and int4, see gather_dequant_int4) KV cache blocks into a
persistent fp16 buffer.

Used by the long-prefill attention paths (plugin._install_pth_prefill_dequant, kv_int4.prefill). One Triton
program per (selected block, slot, kv head) row: out[j, s, h, :] = src[blk[j], s, h, :hs] * scale[blk[j], s, h].
The output buffer is persistent and grows in 1.25x steps, capped at the KV pool's block count and at what the
batch's block table can reference (capping at the pool alone overshot a 64k request by 125 MB,
experiments/0024). So repeated prefill chunks with growing context do not ratchet the caching allocator (a
fresh torch allocation per chunk did: +650 MiB peak at a 50k-token prompt, experiments/0022).
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


def _buffer(tag, device, nb, bsz, nkv, hs, dtype, cap):
    key = (tag, device, bsz, nkv, hs, dtype)
    buf = _bufs.get(key)
    if buf is None or buf.shape[0] < nb:
        grow = max(nb, int((buf.shape[0] if buf is not None else 0) * 1.25) + 1)
        # never beyond the KV pool: a prefill cannot reference more blocks than exist
        grow = min(max((grow + 3) // 4 * 4, nb), max(cap, nb))
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


@triton.jit
def _int4_dequant_kernel(src, scale, blk, out,
                         s_blk, s_slot, s_head,
                         c_blk, c_slot, c_head,
                         o_blk, o_slot, o_head,
                         mul, HALF: tl.constexpr):
    """int4_per_token_head: byte i holds elements 2i (low nibble) and 2i+1; the fp32 scale's low 4 mantissa
    bits are the zero point. out = (nibble - zp) * scale * mul, still in the RHT-rotated domain."""
    j = tl.program_id(0)
    slot = tl.program_id(1)
    head = tl.program_id(2)
    b = tl.load(blk + j).to(tl.int64)
    d = tl.arange(0, HALF)
    p = tl.load(src + b * s_blk + slot * s_slot + head * s_head + d)
    bits = tl.load(scale + b * c_blk + slot * c_slot + head * c_head).to(tl.int32, bitcast=True)
    zp = (bits & 0xF).to(tl.float32)
    sc = (bits & -16).to(tl.float32, bitcast=True) * mul
    lo = ((p & 0xF).to(tl.float32) - zp) * sc
    hi = (((p >> 4) & 0xF).to(tl.float32) - zp) * sc
    o = out + j * o_blk + slot * o_slot + head * o_head + 2 * d
    tl.store(o, lo.to(out.dtype.element_ty, fp_downcast_rounding="rtne"))
    tl.store(o + 1, hi.to(out.dtype.element_ty, fp_downcast_rounding="rtne"))


def gather_dequant_int4(src: torch.Tensor, scale: torch.Tensor, blocks: torch.Tensor, hs: int,
                        dtype: torch.dtype, tag: str, cap: int | None = None) -> torch.Tensor:
    """int4 variant of gather_dequant: src [num_blocks, block_size, nkv, >=hs/2] uint8 view; returns
    [n, block_size, nkv, hs] in the orthonormal rotated domain (the stored RHT has norm sqrt(hs))."""
    n = blocks.numel()
    _, bsz, nkv, _ = src.shape
    out = _buffer(tag, src.device, n, bsz, nkv, hs, dtype, min(src.shape[0], cap or src.shape[0]))
    assert src.stride(-1) == 1 and hs & (hs - 1) == 0
    _int4_dequant_kernel[(n, bsz, nkv)](
        src, scale, blocks, out,
        src.stride(0), src.stride(1), src.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        hs ** -0.5, HALF=hs // 2)
    return out


def gather_dequant(src: torch.Tensor, scale: torch.Tensor, blocks: torch.Tensor, hs: int,
                   dtype: torch.dtype, tag: str, cap: int | None = None) -> torch.Tensor:
    """src: [num_blocks, block_size, nkv, >=hs] int8/fp8 view (any strides, unit last stride);
    scale: [num_blocks, block_size, nkv] fp32 view; blocks: [n] block ids; cap: most blocks a later call can
    need (the block table's size), bounding the buffer's growth steps.
    Returns [n, block_size, nkv, hs] fp16 (a view of a persistent buffer)."""
    n = blocks.numel()
    _, bsz, nkv, _ = src.shape
    out = _buffer(tag, src.device, n, bsz, nkv, hs, dtype, min(src.shape[0], cap or src.shape[0]))
    assert src.stride(-1) == 1
    _pth_dequant_kernel[(n, bsz, nkv)](
        src, scale, blocks, out,
        src.stride(0), src.stride(1), src.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        HS=hs, BLOCK=triton.next_power_of_2(hs))
    return out
