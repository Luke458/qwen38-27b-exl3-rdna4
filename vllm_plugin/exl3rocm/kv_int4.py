"""int4_per_token_head KV cache on gfx1201: vLLM's cache format with faster paths and a better quantizer.

The format is vLLM 0.28's `int4_per_token_head` (vllm/v1/attention/ops/int4_per_token_head.py,
Apache-2.0): each (token, KV head) row is rotated with a randomized Hadamard transform (RHT, unnormalized,
H x D x), quantized to asymmetric 4-bit (two nibbles per byte, element 2i in the low nibble), and its fp32
scale carries the 4-bit zero point in its low mantissa bits. What changes here (experiments/0024):

- Rotations are small Triton GEMMs with cached ±1 matrices (exact in fp16, fp32 accumulate). vLLM's fast
  Hadamard kernel stops at head size 128, so for our 256 it ran a PyTorch butterfly (~50 small kernels per
  call, four calls per attention layer per step); torch.mm would load hipBLASLt (+138 MiB).
- Decode and MTP verify use the split-KV ("3D") kernel for up to 8 query tokens, like the rdna4 image's
  int8 path. vLLM's int4 launcher only splits at 1 query token, so an MTP verify step ran on 12 programs
  that each walked the whole context (38 ms per layer at 32k; here 0.3 ms, vs 0.5 ms for int8).
- The attention kernel (adapted from vLLM's `_attn_packed`) re-interleaves the nibbles to natural order
  with the zero point subtracted ((nibble - zp) is an exact small integer in fp16) and does one fp16 dot
  (fp32 accumulate) per product; vLLM's does two fp32 half-width dots, which gfx12 has no matrix
  instructions for. An MTP verify step (4 tokens x 6 query heads per KV head) runs as one 32-row block.
- Long prefill chunks gather + dequantize the sequence's blocks to fp16 in the (orthonormal) rotated
  domain (kv_dequant.gather_dequant_int4) and run vLLM's fp16 kernel with a rotated q, as for int8.
- The cache write centers K and V with calibrated per-layer channel means (exact: see _load_means) and
  picks each row's 4-bit range by a small clip search instead of min/max. Same bytes, ~31% lower KL
  against an fp16 cache on long text (0.0034 -> 0.0023 prose, 0.0039 -> 0.0027 code).
- Exact rows (after BeeLlama.cpp's KVarN sinks and KV precision tail): the first 4 slots of every block
  and each request's newest 128 positions are also kept in fp16 and replace the 4-bit rows in attention
  (~17 + ~45 MB). Decoding 2 x 16k tokens of held-out text: KL 0.0024 -> 0.0009 (prose), 0.0027 -> 0.0011
  (code), top-1 99.1%. EXL3_KV_INT4_SINKS / EXL3_KV_INT4_TAIL set the sizes (0 disables).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.triton_attention_helpers import (cdiv_fn, compute_tile_loop_bounds, find_seq_idx,
                                                            resolve_seq_and_query_len, softmax_step,
                                                            store_segm_reduce_scalars)

# launch config, tuned on gfx1201 with experiments/0024-kv-int4/attn_bench.py. More warps or larger tiles
# spill (the kernel sits at the 256-VGPR limit); more segments helps long contexts, and the partials
# buffer is rows x query heads x segments x head size fp32 (16 x 24 x 64 x 256 x 4 = 25 MB).
_CFG = {"tile_3d": 16, "tile_2d": 32, "warps": 4, "stages": 1, "segments": 64, "rows_3d": 16,
        "clip_steps": int(os.environ.get("EXL3_KV_INT4_CLIP", "3")),
        # exact (fp16) rows, after BeeLlama.cpp's KVarN sinks / KV precision tail: the first SINKS slots of
        # every cache block, and each request's newest TAIL positions (experiments/0025)
        "sinks": int(os.environ.get("EXL3_KV_INT4_SINKS", "4")),
        "tail": int(os.environ.get("EXL3_KV_INT4_TAIL", "128"))}
MAX_QLEN_3D = 8

_mats: dict = {}
_segm: dict = {}
_ones: dict = {}
_last: dict = {}  # last compiled attention kernel (register / spill stats for tuning)
_means: dict | None = None
_bias: dict = {}
_exact: dict = {}  # layer name -> exact-row buffers (see exact_buffers)
# request slots for the tail ring, fed by plugin._install_kv_int4_request_slots (vLLM's V2 model runner):
# slot[i] = persistent request-state index of batch row i (rows past the batch map to the scratch ring row)
_req = {"slot": None, "rows": 0}


def configure_request_slots(max_reqs: int) -> None:
    """The model runner keeps per-request state in max_reqs fixed slots; ring row max_reqs is scratch."""
    _req["rows"] = max_reqs + 1


def _slot_buffer(device) -> torch.Tensor | None:
    if _req["slot"] is None and _req["rows"] > 0:
        _req["slot"] = torch.full((1024,), _req["rows"] - 1, dtype=torch.int32, device=device)
    return _req["slot"]


def set_batch_slots(idx_mapping: torch.Tensor) -> None:
    """Called before each forward (outside graphs): batch row -> request slot, in the persistent buffer that
    captured kernels read. Rows past the batch (graph padding, dummy runs) point at the scratch row."""
    buf = _slot_buffer(idx_mapping.device)
    if buf is None:
        return
    n = idx_mapping.shape[0]
    buf[:n].copy_(idx_mapping)
    buf[n:].fill_(_req["rows"] - 1)


def request_added(idx: int) -> None:
    """A new request took ring row idx: forget the previous request's tail (tags are positions)."""
    for b in _exact.values():
        if b["tag"] is not None:
            b["tag"][idx].fill_(-1)


def exact_buffers(name: str | None, num_blocks: int, heads: int, d: int, device) -> dict | None:
    """fp16 exact rows for layer `name` in the cache's rotated, centered domain:
    sink_k/v [num_blocks, SINKS, heads, d]: the first SINKS slots of every block, read for each sequence's
    first SINKS positions (keyed by physical block, so prefix-cache hits find them);
    ring_k/v [rows, TAIL, heads, d] + tag [rows, TAIL]: each request's newest TAIL positions, tagged with
    the position they hold (MTP rollbacks and prefix-cache hits leave stale tags, which never match).
    Allocated on first (eager) use; None when disabled or first seen during graph capture."""
    if not name or (_CFG["sinks"] <= 0 and _CFG["tail"] <= 0):
        return None
    if name not in _exact:
        if torch.cuda.is_current_stream_capturing():
            return None
        sk = _CFG["sinks"]
        tail = _CFG["tail"] if _slot_buffer(device) is not None else 0
        rows = max(_req["rows"], 1)
        # one allocation per layer (four separately rounded buffers cost ~2x the ~3.6 MB of data)
        sink_n, ring_n = num_blocks * max(sk, 1) * heads * d, rows * max(tail, 1) * heads * d
        slab = torch.zeros(2 * sink_n + 2 * ring_n, dtype=torch.float16, device=device)
        parts = slab.split([sink_n, sink_n, ring_n, ring_n])
        _exact[name] = {
            "sinks": sk, "tail": tail,
            "sink_k": parts[0].view(num_blocks, max(sk, 1), heads, d),
            "sink_v": parts[1].view(num_blocks, max(sk, 1), heads, d),
            "ring_k": parts[2].view(rows, max(tail, 1), heads, d),
            "ring_v": parts[3].view(rows, max(tail, 1), heads, d),
            "tag": torch.full((rows, max(tail, 1)), -1, dtype=torch.int32, device=device) if tail else None}
    return _exact[name]


def _load_means() -> dict:
    """Per-layer K/V channel means {layer_name: {"k": [kv_heads, D], "v": ...}} for centering, from
    EXL3_KV_MEANS (a path, or 0 to disable) or the file packaged with the plugin (Qwen3.8-27B EXL3,
    tools/calib_kv_means.py). Centering is exact for any means (softmax ignores a per-query constant on the
    keys; the value mean is added back to the output); good means only make the 4-bit ranges tighter."""
    global _means
    if _means is None:
        path = os.environ.get("EXL3_KV_MEANS", os.path.join(os.path.dirname(__file__), "kv_means.pt"))
        _means = torch.load(path, map_location="cpu") if path != "0" and os.path.exists(path) else {}
    return _means


def layer_bias(name: str | None, kv_heads: int, d: int, device) -> dict | None:
    """Rotated-domain write biases -(mean @ F) for K and V [kv_heads, D] and the value mean to add back to
    the output (per query head: see attention). None without means for this layer. First call per layer
    must be eager (it copies to the device)."""
    if not name:
        return None
    key = (name, str(device))
    if key not in _bias:
        m = _load_means().get(name)
        if m is None or tuple(m["k"].shape) != (kv_heads, d):
            _bias[key] = None
        else:
            f = rht_matrices(d, device)["fwd"].float()
            mk, mv = m["k"].to(device, torch.float32), m["v"].to(device, torch.float32)
            _bias[key] = {"k": -(mk @ f), "v": -(mv @ f), "v_mean": mv, "out": {}}
    return _bias[key]


def _out_bias(bias: dict | None, hq: int) -> torch.Tensor | None:
    """Value mean per query head [hq, D] (GQA: query head h reads KV head h // (hq / kv_heads))."""
    if bias is None:
        return None
    if hq not in bias["out"]:
        mv = bias["v_mean"]
        bias["out"][hq] = mv.repeat_interleave(hq // mv.shape[0], dim=0).contiguous()
    return bias["out"][hq]


def rht_matrices(d: int, device) -> dict:
    """fp16 matrices for row vectors: x @ fwd == vLLM single_rht(x); inv_d = (H D) / d un-rotates the
    decode output; fwd_n / inv_n are the orthonormal versions (/ sqrt(d)) for the prefill path. All
    entries are ±2^-k, exact in fp16. First call must be outside graph capture (signs are built on the CPU)."""
    key = (d, str(device))
    if key not in _mats:
        from vllm.v1.attention.ops.int4_per_token_head import _get_hadamard_matrix, _get_rht_signs
        s = _get_rht_signs(d, 0, device).float()
        h = _get_hadamard_matrix(d, torch.float32, device)
        fwd, inv = s[:, None] * h, h * s[None, :]
        _mats[key] = {"fwd": fwd.half(), "inv_d": (inv / d).half(),
                      "fwd_n": (fwd * d ** -0.5).half(), "inv_n": (inv * d ** -0.5).half()}
    return _mats[key]


def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, *, k_scale_cache, v_scale_cache,
                      layer_name: str | None = None, query_start_loc=None, seq_lens=None):
    """Rotate + quantize + pack K and V into the cache (vLLM's reshape_and_cache_int4 format; ranges from the
    fp32 rotation with a clip search), centered with the layer's calibrated means if any. Two launches for
    both sides: a GEMM-style rotation into an fp32 buffer, then one program per row for the quantization
    (a fused version ran one program for a decode step's 4 rows: 56 us per side). With the layer's exact
    buffers, the fp32 rotation is also stored in fp16 for sink slots and (given the batch's query_start_loc /
    seq_lens) for tokens among their request's newest TAIL positions."""
    d = key.shape[-1]
    f = rht_matrices(d, key.device)["fwd"]
    n = min(key.shape[0], slot_mapping.shape[0])
    heads = key.shape[1]
    rows = n * heads
    if rows == 0:
        return
    bias = layer_bias(layer_name, heads, d, key.device)
    y = torch.empty((2, rows, d), dtype=torch.float32, device=key.device)
    _rotate_kv[(triton.cdiv(rows, 16), 2)](
        key, value, f, bias["k"] if bias else f, bias["v"] if bias else f, y, slot_mapping, rows, heads,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        D=d, BLOCK_R=16, BLOCK_K=32, HAS_BIAS=bias is not None, num_warps=4)
    steps = _CFG["clip_steps"]
    ex = exact_buffers(layer_name, key_cache.shape[0], heads, d, key.device)
    sinks = ex["sinks"] if ex else 0
    tail = ex["tail"] if ex is not None and query_start_loc is not None else 0
    dummy = y
    _quant_pack[(rows, 2)](
        y, key_cache, value_cache, k_scale_cache, v_scale_cache, slot_mapping, rows, heads,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2),
        v_scale_cache.stride(0), v_scale_cache.stride(1), v_scale_cache.stride(2),
        ex["sink_k"] if sinks else dummy, ex["sink_v"] if sinks else dummy,
        ex["ring_k"] if tail else dummy, ex["ring_v"] if tail else dummy, ex["tag"] if tail else dummy,
        _req["slot"] if tail else dummy, query_start_loc if tail else dummy, seq_lens if tail else dummy,
        seq_lens.shape[0] if tail else 0,
        BLOCK_SIZE=key_cache.shape[1], D=d, CLIP_STEPS=steps, CLIP_STEP=0.075,
        NC=triton.next_power_of_2(steps * steps), SINKS=sinks, TAIL=tail, num_warps=4)


@triton.jit
def _rotate(x_ptr, m_ptr, b_ptr, y_ptr, rows, heads, x_s0, x_s1, y_s0, y_s1,
            D: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr, HAS_BIAS: tl.constexpr):
    """y[t, h, :] = x[t, h, :] @ M (+ b[h, :]); rows = tokens * heads, fp16 in/out, fp32 accumulate."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < rows
    t, h = r // heads, r % heads
    offs_n = tl.arange(0, D)
    acc = tl.zeros([BLOCK_R, D], dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + t[:, None] * x_s0 + h[:, None] * x_s1 + offs_k[None, :], mask=rm[:, None], other=0.0)
        acc += tl.dot(x.to(tl.float16), tl.load(m_ptr + offs_k[:, None] * D + offs_n[None, :]))
    if HAS_BIAS:
        acc += tl.load(b_ptr + h[:, None] * D + offs_n[None, :], mask=rm[:, None], other=0.0)
    tl.store(y_ptr + t[:, None] * y_s0 + h[:, None] * y_s1 + offs_n[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=rm[:, None])


def rotate(x: torch.Tensor, m: torch.Tensor, out: torch.Tensor | None = None,
           bias: torch.Tensor | None = None) -> torch.Tensor:
    """x [tokens, heads, D] @ m (+ bias [heads, D] fp32) -> out (same shape; any strides with unit last)."""
    t, h, d = x.shape
    out = torch.empty_like(x) if out is None else out
    br = 16
    _rotate[(triton.cdiv(t * h, br),)](x, m, bias if bias is not None else m, out, t * h, h, x.stride(0),
                                        x.stride(1), out.stride(0), out.stride(1), D=d, BLOCK_R=br, BLOCK_K=32,
                                        HAS_BIAS=bias is not None, num_warps=4)
    return out


@triton.jit
def _rotate_kv(k_ptr, v_ptr, m_ptr, kb_ptr, vb_ptr, y_ptr, slot_ptr, rows, heads, k_s0, k_s1, v_s0, v_s1,
               D: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Cache write, step 1: y[side, r] = x[t, h] @ F (+ bias[h]) in fp32 for K (side 0) and V (side 1)."""
    side = tl.program_id(1)
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    t, h = r // heads, r % heads
    rm = (r < rows) & (tl.load(slot_ptr + t, mask=r < rows, other=-1) >= 0)
    if side == 0:
        x_row = k_ptr + t * k_s0 + h * k_s1
        b_ptr = kb_ptr
    else:
        x_row = v_ptr + t * v_s0 + h * v_s1
        b_ptr = vb_ptr
    offs_n = tl.arange(0, D)
    acc = tl.zeros([BLOCK_R, D], dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_row[:, None] + offs_k[None, :], mask=rm[:, None], other=0.0)
        acc += tl.dot(x.to(tl.float16), tl.load(m_ptr + offs_k[:, None] * D + offs_n[None, :]))
    if HAS_BIAS:
        acc += tl.load(b_ptr + h[:, None] * D + offs_n[None, :], mask=rm[:, None], other=0.0)
    tl.store(y_ptr + side * rows * D + r[:, None] * D + offs_n[None, :], acc, mask=rm[:, None])


@triton.jit
def _quant_pack(y_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, slot_ptr, rows, heads,
                kc_blk, kc_slot, kc_head, vc_blk, vc_slot, vc_head, ks_blk, ks_slot, ks_head, vs_blk, vs_slot, vs_head,
                sk_ptr, sv_ptr, rk_ptr, rv_ptr, tag_ptr, rslot_ptr, qsl_ptr, seqlen_ptr, num_seqs,
                BLOCK_SIZE: tl.constexpr, D: tl.constexpr, CLIP_STEPS: tl.constexpr, CLIP_STEP: tl.constexpr,
                NC: tl.constexpr, SINKS: tl.constexpr, TAIL: tl.constexpr):
    """Cache write, step 2, one (row, side) per program: asymmetric 4-bit with vLLM's rounding (half away
    from zero), two nibbles per byte (element 2i low), zero point in the fp32 scale's low mantissa bits.
    Clip search: the row's range [lo, hi] is shrunk toward the row mean by 0, CLIP_STEP, ... at each end
    independently (CLIP_STEPS^2 candidates, evaluated side by side) and the lowest-MSE pair is kept.
    3 x 7.5% + centering: 17% lower attention error than vLLM's min/max (experiments/0024)."""
    r = tl.program_id(0)
    side = tl.program_id(1)
    t, h = r // heads, r % heads
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    offs = tl.arange(0, D)
    y = tl.load(y_ptr + side * rows * D + r * D + offs)
    lo0 = tl.min(y, axis=0)
    hi0 = tl.max(y, axis=0)
    mu = tl.sum(y, axis=0) / D
    c = tl.arange(0, NC)
    lo = mu + (lo0 - mu) * (1.0 - CLIP_STEP * (c // CLIP_STEPS))
    hi = mu + (hi0 - mu) * (1.0 - CLIP_STEP * (c % CLIP_STEPS))
    sc = tl.maximum((hi - lo) / 15.0, 1e-6)
    zr = -lo / sc
    zp = tl.minimum(tl.maximum(tl.where(zr >= 0, zr + 0.5, zr - 0.5).to(tl.int32), 0), 15)
    v = y[None, :] * (1.0 / sc)[:, None] + zp.to(tl.float32)[:, None]
    qn = tl.minimum(tl.maximum(tl.where(v >= 0, v + 0.5, v - 0.5).to(tl.int32), 0), 15)
    e = (qn - zp[:, None]).to(tl.float32) * sc[:, None] - y[None, :]
    err = tl.where(c < CLIP_STEPS * CLIP_STEPS, tl.sum(e * e, axis=1), float("inf"))
    best = tl.argmin(err, axis=0)
    sc_b = tl.sum(tl.where(c == best, sc, 0.0), axis=0)
    zp_b = tl.sum(tl.where(c == best, zp, 0), axis=0)
    vb = y * (1.0 / sc_b) + zp_b.to(tl.float32)
    qb = tl.minimum(tl.maximum(tl.where(vb >= 0, vb + 0.5, vb - 0.5).to(tl.int32), 0), 15)
    ev, od = tl.split(tl.reshape(qb, (D // 2, 2)))
    packed = (ev | (od << 4)).to(tl.uint8)
    blk, sl = slot // BLOCK_SIZE, slot % BLOCK_SIZE
    sbits = ((sc_b.to(tl.int32, bitcast=True) & -16) | zp_b).to(tl.float32, bitcast=True)
    offs_b = tl.arange(0, D // 2)
    if side == 0:
        tl.store(kc_ptr + blk * kc_blk + sl * kc_slot + h * kc_head + offs_b, packed)
        tl.store(ks_ptr + blk * ks_blk + sl * ks_slot + h * ks_head, sbits)
    else:
        tl.store(vc_ptr + blk * vc_blk + sl * vc_slot + h * vc_head + offs_b, packed)
        tl.store(vs_ptr + blk * vs_blk + sl * vs_slot + h * vs_head, sbits)
    if SINKS > 0:  # exact copy of every block's first SINKS slots
        if sl < SINKS:
            e_off = ((blk * SINKS + sl) * heads + h) * D + offs
            if side == 0:
                tl.store(sk_ptr + e_off, y.to(tl.float16))
            else:
                tl.store(sv_ptr + e_off, y.to(tl.float16))
    if TAIL > 0:  # exact copy of the request's newest TAIL positions, in a ring tagged with the position
        row = find_seq_idx(qsl_ptr, t, num_seqs, 1, False)
        q0 = tl.load(qsl_ptr + row)
        q1 = tl.load(qsl_ptr + row + 1)
        seq_len = tl.load(seqlen_ptr + row)
        pos = seq_len - (q1 - q0) + (t - q0)
        if pos >= seq_len - TAIL:
            rs = tl.load(rslot_ptr + row).to(tl.int64)
            ri = pos % TAIL
            r_off = ((rs * TAIL + ri) * heads + h) * D + offs
            if side == 0:
                tl.store(rk_ptr + r_off, y.to(tl.float16))
                if h == 0:
                    tl.store(tag_ptr + rs * TAIL + ri, pos)
            else:
                tl.store(rv_ptr + r_off, y.to(tl.float16))


@triton.jit
def _attn_tile(j, q, m_i, l_i, acc, q_mask, query_abs, offs_t, offs_h, offs_d, max_prefix, bt_row,
               kc_ptr, vc_ptr, ks_ptr, vs_ptr, kv_head_idx, scale,
               k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, ks_s0, ks_s1, ks_s2, vs_s0, vs_s1, vs_s2,
               sk_ptr, sv_ptr, rk_ptr, rv_ptr, tag_ptr, rs, tail_lo,
               BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr, HALF: tl.constexpr, HKV: tl.constexpr,
               SINKS: tl.constexpr, TAIL: tl.constexpr):
    """One TILE_SIZE-token step of _attn_int4's online softmax. SINKS > 0: positions < SINKS read the exact
    sink rows (the sequence's first block); TAIL > 0: positions >= tail_lo whose ring tag matches read the
    exact ring rows. Exact rows are fp16 in the same rotated domain with scale 1."""
    D2: tl.constexpr = 2 * HALF
    seq_off = j * TILE_SIZE + offs_t
    tmask = seq_off < max_prefix
    blk = tl.load(bt_row + seq_off // BLOCK_SIZE).to(tl.int64)
    slot = (seq_off % BLOCK_SIZE).to(tl.int64)
    ksb = tl.load(ks_ptr + blk * ks_s0 + slot * ks_s1 + kv_head_idx * ks_s2, mask=tmask, other=0).to(
        tl.int32, bitcast=True)
    vsb = tl.load(vs_ptr + blk * vs_s0 + slot * vs_s1 + kv_head_idx * vs_s2, mask=tmask, other=0).to(
        tl.int32, bitcast=True)
    # K bytes [TILE, HALF] -> [TILE, 2*HALF] in natural order (low nibble = even element), minus zero point
    kp = tl.load(kc_ptr + blk[:, None] * k_s0 + slot[:, None] * k_s1 + kv_head_idx * k_s2 + offs_h[None, :],
                 mask=tmask[:, None], other=0).to(tl.int32)
    kz = (ksb & 0xF)[:, None]
    k = tl.reshape(tl.join((kp & 0xF) - kz, ((kp >> 4) & 0xF) - kz), (TILE_SIZE, D2)).to(tl.float16)
    k_sc = (ksb & -16).to(tl.float32, bitcast=True)
    if SINKS > 0:
        is_x = (seq_off < SINKS) & tmask
        x_off = ((blk * SINKS + slot) * HKV + kv_head_idx) * D2
        k = tl.where(is_x[:, None], tl.load(sk_ptr + x_off[:, None] + offs_d[None, :], mask=is_x[:, None],
                                            other=0.0), k)
        k_sc = tl.where(is_x, 1.0, k_sc)
    if TAIL > 0:
        ri = seq_off % TAIL
        is_t = tl.load(tag_ptr + rs * TAIL + ri, mask=tmask & (seq_off >= tail_lo), other=-1) == seq_off
        t_off = ((rs * TAIL + ri) * HKV + kv_head_idx) * D2
        k = tl.where(is_t[:, None], tl.load(rk_ptr + t_off[:, None] + offs_d[None, :], mask=is_t[:, None],
                                            other=0.0), k)
        k_sc = tl.where(is_t, 1.0, k_sc)
    s = tl.dot(q, tl.trans(k)) * (scale * k_sc[None, :])
    s = tl.where(q_mask & (seq_off[None, :] <= query_abs), s, float("-inf"))
    m_i, l_i, p, alpha = softmax_step(s, m_i, l_i)
    vp = tl.load(vc_ptr + blk[:, None] * v_s0 + slot[:, None] * v_s1 + kv_head_idx * v_s2 + offs_h[None, :],
                 mask=tmask[:, None], other=0).to(tl.int32)
    vz = (vsb & 0xF)[:, None]
    v = tl.reshape(tl.join((vp & 0xF) - vz, ((vp >> 4) & 0xF) - vz), (TILE_SIZE, D2)).to(tl.float16)
    v_sc = (vsb & -16).to(tl.float32, bitcast=True)
    if SINKS > 0:
        v = tl.where(is_x[:, None], tl.load(sv_ptr + x_off[:, None] + offs_d[None, :], mask=is_x[:, None],
                                            other=0.0), v)
        v_sc = tl.where(is_x, 1.0, v_sc)
    if TAIL > 0:
        v = tl.where(is_t[:, None], tl.load(rv_ptr + t_off[:, None] + offs_d[None, :], mask=is_t[:, None],
                                            other=0.0), v)
        v_sc = tl.where(is_t, 1.0, v_sc)
    pv = (p * v_sc[None, :]).to(tl.float16)
    acc = acc * alpha[:, None] + tl.dot(pv, v)
    return m_i, l_i, acc


@triton.jit
def _attn_int4(out_ptr, segm_out_ptr, segm_max_ptr, segm_sum_ptr, q_ptr, kc_ptr, vc_ptr, bt_ptr, seq_lens_ptr,
               qsl_ptr, ks_ptr, vs_ptr, scale, num_seqs, sk_ptr, sv_ptr, rk_ptr, rv_ptr, tag_ptr, rslot_ptr,
               bt_stride: tl.int64, q_s0: tl.int64, q_s1: tl.int64, o_s0: tl.int64, o_s1: tl.int64,
               k_s0: tl.int64, k_s1: tl.int64, k_s2: tl.int64, v_s0: tl.int64, v_s1: tl.int64, v_s2: tl.int64,
               ks_s0: tl.int64, ks_s1: tl.int64, ks_s2: tl.int64, vs_s0: tl.int64, vs_s1: tl.int64,
               vs_s2: tl.int64,
               num_query_heads: tl.constexpr, num_queries_per_kv: tl.constexpr, BLOCK_SIZE: tl.constexpr,
               TILE_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr, HALF: tl.constexpr,
               BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr, NUM_SEGMENTS: tl.constexpr, IS_3D: tl.constexpr,
               SINKS: tl.constexpr = 0, TAIL: tl.constexpr = 0):
    """Causal paged attention over the int4 cache. q is RHT-rotated (and the scale folded in by the caller);
    the output stays rotated. 3D: per-segment partials for vLLM's reduce_segments. Exact rows (fp16, same
    rotated domain, scale 1) replace the 4-bit ones for the sequence's first SINKS positions and its newest
    TAIL positions whose ring tag matches (see _attn_tile)."""
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0
    seq_idx, q_block_local_idx, q_start, q_len, seq_len = resolve_seq_and_query_len(
        qsl_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q)
    if q_block_local_idx * BLOCK_Q >= q_len:
        return
    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, TILE_SIZE)
    offs_h = tl.arange(0, HALF)
    offs_d = tl.arange(0, 2 * HALF)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    qo0 = q_start + query_pos
    qo1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    qm0 = query_pos < q_len
    qm1 = qo1 < num_query_heads
    q_mask = qm0[:, None] & qm1[:, None]
    q = tl.load(q_ptr + qo0[:, None] * q_s0 + qo1[:, None] * q_s1 + offs_d[None, :], mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, 2 * HALF], dtype=tl.float32)
    context_len = seq_len - q_len
    loop_lo, loop_hi, max_prefix = compute_tile_loop_bounds(
        context_len, seq_len, q_len, q_block_local_idx, segm_idx, tiles_per_segment, TILE_SIZE, BLOCK_M, BLOCK_Q,
        num_queries_per_kv, 0, False, IS_3D)
    bt_row = bt_ptr + seq_idx * bt_stride
    query_abs = context_len + query_pos[:, None]
    HKV: tl.constexpr = num_query_heads // num_queries_per_kv
    tail_lo = seq_len - TAIL
    if TAIL > 0:
        rs = tl.load(rslot_ptr + seq_idx).to(tl.int64)
    else:
        rs = 0

    # three passes so the common tiles run the plain 4-bit body (substituting exact rows inside every tile
    # pushed the kernel to 256 VGPRs with spills: +55% at 32k): tile 0 with the sinks, the body, and the
    # tiles overlapping the tail window (with the sinks too when the sequence is that short)
    tail_start = tl.maximum(tail_lo // TILE_SIZE, 0) if TAIL > 0 else loop_hi
    j_lo = loop_lo
    if SINKS > 0:
        if (loop_lo == 0) & (loop_hi > 0) & (tail_start > 0):
            m_i, l_i, acc = _attn_tile(0, q, m_i, l_i, acc, q_mask, query_abs, offs_t, offs_h, offs_d, max_prefix,
                                       bt_row, kc_ptr, vc_ptr, ks_ptr, vs_ptr, kv_head_idx, scale,
                                       k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, ks_s0, ks_s1, ks_s2, vs_s0, vs_s1, vs_s2,
                                       sk_ptr, sv_ptr, rk_ptr, rv_ptr, tag_ptr, rs, tail_lo,
                                       BLOCK_SIZE, TILE_SIZE, HALF, HKV, SINKS, 0)
            j_lo = 1
    for j in range(j_lo, tl.minimum(loop_hi, tail_start)):
        m_i, l_i, acc = _attn_tile(j, q, m_i, l_i, acc, q_mask, query_abs, offs_t, offs_h, offs_d, max_prefix,
                                   bt_row, kc_ptr, vc_ptr, ks_ptr, vs_ptr, kv_head_idx, scale,
                                   k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, ks_s0, ks_s1, ks_s2, vs_s0, vs_s1, vs_s2,
                                   sk_ptr, sv_ptr, rk_ptr, rv_ptr, tag_ptr, rs, tail_lo,
                                   BLOCK_SIZE, TILE_SIZE, HALF, HKV, 0, 0)
    if TAIL > 0:
        for j in range(tl.maximum(j_lo, tail_start), loop_hi):
            m_i, l_i, acc = _attn_tile(j, q, m_i, l_i, acc, q_mask, query_abs, offs_t, offs_h, offs_d, max_prefix,
                                       bt_row, kc_ptr, vc_ptr, ks_ptr, vs_ptr, kv_head_idx, scale,
                                       k_s0, k_s1, k_s2, v_s0, v_s1, v_s2, ks_s0, ks_s1, ks_s2, vs_s0, vs_s1,
                                       vs_s2, sk_ptr, sv_ptr, rk_ptr, rv_ptr, tag_ptr, rs, tail_lo,
                                       BLOCK_SIZE, TILE_SIZE, HALF, HKV, SINKS, TAIL)

    if IS_3D:
        base = (qo0[:, None].to(tl.int64) * (num_query_heads * NUM_SEGMENTS * HEAD_SIZE_PADDED)
                + qo1[:, None] * (NUM_SEGMENTS * HEAD_SIZE_PADDED) + segm_idx * HEAD_SIZE_PADDED + offs_d[None, :])
        tl.store(segm_out_ptr + base, acc, mask=q_mask)
        store_segm_reduce_scalars(segm_max_ptr, segm_sum_ptr, qo0, qo1, segm_idx, m_i, l_i, qm0, qm1,
                                  num_query_heads, NUM_SEGMENTS)
    else:
        tl.store(out_ptr + qo0[:, None] * o_s0 + qo1[:, None] * o_s1 + offs_d[None, :],
                 (acc / l_i[:, None]).to(out_ptr.dtype.element_ty), mask=q_mask)


def supported(kw) -> bool:
    """Features the kernel leaves out (the model uses none of them): the caller falls back to vLLM."""
    return (kw.get("alibi_slopes") is None and kw.get("sinks") is None and not kw.get("softcap")
            and kw.get("qq_bias") is None and kw.get("mm_prefix_range") is None and kw.get("output_scale") is None
            and tuple(kw.get("window_size", (-1, -1)))[0] < 0 and kw.get("causal") is True
            and kw["q"].dtype == torch.float16 and kw["q"].shape[-1] % 32 == 0)


def _segm_buffers(rows, hq, segs, dpad, device):
    key = (rows, hq, segs, dpad, str(device))
    if key not in _segm:
        _segm[key] = (torch.empty((rows, hq, segs, dpad), dtype=torch.float32, device=device),
                      torch.empty((rows, hq, segs), dtype=torch.float32, device=device),
                      torch.empty((rows, hq, segs), dtype=torch.float32, device=device))
    return _segm[key]


def attention(*, q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, block_table, softmax_scale,
              k_scale_cache, v_scale_cache, tile_size: int | None = None, num_warps: int | None = None,
              segments: int | None = None, block_m: int | None = None, num_stages: int | None = None,
              layer_name: str | None = None, **_):
    """Paged attention over the int4 cache (module docstring); writes the un-rotated result to out.
    Decode / MTP verify (q_len <= 8, <= rows_3d query rows in the batch) use split-KV with our own partials
    buffers; the keyword overrides are for tuning."""
    from vllm.v1.attention.ops.triton_unified_attention import reduce_segments
    d = q.shape[-1]
    mats = rht_matrices(d, q.device)
    qr = rotate(q, mats["fwd"])
    num_seqs = seqused_k.shape[0]
    hq, hkv = q.shape[1], k.shape[2]
    nq = hq // hkv
    use_3d = max_seqlen_q <= MAX_QLEN_3D and q.shape[0] <= _CFG["rows_3d"]
    # one q block per sequence when it fits: an MTP verify step (4 tokens x 6 heads) as one 32-row block
    block_m = block_m or max(16, triton.next_power_of_2(nq * max_seqlen_q if use_3d else nq))
    block_m = min(max(block_m, triton.next_power_of_2(nq)), 32 if use_3d else 16)
    block_q = block_m // nq
    segs = (segments or _CFG["segments"]) if use_3d else 1
    dpad = triton.next_power_of_2(d)
    so, sm, ss = _segm_buffers(_CFG["rows_3d"], hq, segs, dpad, q.device) if use_3d else (None,) * 3
    tile = tile_size or (_CFG["tile_3d"] if use_3d else _CFG["tile_2d"])
    rot = torch.empty_like(q)
    grid = (q.shape[0] // block_q + num_seqs, hkv) + ((segs,) if use_3d else ())
    ex = _exact.get(layer_name) if layer_name else None
    sinks, tail = (ex["sinks"], ex["tail"]) if ex else (0, 0)
    _last["kernel"] = _attn_int4[grid](
        rot, so if use_3d else rot, sm if use_3d else rot, ss if use_3d else rot, qr, k, v, block_table, seqused_k,
        cu_seqlens_q, k_scale_cache, v_scale_cache, softmax_scale / d, num_seqs,
        ex["sink_k"] if sinks else rot, ex["sink_v"] if sinks else rot, ex["ring_k"] if tail else rot,
        ex["ring_v"] if tail else rot, ex["tag"] if tail else rot, _req["slot"] if tail else rot,
        block_table.stride(0), qr.stride(0), qr.stride(1), rot.stride(0), rot.stride(1),
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2),
        v_scale_cache.stride(0), v_scale_cache.stride(1), v_scale_cache.stride(2),
        num_query_heads=hq, num_queries_per_kv=nq, BLOCK_SIZE=v.shape[1], TILE_SIZE=tile, HEAD_SIZE_PADDED=dpad,
        HALF=d // 2, BLOCK_Q=block_q, BLOCK_M=block_m, NUM_SEGMENTS=segs, IS_3D=use_3d, SINKS=sinks, TAIL=tail,
        num_warps=num_warps or _CFG["warps"], num_stages=num_stages or _CFG["stages"])
    if use_3d:
        reduce_segments[(q.shape[0], hq)](
            output_ptr=rot, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=ss, seq_lens_ptr=seqused_k,
            num_seqs=num_seqs, num_query_heads=hq, out_scale_inv=1.0, output_stride_0=rot.stride(0),
            output_stride_1=rot.stride(1), block_table_stride=block_table.stride(0), TILE_SIZE=tile,
            HEAD_SIZE=d, HEAD_SIZE_PADDED=dpad, query_start_len_ptr=cu_seqlens_q, BLOCK_Q=block_q,
            NUM_SEGMENTS_PER_SEQ=segs, USE_FP8=False)
    # un-rotate: out = rot @ (H D) / d (the unnormalized RHT scaled q and v by sqrt(d) each), + value mean
    rotate(rot, mats["inv_d"], out=out, bias=_out_bias(layer_bias(layer_name, hkv, d, q.device), hq))


@triton.jit
def _tail_overlay(kd_ptr, vd_ptr, rk_ptr, rv_ptr, tag_ptr, rslot_ptr, seqlen_ptr, nb, mul,
                  BLOCK_SIZE: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, TAIL: tl.constexpr):
    """Prefill copy (gather_dequant_int4 layout [seq * nb + block, slot, head, D]): overwrite each sequence's
    newest TAIL positions with the exact ring rows whose tag matches, scaled to the orthonormal domain."""
    s = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    seq_len = tl.load(seqlen_ptr + s)
    pos = seq_len - TAIL + i
    if pos < 0:
        return
    rs = tl.load(rslot_ptr + s).to(tl.int64)
    ri = pos % TAIL
    if tl.load(tag_ptr + rs * TAIL + ri) != pos:
        return
    offs = tl.arange(0, D)
    src = ((rs * TAIL + ri) * NKV + h) * D + offs
    j = (s * nb + pos // BLOCK_SIZE).to(tl.int64)
    dst = ((j * BLOCK_SIZE + pos % BLOCK_SIZE) * NKV + h) * D + offs
    tl.store(kd_ptr + dst, (tl.load(rk_ptr + src).to(tl.float32) * mul).to(kd_ptr.dtype.element_ty))
    tl.store(vd_ptr + dst, (tl.load(rv_ptr + src).to(tl.float32) * mul).to(vd_ptr.dtype.element_ty))


def prefill(orig, kw, layer_name: str | None = None):
    """Long prefill chunk: dequantize the sequences' used blocks to fp16 in the orthonormal rotated domain,
    run vLLM's fp16 kernel (orig) with a rotated q, un-rotate the output. q.k and p.v are unchanged."""
    from vllm.v1.kv_cache_interface import KVQuantMode

    from .kv_dequant import gather_dequant_int4
    q, k, v, bt, out = kw["q"], kw["k"], kw["v"], kw["block_table"], kw["out"]
    d = q.shape[-1]
    mats = rht_matrices(d, q.device)
    bsz = k.shape[1]
    nb = (int(kw["max_seqlen_k"]) + bsz - 1) // bsz
    used = bt[:, :nb]
    flat = used.reshape(-1)
    ex = _exact.get(layer_name) if layer_name else None
    kd = gather_dequant_int4(k, kw["k_scale_cache"], flat, d, q.dtype, "k", cap=bt.numel(),
                             sinks=ex["sink_k"] if ex and ex["sinks"] else None, nb=nb)
    vd = gather_dequant_int4(v, kw["v_scale_cache"], flat, d, q.dtype, "v", cap=bt.numel(),
                             sinks=ex["sink_v"] if ex and ex["sinks"] else None, nb=nb)
    if ex and ex["tail"]:
        seq_lens = kw["seqused_k"]
        _tail_overlay[(seq_lens.shape[0], ex["tail"], k.shape[2])](
            kd, vd, ex["ring_k"], ex["ring_v"], ex["tag"], _req["slot"], seq_lens, nb, d ** -0.5,
            BLOCK_SIZE=bsz, NKV=k.shape[2], D=d, TAIL=ex["tail"])
    if q.device not in _ones:
        _ones[q.device] = torch.ones((1, 1), dtype=torch.float32, device=q.device)
    desc = _ones[q.device].expand(bt.shape[0], kd.shape[2])
    rot = torch.empty_like(q)
    orig(**dict(kw, q=rotate(q, mats["fwd_n"]), k=kd, v=vd, out=rot, kv_quant_mode=KVQuantMode.NONE,
                k_scale_cache=None, v_scale_cache=None, k_descale=desc, v_descale=desc,
                block_table=torch.arange(flat.numel(), device=bt.device, dtype=bt.dtype).view(used.shape)))
    rotate(rot, mats["inv_n"], out=out, bias=_out_bias(layer_bias(layer_name, k.shape[2], d, q.device), q.shape[1]))
