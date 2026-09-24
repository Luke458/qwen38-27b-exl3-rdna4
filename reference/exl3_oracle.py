"""Independent CPU/FP32 oracle for the EXL3 mul1 trellis GEMV path.

Implements, from source semantics only (no exllamav3_ext helpers):

  * packed trellis layout        exl3_dq_rdna.hip.h / exl3_gemv_int8_kernel.cuh
  * 16-bit sliding state windows ("exact mul1 state reconstruction")
  * codebook decode (cb 0/1/2)   codebook_rdna.hip.h, incl. fp16 rounding semantics
  * 128-point Hadamard + scales  hadamard_inner.cuh (had_hf_r_128_inner)
  * full GEMV pipeline           A -> had_in -> dot(W_trellis) -> had_out -> C

Reference semantics notes
-------------------------
* A 16x16 (k x n) trellis tile is a 256-position tail-biting ring in
  ``t``-order (see t_to_rowcol below). Position t's trellis state is the
  16-bit window ending at bit ``(t + 1) * bits`` of the tile's bit stream,
  taken modulo the tile's ``256 * bits`` bits.
* mul1 codebook (cb 2): value = k_inv * fp16(0x6400 + byte_sum(w * M)) + k_bias,
  evaluated as a fused fp16 multiply-add (single rounding). k_inv = 0x1eee,
  k_bias = 0xc931 as fp16 bit patterns.
* The Hadamard transform of had_hf_r_128_inner is exactly
  ``y = r_scale * H128 @ (pre ? x * suh : x)`` followed by ``y *= svh``
  where H128 = H32(xor-FWHT over lanes) (x) H4(local butterfly), i.e. the
  Sylvester Hadamard in (t, s) index layout e = 4 * lane + slot.  We build the
  matrix by running the kernel's exact butterfly on unit vectors.

Everything here is numpy float32/float64 + integer bit manipulation.  Exact
integer quantities (state windows, codebook byte sums) are compared bit-exactly
in the test suite; float quantities are compared under a frozen tolerance
policy (configs/acceptance_policy.json).
"""

from __future__ import annotations

import struct
import numpy as np

# ---------------------------------------------------------------------------
# fp16 bit-pattern constants from codebook_rdna.hip.h
# ---------------------------------------------------------------------------
K_INV_BITS = 0x1EEE
K_BIAS_BITS = 0xC931
MUL1_MULT = 0x83DCD12D
CB0_MULT = 89226354
CB0_ADD = 64248484
MCG_MULT = 0xCBAC1ED if False else 0xCBAC1FED

R_SCALE = 0.088388347648  # 1/sqrt(128) as passed by every call site


def fp16(bits: int) -> float:
    """Exact float value of a 16-bit IEEE-754 binary16 bit pattern."""
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


K_INV = fp16(K_INV_BITS)
K_BIAS = fp16(K_BIAS_BITS)


def to_fp16(x: float) -> float:
    """Round a Python float to the nearest fp16 value (float64 in, fp16 out)."""
    return struct.unpack("<e", struct.pack("<e", x))[0]


# ---------------------------------------------------------------------------
# Codebook decode
# ---------------------------------------------------------------------------
def byte_sum(x: np.ndarray) -> np.ndarray:
    """Sum of the four bytes of each uint32 word (wrapping adds happen first)."""
    x = x.astype(np.uint64) & 0xFFFFFFFF
    return ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)).astype(np.int64)


def codebook_lop3(x: np.ndarray) -> np.ndarray:
    """lop3.b32 with imm 0x6a against constants (0x8fff8fff, 0x3b603b60)."""
    x = x.astype(np.uint32)
    b = np.uint32(0x8FFF8FFF)
    c = np.uint32(0x3B603B60)
    return ((~x) & c) | (x & (b ^ c))


def decode_mul1_fused(w: np.ndarray) -> np.ndarray:
    """Exact mul1 decode as the kernel computes it (values in float64).

    value = f16_fma(fp16(0x6400 + s), K_INV, K_BIAS) where s = byte_sum(w * M).
    The integer 0x6400 + s (s <= 1020) is exactly representable in fp16.
    The fused multiply-add has one rounding at fp16 precision, reproduced here
    by computing exactly in float64 and rounding once.
    """
    w = w.astype(np.uint64) & 0xFFFF
    x = (w * MUL1_MULT) & 0xFFFFFFFF
    s = byte_sum(x)
    h = 1024.0 + s.astype(np.float64)          # exact
    return to_fp16_vec(h * K_INV + K_BIAS)     # single fp16 rounding = fused


def bits_to_fp16(x: np.ndarray) -> np.ndarray:
    """Reinterpret uint16 bit patterns as fp16 values (exact)."""
    return np.ascontiguousarray(x.astype(np.uint16)).view(np.float16).astype(np.float64)


def decode_codebook(w: np.ndarray, cb: int) -> np.ndarray:
    """Decode 16-bit states to codebook values, matching decode_3inst<cb>.

    cb 0 and cb 1 go through half2 reinterpret + fp16 add; cb 2 through the
    fused form.  Returns float64 arrays holding exact fp16 values.
    """
    w = (w.astype(np.uint64) & 0xFFFF)
    if cb == 2:
        return decode_mul1_fused(w)
    if cb == 1:
        x = (w * MCG_MULT) & 0xFFFFFFFF
    elif cb == 0:
        x = (w * CB0_MULT + CB0_ADD) & 0xFFFFFFFF
    else:
        raise ValueError(f"cb must be 0/1/2, got {cb}")
    x = codebook_lop3(x.astype(np.uint32))
    lo = bits_to_fp16(x & 0xFFFF)          # low half of the half2
    hi = bits_to_fp16((x >> 16) & 0xFFFF)  # high half
    return to_fp16_vec(lo + hi)            # __hadd: one fp16 rounding


def to_fp16_vec(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float64)


# ---------------------------------------------------------------------------
# Trellis window extraction (exact integer state reconstruction)
# ---------------------------------------------------------------------------
def t_to_rowcol(t: np.ndarray):
    """Map trellis t-position -> (row, col) of the 16x16 (k x n) tile.

    Derived from the fragment layout documented at
    exl3_gemv_kernel_rdna.hip.h (exl3_gemv_dot_tile_direct): lane L handles
    t = 8L .. 8L+7, with
        frag0[0] = (B[r0  ][cA], B[r0+1][cA])
        frag0[1] = (B[r0+8][cA], B[r0+9][cA])
        frag1[0] = (B[r0  ][cB], B[r0+1][cB])
        frag1[1] = (B[r0+8][cB], B[r0+9][cB])
    r0 = 2*(L%4), cA = 2*(L/8) + ((L>>2)&1), cB = cA + 8.
    """
    t = np.asarray(t, dtype=np.int64)
    L = t >> 3
    i = t & 7
    row = 2 * (L % 4) + (i & 1) + 8 * ((i >> 1) & 1)
    col = 2 * (L >> 3) + ((L >> 2) & 1) + 8 * (i >> 2)
    return row, col


def tile_words(tile_u16: np.ndarray) -> np.ndarray:
    """The tile's uint32 view as the kernel sees it (LE: ptr[i] = u16[2i..2i+1])."""
    tile_u16 = np.ascontiguousarray(tile_u16)
    assert tile_u16.dtype == np.uint16
    return tile_u16.view(np.uint32)


def tile_windows(tile_u16: np.ndarray, bits: int) -> np.ndarray:
    """Extract the 16-bit state window w_t for every t-position of one tile.

    Literal transcription of dq()/fshift in exl3_dq_rdna.hip.h:

        b0 = (t + 257) * bits - 16            (== t*bits + bits - 16 + 256*bits)
        b1 = b0 + 16
        i0 = b0 / 32 ; i1 = (b1 - 1) / 32 ; s0 = (i1 + 1) * 32 - b1
        w  = ((ptr[i0] << 32 | ptr[i1]) >> s0) & 0xffff

    with the word indices taken modulo the tile's bits*8 words (the ring).
    State transitions (verified numerically): the state shifts LEFT and the
    position's transition bits enter at the bottom,

        w_{t+1} == ((w_t << bits) & 0xFFFF) | (w_{t+1} & (2**bits - 1))

    i.e. the low `bits` bits of a state ARE that position's transition (also
    documented in conversion/ngram.py: "the low K bits of position i's 16-bit
    trellis state").  The test suite asserts this as the state-transition
    identity on every fixture.
    """
    assert tile_u16.dtype == np.uint16 and tile_u16.size == 16 * bits
    ptr = tile_words(tile_u16).astype(np.uint64)
    n_words = bits * 8

    t = np.arange(256, dtype=np.int64)
    b0 = (t + 257) * bits - 16
    b1 = b0 + 16
    i0 = (b0 // 32) % n_words
    i1 = ((b1 - 1) // 32) % n_words
    s0 = ((b1 - 1) // 32 + 1) * 32 - b1
    merged = (ptr[i0] << np.uint64(32)) | ptr[i1]
    return ((merged >> s0.astype(np.uint64)) & np.uint64(0xFFFF)).astype(np.uint16)


def decode_tile(tile_u16: np.ndarray, bits: int, cb: int = 2):
    """Decode one packed 16x16 tile -> (values[16,16] float, states[16,16] u16).

    values[r, c] is the dequantized weight B[k = r, n = c] of this tile.
    """
    w = tile_windows(tile_u16, bits)
    vals_t = decode_codebook(w, cb)
    rows, cols = t_to_rowcol(np.arange(256))
    vals = np.zeros((16, 16), dtype=np.float64)
    states = np.zeros((16, 16), dtype=np.uint16)
    vals[rows, cols] = vals_t
    states[rows, cols] = w
    return vals, states


def decode_trellis(trellis: np.ndarray, bits: int, cb: int = 2):
    """Decode a full trellis tensor (K_in/16, N/16, 16*bits u16) -> W (K_in, N)."""
    assert trellis.dtype == np.uint16 and trellis.ndim == 3
    k_tiles, n_tiles, per_tile = trellis.shape
    assert per_tile == 16 * bits, (per_tile, bits)
    W = np.zeros((k_tiles * 16, n_tiles * 16), dtype=np.float64)
    rows, cols = t_to_rowcol(np.arange(256))
    for kt in range(k_tiles):
        for nt in range(n_tiles):
            w = tile_windows(trellis[kt, nt], bits)
            tile = np.zeros((16, 16), dtype=np.float64)
            tile[rows, cols] = decode_codebook(w, cb)
            W[kt * 16:(kt + 1) * 16, nt * 16:(nt + 1) * 16] = tile
    return W


def trellis_states(trellis: np.ndarray, bits: int) -> np.ndarray:
    """State windows arranged as (K_in, N) uint16, matching decode_trellis."""
    k_tiles, n_tiles, _ = trellis.shape
    S = np.zeros((k_tiles * 16, n_tiles * 16), dtype=np.uint16)
    rows, cols = t_to_rowcol(np.arange(256))
    for kt in range(k_tiles):
        for nt in range(n_tiles):
            S[kt * 16:(kt + 1) * 16, nt * 16:(nt + 1) * 16][rows, cols] = tile_windows(trellis[kt, nt], bits)
    return S


# ---------------------------------------------------------------------------
# Hadamard transform (had_hf_r_128_inner)
# ---------------------------------------------------------------------------
def _h4() -> np.ndarray:
    # h0 = s0+s1 = v0+v1+v2+v3 ; h1 = v0-v1+v2-v3 ; h2 = v0+v1-v2-v3 ; h3 = v0-v1-v2+v3
    return np.array([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ], dtype=np.float64)


def had128_matrix() -> np.ndarray:
    """The exact 128x128 matrix computed by had_hf_r_128_inner (without r_scale).

    Layout: element e = 4 * lane + slot.  Local H4 acts on slot; xor-FWHT acts
    on lane: y[l] = sum_l0 (-1)^popcount(l & l0) x[l0].
    """
    H32 = np.empty((32, 32), dtype=np.float64)
    for l in range(32):
        for l0 in range(32):
            H32[l, l0] = -1.0 if bin(l & l0).count("1") % 2 else 1.0
    H4 = _h4()
    H = np.kron(H32, H4)
    return H


_H128 = None


def had128(x: np.ndarray, r_scale: float = R_SCALE) -> np.ndarray:
    """Apply the kernel's 128-point Hadamard to blocks of 128 (float64)."""
    global _H128
    if _H128 is None:
        _H128 = had128_matrix()
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    assert n % 128 == 0
    blocks = x.reshape(-1, 128)
    out = blocks @ _H128.T * r_scale
    return out.reshape(x.shape)


def had_in(a: np.ndarray, suh: np.ndarray, faithful: bool = False) -> np.ndarray:
    """Input stage: A_had = had128(a * suh) (pre-scale then transform).

    faithful=True emulates the kernel's rounding: the pre-scale is an fp16
    __hmul2, the butterfly runs in fp32, and the result is stored fp16.
    """
    p = np.asarray(a, dtype=np.float64) * np.asarray(suh, dtype=np.float64)
    if faithful:
        p = to_fp16_vec(p)
    y = had128(p)
    return to_fp16_vec(y) if faithful else y


def had_out(c_rot: np.ndarray, svh: np.ndarray, output_scale: float = 1.0,
            faithful: bool = False) -> np.ndarray:
    """Output stage: C = had128(c_rot) * svh (transform then post-scale).

    faithful=True emulates the kernel: butterfly in fp32, fp16 store, then an
    fp16 __hmul2 post-scale.
    """
    y = had128(np.asarray(c_rot, dtype=np.float64), R_SCALE * output_scale)
    if faithful:
        y = to_fp16_vec(y)
        return to_fp16_vec(y * np.asarray(svh, dtype=np.float64))
    return y * np.asarray(svh, dtype=np.float64)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def gemv_mul1(
    a: np.ndarray,
    trellis: np.ndarray,
    bits: int,
    suh: np.ndarray,
    svh: np.ndarray,
    cb: int = 2,
    faithful: bool = True,
) -> np.ndarray:
    """Complete EXL3 GEMV for m = 1: y = had_out(had_in(a, suh) @ W, svh).

    faithful=True (default) reproduces the kernel's dtype staging exactly:
    fp16 A and scales, fp16 __hmul2 pre-scale, fp32 butterfly with r_scale,
    fp16 A_had store, fp32 accumulation (order-free here), fp16 C store
    between the dot and output kernels, fp32 butterfly, fp16 store + fp16
    svh post-scale.  Residual deviation vs the GPU is then accumulation-order
    rounding only.

    a: [K_in] fp16; trellis: (K_in/16, N/16, 16*bits) u16; suh: [K_in];
    svh: [N]; returns [N] float64 (values on the fp16 grid when faithful).
    """
    a = to_fp16_vec(np.asarray(a, dtype=np.float64))
    suh = to_fp16_vec(np.asarray(suh, dtype=np.float64))
    svh = to_fp16_vec(np.asarray(svh, dtype=np.float64))
    k_tiles, n_tiles, per_tile = trellis.shape
    K_in = k_tiles * 16
    N = n_tiles * 16
    assert a.shape == (K_in,) and suh.shape == (K_in,) and svh.shape == (N,)
    a_had = had_in(a, suh, faithful=faithful)
    rows, cols = t_to_rowcol(np.arange(256))
    c_rot = np.zeros(N, dtype=np.float64)
    for kt in range(k_tiles):
        a_blk = a_had[kt * 16:(kt + 1) * 16]
        for nt0 in range(0, n_tiles, max(1, 1024 // 16)):
            nt1 = min(n_tiles, nt0 + max(1, 1024 // 16))
            W = np.zeros((16, (nt1 - nt0) * 16), dtype=np.float64)
            for nt in range(nt0, nt1):
                w = tile_windows(trellis[kt, nt], bits)
                tile = np.zeros((16, 16))
                tile[rows, cols] = decode_codebook(w, cb)
                W[:, (nt - nt0) * 16:(nt - nt0 + 1) * 16] = tile
            c_rot[nt0 * 16:nt1 * 16] += a_blk @ W
    if faithful:
        c_rot = to_fp16_vec(c_rot)   # the C buffer is fp16 between the kernels
    return had_out(c_rot, svh, faithful=faithful)


# ---------------------------------------------------------------------------
# int8 activation quantization semantics (for the P4 integer path oracle)
# ---------------------------------------------------------------------------
def quantize_splats(a_had: np.ndarray, q: float, residual: bool = False):
    """gemv_int8_stage_splats semantics: symmetric int8 with clamp [-127, 127].

    Returns (v, v2 or None) integer arrays and the exact int sums (s1, s2).
    """
    a = np.asarray(a_had, dtype=np.float64)
    v = np.clip(np.rint(a / q).astype(np.int64), -127, 127)
    s1 = int(v.sum())
    v2 = None
    s2 = 0
    if residual:
        r = a - q * v.astype(np.float64)
        v2 = np.clip(np.rint(r / (q / 254.0)).astype(np.int64), -127, 127)
        s2 = int(v2.sum())
    return v, v2, s1, s2


def gemv_int8_reference(
    a_had: np.ndarray,
    states: np.ndarray,
    q: float,
    svh: np.ndarray,
    residual: bool = False,
    cb: int = 2,
) -> np.ndarray:
    """Integer-path output for the plain/residual int8 mode (blueprint math).

    states: (K_in, N) uint16 state windows; computes the exact integer
    accumulation then the affine epilogue and output transform.

        y[n] = k_inv * (q * acc1 + q2 * acc2) + (1024*k_inv + k_bias) * suma
             (suma = q*s1 + q2*s2, per row over the whole k range)

    Note: the production "sq" kernel quantizes per k-slice; this reference
    implements the single-q (coop) form with a global q.  Per-slice mode uses
    the same equations with one (q_s, acc_s, suma_s) triple per slice, summed
    in fixed slice order.
    """
    a = np.asarray(a_had, dtype=np.float64)
    v, v2, s1, s2 = quantize_splats(a, q, residual)
    K_in, N = states.shape
    assert a.shape == (K_in,)
    s = byte_sum(((states.astype(np.uint64) * MUL1_MULT) & 0xFFFFFFFF)).astype(np.int64)
    acc1 = v.astype(np.int64) @ s
    q2 = q / 254.0
    acc = q * acc1.astype(np.float64)
    suma = q * s1
    if residual:
        acc = acc + q2 * (v2.astype(np.int64) @ s).astype(np.float64)
        suma += q2 * s2
    corr = (1024.0 * K_INV + K_BIAS) * suma
    y_rot = K_INV * acc + corr
    return had_out(y_rot, svh)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def random_trellis(rng: np.random.Generator, k_tiles: int, n_tiles: int, bits: int) -> np.ndarray:
    """Random packed tiles (uniform over exactly 16*bits bits per tile)."""
    return rng.integers(0, 65536, size=(k_tiles, n_tiles, 16 * bits), dtype=np.uint16)


def state_transition_ok(w: np.ndarray, bits: int) -> bool:
    """Check w_{t+1} == ((w_t << bits) & 0xFFFF) | (w_{t+1} & (2**bits - 1))."""
    w = np.asarray(w, dtype=np.uint32)
    nxt = np.roll(w, -1)
    mask = (1 << bits) - 1
    return bool((((w << bits) & 0xFFFF) | (nxt & mask) == nxt).all())
