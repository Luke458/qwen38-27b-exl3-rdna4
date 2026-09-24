"""CPU tests for the independent oracle (no GPU required).

Covers: trellis window extraction (bitfield exactness), state transitions,
mul1 codebook decode exactness, transform normalization, and pipeline identity.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
import exl3_oracle as O  # noqa: E402


RNG = np.random.default_rng(20260923)


class TestTrellisWindows:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_state_transition_identity(self, bits):
        for _ in range(4):
            tile = RNG.integers(0, 65536, size=16 * bits, dtype=np.uint16)
            w = O.tile_windows(tile, bits)
            assert O.state_transition_ok(w, bits), "state transition identity violated"

    @pytest.mark.parametrize("bits", [3, 4])
    def test_boundary_tiles(self, bits):
        tiles = [
            np.zeros(16 * bits, dtype=np.uint16),
            np.full(16 * bits, 0xFFFF, dtype=np.uint16),
            np.arange(16 * bits, dtype=np.uint16),
            (np.arange(16 * bits, dtype=np.uint16) * 7919) & 0xFFFF,
        ]
        for tile in tiles:
            w = O.tile_windows(tile, bits)
            assert w.shape == (256,)
            assert O.state_transition_ok(w, bits)

    def test_transition_bits_are_low_bits(self):
        """The low `bits` bits of a state are that position's transition
        (conversion/ngram.py: 'the low K bits of position i's 16-bit state')."""
        bits = 3
        tile = RNG.integers(0, 65536, size=16 * bits, dtype=np.uint16)
        w = O.tile_windows(tile, bits)
        nxt = np.roll(w, -1)
        # w_{t+1}'s low bits are new; its high bits equal w_t's high bits shifted
        assert (((w.astype(np.uint32) << bits) & 0xFFFF) | (nxt & 7) == nxt).all()

    def test_t_rowcol_bijection(self):
        r, c = O.t_to_rowcol(np.arange(256))
        assert len(set(zip(r.tolist(), c.tolist()))) == 256
        assert (r.min(), r.max(), c.min(), c.max()) == (0, 15, 0, 15)

    def test_window_order_matches_fragment_layout(self):
        """t = 8L+i maps as documented in exl3_gemv_dot_tile_direct."""
        for L in (0, 5, 31):
            r, c = O.t_to_rowcol(np.array([8 * L + i for i in range(8)]))
            r0 = 2 * (L % 4)
            cA = 2 * (L // 8) + ((L >> 2) & 1)
            expect = [
                (r0, cA), (r0 + 1, cA), (r0 + 8, cA), (r0 + 9, cA),
                (r0, cA + 8), (r0 + 1, cA + 8), (r0 + 8, cA + 8), (r0 + 9, cA + 8),
            ]
            assert list(zip(r.tolist(), c.tolist())) == expect


class TestCodebook:
    def test_mul1_constants(self):
        assert O.K_INV == O.fp16(0x1EEE)
        assert O.K_BIAS == O.fp16(0xC931)

    def test_mul1_zero_state(self):
        # w = 0 -> x = 0 -> s = 0 -> value = k_inv * 1024 + k_bias (fused, fp16)
        v = O.decode_codebook(np.array([0], dtype=np.uint16), 2)[0]
        expect = O.to_fp16(O.fp16(0x1EEE) * 1024.0 + O.fp16(0xC931))
        assert v == expect

    def test_mul1_range(self):
        vals = O.decode_codebook(np.arange(65536, dtype=np.uint16), 2)
        assert vals.min() >= -3.5 and vals.max() <= 3.5
        # s in [0, 1020] -> value in [(0-510)/147.7, (1020-510)/147.7] approx
        assert vals.min() > (0 - 510) / 147.7 - 0.01
        assert vals.max() < (1020 - 510) / 147.7 + 0.01

    def test_mul1_affine_in_byte_sum(self):
        w = np.array([0x1234, 0x5678, 0x9ABC, 0xDEF0], dtype=np.uint16)
        s = O.byte_sum((w.astype(np.uint64) * O.MUL1_MULT) & 0xFFFFFFFF)
        vals = O.decode_codebook(w, 2)
        for si, v in zip(s, vals):
            assert v == O.to_fp16(O.K_INV * (1024.0 + float(si)) + O.K_BIAS)

    def test_decode_deterministic(self):
        w = RNG.integers(0, 65536, size=256, dtype=np.uint16)
        assert (O.decode_codebook(w, 2) == O.decode_codebook(w, 2)).all()


class TestTransform:
    def test_hadamard_orthonormal_scaling(self):
        H = O.had128_matrix()
        assert np.allclose(H @ H.T, 128.0 * np.eye(128))
        Hn = H * O.R_SCALE
        assert np.allclose(Hn @ Hn.T, np.eye(128), atol=1e-12)

    def test_involution(self):
        x = RNG.standard_normal(256)
        y = O.had128(O.had128(x))
        assert np.allclose(y, x, atol=1e-10)

    def test_impulse_pattern(self):
        x = np.zeros(128)
        x[0] = 1.0
        y = O.had128(x)
        assert np.allclose(np.abs(y), O.R_SCALE, atol=1e-12)

    def test_pre_post_scale(self):
        x = RNG.standard_normal(128)
        suh = np.abs(RNG.standard_normal(128)) + 0.5
        svh = np.abs(RNG.standard_normal(128)) + 0.5
        assert np.allclose(O.had_in(x, suh), O.had128(x * suh))
        assert np.allclose(O.had_out(x, svh), O.had128(x) * svh)


class TestPipeline:
    def test_gemv_matches_manual_composition(self):
        tr = O.random_trellis(RNG, 8, 8, 3)   # 128 x 128 (Hadamard blocks)
        a = O.to_fp16_vec(RNG.standard_normal(128) * 0.3)
        suh = O.to_fp16_vec(np.abs(RNG.standard_normal(128)) + 0.5)
        svh = O.to_fp16_vec(np.abs(RNG.standard_normal(128)) + 0.5)
        y = O.gemv_mul1(a, tr, 3, suh, svh, faithful=False)
        W = O.decode_trellis(tr, 3)
        assert np.allclose(y, O.had_out(O.had_in(a, suh) @ W, svh), atol=1e-12)

    def test_faithful_mode_lands_on_fp16_grid(self):
        tr = O.random_trellis(RNG, 8, 8, 3)
        a = RNG.standard_normal(128) * 0.3
        suh = np.abs(RNG.standard_normal(128)) + 0.5
        svh = np.abs(RNG.standard_normal(128)) + 0.5
        y = O.gemv_mul1(a, tr, 3, suh, svh, faithful=True)
        assert (O.to_fp16_vec(y) == y).all(), "faithful output must be fp16-representable"

    def test_zero_input(self):
        tr = O.random_trellis(RNG, 8, 8, 3)
        suh = np.abs(RNG.standard_normal(128)) + 0.5
        svh = np.abs(RNG.standard_normal(128)) + 0.5
        y = O.gemv_mul1(np.zeros(128), tr, 3, suh, svh)
        assert np.allclose(y, 0.0, atol=1e-12)

    def test_int8_reference_close_to_fp16(self):
        tr = O.random_trellis(RNG, 8, 8, 3)   # 128 x 128 (Hadamard blocks)
        a = RNG.standard_normal(128)
        suh = np.abs(RNG.standard_normal(128)) + 0.5
        svh = np.abs(RNG.standard_normal(128)) + 0.5
        W = O.decode_trellis(tr, 3)
        S = O.trellis_states(tr, 3)
        a_had = O.had_in(a, suh)
        y_fp = O.had_out(a_had @ W, svh)
        q = max(abs(a_had).max(), 1e-8) / 127.0
        y_i8 = O.gemv_int8_reference(a_had, S, q, svh, residual=False)
        y_r8 = O.gemv_int8_reference(a_had, S, q, svh, residual=True)

        def nrms(u, v):
            return np.sqrt(np.mean((u - v) ** 2)) / np.sqrt(np.mean(v ** 2))

        assert nrms(y_i8, y_fp) < 0.05    # plain int8: ~0.55% on random fixtures
        assert nrms(y_r8, y_fp) < 0.005   # residual: ~0.03%
        assert nrms(y_r8, y_fp) < nrms(y_i8, y_fp)

    def test_alternating_signs_and_outliers(self):
        tr = O.random_trellis(RNG, 8, 8, 3)
        a = np.tile([100.0, -100.0], 64)
        a[0] = 1e4  # activation outlier
        suh = np.ones(128)
        svh = np.ones(128)
        y = O.gemv_mul1(a, tr, 3, suh, svh)
        assert np.isfinite(y).all()
