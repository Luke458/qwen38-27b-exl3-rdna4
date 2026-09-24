"""Integer-accumulation overflow proof for the int8 path (CPU).

Per dp4a term: |i| <= 127 (clamped int8) and byte_sum(w * M) <= 1020, so each
product is bounded by 127 * 1020 = 129540 in magnitude.

The overflow-safe decomposition (which the v1 implementation must use, and
which the plan's staged-ordinary-launch architecture mandates) is:

  * int32 accumulators hold ONE k-slice partial each: at most rows_per * 16
    terms with rows_per <= 768 (coop) or <= 512 (sq) trellis rows;
  * slice partials recombine in float32/float64 in the epilogue.

Note for the record: a single int32 accumulator over the FULL K_in = 17408
range would NOT be provably overflow-free (17408 * 129540 = 2.255e9 > 2^31-1)
under adversarial saturated activations, so the upstream coop kernel's
atomicAdd accumulator region carries a theoretical (unreachable-in-practice)
saturation risk at K_in = 17408. The v1 path avoids it by construction.
"""

import numpy as np

MAX_TERM = 127 * 1020          # |i| * byte_sum
INT32_MAX = 2**31 - 1
SUPPORTED_K_IN = [128, 5120, 17408]
ROWS_PER_CAP = 768             # worst slice height of the staged decomposition


def slice_terms(rows_per: int) -> int:
    return rows_per * 16


def test_full_k_would_overflow_is_documented():
    # the claim we do NOT make: full-K int32 accumulation is safe at K_in=17408
    assert 17408 * MAX_TERM > INT32_MAX


def test_slice_partial_fits_int32():
    for k_in in SUPPORTED_K_IN:
        for rows_per in (8, 16, 64, 320, 512, ROWS_PER_CAP):
            if rows_per * 16 > k_in:
                continue
            worst = slice_terms(rows_per) * MAX_TERM
            assert worst < INT32_MAX, (k_in, rows_per, worst)


def test_residual_second_accumulator_bounded():
    # residual mode: acc1 and acc2 are SEPARATE int32 accumulators; each must
    # independently fit (their float recombination happens in the epilogue)
    for rows_per in (16, 512, ROWS_PER_CAP):
        for acc in (1, 2):
            worst = slice_terms(rows_per) * MAX_TERM
            assert worst < INT32_MAX, (rows_per, acc, worst)


def test_float_recombination_range():
    # epilogue: q * acc_s summed in float; float32 covers the full range
    worst = 17408 * MAX_TERM
    assert np.float32(worst) < np.float32(np.finfo(np.float32).max)
