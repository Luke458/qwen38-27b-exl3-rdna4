# Backend patch profiles

Both profiles start from `CarouselAether/rocm_exl3` commit `311ff5497237a37bea18cf24aa18bc0c573d1d03`. `tools/prepare_backend.py` checks the patch SHA256 values and resulting source hashes. It does not install dependencies, build the extension, or establish runtime correctness.

- `baseline` (default) applies `0000-smem-budget-clamp.patch`, `0001-int8-port-gfx1201-compat.patch` and `0002-wmma-gfx12-asm.patch`. Six upstream files differ. This is the compatibility source used for the frozen baseline plus the gfx12 WMMA fix (the frozen baseline binary predates `0002`; its GEMV paths are identical, its multi-row GEMM paths trapped and are unusable without it).
- `experimental` applies `experimental-full.patch` to a clean pinned checkout. Eight upstream files differ, including the later transform changes and the WMMA fix. It is an explicit research profile; selecting it does not imply a performance or model qualification claim.

The profiles are alternatives, not a patch stack. The full experimental patch already includes the baseline edits. Existing vendor checkouts are verified against the selected profile and are never reset or overwritten. The vendored source is ignored by the project repository; original MIT notices remain in the prepared checkout.
