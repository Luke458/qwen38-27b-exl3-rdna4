# Decode trace findings (2026-09-24/25)

Summary of the kernel-level measurements behind the current plan. Full records are in the
local `experiments/0015-kernel-trace`, `0016-vllm-base` and `0017-exl3-plugin` (not shipped).

## exllamav3 fork, Qwen3.8-27B EXL3, one decode step (34.9 ms)

| component | ms/step |
|---|---:|
| quantized GEMV/dot kernels | 23.7 |
| GPU idle between 1,561 kernel dispatches (~3.8 µs each, eager) | 6.2 |
| GDN core + attention core | 2.9 |
| Hadamard, norm, activation kernels | 1.9 |

- **The dot kernels are decode-ALU-bound, not bandwidth-bound**: ~1.1–1.25 T weights/s at any
  bitrate (2-bit and 3-bit gate/up take 155 vs 159 µs). The ISA of the main loop is ~323 issue slots
  per 32 weights. 40% of that is `v_mul_lo_u32` (codebook hash), which a microbenchmark shows runs at
  ¼ rate on gfx1201. Also costly: un-hoisted 64-bit addressing and ~2.75 bit-ops/weight of extraction.
  Model and measurement agree (1.14 predicted vs 1.12–1.15 T weights/s measured).
- Plain decode reads 9.42 GB of weights per token: 14.6 ms at 644.6 GB/s spec, ~17 ms at the
  ~545 GB/s this card sustains (copy). Reaching it needs ~1.7–1.9 T weights/s at 2.87 bpw.
- HIP-graph replay dispatch costs 2.83 µs/kernel (eager 3.79), so kernel count matters even in graphs.
- System `rocprofv3` cannot trace the host venv (torch wheel bundles its own profiler SDK). Use
  torch.profiler there, or rocprofv3 inside the vLLM image (system-ROCm torch).

## vLLM (rdna4 image) on this card

- Qwen3.5-4B bf16 plain decode: 63.8 tok/s ≈ 549 GB/s. The engine itself runs at bandwidth.
- Native MTP k=3: 2.47 tokens/step and 1.63× less GPU work per token, but only 1.03–1.10× realized
  in the server; host/scheduling side still open. GDN metadata was rebuilt per layer (24×/step,
  12.6 ms host). `vllm_plugin/patches/.../gdn_attn_share_build.patch` builds once (+2%, output identical).
- EXL3 plugin, first light: 19.2 tok/s plain decode; 2,179 kernels/token, 33.4 ms GPU/token.

## Current state (2026-09-25, vLLM + exl3rocm)

| config | decode tok/s |
|---|---:|
| plain | 41.3 |
| native MTP k=3 (bit-exact multi-row verify, V2 runner) | 73.9 (prose 73.7 / code 90.6 / story 60.4) |

- GEMV core (fork patch 0001): the codebook hash uses full-rate `v_mul_u32_u24` pairs instead of the
  quarter-rate `v_mul_lo_u32` (called through the `llvm.amdgcn.mul.u24` intrinsic, marked noconvergent,
  with its operands read from a device global so LLVM cannot re-fuse them). Also: wave-uniform `warp_id`,
  buffer loads with scalar tile offsets, and mask-free pair packing. 6.75 VALU/weight (was ~10.1), outputs bit-identical.
- `GPU_MAX_HW_QUEUES=1`: default multi-queue scheduling cost the vLLM decode graph ~18 ms/step on gfx1201.
- Multi-row GEMV (fork patch 0002): m=4 at 1.07–1.31× the cost of m=1, bitwise equal to m=1 per row.
  MTP output is identical to plain decode.
- Teacher-forced logits vs the fork: top-1 128/128 on both decode and prefill paths.
