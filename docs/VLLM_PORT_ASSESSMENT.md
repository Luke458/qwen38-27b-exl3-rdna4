# vLLM port assessment (2026-09-24)

Question: how much work is an exl3xpu-style vLLM plugin for the RX 9070 XT (gfx1201, 16 GB),
versus continuing on the exllamav3 ROCm fork?

## What exl3xpu actually is (0xSero/exl3xpu @ 6872a30)

- ~2,200 lines of code: `vllm_plugin.py` (325, model-agnostic quantization config + linear
  method + weight loaders), `ops.py` (122, torch custom op), `vllm_patches.py` (267, GDN metadata
  sync fix, fp8-KV prefill attention, XPU block size), `triton_kernels.py` (198, portable fallback),
  ESIMD kernels (~1,300 in `csrc/`).
- vLLM supplies scheduling, paged fp8 KV, attention, GDN, MTP drafting, graphs, the OpenAI API,
  reasoning/tool parsing, and vision.
- Their progress log (`docs/PROGRESS.md`) shows where the speed came from on the B70:
  Triton eager 7.3 → ESIMD GEMV 14.4 → decode graphs 28.4 tok/s (**plain decode ≈ ours today, ~29**)
  → MTP k=3 37.9 → pruned-vocab draft head 48.2 prose → later ~91 (README).
  An M=4 verify costs 35 ms vs 32.5 ms at M=1, because decode dominates and verify rows are nearly free.

## The base engine already exists for this GPU and model

[Capicua25x/vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4) (vLLM 0.28.0, rc12
2026-08-28, docker `capicua25x/vllm-rocm-rdna4:0.28.0-rdna4`) validates **Qwen3.8-27B-FP8 with
native MTP-3 on gfx1201** (R9700 32 GB, TP2): hybrid GDN + full attention hardened, fp8 KV,
`--attention-backend TRITON_ATTN`, spec-decode verify fix, HIP graphs. Their doc lists "gfx1200 (RX 9070 XT)"
as unvalidated, but the 9070 XT is gfx1201, the same target they validated. Not validated there: a 16 GB
card, and any EXL3 quantization. Docker and podman are installed; the user is in the `docker` group.

## Work items for the port

| # | item | size | notes |
|---|---|---|---|
| 0 | Run the rdna4 image on this card with a small Qwen3.5-class model | **done: GO** | Qwen3.5-4B bf16: plain 63.8 tok/s (~549 GB/s, at bandwidth), MTP works (2.47 tokens/step) but the server realizes only ~1.03–1.10× of a 1.63× GPU-work reduction. See `experiments/0016-vllm-base/RESULTS.md`. |
| status | **~80 tok/s MTP-3 single stream, 200 tok/s aggregate at 8 streams (plain), vision + tools + reasoning working (2026-09-25)** | — | GEMV core rewrite + multi-row verify (fork patches in `vllm_plugin/patches/exl3-fork/`), 1 HW queue, V2 runner, GDN patch. Teacher-forced logits vs fork: top-1 128/128. See `docs/DECODE_TRACE.md`. |
| 1–4 | **Phase A done** (`vllm_plugin/`, `experiments/0017-exl3-plugin/RESULTS.md`) | — | 27B EXL3 serves through vLLM with compile + HIP graphs at 19.2 tok/s plain decode; per-group mixed-bit storage works; an Inductor cat/slice GPU fault was found and fixed; fp8 embedding still pending (model hook needed). |
| 1 | Plugin config + weight loading (port `vllm_plugin.py`) | small–medium | **Mixed-bit fused shards**: all 16 `qkv` groups and 36/48 `in_proj_qkvz` groups mix bitrates (e.g. q=2/k,v=5; qkv=2/z=4). exl3xpu asserts equal bits, so this needs per-bitrate shard groups (our mgemv already groups same-K matrices). |
| 2 | **fp8 embedding** | small | Checkpoint stores `embed_tokens` as F8_E4M3 (1.18 GiB). vLLM would upcast to bf16 (2.37 GiB). Needs a custom embedding method that dequantizes gathered rows. |
| 3 | Torch custom ops over the existing HIP kernels | **medium (largest)** | Reuse fork kernels: M≤4 GEMV, M 5–128 via the 0011 WMMA GEMM, M>128 reconstruct + hipBLASLt. Must be graph-safe (no host syncs, allocator-owned workspace), with fake/meta impls for torch.compile. |
| 4 | dtype boundary | small–medium | vLLM runs Qwen3.5 in bf16; EXL3 kernels are fp16. Cast at the op boundary (cheap at small M) or validate an fp16 model run. |
| 5 | 16 GB fit | **risk** | ~10.2 GiB text weights + 0.2 MTP (+0.53 vision, optional) + vLLM runtime/graphs (~1.5–2 GiB, to measure). Leaves ~2–3 GiB for fp8 KV (16 attn layers × 2 × 4 kv heads × 256 = 32 KiB/token → roughly 60–90k tokens) and GDN state per sequence. Measure early. |
| 6 | Correctness | small–medium | Reuse `reference/exl3_oracle.py` (bit-exact reconstruct) and `tools/compare_logits.py` against the exllamav3 path on identical histories. MTP identity checks as in exl3xpu `tests/test_mtp_*`. |
| 7 | Performance kernels | medium | Same on either route: 0015's decode-loop rewrite (quarter-rate `v_mul_lo_u32`, addressing, periodic extraction) and a decode-once multi-row (M=2–8) kernel for MTP verify. |

## Comparison with staying on the exllamav3 fork

Staying means building MTP drafting and verification with GDN state rollback, paged/fp8 KV,
concurrency, tool/reasoning parsing and vision serving in this repo's own `serve.py`. vLLM
provides all of these, and the rdna4 fork has them running on gfx1201 for this model family.
Kernel work (item 7) is identical either way, so it can continue in parallel on the fork's
existing bench harness.

Recommendation: do item 0 first (cheap and decisive). If it passes, port items 1–4 with the existing
kernels, then continue kernel work inside the plugin.
