# exl3rocm: EXL3 weights in vLLM on the RX 9070 XT (gfx1201)

An out-of-tree vLLM quantization plugin that serves EXL3 (trellis-quantized)
checkpoints on RDNA4, on top of the community
[vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4) 0.28.0 image.
It reuses the kernels of the pinned ROCm fork in `vendor/rocm_exl3`. The plugin
structure is adapted from [0xSero/exl3xpu](https://github.com/0xSero/exl3xpu) (MIT).

Status: experimental. It has been tested only with the GestaltLabs Qwen3.8-27B EXL3 11.5 GB checkpoint on one GPU
(text, single images, reasoning and tool-call parsing). Measurements are in
[`docs/VLLM_PORT_ASSESSMENT.md`](../docs/VLLM_PORT_ASSESSMENT.md) and [`docs/DECODE_TRACE.md`](../docs/DECODE_TRACE.md).

## Layout

| path | what |
|---|---|
| `exl3rocm/plugin.py` | `exl3` quantization config, per-checkpoint-tensor ("group") linear method, fp8 embedding method |
| `exl3rocm/ops.py` | opaque torch custom ops over the compiled kernel extension |
| `exl3rocm/kv_dequant.py` | Triton gather + dequantize of 8-bit and 4-bit KV blocks for prefill attention |
| `exl3rocm/kv_int4.py` | 4-bit KV cache (`int4_per_token_head`): cache write, decode / MTP-verify attention, prefill |
| `exl3rocm/kv_means.pt` | per-layer K/V channel means that center the 4-bit cache (from `tools/calib_kv_means.py`) |
| `run_exl3_server.sh` | podman launcher for an EXL3 checkpoint |
| `tools/build_ext_in_image.sh` | build the kernel extension against the image's torch |
| `tools/run_base_server.sh` | launcher for unquantized models (engine baselines) |
| `tools/bench_client.py` | single-stream decode benchmark (OpenAI API) |
| `tools/longctx_bench.py`, `tools/concurrency_bench.py` | long-prompt prefill/decode, concurrent streams |
| `tools/mem_probe.sh` | start a profile, report KV capacity and peak VRAM under stress |
| `tools/calib_kv_means.py` | recompute `kv_means.pt` (run inside the image) |
| `patches/exl3-fork/` | kernel patches on the pinned fork (decode GEMV, multi-row verify GEMV, graph-safe fused Hadamard) |
| `patches/vllm-0.28.0-rdna4/` | vLLM GDN metadata patch, mounted over the image by the launcher |
| `tests/xcheck_ext.py` | bitwise cross-check of two extension builds |
| `tests/sweep_op.py` | fault-isolation sweep of the op over every checkpoint shape |
| `tests/offline_probe.py` | offline load + greedy generation, options as JSON |

## Quick start

See the [top-level quickstart](../README.md#quickstart-vllm-plugin) for the full commands. In short:

```bash
vllm_plugin/tools/build_ext_in_image.sh ~/exl3ext        # fresh pinned fork + patches, built in the image
EXL3_EXT_DIR=~/exl3ext vllm_plugin/run_exl3_server.sh <model-dir> qwen38-27b-exl3 <profile flags below>
```

`run_exl3_server.sh` binds `127.0.0.1:8000`, installs the plugin into the container at start, and forwards
every `EXL3_*` environment variable. `tests/xcheck_ext.py` compares two extension builds bit for bit.

## Serving profiles (RX 9070 XT, Qwen3.8-27B EXL3 11.5 GB, measured 2026-09-26)

Common flags: `--max-num-batched-tokens 1024 --mamba-ssm-cache-dtype float16 --kv-cache-dtype int8_per_token_head`.
The vision tower is loaded in every profile. "Server peak" is the server's own VRAM (whole card minus the desktop
baseline) under a near-max-length prompt, a 1 MP image and 4 concurrent streams (`tools/mem_probe.sh`). Your
desktop's VRAM comes on top of it, and the card has 16,304 MiB, so keep the total below ~16,100 MiB.

| profile | extra flags | KV tokens | server peak | decode | long prompt |
|---|---|---:|---:|---|---|
| **single user, MTP-3, 32k** | `--max-model-len 32768 --max-num-seqs 4 --kv-cache-memory-bytes 1760000000 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | 35,108 | **14,755 MiB** | **~80 tok/s** (prose 80 / code 99 / story 65) | 32k tokens at 1.0k tok/s |
| single user, MTP-3, 40k | same with `--max-model-len 40960 --kv-cache-memory-bytes 2050000000` | 44,063 | 15,122 MiB | ~78 tok/s | 40k tokens at 0.95k tok/s |
| multi-user / long context | `--max-model-len 65536 --max-num-seqs 8 --kv-cache-memory-bytes 2600000000` | 72,238 | 15,171 MiB | 42 tok/s; **199 tok/s** total at 8 streams | 60k tokens at 0.83k tok/s |
| **4-bit KV, MTP-3, 48k** | 32k's MTP flags with `--kv-cache-dtype int4_per_token_head --max-model-len 49152 --kv-cache-memory-bytes 1460000000` | 53,426 | 14,665 MiB (41k prompt + image) | ~80 tok/s; 72 at 41k | 41k tokens at 0.96k tok/s |
| **4-bit KV, MTP-3, 64k** | same with `--max-model-len 65536 --kv-cache-memory-bytes 1760000000` | 70,217 | 14.5–14.8 GiB text-only at 53–62k; ~15.1–15.3 GiB with vision (est.) | ~80 tok/s; 63–68 at 53–62k | 61.8k tokens at 0.78k tok/s |

The 40k and 64k profiles fit with a desktop of up to ~0.9 GiB of VRAM, and the 32k profile with up to ~1.3 GiB. The
MTP 2- and 4-stream totals (111 and 124 tok/s) and the plain 2- and 4-stream totals (70 and 128 tok/s) were
measured with the earlier 16k/2,048-batch settings, which use the same decode kernels.

**32 GB cards (Radeon AI PRO R9700, same gfx1201 chip):** untested. `serve.sh` has `128k` (`--max-model-len 131072
--kv-cache-memory-bytes 5250000000`, ~18,700 MiB estimated peak) and `256k` (`--max-model-len 262144
--kv-cache-memory-bytes 9900000000`, ~23,800 MiB) MTP profiles, sized with the same KV page math (28,853,760-byte
pages, at least 3 spare). See the [top-level README](../README.md#32-gb-cards-radeon-ai-pro-r9700).

**Why 1,024-token batches.** Prefix caching puts vLLM's GDN state cache in "align" mode, where prefill
chunks round down to whole 816-token KV pages. A 2,048 budget gives 1,632-token chunks and a 1,024 budget
gives 816-token chunks. The smaller chunks cut the per-step activation peak by ~380 MiB and were also
faster: 1.5–1.6k tok/s at 1.6–6.5k-token prompts (vs 1.3–1.4k) and 1.0k tok/s at 32k (vs 0.9k).

**KV headroom.** vLLM 0.28 sizes the pool so that exactly one max-model-len request fits, but in align mode a
prompt near max-model-len needs a few more pages. With none spare, a 31.5k prompt at max-model-len 32768
**stalled forever** at 94.6% KV usage. The profiles above leave 3 spare pages, and the plugin logs a warning
(with the KV size or context length that would fit) when a configuration leaves fewer.

**KV cache.** `int8_per_token_head` stores one scale per token and head. Its teacher-forced logits match fp16 KV
(KL 0.0018 vs 0.0055 on the decode path, both 128/128 top-1), and it holds 1.35x as many tokens. Per-tensor
`fp8` is 5x worse (KL 0.029, 127/128 top-1) because the checkpoint has no KV scales. vLLM's Triton attention is slow
on prefill with 8-bit KV, so for prefill chunks the plugin dequantizes the sequence's KV blocks into a persistent fp16
buffer (`exl3rocm/kv_dequant.py`) and runs the fp16 kernel on them. Decode reads int8 directly. Do not set
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`: it caused a GPU memory-access fault here.

**4-bit KV cache** (`--kv-cache-dtype int4_per_token_head`, profiles `48k-int4` / `64k-int4` in `serve.sh`). It
takes half the bytes of int8. Because the GDN state pages cost the same, a KV page holds 1,616 tokens instead of 816.
So 64k with MTP fits in the 32k profile's pool, and 48k fits in less. vLLM has this format, but its path was
unusable here. Its Hadamard transforms ran as a PyTorch butterfly, and its MTP verify step was not split across
the context (38 ms per attention layer at 32k). The plugin replaces the whole path with `exl3rocm/kv_int4.py`
and keeps the cache format:
- the rotations run as small Triton GEMMs;
- a split-KV fp16-dot kernel handles decode and MTP verify (0.3 ms per layer at 32k, vs 0.5 ms for int8);
- prefill uses the same fp16 dequantize path as int8;
- the cache write centers K and V with calibrated per-layer means, which is exact, and picks each row's 4-bit
  range with a small clip search.

Accuracy against an fp16 cache on 2 × 16k tokens of held-out text (prose / code):
- int4: KL 0.0023 / 0.0027, top-1 98.7%, NLL +0.2%;
- vLLM's int4: KL 0.0035 / 0.0039;
- int8: 0.00007.

Speed is within ~1–2% of int8 at short context and faster at long context: without MTP, 36.8 vs 34.4 tok/s at
27k. MTP-3 decodes at 106.8 vs 108.2 tok/s over 16 prompts, with acceptance 3.40 vs 3.44. Switches:
- `EXL3_KV_INT4=0` uses vLLM's int4 path;
- `EXL3_KV_MEANS=0` (or a path) disables or replaces the centering means;
- `EXL3_KV_INT4_CLIP=1` turns off the clip search.

Details are in experiments/0024.

**Text-only compile cache.** torch.compile's cache key does not cover the multimodal settings, and a graph
compiled with `EXL3_TEXT_ONLY=1` fails when a vision-enabled server with the same flags loads it
(`'NoneType' object has no attribute 'size'`). The launcher therefore gives text-only runs their own cache
(`~/.cache/vllm-rdna4-exl3-textonly`).

**Memory.** Model load is 10.86 GiB with MTP and 10.66 GiB without. vLLM always shares the target's embedding and lm_head with the MTP drafter, so the
plugin gives the drafter 0-size placeholders instead of loading them a second time (−0.8 GiB). The vision `attn.qkv`
runs as EXL3 (the checkpoint's bf16 copy is dropped, −0.12 GiB; `EXL3_VISION_QKV_BF16=1` restores it). The launcher caps
image size at 1 MP (`EXL3_MAX_PIXELS`); `EXL3_TEXT_ONLY=1` skips the vision tower. The pruned MTP draft head
reads the first 640 and last 16 vocabulary blocks of the lm_head in place: the lm_head is stored as three column
groups, so no copy is kept (−0.2 GiB). Qwen3.5's rotary cache is clamped to max-model-len + 4096 positions instead
of 262,144 (−0.1 GiB; `EXL3_ROPE_CLAMP=0` disables it), and the logits are bit-identical.
`tests/mem_account.py` breaks down a loaded engine's memory.

Prefill is GEMM-bound at short contexts (`hgemm_recon` ~136 TFLOPS) and attention-bound at long ones. Decode with
MTP stays at 84–95 tok/s with 5–12k-token contexts and ~60 tok/s near 40k. vLLM 0.28 cannot switch speculation
off by batch size (`disable_by_batch_size` is not in V1), so there are two profiles.

**Vision, reasoning and tools** (verified 2026-09-25): the vision tower uses EXL3 6-bit throughout, with the MLP built
at the EXL3-padded width. A synthetic test image (red circle, blue square, "42") was described
correctly. Add `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` for separated
thinking and OpenAI tool calls (both verified).

Launcher defaults (all measured, see `docs/DECODE_TRACE.md`): `GPU_MAX_HW_QUEUES=1`, the V2 model runner,
the GDN metadata patch, the fused output Hadamard, and a pruned MTP draft head (`EXL3_DRAFT_VOCAB_BLOCKS=640`).

Fused vLLM modules are stored as their checkpoint tensors ("groups"), because this checkpoint
mixes bitrates inside fused modules. Multi-group outputs are assembled inside a single
opaque op. Never `torch.cat` per-group outputs in traced code: Inductor mis-strided such a
cat + slice and faulted the GPU.
