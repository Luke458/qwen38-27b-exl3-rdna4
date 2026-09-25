# exl3rocm: EXL3 weights in vLLM on the RX 9070 XT (gfx1201)

An out-of-tree vLLM quantization plugin that serves EXL3 (exllamav3 trellis)
checkpoints on RDNA4, on top of the community
[vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4) 0.28.0 image.
It reuses the exllamav3 ROCm fork's kernels (`vendor/rocm_exl3`). The plugin
structure is adapted from [0xSero/exl3xpu](https://github.com/0xSero/exl3xpu) (MIT).

Status: experimental. Tested only with the GestaltLabs Qwen3.8-27B EXL3 11.5 GB
checkpoint, one GPU (text, single images, reasoning/tool parsing). See `docs/VLLM_PORT_ASSESSMENT.md` for status and measurements.

## Layout

| path | what |
|---|---|
| `exl3rocm/plugin.py` | `exl3` quantization config, per-checkpoint-tensor ("group") linear method, fp8 embedding method |
| `exl3rocm/ops.py` | opaque torch custom ops over `exllamav3_ext` |
| `exl3rocm/kv_dequant.py` | Triton gather + dequantize of 8-bit KV blocks for prefill attention |
| `run_exl3_server.sh` | podman launcher for an EXL3 checkpoint |
| `tools/build_ext_in_image.sh` | build `exllamav3_ext` against the image's torch |
| `tools/run_base_server.sh` | launcher for unquantized models (engine baselines) |
| `tools/bench_client.py` | single-stream decode benchmark (OpenAI API) |
| `tools/longctx_bench.py`, `tools/concurrency_bench.py` | long-prompt prefill/decode, concurrent streams |
| `tools/mem_probe.sh` | start a profile, report KV capacity and peak VRAM under stress |
| `patches/vllm-0.28.0-rdna4/` | vLLM source patches, mounted over the image (`EXTRA_MOUNTS`) |
| `tests/xcheck_ext.py` | bitwise cross-check of two extension builds |
| `tests/sweep_op.py` | fault-isolation sweep of the op over every checkpoint shape |
| `tests/offline_probe.py` | offline load + greedy generation, options as JSON |

## Quick start

```bash
vllm_plugin/tools/build_ext_in_image.sh ~/exl3ext            # ~6 min
# validate: host control binary writes reference, image build must match bit-for-bit
EXL3_EXT_DIR=~/exl3ext vllm_plugin/run_exl3_server.sh /path/to/qwen3.8-27b-exl3 qwen38-27b-exl3 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 1500000000
python3 vllm_plugin/tools/bench_client.py --model qwen38-27b-exl3
```

## Serving profiles (RX 9070 XT, Qwen3.8-27B EXL3 11.5 GB, measured 2026-09-25)

Common flags: `--max-num-batched-tokens 2048 --mamba-ssm-cache-dtype float16 --kv-cache-dtype int8_per_token_head`.
The vision tower is loaded in both profiles. Peaks are whole-card VRAM with a desktop session (~480 MiB), measured under
a long prompt, a 1 MP image and 4 concurrent streams (`tools/mem_probe.sh`).

| profile | extra flags | KV tokens | peak | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---:|---:|---:|---:|---:|---:|
| single-user (MTP) | `--max-model-len 16384 --max-num-seqs 4 --kv-cache-memory-bytes 1950000000 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | 30,492 | 15,950 MiB | **~80 tok/s** (prose 80 / code 99 / story 65) | 111 agg | 124 agg | - |
| multi-user / long context (plain) | `--max-model-len 65536 --max-num-seqs 8 --kv-cache-memory-bytes 2600000000` | 72,238 | 15,950 MiB (50k prompt) | 42 tok/s | 70 agg | 128 agg | **199 agg** |

With int8 KV, the 1-stream MTP, 2-stream MTP and 8-stream plain figures were re-measured. The other cells are from fp16 KV,
where the decode kernels are the same.

**KV cache.** `int8_per_token_head` stores one scale per token and head. Its teacher-forced logits match fp16 KV
(KL 0.0018 vs 0.0055 on the decode path, both 128/128 top-1), and it holds 1.35x as many tokens. Per-tensor
`fp8` is 5x worse (KL 0.029, 127/128 top-1) because the checkpoint has no KV scales. vLLM's Triton attention is slow
on prefill with 8-bit KV, so for prefill chunks the plugin dequantizes the sequence's KV blocks into a persistent fp16
buffer (`exl3rocm/kv_dequant.py`) and runs the fp16 kernel on them. Decode reads int8 directly. Do not set
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`: it caused a GPU memory-access fault here.

**Memory.** Model load is 11.17 GiB. vLLM always shares the target's embedding and lm_head with the MTP drafter, so the
plugin gives the drafter 0-size placeholders instead of loading them a second time (−0.8 GiB). The vision `attn.qkv`
runs as EXL3 (the checkpoint's bf16 copy is dropped, −0.12 GiB; `EXL3_VISION_QKV_BF16=1` restores it). The launcher caps
image size at 1 MP (`EXL3_MAX_PIXELS`); `EXL3_TEXT_ONLY=1` skips the vision tower.

Prefill is about 1.3–1.9k tok/s (2,048-token chunks; GEMM-bound, `hgemm_recon` ~136 TFLOPS) and ~1.1k tok/s at a
50k-token prompt. Decode with MTP stays at 84–95 tok/s with 5–12k-token contexts. vLLM 0.28 cannot switch speculation
off by batch size (`disable_by_batch_size` is not in V1), so there are two profiles.

**Vision, reasoning and tools** (verified 2026-09-25): the vision tower uses EXL3 6-bit throughout, with the MLP built
at the EXL3-padded width, as exllamav3 does. A synthetic test image (red circle, blue square, "42") was described
correctly. Add `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` for separated
thinking and OpenAI tool calls (both verified).

Launcher defaults (all measured, see `docs/DECODE_TRACE.md`): `GPU_MAX_HW_QUEUES=1`, the V2 model runner,
the GDN metadata patch, the fused output Hadamard, and a pruned MTP draft head (`EXL3_DRAFT_VOCAB_BLOCKS=640`).

Fused vLLM modules are stored as their checkpoint tensors ("groups"), because this checkpoint
mixes bitrates inside fused modules. Multi-group outputs are assembled inside a single
opaque op. Never `torch.cat` per-group outputs in traced code: Inductor mis-strided such a
cat + slice and faulted the GPU.
