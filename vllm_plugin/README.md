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
| `run_exl3_server.sh` | podman launcher for an EXL3 checkpoint |
| `tools/build_ext_in_image.sh` | build `exllamav3_ext` against the image's torch |
| `tools/run_base_server.sh` | launcher for unquantized models (engine baselines) |
| `tools/bench_client.py` | single-stream decode benchmark (OpenAI API) |
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

Common flags: `--max-model-len 16384 --max-num-batched-tokens 2048 --mamba-ssm-cache-dtype float16`
(fp16 GDN state: KV capacity 16.8k -> 21.4k tokens; single-stream MTP output unchanged in the checks).

| profile | extra flags | 1 stream | 2 streams | 4 streams | 8 streams |
|---|---|---:|---:|---:|---:|
| single-user (MTP) | `--max-num-seqs 4 --kv-cache-memory-bytes 2100000000 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | **~80 tok/s** (prose 80 / code 99 / story 65) | 107 agg | 124 agg | - |
| multi-user (plain) | `--max-num-seqs 8 --kv-cache-memory-bytes 2300000000` | 42 tok/s | 70 agg | 128 agg | **200 agg** |

Prefill is about 1.4k tok/s (2,048-token chunks; GEMM-bound, `hgemm_recon` ~136 TFLOPS). Decode with MTP stays
at 84-95 tok/s with 5-9k-token contexts. vLLM 0.28 cannot switch speculation off by batch size
(`disable_by_batch_size` is not in V1), so there are two profiles.

**Vision, reasoning and tools** (verified 2026-09-25): the vision tower loads by default (EXL3 6-bit
proj/MLP/merger through the plugin; the checkpoint's bf16 fused `attn.qkv`; MLP built at the EXL3-padded
width, as exllamav3 does). A synthetic test image (red circle, blue square, "42") was described correctly.
The vision tower costs ~0.7 GiB, so use `--max-model-len 8192 --kv-cache-memory-bytes 1500000000`
with MTP, or `EXL3_TEXT_ONLY=1` to skip it. Add `--reasoning-parser qwen3 --enable-auto-tool-choice
--tool-call-parser qwen3_coder` for separated thinking and OpenAI tool calls (both verified).

Launcher defaults (all measured, see `docs/DECODE_TRACE.md`): `GPU_MAX_HW_QUEUES=1`, the V2 model runner,
the GDN metadata patch, the fused output Hadamard, and a pruned MTP draft head (`EXL3_DRAFT_VOCAB_BLOCKS=640`).

Fused vLLM modules are stored as their checkpoint tensors ("groups"), because this checkpoint
mixes bitrates inside fused modules. Multi-group outputs are assembled inside a single
opaque op. Never `torch.cat` per-group outputs in traced code: Inductor mis-strided such a
cat + slice and faulted the GPU.
