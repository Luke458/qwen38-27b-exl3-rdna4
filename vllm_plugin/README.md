# exl3rocm: EXL3 weights in vLLM on the RX 9070 XT (gfx1201)

An out-of-tree vLLM quantization plugin that serves EXL3 (exllamav3 trellis)
checkpoints on RDNA4, on top of the community
[vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4) 0.28.0 image.
It reuses the exllamav3 ROCm fork's kernels (`vendor/rocm_exl3`). The plugin
structure is adapted from [0xSero/exl3xpu](https://github.com/0xSero/exl3xpu) (MIT).

Status: experimental. Tested only with the GestaltLabs Qwen3.8-27B EXL3 11.5 GB
checkpoint, text-only, one GPU. See `docs/VLLM_PORT_ASSESSMENT.md` for status and measurements.

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

Fused vLLM modules are stored as their checkpoint tensors ("groups"), because this checkpoint
mixes bitrates inside fused modules. Multi-group outputs are assembled inside a single
opaque op. Never `torch.cat` per-group outputs in traced code: Inductor mis-strided such a
cat + slice and faulted the GPU.
