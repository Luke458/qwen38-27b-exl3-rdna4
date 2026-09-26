# Qwen3.8-27B EXL3 on RDNA4

Fast local serving of **Qwen3.8-27B (EXL3, 11.5 GB)** on an **AMD Radeon RX 9070 XT
(gfx1201, 16 GB)**: an OpenAI-compatible endpoint with speculative decoding, vision,
reasoning and tool calls, all within 16 GB of VRAM. The **Radeon AI PRO R9700 (32 GB)** is the same
Navi 48 / gfx1201 chip, so the same build runs there, with room for up to the model's full 262k context
(see [32 GB cards](#32-gb-cards-radeon-ai-pro-r9700)).

The main route is a vLLM plugin ([`vllm_plugin/`](vllm_plugin/README.md)) that runs the
checkpoint's trellis-quantized weights through custom gfx1201 kernels inside the community
[vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4) image. The repo also keeps
the original standalone server and the kernel research that led to the plugin.
No weights or compiled binaries are distributed here.

| profile | context | decode | notes |
|---|---:|---:|---|
| single user (MTP-3 speculative decoding) | 32k (40k option) | **~80 tok/s** (code ~99, prose ~80) | vision loaded |
| multi-user / long context | 64k | 42 tok/s single stream, **~200 tok/s** total at 8 streams | vision loaded |
| R9700 32 GB, long context (MTP-3) | 128k / 262k | expected ~80 tok/s at short context | sized, not yet run on a 32 GB card |

Prompt processing runs at about 1.5–1.6k tok/s, and ~1k tok/s near 32k tokens. The KV cache is int8, and its
teacher-forced logits match an fp16 KV cache. The server peaks at 14.4–14.8 GiB of the card's 15.9 GiB, depending on the profile,
leaving room for a desktop session on the same card. Measured on 2026-09-26; details are in the
[plugin README](vllm_plugin/README.md).

This is an experimental, narrowly tested setup: one card, one checkpoint, one GPU.

## Quickstart (vLLM plugin)

Prerequisites: Linux with the amdgpu driver and access to `/dev/kfd` and `/dev/dri`,
rootless [podman](https://podman.io/), Git, Python 3 and [uv](https://docs.astral.sh/uv/).
ROCm does not need to be installed on the host because the container brings its own. Allow about
45 GB of disk: 28 GB for the vLLM image, 12 GB for the model and a few GB for the build.

```bash
git clone https://github.com/Luke458/qwen38-27b-exl3-rdna4.git
cd qwen38-27b-exl3-rdna4

# 1. build the kernel extension inside the vLLM image (~6 min; pulls the image on first use)
vllm_plugin/tools/build_ext_in_image.sh ~/exl3ext

# 2. download the tested checkpoint
uvx --from huggingface_hub hf download GestaltLabs/Qwen3.8-27B-EXL3-11.5GB \
  --revision 0e6c4a863b945dbaf9657343fedd876c45e64dd5 --local-dir ./models/qwen38-27b-exl3

# 3. serve (single-user MTP profile, 32k context)
EXL3_EXT_DIR=~/exl3ext vllm_plugin/run_exl3_server.sh ./models/qwen38-27b-exl3 qwen38-27b-exl3 \
  --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --kv-cache-dtype int8_per_token_head --mamba-ssm-cache-dtype float16 \
  --kv-cache-memory-bytes 1760000000 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Step 3 is also available as `./serve.sh`, which runs the profiles with the defaults above: `./serve.sh`
(32k with MTP), `./serve.sh 40k` or `./serve.sh 64k`, and on a 32 GB card `./serve.sh 128k` or `./serve.sh 256k`.
It checks VRAM headroom before starting. Set `MODEL_DIR` if the
model is somewhere other than `~/models/qwen3.8-27b-exl3-11.5gb`.

The build prints `source tree matches the tested build` when the fork pin and patches are the ones
that were measured. The first start compiles and captures graphs for a few minutes, and later starts reuse
`~/.cache/vllm-rdna4-exl3`. When the log shows `Application startup complete`, the endpoint is
**`http://127.0.0.1:8000/v1`** with model ID **`qwen38-27b-exl3`** (loopback only, no API key):

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b-exl3","messages":[{"role":"user","content":"Write a haiku about GPUs."}],
       "max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
python3 vllm_plugin/tools/bench_client.py --model qwen38-27b-exl3   # decode tok/s
```

Thinking is on by default: drop `chat_template_kwargs` and the model thinks first, returning its thoughts
in the message's `reasoning` field (allow a larger `max_tokens`). OpenAI-style `tools` come back as
`tool_calls`, and images are accepted as `image_url` content parts (capped at 1 MP).

For the multi-user / 64k profile, replace `--max-model-len`, `--max-num-seqs`, `--kv-cache-memory-bytes`
and `--speculative-config` with `--max-model-len 65536 --max-num-seqs 8 --kv-cache-memory-bytes 2600000000`.
`EXL3_TEXT_ONLY=1` skips the vision tower. Stop the server with Ctrl-C or `podman stop vllm-exl3`.

The server itself peaks at about 14.4 GiB in this profile. Whatever your desktop uses comes on top of that, out of
the card's 15.9 GiB (`rocm-smi --showmeminfo vram` shows it). For 40k context use `--max-model-len 40960
--kv-cache-memory-bytes 2050000000` (server peak 14.8 GiB). See the [plugin README](vllm_plugin/README.md) for profiles, memory notes and the tools.

## 32 GB cards (Radeon AI PRO R9700)

The R9700 uses the same Navi 48 chip as the RX 9070 XT: gfx1201, 64 compute units and a 256-bit GDDR6 bus. So the
extension build, the plugin and the image are the same, and decode speed, which is limited by memory bandwidth,
should match (~80 tok/s with MTP, ~42 without). The extra 16 GB goes to KV cache:

| `./serve.sh` profile | context | KV pool | estimated server peak |
|---|---:|---:|---:|
| `128k` | 131,072 | 5.25 GB | ~18,700 MiB |
| `256k` | 262,144 (the model's maximum) | 9.9 GB | ~23,800 MiB |

These are **untested**: no 32 GB card was available. The KV pools use the same page math as the measured 16 GB
profiles (28,853,760-byte pages, with at least 3 spare so a full-length prompt cannot stall), and the peaks are
the measured non-KV footprint plus the pool plus the prefill-time fp16 copy of the context's KV. The script
refuses these profiles on cards with less than ~30 GB. Very long prompts are slow to read the first time: on
current measurements, roughly 4 minutes for 128k tokens and 15 minutes for 262k (prefix caching makes follow-up
turns fast). Decode also slows as the context fills (~60 tok/s at 40k). The plugin is single-GPU: two cards
cannot be pooled with tensor parallelism. On a 32 GB card you can also raise `--max-num-seqs` for more concurrent
long requests. The plugin's startup check warns if the KV pool is too small for the chosen context.

### Higher-bitrate quants on 32 GB

A 32 GB card can also hold less-quantized versions of the model. turboderp's official
[Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) has `4.00bpw`, `5.00bpw` and `6.00bpw` branches.
They use the same `mul1` codebook as the tested checkpoint (which averages ~2.9 bpw in its decoder layers) and
include the MTP head. They are **untested** here; the estimates below scale the measured speed by weight size:

| branch | download | context that fits (est.) | decode without / with MTP (est.) |
|---|---:|---|---:|
| `4.00bpw` | 15.7 GiB | full 262k | ~31 / ~58 tok/s |
| `5.00bpw` | 18.5 GiB | ~128k (262k marginal) | ~25 / ~48 tok/s |
| `6.00bpw` | 21.4 GiB | ~128k | ~21 / ~40 tok/s |

4.0 bpw is likely the best balance: a clear quality step up from ~3 bpw, still fast, and full context. These
quants keep the input embedding in bf16 (2.4 GiB on the GPU, vs 1.2 GiB for the tested checkpoint's fp8 table)
and the vision tower unquantized; the plugin handles both. The `serve.sh` profiles are sized for the tested
checkpoint, so for these run `MODEL_DIR=... ./serve.sh 128k` and adjust `--kv-cache-memory-bytes` if needed
(the plugin warns at startup when the pool is too small for the context). None of them fit on a 16 GB card.

## Standalone server (original route)

The original server (`tools/serve.py`) runs the same checkpoint directly on the ROCm fork's
runtime, without vLLM. It is single-user, has no speculative decoding and decodes at about 29 tok/s.
It is kept for reference and for the kernel experiments. In it, the experimental fused/unrolled FP16
implementation measured **1.0416× decode throughput** in ten paired trials, below the predeclared 1.05×
promotion threshold, so the baseline remains its default. The separate int8 experiment was slower
and stays disabled. See [measured status](docs/STATUS.md) and [optimisation notes](docs/OPTIMIZATION.md).

### Quickstart

Prerequisites: Linux with a working ROCm **7.2.4** installation at `/opt/rocm`,
RX 9070 XT device access, Git, Python 3, and [uv](https://docs.astral.sh/uv/).
Allow roughly 30 GB of free disk for the model, environment, and build.
The installer does not change drivers, system packages, clocks, or permissions.

```bash
git clone https://github.com/Luke458/qwen38-27b-exl3-rdna4.git
cd qwen38-27b-exl3-rdna4
bash tools/install.sh
```

This prepares the pinned fork plus baseline compatibility patches, installs the
ROCm PyTorch build into `.venv`, and compiles the extension for `gfx1201`.
Compilation can take several minutes. It refuses to overwrite an existing
checkout whose source differs from the requested profile.

Download the [tested checkpoint](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB)
separately, or use an existing copy:

```bash
.venv/bin/hf download GestaltLabs/Qwen3.8-27B-EXL3-11.5GB \
  --revision 0e6c4a863b945dbaf9657343fedd876c45e64dd5 \
  --local-dir ./models/qwen38-27b-exl3

.venv/bin/python tools/serve.py --model ./models/qwen38-27b-exl3
```

The endpoint is **`http://127.0.0.1:8000/v1`**; model ID is
**`qwen38-27b-exl3`**. Defaults: 8192-token cache, one active sequence,
256-token prefill chunks, 2048-token response cap, thinking disabled. The first
request may compile additional kernels. Stop with Ctrl-C.

> Serving is smoke-validated on gfx1201 (2026-09-24): health, models, chat,
> streaming and API-key auth all pass end-to-end. Builds made before
> `patches/0002-wmma-gfx12-asm.patch` will fault on the first request — see
> [docs/STATUS.md](docs/STATUS.md).

In another terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b-exl3","messages":[{"role":"user","content":"Explain EXL3 in two sentences."}],"max_tokens":128,"temperature":0.7,"stream":false}'

.venv/bin/python tools/smoke_api.py
```

Set `"stream": true` for SSE streaming. For an OpenAI-compatible client, use the
base URL above and the model ID above; use any placeholder API key if you have
not configured authentication. The compatibility target is
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
not the Responses API. See [serving details](docs/SERVING.md).

#### Experimental optimisation profile

Use a **separate clone/environment** and run `bash tools/install.sh experimental`
instead of the default installer. Then launch with:

```bash
EXL3_GEMV_FUSED_HAD=1 .venv/bin/python tools/serve.py --model /path/to/checkpoint
```

This builds the unroll-4 experiment and enables its fused output transform.
It is opt-in, not promoted. Setting the fusion variable on a baseline build
does not add the experimental kernel. Switching off fusion on an experimental
build does not remove its compile-time unrolling; use the baseline environment
for a true baseline comparison.

### Tests and research

```bash
uv pip install -p .venv pytest
.venv/bin/python -m pytest tests/test_oracle.py tests/test_arithmetic_bounds.py \
  tests/test_runner_failures.py tests/test_model_gates.py tests/test_serve.py -q
```

GPU tests require a built backend. Run them only with the server stopped and
the GPU otherwise idle. Research tools are in `bench/`, `reference/`, and
`tools/`; machine-specific baseline binaries, frozen policies, full logits, and
experiment history are intentionally not shipped. Start new local research
inputs from `configs/acceptance_policy.example.json`, generate a shape manifest
with `tools/build_shape_manifest.py --model /path/to/checkpoint`, and calibrate
and freeze your own gates before scoring candidates. Historical measured
summaries are evidence, not portable performance guarantees.

## Scope and safety

- Linux and one gfx1201 GPU: RX 9070 XT (tested) or Radeon AI PRO R9700 (same chip, untested). The plugin uses the `capicua25x/vllm-rocm-rdna4:0.28.0-rdna4`
  image (vLLM 0.28, ROCm 7.2.3). The standalone server uses Python 3.12, ROCm 7.2.4 and PyTorch 2.13.0+rocm7.2.
- Both servers bind to loopback. A bearer key is not a substitute for TLS, access controls, or a
  hardened proxy when exposing a server remotely.
- Do not set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` for the plugin, and do not force
  `EXL3_GEMV=0` in the standalone server. Both caused GPU faults on this card.
- The standalone server supports one active sequence and a subset of Chat Completions (no
  multimodal input or tool calls). Multi-row prefill there uses reconstruction plus dense
  multiplication, which avoids the fork's unsupported gfx12 WMMA path.

## Upstream and licensing

MIT-licensed (see [LICENSE](LICENSE)). The kernels build on
[CarouselAether/rocm_exl3](https://github.com/CarouselAether/rocm_exl3) at a pinned revision,
itself a port of [ExLlamaV3](https://github.com/turboderp-org/exllamav3). The vLLM plugin's
structure is adapted from [0xSero/exl3xpu](https://github.com/0xSero/exl3xpu), and it runs on
[vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4). See
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights are downloaded separately and
remain subject to their own license.
