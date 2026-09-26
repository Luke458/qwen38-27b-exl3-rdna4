# Qwen3.8-27B EXL3 on RDNA4

Fast local serving of **Qwen3.8-27B (EXL3, 11.5 GB)** on an **AMD Radeon RX 9070 XT
(gfx1201, 16 GB)**: an OpenAI-compatible endpoint with speculative decoding, vision,
reasoning and tool calls, all within 16 GB of VRAM. The **Radeon AI PRO R9700 (32 GB)** is the same
Navi 48 / gfx1201 chip, so the same build runs there, with room for up to the model's full 262k context
(see [32 GB cards](#32-gb-cards-radeon-ai-pro-r9700)).

It is a vLLM plugin ([`vllm_plugin/`](vllm_plugin/README.md)) that runs the checkpoint's trellis-quantized
weights through custom gfx1201 kernels inside the community
[vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4) image. No weights or compiled binaries are
distributed here.

| profile | context | decode | notes |
|---|---:|---:|---|
| single user (MTP-3 speculative decoding) | 32k (40k option) | **~80 tok/s** (code ~99, prose ~80) | int8 KV cache |
| single user, 4-bit KV cache (MTP-3) | **48k** in the same VRAM as 32k, or **64k** | ~80 tok/s; 63–72 at 41–62k | slightly less accurate |
| multi-user / long context | 64k | 42 tok/s single stream, **~200 tok/s** total at 8 streams | int8 KV cache |
| R9700 32 GB, long context (MTP-3) | 128k / 262k | expected ~80 tok/s at short context | sized, not yet run on a 32 GB card |

All profiles load the vision tower. Prompt processing runs at about 1.5–1.6k tok/s, ~1k tok/s near 32k tokens and
~0.8k tok/s near 60k. Measured on 2026-09-26; details are in the [plugin README](vllm_plugin/README.md).

This is an experimental, narrowly tested setup: one card, one checkpoint, one GPU.

## Quickstart

Prerequisites: Linux with the amdgpu driver and access to `/dev/kfd` and `/dev/dri`,
rootless [podman](https://podman.io/), Git, Python 3 and [uv](https://docs.astral.sh/uv/).
ROCm does not need to be installed on the host because the container brings its own. Allow about
45 GB of disk: 28 GB for the vLLM image, 12 GB for the model and a few GB for the build.

```bash
git clone https://github.com/Luke458/qwen38-27b-exl3-rdna4.git
cd qwen38-27b-exl3-rdna4

# 1. build the kernel extension inside the vLLM image (~6 min; pulls the image on first use)
vllm_plugin/tools/build_ext_in_image.sh ~/models/exl3ext

# 2. download the tested checkpoint
uvx --from huggingface_hub hf download GestaltLabs/Qwen3.8-27B-EXL3-11.5GB \
  --revision 0e6c4a863b945dbaf9657343fedd876c45e64dd5 --local-dir ~/models/qwen3.8-27b-exl3-11.5gb

# 3. serve (32k context with MTP; see the profiles below)
./serve.sh
```

The build prints `source tree matches the tested build` when the fork pin and patches are the ones
that were measured. `serve.sh` checks VRAM headroom, then starts the server. Set `MODEL_DIR` or `EXL3_EXT_DIR`
if the model or the extension is somewhere other than the paths above. The first start compiles and
captures graphs for a few minutes; later starts reuse `~/.cache/vllm-rdna4-exl3`.

When the log shows `Application startup complete`, the endpoint is **`http://127.0.0.1:8000/v1`** with
model ID **`qwen38-27b-exl3`** (loopback only, no API key):

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b-exl3","messages":[{"role":"user","content":"Write a haiku about GPUs."}],
       "max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
python3 vllm_plugin/tools/bench_client.py --model qwen38-27b-exl3   # decode tok/s
```

Thinking is on by default: drop `chat_template_kwargs` and the model thinks first, returning its thoughts
in the message's `reasoning` field (allow a larger `max_tokens`). OpenAI-style `tools` come back as
`tool_calls`, and images are accepted as `image_url` content parts (capped at 1 MP).
Stop the server with Ctrl-C or `podman stop vllm-exl3`.

## Profiles (16 GB)

`./serve.sh <profile>` picks one. The server's peak VRAM is measured with a near-maximum prompt, a 1 MP image
and 4 concurrent streams. Your desktop's VRAM use (`rocm-smi --showmeminfo vram` before starting) comes on
top of it, out of the card's 16,304 MiB; the last column is how much that leaves for it.

| profile | context | KV cache | decode | server peak | desktop VRAM that fits |
|---|---:|---|---|---:|---:|
| `32k` (default) | 32,768 | int8 | ~80 tok/s with MTP-3 | 14,755 MiB | ~1.3 GiB |
| `40k` | 40,960 | int8 | ~78 tok/s with MTP-3 | 15,122 MiB | ~0.9 GiB |
| `48k-int4` | 49,152 | 4-bit | ~80 tok/s with MTP-3, 72 at 41k | ~14,800 MiB | ~1.3 GiB |
| `64k-int4` | 65,536 | 4-bit | ~80 tok/s with MTP-3, 63 at 62k | ~15,300 MiB (est.) | ~0.8 GiB |
| `64k` | 65,536 | int8 | 42 tok/s; ~200 total at 8 streams | 15,171 MiB | ~0.9 GiB |

- **int8 KV cache:** its teacher-forced logits match an fp16 cache.
- **4-bit KV cache:** holds twice as many tokens per byte. It keeps each sequence's first 4 tokens and newest
  128 tokens exact, and decodes close to fp16 but measurably different: KL 0.0009–0.0012 on long text (int8:
  0.00007), top-1 agreement 99.1–99.2%. Speed matches int8 at short context and is faster at long context.
- **64k-int4 peak:** the ~ figure is estimated (text-only measured, plus the vision tower); the others are
  measured.
- **Text-only:** `EXL3_TEXT_ONLY=1 ./serve.sh ...` skips the vision tower and saves ~0.45 GiB.
- **Extra arguments** after the profile go to `vllm serve`, e.g. `./serve.sh 32k --generation-config vllm`.
- **Manual launch:** `serve.sh` wraps `vllm_plugin/run_exl3_server.sh`; the
  [plugin README](vllm_plugin/README.md) lists each profile's flags, memory notes and the tools.

## 32 GB cards (Radeon AI PRO R9700)

The R9700 uses the same Navi 48 chip as the RX 9070 XT: gfx1201, 64 compute units and a 256-bit GDDR6 bus. So the
extension build, the plugin and the image are the same, and decode speed, which is limited by memory bandwidth,
should match (~80 tok/s with MTP, ~42 without). The extra 16 GB goes to KV cache:

| `./serve.sh` profile | context | KV pool | estimated server peak |
|---|---:|---:|---:|
| `128k` | 131,072 | 5.25 GB | ~18,700 MiB |
| `256k` | 262,144 (the model's maximum) | 9.9 GB | ~23,800 MiB |

These are **untested**: no 32 GB card was available.

- **Sizing:** the KV pools use the same page math as the measured 16 GB profiles (28,853,760-byte pages, with
  at least 3 spare so a full-length prompt cannot stall). The peaks are the measured non-KV footprint plus
  the pool plus the prefill-time fp16 copy of the context's KV. The script refuses these profiles on cards
  with less than ~30 GB.
- **4-bit KV cache:** `--kv-cache-dtype int4_per_token_head` would roughly halve these pools (262k in
  ~5.3 GB), also untested.
- **Speed at long context:** very long prompts are slow to read the first time, roughly 4 minutes for 128k
  tokens and 15 minutes for 262k on current measurements (prefix caching makes follow-up turns fast).
  Decode also slows as the context fills (~60–70 tok/s at 40–60k).
- **Limits:** the plugin is single-GPU, so two cards cannot be pooled with tensor parallelism. On a 32 GB card
  you can raise `--max-num-seqs` for more concurrent long requests. The plugin's startup check warns if the
  KV pool is too small for the chosen context.

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

## Other routes and research

- [Standalone server](docs/STANDALONE.md): the original single-user server on the ROCm fork's runtime without
  vLLM (~29 tok/s), kept for reference and for kernel experiments.
- [Kernel status](docs/STATUS.md), [optimisation notes](docs/OPTIMIZATION.md), [decode trace](docs/DECODE_TRACE.md)
  and the [vLLM port assessment](docs/VLLM_PORT_ASSESSMENT.md): the research that led to the plugin.

## Scope and safety

- Linux and one gfx1201 GPU: RX 9070 XT (tested) or Radeon AI PRO R9700 (same chip, untested), with the
  `capicua25x/vllm-rocm-rdna4:0.28.0-rdna4` image (vLLM 0.28, ROCm 7.2.3).
- The server binds to loopback. Exposing it remotely needs TLS, access controls or a hardened proxy.
- Do not set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`: it caused a GPU fault on this card.

## Upstream and licensing

MIT-licensed (see [LICENSE](LICENSE)). The kernels build on
[CarouselAether/rocm_exl3](https://github.com/CarouselAether/rocm_exl3) at a pinned revision,
itself a port of [ExLlamaV3](https://github.com/turboderp-org/exllamav3). The vLLM plugin's
structure is adapted from [0xSero/exl3xpu](https://github.com/0xSero/exl3xpu), and it runs on
[vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4); its 4-bit KV attention kernel is adapted
from vLLM's (Apache-2.0). See [third-party notices](THIRD_PARTY_NOTICES.md). Model weights are downloaded
separately and remain subject to their own license.
