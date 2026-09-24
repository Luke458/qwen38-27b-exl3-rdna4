# Qwen3.8-27B EXL3 on RDNA4

Run a local OpenAI-compatible chat endpoint for **Qwen3.8-27B EXL3 on an AMD
Radeon RX 9070 XT (gfx1201, 16 GB)**, with reproducible kernel experiments.

This is a narrowly tested research backend, not a general ROCm serving stack.
The target checkpoint is mixed-bitrate; kernel optimisation focuses on its
**3-bit `mul1` weights**. No weights or compiled binaries are distributed here.

## Status

Single-user inference works. The experimental fused/unrolled FP16 implementation
measured **1.0416× decode throughput** in ten paired trials, below the predeclared
1.05× promotion threshold. The baseline remains the default. The separate int8
experiment was slower and stays disabled. See [measured status](docs/STATUS.md)
and [next optimisation work](docs/OPTIMIZATION.md).

## Quickstart

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

### Experimental optimisation profile

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

- Linux, Python 3.12, ROCm 7.2.4, PyTorch 2.13.0+rocm7.2, gfx1201.
- One GPU and one active generation sequence. No MTP/speculative decoding.
- Multi-row prefill uses reconstruction plus dense multiplication to avoid the
  fork's unsupported gfx12 WMMA path. Packed single-token decode is retained.
- Do not force `EXL3_GEMV=0`: that diagnostic path can trap on this GPU.
- Bind to loopback by default. A local bearer key is not a substitute for TLS,
  access controls, or a hardened proxy when exposing a server remotely.
- Chat Completions compatibility is a subset, not the entire OpenAI API;
  Responses, multimodal input, and tool-call execution are not release targets.

## Upstream and licensing

Based on [CarouselAether/rocm_exl3](https://github.com/CarouselAether/rocm_exl3)
at a pinned revision, itself a port of
[ExLlamaV3](https://github.com/turboderp-org/exllamav3).
See [third-party notices](THIRD_PARTY_NOTICES.md). Model weights are downloaded
separately and remain subject to their own license.
