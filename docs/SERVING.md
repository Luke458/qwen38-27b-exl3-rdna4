# Serving

`tools/serve.py` adapts the pinned fork's existing server rather than introducing
a second generation engine. It bounds the loader and dynamic generator for this
checkpoint on the RX 9070 XT. Keep the source and compiled extension from the
same prepared profile.

## Authentication and network access

Loopback without authentication is the default. To require a local API key,
set `EXL3_API_KEY` in the server environment; clients must send
`Authorization: Bearer <your-key>`. The key is not a paid OpenAI credential.
The smoke test reads the same environment variable. Do not commit it.

Binding with `--host 0.0.0.0` requires that variable. This opens a listener on
all interfaces; it does not install firewall rules or TLS. Use a trusted network
or an authenticated TLS proxy, and rate/request-size limits, before remote
access. This small server is not hardened for untrusted Internet traffic.
Unrestricted upstream CORS is removed; browser frontends need an intentional
same-origin proxy. `/health` is unauthenticated; other routes require the key
when configured.

## Supported release surface

- `GET /health`, `GET /v1/models`.
- `POST /v1/chat/completions`: text messages, non-streaming JSON or SSE,
  temperature/top-p, stop strings, maximum completion tokens.
- Upstream raw completion and tokenisation routes remain available, but the
  chat route is the main tested integration.
- One completion per request (`n=1`); one active generation sequence. Multiple
  requests can queue. This is not a concurrency-throughput qualification.
- Tool calls and multimodal messages are rejected by the wrapper. There is no
  Responses, embeddings, speech, or image endpoint.
- Unknown upstream request fields may be ignored. Do not assume JSON-schema
  constrained output or every OpenAI sampling option is implemented.

Thinking is off by default. Use `--enable-thinking` to opt in, with an appropriate
token budget. The response cap is 2048 tokens and total history must fit the
8192-token cache; long conversations must be trimmed by the client. The server
does not silently increase its cache to the model's much larger advertised
context length.

## Local development binaries

The normal launcher uses the installed extension. The original research
workspace can use `--backend baseline` to select its hash-verified private
baseline artifact, or `--backend candidate --candidate DIR` for a compatible
build manifest. These artifacts are not included in GitHub and are not needed
after running the installer.

Do not disable ROCm compatibility patches. The wrapper strips the known unsafe
`EXL3_GEMV=0` diagnostic override and forces the int8 experiment off. It does
not make every upstream experimental environment switch safe.

## Troubleshooting

- Build errors: confirm ROCm 7.2.4, the ROCm PyTorch wheel, Python 3.12, and a
  `gfx1201` target. Do not replace ROCm PyTorch with a CUDA wheel.
- Out of memory: stop other GPU workloads; do not raise batch/chunk/cache sizes.
- First request slow: JIT compilation can add startup latency; later requests
  are warm. Direct-model benchmark speed is not a server latency guarantee.
- GPU fault: stop GPU work, save logs, and run `tools/preflight.py` before
  resuming. Do not automatically reset the GPU or kill unrelated applications.
- Changing build profile: use another clone/environment. Preparation refuses
  mismatched or dirty sources rather than resetting your changes.
