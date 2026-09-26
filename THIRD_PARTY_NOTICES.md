# Third-party notices

This repository contains patches to [CarouselAether/rocm_exl3](https://github.com/CarouselAether/rocm_exl3), pinned to commit `311ff5497237a37bea18cf24aa18bc0c573d1d03`. The fork is not included in this source repository; `tools/prepare_backend.py` clones it into the ignored `vendor/rocm_exl3` directory and applies the selected patch profile.

The fork is distributed under the MIT License, copyright (c) 2025 Turboderp. Its license text is retained at `vendor/rocm_exl3/LICENSE` after preparation. Its bundled `exllamav3/vendor/fla` code has a separate MIT License, copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li, retained at `vendor/rocm_exl3/exllamav3/vendor/fla/LICENSE`.

The upstream MIT notice is also included in `patches/UPSTREAM-LICENSE` because
the patch files contain upstream source context.

The EXL3 model weights are not included. The tested
[GestaltLabs checkpoint](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB)
is listed under Apache-2.0; obtain its license and attribution alongside the
weights. This project's MIT license does not replace model or dependency terms.

## 0xSero/exl3xpu (MIT)

`vllm_plugin/exl3rocm/plugin.py` adapts the vLLM quantization-plugin structure of
[0xSero/exl3xpu](https://github.com/0xSero/exl3xpu) (commit 6872a30): checkpoint-driven
EXL3 module discovery, fused-module shard handling with per-shard input scales, and the
plugin entry point. `vllm_plugin/patches/vllm-0.28.0-rdna4/gdn_attn_mask_sync.patch` ports
its GDN metadata mask-index fix. License text:

```
MIT License

Copyright (c) 2026 0xSero

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## vLLM (Apache-2.0)

`vllm_plugin/patches/vllm-0.28.0-rdna4/gdn_attn.py` is a modified copy of
`vllm/v1/attention/backends/gdn_attn.py` from vLLM 0.28.0 (as shipped in the
`capicua25x/vllm-rocm-rdna4:0.28.0-rdna4` image), copyright contributors to the vLLM
project, licensed under the Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0). The changes are the GDN metadata
mask-index fix and one metadata build per step shared across GDN layers; the
`.patch` files next to it show the exact differences.

`vllm_plugin/exl3rocm/kv_int4.py` uses the cache format of vLLM 0.28.0's
`vllm/v1/attention/ops/int4_per_token_head.py` and adapts its attention kernel (`_attn_packed`, same
copyright and license): the nibble layout, zero-point-in-scale encoding, randomized Hadamard sign vector and
the split-KV / online-softmax structure come from it. The plugin otherwise only imports vLLM at runtime and
does not redistribute it.
