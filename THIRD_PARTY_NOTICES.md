# Third-party notices

This repository contains patches to [CarouselAether/rocm_exl3](https://github.com/CarouselAether/rocm_exl3), pinned to commit `311ff5497237a37bea18cf24aa18bc0c573d1d03`. The fork is not included in this source repository; `tools/prepare_backend.py` clones it into the ignored `vendor/rocm_exl3` directory and applies the selected patch profile.

The fork is distributed under the MIT License, copyright (c) 2025 Turboderp. Its license text is retained at `vendor/rocm_exl3/LICENSE` after preparation. Its bundled `exllamav3/vendor/fla` code has a separate MIT License, copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li, retained at `vendor/rocm_exl3/exllamav3/vendor/fla/LICENSE`.

The upstream MIT notice is also included in `patches/UPSTREAM-LICENSE` because
the patch files contain upstream source context.

The EXL3 model weights are not included. The tested
[GestaltLabs checkpoint](https://huggingface.co/GestaltLabs/Qwen3.8-27B-EXL3-11.5GB)
is listed under Apache-2.0; obtain its license and attribution alongside the
weights. This project's MIT license does not replace model or dependency terms.
