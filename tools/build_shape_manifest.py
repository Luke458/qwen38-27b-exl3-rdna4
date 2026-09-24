#!/usr/bin/env python3
"""Generate configs/shape_manifest.json from the actual checkpoint files.

An inventory generated from actual files is authoritative (IMPLEMENTATION_PLAN):
reads quantization_config.json's tensor_storage (per-module stored tensors,
explicit bits_per_weight / quant_format / mul1_multiplier) plus model config,
and emits every linear module's name, M/N/K_in, actual bitrate, codebook,
packed layout, transforms, dtype and per-decode-step invocation count.

Usage: python tools/build_shape_manifest.py [--model DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

DEFAULT_MODEL = os.path.expanduser("~/models/qwen3.8-27b-exl3-11.5gb")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_dims(text_cfg: dict, name: str, shapes: dict) -> dict:
    """Infer (N, K_in) from stored tensor shapes and the model config.

    Convention (verified against the checkpoint): for module M with
    trellis (K_in/16, N/16, 16*bits), suh [K_in], svh [N].
    """
    suh = shapes.get("suh")
    svh = shapes.get("svh")
    trellis = shapes.get("trellis")
    out = {}
    if suh and svh and trellis:
        out["K_in"] = suh[0]
        out["N"] = svh[0]
        out["trellis_shape"] = trellis
        # cross-check against trellis layout (K_in/16, N/16, per-tile u16)
        assert trellis[0] * 16 == out["K_in"] and trellis[1] * 16 == out["N"], (name, trellis, out)
    return out


def role_of(name: str) -> str:
    return name.split(".")[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="configs/shape_manifest.json")
    args = ap.parse_args()

    qpath = os.path.join(args.model, "quantization_config.json")
    cpath = os.path.join(args.model, "config.json")
    q = json.load(open(qpath))
    cfg = json.load(open(cpath))
    tcfg = cfg.get("text_config", cfg)

    ts = q["tensor_storage"]
    modules = []
    bits_hist = Counter()
    for name, entry in sorted(ts.items()):
        shapes = {
            t.split(".")[-1]: info["shape"]
            for t, info in entry["stored_tensors"].items()
        }
        dtypes = {
            t.split(".")[-1]: info["dtype"]
            for t, info in entry["stored_tensors"].items()
        }
        bpw = entry.get("bits_per_weight")
        if bpw is None:
            continue  # non-exl3 module (embeddings, norms, biases)
        dims = infer_dims(tcfg, name, shapes)
        bits_hist[bpw] += 1
        per_tile = 16 * bpw  # u16 words per 16x16 tile
        assert shapes["trellis"][2] == per_tile, (name, shapes["trellis"], per_tile)

        role = role_of(name)
        layers = int(tcfg["num_hidden_layers"])
        full_iv = int(tcfg.get("full_attention_interval", 4))
        is_mtp = name.startswith("mtp.")
        is_vision = name.startswith(("model.visual", "vision_model"))
        # decode-step invocation counts (M=1 single request)
        invoke = 0 if (is_mtp or is_vision) else 1

        modules.append({
            "name": name,
            "role": role,
            "M": 1,
            "N": dims["N"],
            "K_in": dims["K_in"],
            "bits": bpw,
            "codebook": "mul1" if entry.get("mul1_multiplier") is not None else q.get("codebook"),
            "mul1_multiplier": entry.get("mul1_multiplier"),
            "quant_format": entry.get("quant_format"),
            "trellis_shape": dims["trellis_shape"],
            "trellis_dtype": dtypes.get("trellis"),
            "suh_shape": shapes.get("suh"),
            "svh_shape": shapes.get("svh"),
            "suh_dtype": dtypes.get("suh"),
            "transforms": "suh pre-scale + Had128 in; Had128 + svh post-scale out (r_scale=1/sqrt(128))",
            "invocations_per_decode_step": invoke,
            "eligible_v1": bool(
                bpw == 3
                and entry.get("mul1_multiplier") is not None
                and dims["K_in"] % 128 == 0
                and dims["N"] % 128 == 0
                and not is_mtp and not is_vision
            ),
        })

    eligible = [m for m in modules if m["eligible_v1"]]
    # weighted eligible time share proxy: bits-scaled weight bytes per decode step
    total_proxy = sum(m["K_in"] * m["N"] * m["bits"] for m in modules if m["invocations_per_decode_step"])
    elig_proxy = sum(m["K_in"] * m["N"] * m["bits"] for m in eligible)

    groups = Counter(
        (m["role"], m["N"], m["K_in"], m["bits"], m["eligible_v1"]) for m in modules
    )
    manifest = {
        "model": args.model,
        "model_sha256_quantization_config": sha256_file(qpath),
        "model_sha256_config": sha256_file(cpath),
        "generated_by": "tools/build_shape_manifest.py",
        "architecture": cfg.get("architectures"),
        "text_config": {
            k: tcfg.get(k) for k in (
                "hidden_size", "intermediate_size", "num_hidden_layers",
                "full_attention_interval", "num_attention_heads", "num_key_value_heads",
                "head_dim", "vocab_size", "mtp_num_hidden_layers",
            )
        },
        "quant_header": {k: v for k, v in q.items() if k != "tensor_storage"},
        "bits_histogram_modules": dict(sorted(bits_hist.items())),
        "n_modules": len(modules),
        "n_eligible_v1": len(eligible),
        "eligible_weighted_share_proxy": (elig_proxy / total_proxy) if total_proxy else None,
        "shape_groups": [
            {"role": role, "M": 1, "N": n, "K_in": k, "bits": bits,
             "eligible_v1": elig, "count": count}
            for (role, n, k, bits, elig), count in sorted(groups.items())
        ],
        "modules": modules,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, args.out)

    print(f"wrote {args.out}: {len(modules)} exl3 modules, {len(eligible)} eligible v1")
    for m in eligible[:6]:
        print(f"  eligible: {m['name']} N={m['N']} K_in={m['K_in']} bits={m['bits']}")
    print(f"  eligible weighted share (bits*N*K proxy): {manifest['eligible_weighted_share_proxy']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
