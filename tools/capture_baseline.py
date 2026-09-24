#!/usr/bin/env python3
"""Capture the immutable baseline record set under artifacts/baseline/.

Records binary/patch/policy/workload/model/fixture hashes, hardware manifest,
and (optionally) fresh benchmark snapshots. The baseline binary and records are
frozen from this point; candidates must never modify them.

  python tools/capture_baseline.py [--run-bench]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.expanduser("~/models/qwen3.8-27b-exl3-11.5gb")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-bench", action="store_true")
    args = ap.parse_args()

    so = os.path.join(ROOT, ".venv/lib/python3.12/site-packages/exllamav3_ext.cpython-312-x86_64-linux-gnu.so")
    baseline_dir = os.path.join(ROOT, "artifacts", "baseline")
    out = os.path.join(baseline_dir, "records.json")
    if os.path.exists(out):
        print("BLOCKED: baseline records already frozen", file=sys.stderr)
        return 2
    if not os.path.isfile(so):
        print("BLOCKED: installed extension missing", file=sys.stderr)
        return 2
    digest = sha256_file(so)
    frozen_binary = os.path.join(baseline_dir, "binary", os.path.basename(so))
    os.makedirs(os.path.dirname(frozen_binary), exist_ok=True)
    if os.path.exists(frozen_binary):
        if sha256_file(frozen_binary) != digest:
            print("BLOCKED: existing frozen binary differs", file=sys.stderr)
            return 2
    else:
        with open(so, "rb") as src, open(frozen_binary, "xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
    rec = {
        "ts": time.time(),
        "source_commit": "311ff5497237a37bea18cf24aa18bc0c573d1d03",
        "patches": {
            fn: sha256_file(os.path.join(ROOT, "artifacts/baseline/patches", fn))
            for fn in sorted(os.listdir(os.path.join(ROOT, "artifacts/baseline/patches")))
        },
        "binary_sha256": digest,
        "binary": os.path.relpath(frozen_binary, ROOT),
        "env": {
            "torch": subprocess.run([os.path.join(ROOT, ".venv/bin/python"), "-c",
                                     "import torch; print(torch.__version__, torch.version.hip)"],
                                    capture_output=True, text=True).stdout.strip(),
            "rocm": open("/opt/rocm/.info/version").read().strip(),
            "python": subprocess.run([os.path.join(ROOT, ".venv/bin/python"), "--version"],
                                     capture_output=True, text=True).stdout.strip(),
        },
        "model": {
            "path": MODEL,
            "quantization_config_sha256": sha256_file(os.path.join(MODEL, "quantization_config.json")),
            "config_sha256": sha256_file(os.path.join(MODEL, "config.json")),
            "index_sha256": sha256_file(os.path.join(MODEL, "model.safetensors.index.json")),
            "tokenizer_sha256": sha256_file(os.path.join(MODEL, "tokenizer.json")),
            "safetensors_sha256": {
                fn: sha256_file(os.path.join(MODEL, fn))
                for fn in sorted(os.listdir(MODEL)) if fn.endswith(".safetensors")
            },
        },
        "frozen_inputs": {
            fn: sha256_file(os.path.join(ROOT, "configs", fn))
            for fn in ("workloads.json", "shape_manifest.json", "acceptance_policy.frozen.json")
            if os.path.isfile(os.path.join(ROOT, "configs", fn))
        },
        "fixtures": {
            fn: sha256_file(os.path.join(ROOT, "fixtures", fn))
            for fn in sorted(os.listdir(os.path.join(ROOT, "fixtures")))
        },
        "gpu_parity_stats": json.load(open(os.path.join(ROOT, "artifacts/baseline/gpu_parity_stats.json")))
        if os.path.isfile(os.path.join(ROOT, "artifacts/baseline/gpu_parity_stats.json")) else None,
    }

    if args.run_bench:
        bench_out = os.path.join(ROOT, "artifacts/baseline/bench_layer-captured.json")
        subprocess.run([os.path.join(ROOT, ".venv/bin/python"),
                        os.path.join(ROOT, "bench/bench_layer.py"), "--out", bench_out], check=True)
        rec["bench_layer"] = json.load(open(bench_out))

    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, out)
    print(f"wrote {out}")
    print("binary sha256:", rec["binary_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
