#!/usr/bin/env python3
"""Decode-step time breakdown: where does one token's ~36 ms actually go?

Nesting-aware exclusive GPU timing per module category across N steady decode
steps. Every exllamav3 module forward is bracketed by a GPU event pair; the
call-order sequence reconstructs the module tree (stack parsing), and each
node's exclusive time is its duration minus its wrapped children's. Categories:

  gemv_3bit_eligible   LinearEXL3 in v1 scope (bits=3, mul1) -- the replaceable share
  gemv_fallback        LinearEXL3 outside it (lm_head, bits!=3, ...)
  linear_attn          GDN layers (core time beyond their projections)
  attn_full            full-attention layers (core time beyond their projections)
  other_forwards       norms/gates/embeddings/sampling (exclusive)
  unattributed         wall - GPU (Python/launch gaps)

Run via the hash-verified launcher:

  .venv/bin/python tools/run_backend.py -- bench/bench_breakdown.py \
      --out artifacts/.../decode_breakdown.json --steps 64
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

DEFAULT_MODEL = os.path.expanduser("~/models/qwen3.8-27b-exl3-11.5gb")

_EVENT_STATE = {"events": None, "seq": 0, "cats": {}}


def install_class_hooks():
    """Patch forward() at CLASS level for every exllamav3 module class.

    Must run BEFORE model construction: some modules capture bound methods at
    __init__ and some call paths are class-qualified, so instance-attribute
    wrapping is silently bypassed. Per-instance categories come from the
    registry filled during the tree walk; unregistered instances pass through.
    """
    import exllamav3
    import pkgutil, importlib, inspect
    seen_cls = set()
    for modinfo in pkgutil.walk_packages(exllamav3.__path__, "exllamav3."):
        try:
            m = importlib.import_module(modinfo.name)
        except Exception:
            continue
        for obj in vars(m).values():
            if not (isinstance(obj, type) and obj.__module__.startswith("exllamav3")):
                continue
            if not hasattr(obj, "forward") or obj in seen_cls:
                continue
            seen_cls.add(obj)
            orig = obj.forward

            def make(orig=orig):
                def forward(self, *a, **kw):
                    st = _EVENT_STATE
                    cat = st["cats"].get(id(self))
                    if cat is None or st["events"] is None:
                        return orig(self, *a, **kw)
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    en = st["seq"]; st["seq"] += 1
                    out = orig(self, *a, **kw)
                    ev1.record()
                    ex = st["seq"]; st["seq"] += 1
                    st["events"].append((cat, ev0, ev1, en, ex))
                    return out
                return forward

            obj.forward = make()
    return len(seen_cls)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classify(module, name: str, eligible: set) -> str:
    cls = type(module).__name__
    if "LinearEXL3" in cls or cls == "Linear":
        return "gemv_3bit_eligible" if name in eligible else "gemv_fallback"
    if cls == "GatedMLP":
        # C++ BC path: gate/up fused + down GEMVs run inside this forward; its
        # exclusive time is the (mostly eligible) MLP projection work
        return "mlp_gemv_host"
    if "GatedDelta" in cls or "GDN" in cls or "Gla" in cls or "LinearAttn" in cls:
        return "linear_attn"
    if "Attention" in cls or "Attn" in cls or "QSA" in cls or "MLA" in cls or "Sliding" in cls:
        return "attn_full"
    return "other_forwards"


def module_children(mod):
    out = []
    seen = set()

    def consider(v):
        if hasattr(v, "forward") and id(v) not in seen and not isinstance(v, type):
            tn = type(v).__name__
            if type(v).__module__.startswith("exllamav3") or "LinearEXL3" in tn:
                seen.add(id(v))
                out.append(v)

    def scan(v):
        if isinstance(v, (list, tuple)):
            for x in v:
                scan(x)
        elif isinstance(v, dict):
            for x in v.values():
                scan(x)
        else:
            consider(v)

    for src in (getattr(mod, "modules", None) or []):
        scan(src)
    for v in vars(mod).values():
        scan(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--manifest", default="configs/shape_manifest.json")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.cache import Cache, CacheLayer_fp16

    manifest = json.load(open(os.path.join(ROOT, args.manifest)))
    eligible = {m["name"] for m in manifest["modules"] if m.get("eligible_v1")}

    n_hooked = install_class_hooks()
    cfg = Config.from_directory(args.model)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=8192, layer_type=CacheLayer_fp16,
                  max_history=0, max_batch_size=1)
    model.load(progressbar=False, max_chunk_size=256)
    tok = Tokenizer.from_config(cfg)

    _EVENT_STATE["events"] = events = []
    instance_cat = _EVENT_STATE["cats"]
    wrapped = set()

    def walk(mod, name):
        if id(mod) in wrapped:
            return
        wrapped.add(id(mod))
        instance_cat[id(mod)] = classify(mod, name, eligible)
        for child in module_children(mod):
            walk(child, getattr(child, "key", name + "/" + type(child).__name__))

    for top in (getattr(model, "modules", None) or []):
        walk(top, getattr(top, "key", type(top).__name__))
    for attr in vars(model).values():
        if hasattr(attr, "forward") and type(attr).__module__.startswith("exllamav3"):
            walk(attr, getattr(attr, "key", type(attr).__name__))

    # ---- prefill (cache mode; recurrent states threaded) ----
    capacity = 8192
    text = "The quick brown fox jumps over the lazy dog and keeps running until"
    ids = tok.encode(text)[0].tolist()
    seq = (ids * (args.prompt_tokens // len(ids) + 2))[:args.prompt_tokens]
    input_ids = torch.tensor([seq], dtype=torch.long)
    seq_in = input_ids[:, :-1]
    states = None
    for pos in range(0, seq_in.shape[1], 128):
        chunk = seq_in[:, pos:min(pos + 128, seq_in.shape[1])]
        p = {"attn_mode": "flash_attn", "cache": cache,
             "batch_shape": (1, capacity), "past_len": pos}
        if states is not None:
            p["recurrent_states"] = states
        model.prefill(chunk, p)
        states = p.get("recurrent_states")
    torch.cuda.synchronize()

    # ---- steady decode steps ----
    walls = []
    step = input_ids[:, -1:]
    pos = seq_in.shape[1]

    def forward_token(token, pos, states):
        p = {"attn_mode": "flash_attn", "cache": cache,
             "batch_shape": (1, capacity), "past_len": pos,
             "recurrent_states": states, "last_tokens_only": 1}
        lo = model.forward(token, p)
        return lo, p.get("recurrent_states")

    lo, states = forward_token(step, pos, states)   # prime
    step = lo[:, -1, :].argmax(dim=-1, keepdim=True)
    pos += 1
    torch.cuda.synchronize()
    events.clear()
    _EVENT_STATE["seq"] = 0

    for _ in range(args.steps):
        t0 = time.perf_counter()
        lo, states = forward_token(step, pos, states)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        step = lo[:, -1, :].argmax(dim=-1, keepdim=True)
        pos += 1

    # ---- stack-parse the call sequence into a tree, compute exclusive times ----
    marks = []
    for cat, ev0, ev1, en, ex in events:
        marks.append((en, "enter", cat, ev0, ev1))
        marks.append((ex, "exit", cat, ev0, ev1))
    marks.sort(key=lambda m: m[0])
    node_of = {}
    stack = []
    for _, kind, cat, ev0, ev1 in marks:
        if kind == "enter":
            node = {"cat": cat, "ev0": ev0, "ev1": ev1, "children": []}
            if stack:
                stack[-1]["children"].append(node)
            stack.append(node)
            node_of[(id(ev0), id(ev1))] = node
        else:
            stack.pop()
    roots = [n for k, n in node_of.items() if not any(k in id(p) for p in [])]  # placeholder
    # recompute roots properly: nodes not appended to any parent
    all_nodes = list(node_of.values())
    child_ids = {id(c) for n in all_nodes for c in n["children"]}
    roots = [n for n in all_nodes if id(n) not in child_ids]

    def dur(ev0, ev1):
        return ev0.elapsed_time(ev1)

    def excl(n):
        d = dur(n["ev0"], n["ev1"])
        for c in n["children"]:
            d -= dur(c["ev0"], c["ev1"])
        return d

    gpu_cat = {}
    call_cat = {}
    for n in all_nodes:
        gpu_cat[n["cat"]] = gpu_cat.get(n["cat"], 0.0) + excl(n)
        call_cat[n["cat"]] = call_cat.get(n["cat"], 0) + 1

    nsteps = args.steps
    rec = {
        "steps": nsteps,
        "step_wall_ms_mean": float(np.mean(walls)) * 1e3,
        "step_wall_ms_median": float(np.median(walls)) * 1e3,
        "gpu_exclusive_ms_per_step": {k: v / nsteps for k, v in sorted(gpu_cat.items())},
        "calls_per_step": {k: c / nsteps for k, c in sorted(call_cat.items())},
    }
    gpu_sum = sum(rec["gpu_exclusive_ms_per_step"].values())
    rec["gpu_sum_exclusive_ms_per_step"] = gpu_sum
    rec["unattributed_ms_per_step"] = rec["step_wall_ms_median"] - gpu_sum
    rec["in_scope_replaceable_share"] = (
        (rec["gpu_exclusive_ms_per_step"].get("gemv_3bit_eligible", 0.0)
         + rec["gpu_exclusive_ms_per_step"].get("mlp_gemv_host", 0.0))
        / rec["step_wall_ms_median"]
    )

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(out + ".tmp", out)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
