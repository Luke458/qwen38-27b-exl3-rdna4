#!/usr/bin/env python3
"""Batch-one generation benchmark and identical-history teacher-forced logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.expanduser("~/models/qwen3.8-27b-exl3-11.5gb")
PREFILL_CHUNK = 128


def load_prompt(split: str, index: int) -> str:
    path = os.path.join(ROOT, "fixtures", f"prompts_{split}.jsonl")
    with open(path) as f:
        prompts = [json.loads(line)["text"] for line in f if line.strip()]
    if not 0 <= index < len(prompts):
        raise ValueError(f"prompt index {index} outside {split} fixture range")
    return prompts[index]


def fixed_tokens(encoded: list[int], length: int) -> list[int]:
    if not encoded or length < 1:
        raise ValueError("tokenized prompt must be nonempty and length positive")
    return (encoded * math.ceil(length / len(encoded)))[:length]


def prefill(model, cache, ids: torch.Tensor, capacity: int, states=None):
    """Fill cache through the penultimate token, preserving recurrent state."""
    elapsed = 0.0
    for pos in range(0, ids.shape[1] - 1, PREFILL_CHUNK):
        chunk = ids[:, pos:min(pos + PREFILL_CHUNK, ids.shape[1] - 1)]
        params = {"attn_mode": "flash_attn", "cache": cache,
                  "batch_shape": (1, capacity), "past_len": pos}
        if states is not None:
            params["recurrent_states"] = states
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model.prefill(chunk, params)
        end.record()
        end.synchronize()
        elapsed += start.elapsed_time(end)
        states = params.get("recurrent_states")
    return elapsed, states


def forward_one(model, cache, token: torch.Tensor, pos: int, capacity: int, states):
    params = {"attn_mode": "flash_attn", "cache": cache,
              "batch_shape": (1, capacity), "past_len": pos,
              "recurrent_states": states, "last_tokens_only": 1}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    logits = model.forward(token, params)
    end.record()
    end.synchronize()
    return logits, start.elapsed_time(end)


def decode_one(model, cache, token: torch.Tensor, pos: int, capacity: int, states):
    """Time the model forward plus greedy selection for one real token."""
    params = {"attn_mode": "flash_attn", "cache": cache,
              "batch_shape": (1, capacity), "past_len": pos,
              "recurrent_states": states, "last_tokens_only": 1}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    logits = model.forward(token, params)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    end.record()
    end.synchronize()
    return next_token, start.elapsed_time(end)


def release(states) -> None:
    if states:
        for state in states:
            state.free()


def run_generation(model, cache, prompt: list[int], steps: int, capacity: int) -> dict:
    ids = torch.tensor([prompt], dtype=torch.long)
    states = None
    torch.cuda.reset_peak_memory_stats()
    try:
        prefill_ms, states = prefill(model, cache, ids, capacity)
        step = ids[:, -1:]
        durations, generated = [], []
        for i in range(steps):
            step, duration = decode_one(model, cache, step, len(prompt)-1+i,
                                        capacity, states)
            durations.append(duration)
            generated.append(int(step.item()))
            # The fork's direct-model example feeds CPU token IDs; keep the
            # next input on CPU even when the logits were produced on GPU.
            step = step.cpu()
        steady = np.asarray(durations[1:], dtype=np.float64)
        return {"prefill_ms": prefill_ms,
                "first_decode_ms": durations[0],
                "ttft_ms": prefill_ms + durations[0],
                "tpot_ms_raw": durations[1:],
                "tpot_median_ms": float(np.median(steady)) if steady.size else None,
                "tokens_per_s": float(1000.0 / steady.mean()) if steady.size else None,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "peak_vram_method": "torch.cuda.max_memory_allocated allocator high-water",
                "generated_head": generated[:16], "decode_steps": steps}
    finally:
        release(states)


def capture_logits(model, cache, history: list[int], prompt_len: int,
                   capacity: int) -> np.ndarray:
    """Predict the fixed continuation, independent of candidate generation."""
    ids = torch.tensor([history[:prompt_len]], dtype=torch.long)
    states, rows = None, []
    try:
        _, states = prefill(model, cache, ids, capacity)
        for pos in range(prompt_len - 1, len(history) - 1):
            token = torch.tensor([[history[pos]]], dtype=torch.long)
            logits, _ = forward_one(model, cache, token, pos, capacity, states)
            rows.append(logits[0, -1, :].to(torch.float16).cpu().numpy())
        return np.stack(rows)
    finally:
        release(states)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workloads", default=os.path.join(ROOT, "configs/workloads.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["calibration", "holdout"], default="calibration")
    ap.add_argument("--teacher-forced-out")
    ap.add_argument("--mode", choices=["eager", "graph"], default="eager")
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--trace-modules", action="store_true",
                    help="diagnostic only: synchronize and log each module boundary")
    ap.add_argument("--workload-id", help="run one workload (for smoke checks)")
    ap.add_argument("--teacher-steps", type=int,
                    help="capture this many fixed-history logits instead of all decode steps")
    ap.add_argument("--decode-steps", type=int,
                    help="run this many decode steps instead of the frozen workload length (smoke only)")
    args = ap.parse_args()
    if args.mode == "graph":
        ap.error("graph mode is not implemented by this direct-model runner")

    with open(args.workloads, "rb") as f:
        workload_bytes = f.read()
    workloads = json.loads(workload_bytes)
    selected = [w for w in workloads["workloads"]
                if args.workload_id is None or w["id"] == args.workload_id]
    if not selected:
        ap.error(f"unknown workload ID: {args.workload_id}")
    if args.teacher_steps is not None and args.teacher_steps < 1:
        ap.error("--teacher-steps must be positive")
    if args.decode_steps is not None and args.decode_steps < 2:
        ap.error("--decode-steps must be at least 2 for steady TPOT")
    text = load_prompt(args.split, args.prompt_index)

    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.cache import Cache, CacheLayer_fp16

    cfg = Config.from_directory(args.model)
    model = Model.from_config(cfg)
    if args.trace_modules:
        for module in model:
            original = module.forward
            def traced(*a, _original=original, _key=module.key, **kw):
                torch.cuda.synchronize()
                print(f"ENTER {_key} shape={tuple(a[0].shape) if a else None}", flush=True)
                value = _original(*a, **kw)
                torch.cuda.synchronize()
                print(f"EXIT {_key}", flush=True)
                return value
            module.forward = traced
    capacity = max(math.ceil((w["prompt_tokens"] + max(args.decode_steps or w["decode_tokens"],
                                                       args.teacher_steps or 0)) / 256) * 256
                   for w in selected)
    cache = Cache(model, max_num_tokens=capacity, layer_type=CacheLayer_fp16,
                  max_history=0, max_batch_size=1)
    t0 = time.monotonic()
    model.load(progressbar=False, max_chunk_size=256, max_output_size=1)
    load_s = time.monotonic() - t0
    tokenizer = Tokenizer.from_config(cfg)
    encoded = tokenizer.encode(text)[0].tolist()
    metrics = {"mode": args.mode, "split": args.split,
               "prompt_index": args.prompt_index, "model_load_s": load_s,
               "smoke_override": args.decode_steps is not None or args.teacher_steps is not None,
               "workloads_sha256": hashlib.sha256(workload_bytes).hexdigest(),
               "vram_after_load_bytes": torch.cuda.memory_allocated(), "workloads": {}}

    run_generation(model, cache, fixed_tokens(encoded, 128), 2, capacity)
    for w in selected:
        n_prompt = w["prompt_tokens"]
        n_decode = args.decode_steps or w["decode_tokens"]
        if n_prompt + n_decode > capacity:
            metrics["workloads"][w["id"]] = {"status": "unsupported_context"}
            continue
        prompt = fixed_tokens(encoded, n_prompt)
        # Warm the same prefill shape/context before timed measurement. A
        # 128-token warmup alone does not warm attention's long-context JITs.
        run_generation(model, cache, prompt, 2, capacity)
        metrics["workloads"][w["id"]] = run_generation(model, cache, prompt,
                                                         n_decode, capacity)
        if args.teacher_forced_out:
            os.makedirs(args.teacher_forced_out, exist_ok=True)
            n_teacher = args.teacher_steps or n_decode
            history = fixed_tokens(encoded, n_prompt + n_teacher)
            logits = capture_logits(model, cache, history, n_prompt, capacity)
            filename = f"logits_{w['id']}_p{args.prompt_index}_{args.split}.npz"
            np.savez_compressed(os.path.join(args.teacher_forced_out, filename),
                                logits=logits, input_ids=np.asarray(history, dtype=np.int32))
        m = metrics["workloads"][w["id"]]
        print(f"{w['id']}: prefill {m['prefill_ms']:.1f} ms, "
              f"TTFT {m['ttft_ms']:.1f} ms, TPOT {m['tpot_median_ms']:.3f} ms")

    metrics["vram_end_bytes"] = torch.cuda.memory_allocated()
    metrics["per_workload_warmup_decode_steps"] = 2
    metrics["internal_graphs"] = "fork defaults; int8 path declines graph capture"
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=1)
    os.replace(tmp, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
