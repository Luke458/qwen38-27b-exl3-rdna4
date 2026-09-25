#!/usr/bin/env python3
"""Offline vLLM load + short greedy generation of an EXL3 checkpoint through exl3rocm.
Engine options come from a JSON argument so fault reproductions are exact.

  python offline_probe.py '{"enforce_eager": true, "max_num_batched_tokens": 512}'
"""
import json
import sys
import time

from vllm import LLM, SamplingParams

opts = dict(model="/model", dtype="float16", max_model_len=8192, gpu_memory_utilization=0.93,
            max_num_seqs=4, attention_backend="TRITON_ATTN",
            limit_mm_per_prompt={"image": 0, "video": 0})
opts.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
probe_tokens = int(opts.pop("probe_tokens", 64))
llm = LLM(**opts)
sp = SamplingParams(temperature=0.0, max_tokens=probe_tokens, ignore_eos=True)
msgs = [{"role": "user", "content": "Explain in three sentences what a CPU cache is."}]
t = time.perf_counter()
out = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
dt = time.perf_counter() - t
o = out[0].outputs[0]
print(json.dumps({"tokens": len(o.token_ids), "seconds": dt, "text": o.text}), flush=True)
