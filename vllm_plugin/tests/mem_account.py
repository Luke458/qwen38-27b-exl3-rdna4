#!/usr/bin/env python3
"""Where does GPU memory go after vLLM loads the EXL3 checkpoint? Builds an eager engine, then
prints device-used memory against torch-allocated/reserved, and groups every parameter and buffer
(unique storages) of the loaded modules by prefix, next to the KV cache tensors.

  python mem_account.py '{"kv_cache_memory_bytes": 1650000000, ...}'
"""
import collections
import gc
import json
import sys

import torch
from vllm import LLM

GiB = 2 ** 30
opts = dict(model="/model", dtype="float16", max_model_len=4096, max_num_seqs=4,
            attention_backend="TRITON_ATTN", enforce_eager=True)
opts.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
free0, total = torch.cuda.mem_get_info()
llm = LLM(**opts)
free1, _ = torch.cuda.mem_get_info()
print(f"device used by this process: {(free0 - free1) / GiB:.3f} GiB "
      f"(torch allocated {torch.cuda.memory_allocated() / GiB:.3f}, reserved {torch.cuda.memory_reserved() / GiB:.3f})")

def find_runner(obj, depth=0, seen_ids=None):
    """Walk engine attributes to the GPU model runner (layout differs between V1/V2 runners)."""
    seen_ids = seen_ids if seen_ids is not None else set()
    if depth > 7 or id(obj) in seen_ids:
        return None
    seen_ids.add(id(obj))
    if hasattr(obj, "model") and isinstance(getattr(obj, "model"), torch.nn.Module) and hasattr(obj, "kv_caches"):
        return obj
    for name in ("llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner"):
        child = getattr(obj, name, None)
        if child is not None:
            r = find_runner(child, depth + 1, seen_ids)
            if r is not None:
                return r
    return None


runner = find_runner(llm)
print("runner:", type(runner).__name__ if runner else None)
roots = [runner.model]
for name in ("drafter", "speculator"):
    d = getattr(runner, name, None)
    if d is not None and isinstance(getattr(d, "model", None), torch.nn.Module):
        roots.append(d.model)
seen = set()
groups = collections.Counter()
for root in roots:
    rname = type(root).__name__
    for name, t in list(root.named_parameters(remove_duplicate=False)) + list(root.named_buffers(remove_duplicate=False)):
        if not t.is_cuda:
            continue
        key = t.untyped_storage().data_ptr()
        if key in seen or key == 0:
            continue
        seen.add(key)
        parts = name.split(".")
        pre = ".".join(p for p in parts[:3] if not p.isdigit())
        leaf = parts[-1]
        groups[(rname, pre, leaf)] += t.untyped_storage().nbytes()
kv = 0
for t in getattr(runner, "kv_caches", []) or []:
    for x in (t if isinstance(t, (list, tuple)) else [t]):
        if isinstance(x, torch.Tensor) and x.untyped_storage().data_ptr() not in seen:
            seen.add(x.untyped_storage().data_ptr()); kv += x.untyped_storage().nbytes()
print(f"kv cache tensors: {kv / GiB:.3f} GiB")
from exl3rocm import kv_dequant, ops
print("recon buffer:", {str(k): round(v.untyped_storage().nbytes() / GiB, 3) for k, v in ops._wbuf.items()})
# tensors held as plain module attributes (not registered parameters/buffers)
attr = collections.Counter()
for root in roots:
    for mname, m in root.named_modules():
        for k, v in vars(m).items():
            vs = v if isinstance(v, (list, tuple)) else [v]
            for x in vs:
                if isinstance(x, torch.Tensor) and x.is_cuda and x.untyped_storage().data_ptr() not in seen:
                    seen.add(x.untyped_storage().data_ptr())
                    attr[(type(root).__name__, type(m).__name__, k)] += x.untyped_storage().nbytes()
print(f"module attribute tensors: {sum(attr.values()) / GiB:.3f} GiB")
for k, n in attr.most_common(12):
    print(f"  {n / GiB:7.3f} GiB  {k}")
# other live CUDA tensors not owned by the modules (KV cache, plugin buffers, workspaces)
other = collections.Counter()
for o in gc.get_objects():
    try:
        if isinstance(o, torch.Tensor) and o.is_cuda:
            key = o.untyped_storage().data_ptr()
            if key in seen or key == 0:
                continue
            seen.add(key)
            other[(str(o.dtype), tuple(o.shape)[:2])] += o.untyped_storage().nbytes()
    except Exception:
        pass

by_root = collections.Counter()
for (r, pre, leaf), n in groups.items():
    by_root[r] += n
print("module tensors:", {r: round(n / GiB, 3) for r, n in by_root.items()})
for (r, pre, leaf), n in groups.most_common(30):
    print(f"  {n / GiB:7.3f} GiB  {r}  {pre} .{leaf}")
print(f"other live CUDA tensors: {sum(other.values()) / GiB:.3f} GiB")
for k, n in other.most_common(15):
    print(f"  {n / GiB:7.3f} GiB  {k}")
