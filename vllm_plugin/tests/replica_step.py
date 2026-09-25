#!/usr/bin/env python3
"""Replica of one decode step's quantized linears: every EXL3 module of the checkpoint (real
shapes/bitrates, random trellis, so ~9.4 GB DRAM-resident weights), grouped like vLLM's fused
modules, called through the plugin ops in layer order at M=1. Times eager and graph replay.

  python replica_step.py /model
"""
import re
import sys

import torch

sys.path.insert(0, "/plugin")
from exl3rocm.plugin import FUSED, _read_checkpoint_layout  # noqa: E402
from exl3rocm import ops  # noqa: E402,F401

model = sys.argv[1]
mods, _, _ = _read_checkpoint_layout(model)
g = torch.Generator(device="cuda").manual_seed(9)

members_to_fused = {m: f for f, ms in FUSED.items() for m in ms}
groups = {}   # fused key -> list of member keys, in checkpoint order
order = []
for key in sorted((k for k in mods if k.startswith("layers.")),
                  key=lambda k: (int(k.split(".")[1]), k)):
    base, _, leaf = key.rpartition(".")
    fused = members_to_fused.get(leaf)
    fk = f"{base}.{fused}" if fused else key
    if fk not in groups:
        groups[fk] = []
        order.append(fk)
    groups[fk].append(key)
if "lm_head" in mods:
    groups["lm_head"] = ["lm_head"]
    order.append("lm_head")


def within_layer_rank(fk):
    leaf = fk.rsplit(".", 1)[-1]
    return ["in_proj_qkvz", "qkv_proj", "o_proj", "out_proj", "gate_up_proj", "down_proj"].index(leaf) \
        if leaf in ["in_proj_qkvz", "qkv_proj", "o_proj", "out_proj", "gate_up_proj", "down_proj"] else 9


order.sort(key=lambda fk: (int(fk.split(".")[1]) if fk.startswith("layers.") else 10 ** 6, within_layer_rank(fk)))

calls = []
total_bytes = 0
for fk in order:
    mem = [FUSED.get(m.rsplit(".", 1)[-1]) and m for m in groups[fk]]
    specs = [mods[m] for m in groups[fk]]
    # FUSED member order
    leaf = fk.rsplit(".", 1)[-1]
    if leaf in FUSED:
        base = fk.rsplit(".", 1)[0]
        names = [f"{base}.{p}" for p in FUSED[leaf] if f"{base}.{p}" in mods]
        specs = [mods[n] for n in names]
    tr, su, sv, Ks = [], [], [], []
    for K, k, n in specs:
        tr.append(torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                                generator=g).to(torch.int16))
        su.append(torch.ones(k, device="cuda", dtype=torch.float16))
        sv.append(torch.full((n,), 0.01, device="cuda", dtype=torch.float16))
        Ks.append(K)
        total_bytes += k * n * K // 8
    ptrs = (None, None, None)
    if len(specs) > 1 and len(set(Ks)) == 1 and len({s[2] for s in specs}) == 1:
        ptrs = tuple(torch.tensor([t.data_ptr() for t in lst], dtype=torch.long, device="cuda") for lst in (tr, su, sv))
    x = torch.randn((1, specs[0][1]), device="cuda", dtype=torch.float16)
    calls.append((x, tr, su, sv, Ks, ptrs))
    ops.reserve_weight_buffer("cuda:0", specs[0][1] * min(max(s[2] for s in specs), ops.RECON_SLICE_N))

print(f"{len(calls)} module calls, {total_bytes / 1e9:.2f} GB weights", flush=True)


def step():
    for x, tr, su, sv, Ks, ptrs in calls:
        torch.ops.exl3rocm.linear_groups(x, tr, su, sv, Ks, False, True, *ptrs)


for _ in range(3):
    step()
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(5):
    step()
e1.record()
torch.cuda.synchronize()
print(f"eager: {e0.elapsed_time(e1) / 5:.2f} ms/step", flush=True)

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    step()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    step()
for _ in range(3):
    graph.replay()
torch.cuda.synchronize()
e0.record()
for _ in range(10):
    graph.replay()
e1.record()
torch.cuda.synchronize()
ms = e0.elapsed_time(e1) / 10
print(f"graph: {ms:.2f} ms/step  ({total_bytes / ms / 1e6:.0f} GB/s effective)", flush=True)
