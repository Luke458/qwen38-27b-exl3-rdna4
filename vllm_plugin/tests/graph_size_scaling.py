#!/usr/bin/env python3
"""HIP-graph replay cost per node vs node count (tiny kernels): detects a queue-size cliff."""
import torch


def per_node_us(n):
    x = torch.zeros(1024, device="cuda", dtype=torch.float16)

    def fn():
        for _ in range(n):
            x.add_(1.0)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(5):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 5 / n * 1e3


for n in (256, 512, 1000, 1024, 1100, 1536, 2048, 4096):
    print(f"{n:5d} nodes: {per_node_us(n):6.2f} us/node", flush=True)
