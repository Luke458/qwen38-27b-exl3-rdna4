#!/usr/bin/env python3
"""Per-call cost of EXL3 M=1 linears eager vs HIP-graph replay, on one realistic layer mix.

A decode step in vLLM is one FULL graph of ~1,700 kernels; if graph replay adds a large
per-node cost for these kernels, it shows up here as graph time >> eager-back-to-back time.
"""
import torch

import exllamav3_ext as E

SHAPES = [(5120, 10240, 4), (5120, 6144, 4), (6144, 5120, 4), (17408, 5120, 3), (5120, 1024, 5)]
REPS = 40


def main():
    g = torch.Generator(device="cuda").manual_seed(5)
    ops = []
    for k, n, K in SHAPES:
        tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), device="cuda", dtype=torch.int32,
                           generator=g).to(torch.int16)
        suh = torch.ones(k, device="cuda", dtype=torch.float16)
        svh = torch.full((n,), 0.01, device="cuda", dtype=torch.float16)
        x = torch.randn((1, k), device="cuda", dtype=torch.float16)
        xh = torch.empty_like(x)
        y = torch.empty((1, n), device="cuda", dtype=torch.float16)
        ops.append((x, tr, y, suh, xh, svh))

    def seq():
        for _ in range(REPS):
            for x, tr, y, suh, xh, svh in ops:
                E.exl3_gemm(x, tr, y, suh, xh, svh, -1, False, True, 0)

    for _ in range(3):
        seq()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); seq(); e1.record(); torch.cuda.synchronize()
    eager = e0.elapsed_time(e1)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        seq()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        seq()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    e0.record(); graph.replay(); e1.record(); torch.cuda.synchronize()
    gr = e0.elapsed_time(e1)
    calls = REPS * len(SHAPES)
    kernels = calls * 3
    print(f"{calls} linears ({kernels} kernels): eager {eager:.2f} ms ({eager / calls * 1e3:.1f} us/linear), "
          f"graph replay {gr:.2f} ms ({gr / calls * 1e3:.1f} us/linear)")


if __name__ == "__main__":
    main()
