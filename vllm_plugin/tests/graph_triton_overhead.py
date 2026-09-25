#!/usr/bin/env python3
"""HIP-graph replay cost per node: native torch kernels vs Inductor/Triton kernels vs exl3 kernels.
Small tensors, so per-node cost ~ dispatch overhead."""
import torch

N = 600


def timed_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 10


def main():
    x = torch.randn(5120, device="cuda", dtype=torch.float16)
    w = torch.randn(5120, device="cuda", dtype=torch.float16)

    def native():
        y = x
        for _ in range(N):
            y = y.add(1.0)
        return y

    @torch.compile(fullgraph=True, dynamic=False)
    def rms(t, w):
        v = t.float().pow(2).mean(-1, keepdim=True)
        return (t.float() * torch.rsqrt(v + 1e-6) * w.float()).half()

    def triton_seq():
        y = x
        for _ in range(N):
            y = rms(y, w)
        return y

    t_native = timed_graph(native)
    t_triton = timed_graph(triton_seq)
    print(f"{N} native torch kernels: {t_native:.2f} ms ({t_native / N * 1e3:.2f} us/node)")
    print(f"{N} inductor/triton kernels: {t_triton:.2f} ms ({t_triton / N * 1e3:.2f} us/node)")


if __name__ == "__main__":
    main()
