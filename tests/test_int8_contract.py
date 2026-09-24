"""Concurrency and capture fallback contracts for the isolated candidate."""
import os
import numpy as np
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext


def fixture(seed=0):
    rng = np.random.default_rng(seed)
    a = torch.tensor(rng.normal(0, .2, (1, 256)), dtype=torch.float16, device='cuda')
    b = torch.tensor(rng.integers(-32768, 32767, (16, 16, 48)), dtype=torch.int16, device='cuda')
    su = torch.ones(256, dtype=torch.float16, device='cuda')
    sv = su.clone()
    return a, b, su, sv


def launch(data, output, scratch):
    a, b, su, sv = data
    ext.exl3_gemm(a, b, output, su, scratch, sv, -1, False, True, 0)


def test_concurrent_stream_scratch(monkeypatch):
    monkeypatch.setenv('EXL3_ROCM_GFX1201_INT8', '2')
    inputs = [fixture(71+i) for i in range(4)]
    outputs = [torch.empty_like(x[0]) for x in inputs]
    scratch = [torch.empty_like(x[0]) for x in inputs]
    expected = []
    for data, out, ah in zip(inputs, outputs, scratch):
        launch(data, out, ah)
        expected.append(out.clone())
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in inputs]
    for _ in range(20):
        for stream, data, out, ah in zip(streams, inputs, outputs, scratch):
            with torch.cuda.stream(stream):
                launch(data, out, ah)
    torch.cuda.synchronize()
    for out, ref in zip(outputs, expected):
        assert torch.equal(out, ref)


@pytest.mark.parametrize('mode', ['0', '1', '7'])
def test_unsupported_modes_preserve_baseline(monkeypatch, mode):
    data = fixture(12)
    out, ref, scratch = (torch.empty_like(data[0]) for _ in range(3))
    monkeypatch.setenv('EXL3_ROCM_GFX1201_INT8', '0')
    launch(data, ref, scratch)
    monkeypatch.setenv('EXL3_ROCM_GFX1201_INT8', mode)
    launch(data, out, scratch)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    assert ext.exl3_gemv_int8_max_k(0) == 0


def test_graph_falls_back_and_replays(monkeypatch):
    data = fixture(44)
    out, ref, scratch = (torch.empty_like(data[0]) for _ in range(3))
    monkeypatch.setenv('EXL3_ROCM_GFX1201_INT8', '0')
    launch(data, ref, scratch)
    torch.cuda.synchronize()
    monkeypatch.setenv('EXL3_ROCM_GFX1201_INT8', '2')
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(data, out, scratch)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
