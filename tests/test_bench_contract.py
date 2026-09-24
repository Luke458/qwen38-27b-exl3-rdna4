"""CPU contract checks for the direct-model benchmark's cache histories."""

import importlib.util
from pathlib import Path

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location(
    "bench_generate", Path(__file__).resolve().parents[1] / "bench" / "bench_generate.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class Event:
    def __init__(self, enable_timing):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 1.0


class State:
    def __init__(self):
        self.position = 0
        self.freed = False

    def free(self):
        self.freed = True


class Model:
    def __init__(self):
        self.calls = []
        self.state = State()

    def prefill(self, ids, params):
        assert params["attn_mode"] == "flash_attn"
        assert params["past_len"] == self.state.position
        if self.state.position:
            assert params["recurrent_states"] == [self.state]
        params["recurrent_states"] = [self.state]
        self.calls.append(("prefill", params["past_len"], ids.shape[1]))
        self.state.position += ids.shape[1]

    def forward(self, ids, params):
        assert params["attn_mode"] == "flash_attn"
        assert params["past_len"] == self.state.position
        assert params["recurrent_states"] == [self.state]
        self.calls.append(("forward", params["past_len"], int(ids.item())))
        self.state.position += 1
        return torch.tensor([[[float(ids.item()), 1.0]]])


def test_fixed_tokens_are_deterministic_and_exact_length():
    assert BENCH.fixed_tokens([2, 5, 7], 8) == [2, 5, 7, 2, 5, 7, 2, 5]


def test_teacher_force_keeps_one_history_and_all_logits(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", Event)
    model = Model()
    history = list(range(300)) + [42, 43, 44]
    rows = BENCH.capture_logits(model, object(), history, 300, 512)
    assert rows.shape == (3, 2)
    np.testing.assert_array_equal(rows[:, 0], [299, 42, 43])
    assert model.calls[:3] == [("prefill", 0, 128),
                               ("prefill", 128, 128),
                               ("prefill", 256, 43)]
    assert model.calls[3:] == [("forward", 299, 299),
                                ("forward", 300, 42),
                                ("forward", 301, 43)]
    assert model.state.freed
