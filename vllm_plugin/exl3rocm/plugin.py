"""
vLLM out-of-tree quantization plugin: EXL3 (exllamav3 trellis) weights on ROCm gfx1201.

Registered via the `vllm.general_plugins` entry point so every vLLM process imports it
before model construction. Structure (config discovery from the checkpoint, fused-module
shard handling, per-shard suh) adapted from 0xSero/exl3xpu (MIT); see THIRD_PARTY_NOTICES.

Differences from exl3xpu, driven by the GestaltLabs Qwen3.8-27B checkpoint:
- vLLM fuses sibling projections (qkv, gate/up, GDN in_proj_qkvz), but this checkpoint
  mixes bitrates inside them (e.g. q=2 / k,v=5 bits; qkv=2 / z=4). Weights are therefore
  kept per checkpoint tensor ("group") with that group's own K, trellis, suh and svh, and
  the fused output is the concatenation of per-group results.
- K per group is read from the safetensors headers (trellis last dim / 16), not from
  quantization_config.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
import struct
from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger("vllm.exl3rocm")

# vLLM fused module -> checkpoint constituents, in output order
FUSED = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
QKV_IDS = {"q": 0, "k": 1, "v": 2}


def _norm_key(k: str) -> str:
    """Module name shared by checkpoint keys and vLLM prefixes:
       'model.language_model.layers.3.mlp.gate_proj' / 'language_model.model.layers.3.mlp.gate_proj'
           -> 'layers.3.mlp.gate_proj';  '...mtp.layers.0.x' -> 'mtp.layers.0.x';  '...lm_head' -> 'lm_head'"""
    parts = k.split(".")
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


def _read_checkpoint_layout(model_dir: str):
    """({normalized module: (K, k, n)} for every '<module>.trellis' tensor,
        {tensor name: dtype string}, {tensor name: shape} for every tensor)."""
    out, dtypes, shapes = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, v in header.items():
            if name == "__metadata__":
                continue
            dtypes[name] = v["dtype"]
            shapes[name] = tuple(v["shape"])
            if name.endswith(".trellis") and "visual" not in name.split("."):
                s = v["shape"]
                out[_norm_key(name[: -len(".trellis")])] = (s[-1] // 16, s[0] * 16, s[1] * 16)
    return out, dtypes, shapes


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):

    def __init__(self, codebook: str = "mul1"):
        super().__init__()
        self.codebook = codebook
        self.modules: dict[str, tuple[int, int, int]] = {}
        self.fp8_embedding = False
        self.embed_shape = None

    def __repr__(self):
        return f"Exl3Config(codebook={self.codebook}, modules={len(self.modules)})"

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self):
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(codebook=config.get("codebook", "mul1"))

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        if not os.path.isdir(model_name):
            raise NotImplementedError("exl3rocm: pass a local checkpoint directory")
        self.modules, dtypes, shapes = _read_checkpoint_layout(model_name)
        emb = [n for n, dt in dtypes.items() if n.endswith("embed_tokens.weight") and dt == "F8_E4M3"
               and "visual" not in n.split(".")]
        self.fp8_embedding = len(emb) == 1
        self.embed_shape = shapes[emb[0]] if self.fp8_embedding else None
        logger.info("exl3rocm: %d EXL3 modules (%d in MTP head), fp8 input embedding: %s", len(self.modules),
                    sum(1 for k in self.modules if k.startswith("mtp.")), self.fp8_embedding)

    def _groups_for(self, prefix: str):
        """Checkpoint members of the (possibly fused) vLLM module at `prefix`, as
        [(member_key, K, k, n)], or None when the module is not EXL3 (e.g. in_proj_ba)."""
        parts = prefix.split(".")
        if "visual" in parts or "vision_tower" in parts:
            return None
        key = _norm_key(prefix)
        base, _, leaf = key.rpartition(".")
        members = [f"{base}.{p}" if base else p for p in FUSED.get(leaf, [leaf])]
        found = [self.modules.get(m) for m in members]
        if all(f is None for f in found):
            return None
        assert all(f is not None for f in found), f"exl3rocm: partially quantized fused module {prefix}"
        return [(m, *f) for m, f in zip(members, found)]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            groups = self._groups_for(prefix)
            if groups is None:
                return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None
            return Exl3LinearMethod(self, groups)
        if isinstance(layer, VocabParallelEmbedding) and self.fp8_embedding and prefix.endswith("embed_tokens"):
            return Fp8EmbeddingMethod()
        return None


class Fp8EmbeddingMethod(QuantizeMethodBase):
    """Input embedding kept in the checkpoint's float8_e4m3fn (no scale; exllamav3 treats it
    as a plain cast). Rows are gathered as bytes and cast to fp16, halving embedding memory
    (1.18 GiB instead of 2.37 GiB for a 248320 x 5120 table)."""

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        weight = Parameter(torch.empty(sum(output_partition_sizes), input_size_per_partition,
                                       dtype=torch.float8_e4m3fn), requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self, layer, x, bias=None):
        raise NotImplementedError("exl3rocm: fp8 input embedding cannot be used as an output head")

    def embedding(self, layer, input_):
        from . import ops  # noqa: F401
        return torch.ops.exl3rocm.fp8_embedding(layer.weight, input_)


class Exl3LinearMethod(LinearMethodBase):
    """Per-group EXL3 storage. The registered Parameters `trellis`, `suh`, `svh`, `mul1`
    are 0-size load targets (what vLLM's name mapping resolves to); their weight_loader
    routes each checkpoint tensor into that group's non-persistent buffers."""

    def __init__(self, config: Exl3Config, groups):
        self.config = config
        self.groups = groups  # [(member_key, K, k, n)]
        # pruned draft head only when a speculative (MTP) config is active
        self._drafting = False
        try:
            from vllm.config import get_current_vllm_config_or_none
            cfg = get_current_vllm_config_or_none()
            self._drafting = cfg is not None and cfg.speculative_config is not None
        except Exception:
            pass

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        k = input_size_per_partition
        if k != input_size:
            raise NotImplementedError("exl3rocm: tensor parallelism is not supported")
        parts = list(output_partition_sizes)
        # map vLLM output partitions onto checkpoint groups (a group may span partitions,
        # e.g. in_proj_qkv covers q, k, v of in_proj_qkvz)
        group_n = [g[3] for g in self.groups]
        part_to_group, gi, acc = [], 0, 0
        for p in parts:
            assert gi < len(group_n), f"exl3rocm: partitions {parts} do not tile groups {group_n}"
            part_to_group.append(gi)
            acc += p
            assert acc <= group_n[gi], f"exl3rocm: partitions {parts} do not tile groups {group_n}"
            if acc == group_n[gi]:
                gi, acc = gi + 1, 0
        assert gi == len(group_n), f"exl3rocm: partitions {parts} do not tile groups {group_n}"
        layer.exl3_part_to_group = part_to_group
        layer.exl3_K = [g[1] for g in self.groups]
        layer.exl3_n = [g[3] for g in self.groups]
        for i, (_, K, gk, n) in enumerate(self.groups):
            assert gk == k, f"exl3rocm: {self.groups[i][0]} k={gk} vs layer k={k}"
            layer.register_buffer(f"exl3_trellis_{i}", torch.empty((k // 16, n // 16, 16 * K), dtype=torch.int16),
                                  persistent=False)
            layer.register_buffer(f"exl3_suh_{i}", torch.empty((k,), dtype=torch.float16), persistent=False)
            layer.register_buffer(f"exl3_svh_{i}", torch.empty((n,), dtype=torch.float16), persistent=False)
        layer.exl3_loaded = set()
        layer.exl3_mul1 = None
        layer.exl3_mcg = None

        def reg(name, loader):
            p = Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False)
            p.weight_loader = loader
            layer.register_parameter(name, p)

        reg("trellis", self._loader(layer, "trellis"))
        reg("suh", self._loader(layer, "suh"))
        reg("svh", self._loader(layer, "svh"))
        # only the codebook scalar the checkpoint actually stores (vLLM rejects
        # registered parameters that are never loaded)
        if self.config.codebook in ("mul1", "mcg"):
            reg(self.config.codebook, self._loader(layer, self.config.codebook))

    @staticmethod
    def _group_of(layer, shard_id) -> int:
        if shard_id is None:
            assert len(layer.exl3_n) == 1, "exl3rocm: unsharded load into fused module"
            return 0
        if isinstance(shard_id, str):
            parts = [QKV_IDS[shard_id]]
        elif isinstance(shard_id, int):
            parts = [shard_id]
        else:
            parts = list(shard_id)
        gs = {layer.exl3_part_to_group[p] for p in parts}
        assert len(gs) == 1, f"exl3rocm: shard {shard_id} spans groups {gs}"
        return gs.pop()

    def _loader(self, layer, kind):
        def load(param, w, shard_id=None, *args, **kwargs):
            g = self._group_of(layer, shard_id)
            if kind in ("mul1", "mcg"):
                val = int(w.item())
                prev = getattr(layer, f"exl3_{kind}")
                assert prev is None or prev == val, f"exl3rocm: {kind} differs across groups"
                setattr(layer, f"exl3_{kind}", val)
            else:
                buf = getattr(layer, f"exl3_{kind}_{g}")
                assert tuple(w.shape) == tuple(buf.shape), \
                    f"exl3rocm: {kind} shape {tuple(w.shape)} vs {tuple(buf.shape)} (group {g})"
                buf.copy_(w)
            layer.exl3_loaded.add((kind, g))
        return load

    def process_weights_after_loading(self, layer):
        ng = len(layer.exl3_n)
        missing = [(k, g) for k in ("trellis", "suh", "svh") for g in range(ng) if (k, g) not in layer.exl3_loaded]
        assert not missing, f"exl3rocm: missing tensors {missing}"
        layer.exl3_is_mul1 = layer.exl3_mul1 is not None
        layer.exl3_is_mcg = layer.exl3_mcg is not None
        from .ops import reserve_weight_buffer
        k = layer.exl3_trellis_0.shape[0] * 16
        dev = layer.exl3_trellis_0.device
        reserve_weight_buffer(dev, max(k * min(n, ops_slice_n()) for n in layer.exl3_n))
        # grouped M=1 path (exl3_mgemm) for modules whose groups share K and width (gate/up)
        layer.exl3_mgemm_ptrs = None
        if ng > 1 and len(set(layer.exl3_K)) == 1 and len(set(layer.exl3_n)) == 1 \
                and os.environ.get("EXL3_MGEMM", "1") != "0":
            def ptrs(kind):
                t = torch.tensor([getattr(layer, f"exl3_{kind}_{i}").data_ptr() for i in range(ng)],
                                 dtype=torch.long, device=dev)
                layer.register_buffer(f"exl3_ptrs_{kind}", t, persistent=False)
                return t
            layer.exl3_mgemm_ptrs = [ptrs("trellis"), ptrs("suh"), ptrs("svh")]
        if isinstance(layer, ParallelLMHead) and self._drafting:
            _build_draft_head(layer)
        # drop the 0-size load targets so nothing downstream mistakes them for weights
        for name in ("trellis", "suh", "svh", "mul1", "mcg"):
            if name in layer._parameters:
                del layer._parameters[name]

    def apply(self, layer, x, bias=None):
        ng = len(layer.exl3_n)
        # one opaque op per module: never concatenate per-group outputs in traced code
        y = torch.ops.exl3rocm.linear_groups(
            x,
            [getattr(layer, f"exl3_trellis_{i}") for i in range(ng)],
            [getattr(layer, f"exl3_suh_{i}") for i in range(ng)],
            [getattr(layer, f"exl3_svh_{i}") for i in range(ng)],
            list(layer.exl3_K), layer.exl3_is_mcg, layer.exl3_is_mul1,
            *(layer.exl3_mgemm_ptrs or (None, None, None)))
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        if bias is not None:
            y = y + bias
        return y

    def embedding(self, layer, input_):
        raise NotImplementedError("exl3rocm: quantized input embeddings are not supported")


def ops_slice_n() -> int:
    from .ops import RECON_SLICE_N
    return RECON_SLICE_N


def _install_debug_module_sync():
    """EXL3_DEBUG_MODULE_SYNC=1: after model load, synchronize and log (flushed, to stderr)
    before every module forward, so a GPU fault names the module that issued it.
    Eager mode only (compiled graphs bypass module hooks)."""
    import sys
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    orig = GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        for name, mod in self.model.named_modules():
            def pre(m, args, name=name):
                torch.cuda.synchronize()
                shapes = [tuple(t.shape) for t in args if isinstance(t, torch.Tensor)]
                print(f"[exl3dbg] enter {name} {type(m).__name__} {shapes}", file=sys.stderr, flush=True)
            mod.register_forward_pre_hook(pre)
        logger.warning("exl3rocm: debug module sync hooks installed")
        return r

    GPUModelRunner.load_model = load_model


def _install_fp8_embedding_hook():
    """Qwen3_5Model builds `embed_tokens = VocabParallelEmbedding(vocab, hidden)` without its
    quant_config, so no quantization plugin can choose the embedding's storage and vLLM would
    upcast the checkpoint's fp8 table to fp16 (+1.18 GiB). Inject the config for a plain
    (non-lm_head) embedding of exactly the checkpoint's embed shape, constructed while an
    exl3 model with an fp8 embedding is being built. Anything else is untouched."""
    import inspect
    from vllm.config import get_current_vllm_config_or_none
    orig = VocabParallelEmbedding.__init__
    sig = inspect.signature(orig)

    def __init__(self, *args, **kwargs):
        if type(self) is VocabParallelEmbedding:
            b = sig.bind(self, *args, **kwargs)
            cfg = get_current_vllm_config_or_none()
            qc = getattr(cfg, "quant_config", None) if cfg is not None else None
            if (isinstance(qc, Exl3Config) and qc.fp8_embedding and b.arguments.get("quant_config") is None
                    and (b.arguments["num_embeddings"], b.arguments["embedding_dim"]) == qc.embed_shape):
                b.arguments["quant_config"] = qc
                b.arguments["prefix"] = b.arguments.get("prefix") or "embed_tokens"
                return orig(*b.args, **b.kwargs)
        return orig(self, *args, **kwargs)

    VocabParallelEmbedding.__init__ = __init__


def _draft_blocks(n_blocks_total: int) -> list[int]:
    """Output blocks (128 columns = 1 Hadamard block) scored by the MTP drafter: the first
    EXL3_DRAFT_VOCAB_BLOCKS blocks (BPE ids are roughly frequency-ordered; the first 640 cover
    >99.9% of tokens in local prose/code samples) plus the last 16 blocks (special / chat tokens)."""
    n = int(os.environ.get("EXL3_DRAFT_VOCAB_BLOCKS", "640"))
    if n <= 0 or n >= n_blocks_total:
        return []
    return sorted(set(range(n)) | set(range(max(n, n_blocks_total - 16), n_blocks_total)))


def _build_draft_head(layer):
    """Pruned copy of the lm_head for the MTP drafter (target verification keeps the full head,
    so generated text is unchanged; only draft acceptance can drop). Adapted from exl3xpu."""
    if len(layer.exl3_n) != 1:
        return
    n = layer.exl3_n[0]
    blocks = _draft_blocks(n // 128)
    if not blocks:
        return
    dev = layer.exl3_trellis_0.device
    b = torch.tensor(blocks, dtype=torch.long, device=dev)
    tiles = (b[:, None] * 8 + torch.arange(8, device=dev)[None, :]).flatten()
    cols = (b[:, None] * 128 + torch.arange(128, device=dev)[None, :]).flatten()
    layer.register_buffer("exl3_draft_trellis", layer.exl3_trellis_0.index_select(1, tiles).contiguous(),
                          persistent=False)
    layer.register_buffer("exl3_draft_svh", layer.exl3_svh_0.index_select(0, cols).contiguous(), persistent=False)
    layer.register_buffer("exl3_draft_cols", cols, persistent=False)
    logger.info("exl3rocm: MTP draft head scores %d of %d vocab blocks (%.1f%%)", len(blocks), n // 128,
                100.0 * len(blocks) / (n // 128))


def _install_draft_logits_patch():
    try:
        from vllm.model_executor.models import qwen3_5_mtp
    except Exception:
        return
    cls = qwen3_5_mtp.Qwen3_5MTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig = cls.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        lm = self.lm_head
        if getattr(lm, "exl3_draft_trellis", None) is None:
            return orig(self, hidden_states, spec_step_idx)
        from . import ops  # noqa: F401
        sub = torch.ops.exl3rocm.linear(hidden_states, lm.exl3_draft_trellis, lm.exl3_suh_0, lm.exl3_draft_svh,
                                        lm.exl3_K[0], lm.exl3_is_mcg, lm.exl3_is_mul1)
        logits = sub.new_full((sub.shape[0], lm.exl3_n[0]), float("-inf"))
        logits.index_copy_(1, lm.exl3_draft_cols, sub)
        return logits[:, : self.config.vocab_size].to(hidden_states.dtype)

    cls.compute_logits = compute_logits

    # greedy drafting goes through get_top_tokens (LocalArgmaxMixin), not compute_logits:
    # argmax over the pruned columns, mapped back to vocab ids (no full-vocab tensor at all)
    orig_top = getattr(cls, "get_top_tokens", None)

    def get_top_tokens(self, hidden_states):
        lm = self.lm_head
        if getattr(lm, "exl3_draft_trellis", None) is None or orig_top is None:
            return orig_top(self, hidden_states)
        from . import ops  # noqa: F401
        sub = torch.ops.exl3rocm.linear(hidden_states, lm.exl3_draft_trellis, lm.exl3_suh_0, lm.exl3_draft_svh,
                                        lm.exl3_K[0], lm.exl3_is_mcg, lm.exl3_is_mul1)
        vocab = self.config.vocab_size
        cols = lm.exl3_draft_cols
        sub = sub.masked_fill((cols >= vocab)[None, :], float("-inf"))
        return cols[sub.argmax(dim=-1)]

    if orig_top is not None:
        cls.get_top_tokens = get_top_tokens
    cls._exl3_patched = True


def _install_debug_graph_timing():
    """EXL3_DEBUG_GRAPH_TIME=N: time every CUDA/HIP graph replay with GPU events and log the
    median replay time and the median interval between replay starts every N replays."""
    import sys
    import time
    n_report = int(os.environ["EXL3_DEBUG_GRAPH_TIME"])
    orig = torch.cuda.CUDAGraph.replay
    state = {"ev": [], "wall": [], "last": None}

    def replay(self):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        now = time.perf_counter()
        if state["last"] is not None:
            state["wall"].append(now - state["last"])
        state["last"] = now
        e0.record()
        h0 = time.perf_counter()
        r = orig(self)
        state.setdefault("host", []).append(time.perf_counter() - h0)
        e1.record()
        state["ev"].append((e0, e1))
        if len(state["ev"]) >= n_report:
            torch.cuda.synchronize()
            gpu = sorted(a.elapsed_time(b) for a, b in state["ev"])
            wall = sorted(state["wall"]) or [0.0]
            seq = [round(a.elapsed_time(b), 1) for a, b in state["ev"]]
            pct = lambda q: gpu[min(len(gpu) - 1, int(q * len(gpu)))]
            print(f"[exl3dbg] graph replay GPU ms: p10 {pct(.1):.2f} p25 {pct(.25):.2f} median {pct(.5):.2f} "
                  f"p75 {pct(.75):.2f} max {gpu[-1]:.2f} | replay-start interval ms: median "
                  f"{wall[len(wall) // 2] * 1e3:.2f} | host replay() call ms: median "
                  f"{sorted(state['host'])[len(state['host']) // 2] * 1e3:.2f} | first 8: {seq[:8]}",
                  file=sys.stderr, flush=True)
            state["ev"].clear(); state["wall"].clear(); state["host"].clear()
        return r

    torch.cuda.CUDAGraph.replay = replay


def register():
    """vllm.general_plugins entry point (runs in every vLLM process)."""
    if os.environ.get("EXL3_DEBUG_GRAPH_TIME"):
        _install_debug_graph_timing()
    from . import ops  # noqa: F401  (registers torch.ops.exl3rocm.linear)
    if os.environ.get("EXL3_FP8_EMBEDDING", "1") != "0":
        _install_fp8_embedding_hook()
    if os.environ.get("EXL3_DRAFT_VOCAB_BLOCKS", "640") != "0":
        _install_draft_logits_patch()
    if os.environ.get("EXL3_DEBUG_MODULE_SYNC") == "1":
        _install_debug_module_sync()
    return None
