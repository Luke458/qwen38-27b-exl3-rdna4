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
    "qkv": ["q_proj", "k_proj", "v_proj"],          # vision attention (language uses qkv_proj)
}
QKV_IDS = {"q": 0, "k": 1, "v": 2}


def _norm_key(k: str) -> str:
    """Module name shared by checkpoint keys and vLLM prefixes:
       'model.language_model.layers.3.mlp.gate_proj' / 'language_model.model.layers.3.mlp.gate_proj'
           -> 'layers.3.mlp.gate_proj';  '...mtp.layers.0.x' -> 'mtp.layers.0.x';  '...lm_head' -> 'lm_head'"""
    parts = k.split(".")
    if "visual" in parts:
        return "visual." + ".".join(parts[parts.index("visual") + 1:])
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
            if name.endswith(".trellis"):
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
        # EXL3 pads linear widths to a multiple of 128 (vision MLP: 4304 -> 4352, with zero-padded
        # weights and biases); exllamav3 runs the padded width end to end (fc1 out -> fc2 in), so
        # build vLLM's vision MLP at the stored width too.
        vc = getattr(hf_config, "vision_config", None) if hf_config is not None else None
        fc1 = self.modules.get("visual.blocks.0.mlp.linear_fc1")
        if vc is not None and fc1 is not None and getattr(vc, "intermediate_size", None) not in (None, fc1[2]):
            logger.info("exl3rocm: vision intermediate_size %d -> %d (EXL3 padded width)",
                        vc.intermediate_size, fc1[2])
            vc.intermediate_size = fc1[2]
        logger.info("exl3rocm: %d EXL3 modules (%d in MTP head), fp8 input embedding: %s", len(self.modules),
                    sum(1 for k in self.modules if k.startswith("mtp.")), self.fp8_embedding)

    def _groups_for(self, prefix: str):
        """Checkpoint members of the (possibly fused) vLLM module at `prefix`, as
        [(member_key, K, k, n)], or None when the module is not EXL3 (e.g. in_proj_ba)."""
        # vision tower: proj / linear_fc1 / linear_fc2 / merger are EXL3; attn.qkv is the 6-bit
        # q/k/v groups (FUSED "qkv"), or unquantized bf16 with EXL3_VISION_QKV_BF16=1.
        key = _norm_key(prefix)
        if key.endswith(".attn.qkv") and key.startswith("visual.") and os.environ.get("EXL3_VISION_QKV_BF16") == "1":
            return None
        base, _, leaf = key.rpartition(".")
        members = [f"{base}.{p}" if base else p for p in FUSED.get(leaf, [leaf])]
        found = [self.modules.get(m) for m in members]
        if all(f is None for f in found):
            return None
        assert all(f is not None for f in found), f"exl3rocm: partially quantized fused module {prefix}"
        return [(m, *f) for m, f in zip(members, found)]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if _BUILDING_DRAFTER["on"] and isinstance(layer, VocabParallelEmbedding) \
                and os.environ.get("EXL3_DRAFTER_PLACEHOLDERS", "1") != "0":
            return SharedPlaceholderMethod()     # embed_tokens and lm_head: vLLM shares the target's
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            groups = self._groups_for(prefix)
            if groups is None:
                return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None
            return Exl3LinearMethod(self, groups)
        if isinstance(layer, VocabParallelEmbedding) and self.fp8_embedding and prefix.endswith("embed_tokens"):
            return Fp8EmbeddingMethod()
        return None


class SharedPlaceholderMethod(QuantizeMethodBase):
    """For the MTP drafter's own embed_tokens / lm_head. vLLM (V1 and V2 runners) always replaces
    both with the target's modules after loading a Qwen3.5 MTP drafter (it declares neither
    has_own_embed_tokens nor has_own_lm_head), so allocating and loading them only fragments
    memory (~2 GiB transient here: 1.18 GiB fp8 embedding, 0.64 GiB lm_head, 0.21 GiB draft head).
    Registers a 0-size weight whose loader discards the checkpoint tensor; any use raises."""

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        # the names the checkpoint provides: EXL3 lm_head tensors, or a plain embedding weight
        names = ("trellis", "suh", "svh", "mul1") if isinstance(layer, ParallelLMHead) else ("weight",)
        for name in names:
            w = Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)
            w.weight_loader = lambda *a, **kw: None
            layer.register_parameter(name, w)
        layer.exl3_placeholder = True

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        raise RuntimeError("exl3rocm: MTP drafter lm_head placeholder used; vLLM did not share the target's")

    def embedding(self, layer, input_):
        raise RuntimeError("exl3rocm: MTP drafter embedding placeholder used; vLLM did not share the target's")


_BUILDING_DRAFTER = {"on": False}


def _install_drafter_placeholder_hook():
    try:
        from vllm.model_executor.models import qwen3_5_mtp
    except Exception:
        return
    for name in ("Qwen3_5MTP", "Qwen3_5MoeMTP"):
        cls = getattr(qwen3_5_mtp, name, None)
        if cls is None or getattr(cls, "_exl3_placeholder_hook", False):
            continue
        orig = cls.__init__

        def __init__(self, *a, orig=orig, **kw):
            _BUILDING_DRAFTER["on"] = True
            try:
                orig(self, *a, **kw)
            finally:
                _BUILDING_DRAFTER["on"] = False

        cls.__init__ = __init__
        cls._exl3_placeholder_hook = True


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
            if shard_id is None:
                shard_id = getattr(w, "exl3_shard", None)
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
        if isinstance(layer, ParallelLMHead) and self._drafting:
            _split_lm_head_for_draft(layer)
            ng = len(layer.exl3_n)
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


def _split_lm_head_for_draft(layer):
    """Pruned MTP draft head without a copy (was 0.20 GiB): store the lm_head as three column
    groups, [drafted prefix | middle | drafted tail]. The target's full head runs all three
    (concatenated in order inside linear_groups, so logits keep their column order); the drafter
    runs groups 0 and 2 only. Target verification keeps the full head, so generated text is
    unchanged; only draft acceptance can drop. Pruning adapted from exl3xpu."""
    if len(layer.exl3_n) != 1:
        return
    n = layer.exl3_n[0]
    blocks = _draft_blocks(n // 128)
    if not blocks:
        return
    nb = n // 128
    p = next((i for i, blk in enumerate(blocks) if blk != i), len(blocks))  # prefix blocks [0, p)
    if p == 0 or p == len(blocks) or blocks[p:] != list(range(blocks[p], nb)):
        return
    a, b = 128 * p, 128 * blocks[p]  # drafted columns: [0, a) and [b, n)
    K, t, su, sv = layer.exl3_K[0], layer.exl3_trellis_0, layer.exl3_suh_0, layer.exl3_svh_0
    bounds = [0, a, b, n]
    parts = [(t[:, bounds[i] // 16:bounds[i + 1] // 16].contiguous(), sv[bounds[i]:bounds[i + 1]].contiguous())
             for i in range(3)]
    for name in ("exl3_trellis_0", "exl3_svh_0"):
        del layer._buffers[name]
    del t, sv
    for i, (ti, svi) in enumerate(parts):
        layer.register_buffer(f"exl3_trellis_{i}", ti, persistent=False)
        layer.register_buffer(f"exl3_suh_{i}", su, persistent=False)
        layer.register_buffer(f"exl3_svh_{i}", svi, persistent=False)
    del parts
    layer.exl3_K = [K] * 3
    layer.exl3_n = [a, b - a, n - b]
    layer.exl3_draft_groups = (0, 2)
    dev = su.device
    layer.register_buffer("exl3_draft_cols", torch.cat([torch.arange(0, a, device=dev),
                                                        torch.arange(b, n, device=dev)]), persistent=False)
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.empty_cache()
    logger.info("exl3rocm: MTP draft head scores %d of %d vocab blocks (%.1f%%), no copy",
                (a + n - b) // 128, n // 128, 100.0 * (a + n - b) / n)


def _draft_logits(lm, hidden_states):
    gs = lm.exl3_draft_groups
    return torch.ops.exl3rocm.linear_groups(
        hidden_states, [getattr(lm, f"exl3_trellis_{g}") for g in gs], [getattr(lm, f"exl3_suh_{g}") for g in gs],
        [getattr(lm, f"exl3_svh_{g}") for g in gs], [lm.exl3_K[g] for g in gs], lm.exl3_is_mcg, lm.exl3_is_mul1)


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
        if getattr(lm, "exl3_draft_groups", None) is None:
            return orig(self, hidden_states, spec_step_idx)
        from . import ops  # noqa: F401
        sub = _draft_logits(lm, hidden_states)
        logits = sub.new_full((sub.shape[0], sum(lm.exl3_n)), float("-inf"))
        logits.index_copy_(1, lm.exl3_draft_cols, sub)
        return logits[:, : self.config.vocab_size].to(hidden_states.dtype)

    cls.compute_logits = compute_logits

    # greedy drafting goes through get_top_tokens (LocalArgmaxMixin), not compute_logits:
    # argmax over the pruned columns, mapped back to vocab ids (no full-vocab tensor at all)
    orig_top = getattr(cls, "get_top_tokens", None)

    def get_top_tokens(self, hidden_states):
        lm = self.lm_head
        if getattr(lm, "exl3_draft_groups", None) is None or orig_top is None:
            return orig_top(self, hidden_states)
        from . import ops  # noqa: F401
        sub = _draft_logits(lm, hidden_states)
        vocab = self.config.vocab_size
        cols = lm.exl3_draft_cols
        sub = sub.masked_fill((cols >= vocab)[None, :], float("-inf"))
        return cols[sub.argmax(dim=-1)]

    if orig_top is not None:
        cls.get_top_tokens = get_top_tokens
    cls._exl3_patched = True


_VISUAL_QKV_SPLIT = re.compile(r"^(.*\bvisual\.blocks\.\d+\.attn\.)([qkv])_proj\.(trellis|suh|svh|mul1|bias)$")
_VISUAL_QKV_FUSED_W = re.compile(r"\bvisual\.blocks\.\d+\.attn\.qkv\.weight$")


def _visual_qkv_stream(weights):
    """The checkpoint stores each vision attention's q/k/v twice: 6-bit EXL3 q_proj/k_proj/v_proj
    and a bf16 fused qkv.weight (+ bf16 fused qkv.bias). EXL3 mode (default, what exllamav3 runs,
    ~134 MB less VRAM): rename q/k/v EXL3 tensors onto attn.qkv with their shard as a tensor
    attribute and drop the bf16 weight; the per-matrix fp16 biases are dropped in favour of the
    fused bias. EXL3_VISION_QKV_BF16=1: keep the bf16 fused weight, drop the EXL3 copies."""
    bf16 = os.environ.get("EXL3_VISION_QKV_BF16") == "1"
    # checkpoints whose vision tower is not EXL3 (e.g. turboderp's uniform-bpw quants) only have
    # the bf16 fused weight: pass everything through untouched
    try:
        from vllm.config import get_current_vllm_config_or_none
        cfg = get_current_vllm_config_or_none()
        qc = getattr(cfg, "quant_config", None) if cfg is not None else None
        if isinstance(qc, Exl3Config) and not any(k.startswith("visual.") and k.endswith("attn.q_proj")
                                                  for k in qc.modules):
            yield from weights
            return
    except Exception:
        pass
    for n, w in weights:
        m = _VISUAL_QKV_SPLIT.search(n)
        if m:
            if bf16 or m.group(3) == "bias":
                continue
            w.exl3_shard = m.group(2)
            yield m.group(1) + "qkv." + m.group(3), w
        elif not bf16 and _VISUAL_QKV_FUSED_W.search(n):
            continue
        else:
            yield n, w


def _install_visual_qkv_filter():
    """Route the vision q/k/v tensors (see _visual_qkv_stream)."""
    try:
        from vllm.model_executor.models import qwen3_5
    except Exception:
        return
    for cls in (getattr(qwen3_5, "Qwen3_5ForConditionalGeneration", None),
                getattr(qwen3_5, "Qwen3_5MoeForConditionalGeneration", None)):
        if cls is None or getattr(cls, "_exl3_visual_filter", False):
            continue
        orig = cls.load_weights

        def load_weights(self, weights, orig=orig):
            return orig(self, _visual_qkv_stream(weights))

        cls.load_weights = load_weights
        cls._exl3_visual_filter = True


def _install_pth_prefill_dequant():
    """Per-token-head int8/fp8 KV cache: the Triton unified attention kernel dequantizes K/V
    inline per query tile, which is cheap for decode but halves long-prompt prefill speed (every
    query tile of a 2048-token chunk re-converts the whole context). For long queries, gather the
    batch's blocks once, dequantize to fp16 (value * its per-token-head scale), and run the fp16
    path on a remapped block table. Decode / MTP verify (short queries, graph-captured) keep the
    inline path. Same idea as exl3xpu's fp8-KV prefill fix."""
    try:
        from vllm.v1.attention.backends import triton_attn as ta
        from vllm.v1.kv_cache_interface import KVQuantMode
    except Exception:
        return
    if getattr(ta, "_exl3_pth_prefill", False):
        return
    orig = ta.unified_attention
    thr = int(os.environ.get("EXL3_PTH_PREFILL_MIN_Q", "32"))
    # the fp16 K+V copy is transient and grows with context (~268 MB at 64k tokens). The inline
    # fallback is ~4.5x slower at 38k context (experiments/0022), so the default is no limit and
    # KV sizing leaves room for the copy; EXL3_PTH_PREFILL_MAX_MB caps it if memory is tighter.
    budget = int(float(os.environ.get("EXL3_PTH_PREFILL_MAX_MB", "1e9")) * 1048576)
    modes = (KVQuantMode.INT8_PER_TOKEN_HEAD, KVQuantMode.FP8_PER_TOKEN_HEAD)
    ones = {}

    def unified_attention(*args, **kw):
        mode = kw.get("kv_quant_mode", KVQuantMode.NONE)
        if args or mode not in modes or kw.get("max_seqlen_q", 0) < thr \
                or torch.cuda.is_current_stream_capturing():
            return orig(*args, **kw)
        q, k, v, bt = kw["q"], kw["k"], kw["v"], kw["block_table"]
        ks, vs = kw["k_scale_cache"], kw["v_scale_cache"]
        hs = q.shape[-1]
        bsz = k.shape[1]
        nb = (int(kw["max_seqlen_k"]) + bsz - 1) // bsz
        if bt.shape[0] * nb * bsz * k.shape[2] * hs * 2 * q.element_size() > budget:
            return orig(*args, **kw)
        used = bt[:, :nb]
        flat = used.reshape(-1)
        from .kv_dequant import gather_dequant
        kd = gather_dequant(k, ks, flat, hs, q.dtype, "k", cap=bt.numel())
        vd = gather_dequant(v, vs, flat, hs, q.dtype, "v", cap=bt.numel())
        key = (q.device, kd.shape[2])
        if key not in ones:
            ones[key] = torch.ones((1, 1), dtype=torch.float32, device=q.device)
        desc = ones[key].expand(bt.shape[0], kd.shape[2])
        kw = dict(kw, k=kd, v=vd, kv_quant_mode=KVQuantMode.NONE, k_scale_cache=None, v_scale_cache=None,
                  k_descale=desc, v_descale=desc,
                  block_table=torch.arange(flat.numel(), device=bt.device, dtype=bt.dtype).view(used.shape))
        return orig(**kw)

    ta.unified_attention = unified_attention
    ta._exl3_pth_prefill = True


def _install_kv_int4():
    """int4_per_token_head KV: route the cache write, decode / MTP-verify attention and long prefill chunks
    through exl3rocm.kv_int4 (same cache format). vLLM's int4 path ran its head-256 Hadamard transforms as a
    PyTorch butterfly and an MTP verify step unsplit (38 ms per attention layer at 32k, experiments/0024).
    Anything the kernel does not implement (sinks, ALiBi, sliding window, ...) still goes to vLLM."""
    try:
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends import triton_attn as ta
        from vllm.v1.kv_cache_interface import KVQuantMode
    except Exception:
        return
    if getattr(ta, "_exl3_kv_int4", False):
        return
    from . import kv_int4
    int4 = KVQuantMode.INT4_PER_TOKEN_HEAD
    inner = ta.unified_attention
    thr = int(os.environ.get("EXL3_PTH_PREFILL_MIN_Q", "32"))
    budget = int(float(os.environ.get("EXL3_PTH_PREFILL_MAX_MB", "1e9")) * 1048576)

    cur = {"layer": None}  # layer whose forward is running (its calibrated means center K / V)

    def unified_attention(*args, **kw):
        if args or kw.get("kv_quant_mode") != int4 or not kv_int4.supported(kw):
            return inner(*args, **kw)
        if kw.get("max_seqlen_q", 0) >= thr and not torch.cuda.is_current_stream_capturing():
            k, q, bt = kw["k"], kw["q"], kw["block_table"]
            nb = (int(kw["max_seqlen_k"]) + k.shape[1] - 1) // k.shape[1]
            if bt.shape[0] * nb * k.shape[1] * k.shape[2] * q.shape[-1] * 2 * q.element_size() <= budget:
                return kv_int4.prefill(inner, kw, layer_name=cur["layer"])
        return kv_int4.attention(**kw, layer_name=cur["layer"])

    ta.unified_attention = unified_attention
    cls = ta.TritonAttentionImpl
    orig_update = cls.do_kv_cache_update
    orig_forward = cls.forward

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self._kv_quant_mode != int4 or self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return orig_update(self, layer, key, value, kv_cache, slot_mapping)
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        kv_int4.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                                  k_scale_cache=self._k_scale_cache, v_scale_cache=self._v_scale_cache,
                                  layer_name=getattr(layer, "layer_name", None))

    def forward(self, layer, *args, **kw):
        cur["layer"] = getattr(layer, "layer_name", None)
        try:
            return orig_forward(self, layer, *args, **kw)
        finally:
            cur["layer"] = None

    cls.do_kv_cache_update = do_kv_cache_update
    cls.forward = forward
    ta._exl3_kv_int4 = True


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


def _install_rope_clamp():
    """Qwen3.5 sizes its rotary cos/sin cache by config.max_position_embeddings (262,144 rows,
    0.125 GiB) whatever max_model_len is. For plain rope and mrope ("default" rope type, no
    dual-chunk attention) the cache is only indexed by position, and positions stay below the
    sequence length, so clamp it to max_model_len plus a margin. Scaled rope types are untouched."""
    import sys
    from vllm.config import get_current_vllm_config_or_none
    from vllm.model_executor.layers import rotary_embedding
    orig = rotary_embedding.get_rope
    if getattr(orig, "_exl3_clamped", False):
        return

    def get_rope(head_size, max_position, *args, **kwargs):
        cfg = get_current_vllm_config_or_none()
        rp = kwargs.get("rope_parameters") or {}
        if (cfg is not None and isinstance(getattr(cfg, "quant_config", None), Exl3Config)
                and kwargs.get("dual_chunk_attention_config") is None and not args
                and rp.get("rope_type", "default") == "default" and not rp.get("use_fope")):
            cap = cfg.model_config.max_model_len + 4096
            if max_position > cap:
                max_position = cap
        return orig(head_size, max_position, *args, **kwargs)

    get_rope._exl3_clamped = True
    rotary_embedding.get_rope = get_rope
    # model modules imported before this hook hold their own reference
    for mod in list(sys.modules.values()):
        if getattr(mod, "get_rope", None) is orig and mod is not rotary_embedding:
            mod.get_rope = get_rope


def _install_kv_headroom_check():
    """vLLM 0.28 sizes a hybrid (GDN) model's KV pool so exactly one max_model_len request fits,
    counting 2 + num_speculative_blocks mamba pages per group in "align" mode (the default with
    prefix caching). In practice a prompt near max_model_len needed ~3 more pages: with MTP-3 at
    max_model_len 32768 and a 1.65 GB pool, a 31.5k-token prompt stalled forever at 94.6% KV
    usage with nothing to preempt (experiments/0023). Warn at startup when the pool leaves fewer
    than _KV_HEADROOM_BLOCKS spare pages, with the KV size or context length that would fit."""
    import sys
    from vllm.v1.core import kv_cache_utils
    orig = kv_cache_utils.update_kv_cache_capacity
    if getattr(orig, "_exl3_wrapped", False):
        return

    def update_kv_cache_capacity(vllm_config, kv_cache_config):
        orig(vllm_config, kv_cache_config)
        try:
            from vllm.utils.math_utils import cdiv
            from vllm.v1.kv_cache_interface import MambaSpec
            groups = kv_cache_config.kv_cache_groups
            if not any(isinstance(g.kv_cache_spec, MambaSpec) for g in groups):
                return
            per_req = sum(cdiv(g.kv_cache_spec.max_memory_usage_bytes(vllm_config), g.kv_cache_spec.page_size_bytes)
                          for g in groups)
            spare = kv_cache_config.num_blocks - 1 - per_req  # block 0 is the null block
            if spare >= _KV_HEADROOM_BLOCKS:
                return
            short = _KV_HEADROOM_BLOCKS - spare
            bsz = min(g.kv_cache_spec.block_size for g in groups if not isinstance(g.kv_cache_spec, MambaSpec))
            fit_len = vllm_config.model_config.max_model_len - short * bsz
            try:
                extra = f"about {short * kv_cache_utils._pool_bytes_per_block(vllm_config, groups):,} more bytes of " \
                        "--kv-cache-memory-bytes"
            except Exception:
                extra = f"{short} more KV blocks"
            logger.warning(
                "exl3rocm: the KV pool holds one max_model_len request with only %d spare blocks (%d-token "
                "pages); vLLM can stall a prompt near max_model_len indefinitely in this state. Use %s, or "
                "--max-model-len %d or less.", max(spare, 0), bsz, extra, max(fit_len, bsz))
        except Exception as e:  # never block startup on a diagnostic
            logger.debug("exl3rocm: KV headroom check skipped: %s", e)

    update_kv_cache_capacity._exl3_wrapped = True
    kv_cache_utils.update_kv_cache_capacity = update_kv_cache_capacity
    for mod in list(sys.modules.values()):
        if getattr(mod, "update_kv_cache_capacity", None) is orig and mod is not kv_cache_utils:
            mod.update_kv_cache_capacity = update_kv_cache_capacity


_KV_HEADROOM_BLOCKS = 3  # tested: 3 spare served a 32,344-token prompt at max_model_len 32768; 0 stalled


def _install_kv_int4_emulation(which):
    """EXL3_KV_EMU_INT4=k|v|kv (accuracy experiments only): before the per-token-head KV write,
    round-trip K and/or V through vLLM's int4_per_token_head quantizer in torch (same RHT, asymmetric
    4-bit, round-half-away, per token and head) and rotate back. The 8-bit cache then holds the
    4-bit-quantized values, which measures e.g. 4-bit keys + 8-bit values without a mixed-width layout.
    Saves no memory. EXL3_KV_EMU_CENTER=means.pt subtracts per-layer, per-head channel means before the
    round trip and adds them back (what exact K/V centering would do). EXL3_KV_STATS=out.pt accumulates those
    means from eager KV writes (tools/calib_kv_means.py). EXL3_KV_SAMPLES=out.pt saves each layer's first
    2048 keys / values and the 512 queries attending to them (offline quantizer studies). experiments/0024."""
    from vllm.v1.attention.backends import triton_attn as ta
    from vllm.v1.attention.ops.int4_per_token_head import single_rht
    cls = ta.TritonAttentionImpl
    if getattr(cls, "_exl3_kv_emu", False):
        return
    orig = cls.do_kv_cache_update

    def rnd(x):  # the int4 kernel's round-half-away-from-zero
        return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

    def fq(x):
        r = single_rht(x.float()).to(x.dtype).float()
        lo, hi = r.amin(-1, keepdim=True), r.amax(-1, keepdim=True)
        sc = ((hi - lo) / 15.0).clamp_min(1e-6)
        zp = rnd(-lo / sc).clamp(0, 15)
        q = rnd(r * (1.0 / sc) + zp).clamp(0, 15)
        return (single_rht((q - zp) * sc, inverse=True) / x.shape[-1]).to(x.dtype)

    means = torch.load(os.environ["EXL3_KV_EMU_CENTER"]) if os.environ.get("EXL3_KV_EMU_CENTER") else None
    stats_path = os.environ.get("EXL3_KV_STATS")
    stats, seen = {}, {"tokens": 0, "saved": 0}
    dev_means = {}

    def center(x, name, side):
        if means is None or name not in means:
            return fq(x)
        if (name, side) not in dev_means:  # first use is eager (warmup), before graph capture
            dev_means[(name, side)] = means[name][side].to(x.device, torch.float32)
        mu = dev_means[(name, side)]
        return (fq((x.float() - mu).to(x.dtype)).float() + mu).to(x.dtype)

    samples_path = os.environ.get("EXL3_KV_SAMPLES")
    samples, cur = {}, {"name": None}
    n_samp = 2048

    if samples_path:  # queries of the chunk that completes a layer's n_samp sampled keys
        inner = ta.unified_attention

        def unified_attention(*args, **kw):
            d = samples.get(cur["name"])
            if d is not None and "q" not in d and sum(x.shape[0] for x in d["k"]) >= n_samp:
                d["q"] = kw["q"][-512:].half().cpu()
                if all("q" in x for x in samples.values()) and len(samples) >= 16:
                    torch.save({n: {"k": torch.cat(x["k"])[:n_samp], "v": torch.cat(x["v"])[:n_samp], "q": x["q"]}
                                for n, x in samples.items()}, samples_path)
            return inner(*args, **kw)

        ta.unified_attention = unified_attention

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        name = getattr(layer, "layer_name", "")
        cur["name"] = name
        if samples_path and not torch.cuda.is_current_stream_capturing():
            d = samples.setdefault(name, {"k": [], "v": []})
            if sum(x.shape[0] for x in d["k"]) < n_samp:
                valid = slot_mapping[: key.shape[0]] >= 0
                d["k"].append(key[valid].half().cpu())
                d["v"].append(value[valid].half().cpu())
        if stats_path and not torch.cuda.is_current_stream_capturing():
            valid = slot_mapping[: key.shape[0]] >= 0
            st = stats.setdefault(name, {"k": 0.0, "v": 0.0, "n": 0})
            st["k"] = st["k"] + key[valid].double().sum(0)
            st["v"] = st["v"] + value[valid].double().sum(0)
            st["n"] += int(valid.sum())
            if name == min(stats):
                seen["tokens"] += int(valid.sum())
                if seen["tokens"] - seen["saved"] >= 2048:
                    seen["saved"] = seen["tokens"]
                    torch.save({n: {"k": (d["k"] / d["n"]).float().cpu(), "v": (d["v"] / d["n"]).float().cpu(),
                                    "n": d["n"]} for n, d in stats.items() if d["n"]}, stats_path)
        if self._is_per_token_head_quant:
            if "k" in which:
                key = center(key, name, "k")
            if "v" in which:
                value = center(value, name, "v")
        return orig(self, layer, key, value, kv_cache, slot_mapping)

    cls.do_kv_cache_update = do_kv_cache_update
    cls._exl3_kv_emu = True


def register():
    """vllm.general_plugins entry point (runs in every vLLM process)."""
    if os.environ.get("EXL3_DEBUG_GRAPH_TIME"):
        _install_debug_graph_timing()
    from . import ops  # noqa: F401  (registers torch.ops.exl3rocm.linear)
    if os.environ.get("EXL3_FP8_EMBEDDING", "1") != "0":
        _install_fp8_embedding_hook()
    _install_visual_qkv_filter()
    if os.environ.get("EXL3_DRAFT_VOCAB_BLOCKS", "640") != "0":
        _install_draft_logits_patch()
    _install_drafter_placeholder_hook()
    if os.environ.get("EXL3_ROPE_CLAMP", "1") != "0":
        _install_rope_clamp()
    _install_kv_headroom_check()
    if os.environ.get("EXL3_PTH_PREFILL_DEQUANT", "1") != "0":
        _install_pth_prefill_dequant()
    if os.environ.get("EXL3_KV_INT4", "1") != "0":
        _install_kv_int4()
    if os.environ.get("EXL3_DEBUG_MODULE_SYNC") == "1":
        _install_debug_module_sync()
    if os.environ.get("EXL3_KV_EMU_INT4") or os.environ.get("EXL3_KV_STATS") or os.environ.get("EXL3_KV_SAMPLES"):
        _install_kv_int4_emulation(os.environ.get("EXL3_KV_EMU_INT4", ""))
    return None
