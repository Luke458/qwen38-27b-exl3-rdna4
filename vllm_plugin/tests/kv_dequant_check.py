#!/usr/bin/env python3
"""gather_dequant vs a torch reference on a padded, non-contiguous per-token-head cache layout."""
import sys
import torch
sys.path.insert(0, "/plugin")
from exl3rocm.kv_dequant import gather_dequant
nb, bsz, nkv, hs, pad = 40, 784, 4, 256, 4
raw = torch.randint(-127, 127, (nb, nkv, bsz, 2 * (hs + pad)), dtype=torch.int8, device="cuda")
kv = raw.transpose(1, 2)                     # (B, N, H, content) view, like _pth_key_value_caches
k = kv[..., : hs + pad]
scale = torch.rand((nb, bsz, nkv), device="cuda") * 0.05
blocks = torch.tensor([3, 17, 0, 39, 5], dtype=torch.int32, device="cuda")
out = gather_dequant(k, scale, blocks, hs, torch.float16, "k")
ref = (k.index_select(0, blocks.long())[..., :hs].float() * scale.index_select(0, blocks.long()).unsqueeze(-1)).half()
print("gather_dequant exact:", torch.equal(out, ref), tuple(out.shape))
d = (out.float() - ref.float()).abs()
print("max abs diff", d.max().item(), "n diff", int((d > 0).sum()), "of", d.numel(),
      "max rel", (d / ref.float().abs().clamp_min(1e-6)).max().item())
