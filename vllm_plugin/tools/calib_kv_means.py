#!/usr/bin/env python3
"""Per-layer K/V channel means for int4 KV centering (exl3rocm/kv_int4.py), including the MTP layer.

Run inside the serving image with the plugin installed, the model at /model and text to read:
  python calib_kv_means.py OUT.pt [CORPUS_DIR]

Reads ~24k tokens of CORPUS_DIR's markdown and Python (default /repo: this repository's docs and code) and
averages each attention layer's post-RoPE keys and values (EXL3_KV_STATS hook). Centering is exact for any
means, so the corpus only affects how tight the 4-bit ranges are (experiments/0024: 17% lower attention
error with centering + clip search). The packaged file is exl3rocm/kv_means.pt.
"""
import glob
import os
import sys


def main():
    out = os.path.abspath(sys.argv[1])
    root = sys.argv[2] if len(sys.argv) > 2 else "/repo"
    os.environ["EXL3_KV_STATS"] = out
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model="/model", dtype="float16", max_model_len=8448, max_num_seqs=1, attention_backend="TRITON_ATTN",
              enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0}, enforce_eager=True,
              kv_cache_memory_bytes=1000000000, max_num_batched_tokens=512, mamba_ssm_cache_dtype="float16",
              kv_cache_dtype="int8_per_token_head",
              speculative_config={"method": "mtp", "num_speculative_tokens": 3})
    tok = llm.get_tokenizer()
    for pat in ("docs/*.md", "vendor/rocm_exl3/exllamav3/modules/*.py", "vllm_plugin/**/*.py"):
        files = sorted(glob.glob(os.path.join(root, pat), recursive=True))
        text = "\n\n".join(open(f, errors="replace").read() for f in files)
        ids = tok.encode(text, add_special_tokens=False)[:8192]
        if ids:
            llm.generate(TokensPrompt(prompt_token_ids=ids), SamplingParams(max_tokens=1))
            print(f"{pat}: {len(ids)} tokens")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
