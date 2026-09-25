#!/usr/bin/env python3
"""Teacher-forced comparison: vLLM + exl3rocm prompt logprobs vs the exllamav3 fork's logits.

  # inside the serving image (plugin installed):
  python logits_compare.py vllm REF.npz OUT.json '{"max_num_batched_tokens": 8}'
  # anywhere with numpy:
  python logits_compare.py compare REF.npz OUT.json

REF.npz comes from bench/bench_generate.py --teacher-forced-out: logits[i] is the fork's
distribution after history[: P + i], P = len(input_ids) - len(logits). vLLM's
prompt_logprobs[j] is the distribution for token j, so row i <-> j = P + i.
"""
import json
import sys

import numpy as np


def run_vllm(ref, out, opts_json):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    d = np.load(ref)
    ids = d["input_ids"].tolist()
    P = len(ids) - d["logits"].shape[0]
    opts = dict(model="/model", dtype="float16", max_model_len=4096, gpu_memory_utilization=0.93,
                max_num_seqs=1, attention_backend="TRITON_ATTN", enable_prefix_caching=False,
                limit_mm_per_prompt={"image": 0, "video": 0}, kv_cache_memory_bytes=1500000000)
    opts.update(json.loads(opts_json))
    llm = LLM(**opts)
    res = llm.generate(TokensPrompt(prompt_token_ids=ids),
                       SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=20))
    pl = res[0].prompt_logprobs
    rows = []
    for j in range(P, len(ids)):
        dist = pl[j]
        rows.append({str(t): lp.logprob for t, lp in dist.items()})
    json.dump({"P": P, "rows": rows, "opts": opts}, open(out, "w"))
    print(f"wrote {len(rows)} rows to {out}")


def compare(ref, out):
    d = np.load(ref)
    logits = d["logits"].astype(np.float32)
    v = json.load(open(out))
    x = logits - logits.max(-1, keepdims=True)
    lsm = x - np.log(np.exp(x).sum(-1, keepdims=True))
    top1 = agree = 0
    dlp, overlap5, kl = [], [], []
    for i, row in enumerate(v["rows"]):
        vl = {int(t): lp for t, lp in row.items()}
        ref_top = np.argsort(-lsm[i])[:20]
        v_top = sorted(vl, key=lambda t: -vl[t])
        top1 += 1
        agree += int(ref_top[0] == v_top[0])
        if ref_top[0] in vl:
            dlp.append(abs(float(lsm[i][ref_top[0]]) - vl[ref_top[0]]))
        overlap5.append(len(set(ref_top[:5]) & set(v_top[:5])) / 5)
        # KL(ref || vllm) restricted to tokens both report in their top-20 (renormalized)
        common = [t for t in ref_top if t in vl]
        p = np.exp(np.array([lsm[i][t] for t in common])); p /= p.sum()
        q = np.exp(np.array([vl[t] for t in common])); q /= q.sum()
        kl.append(float((p * (np.log(p) - np.log(q))).sum()))
    res = {"positions": top1, "top1_agreement": agree / top1,
           "ref_argmax_logprob_absdiff_mean": float(np.mean(dlp)), "ref_argmax_logprob_absdiff_max": float(np.max(dlp)),
           "top5_overlap_mean": float(np.mean(overlap5)), "kl_top20_renorm_mean": float(np.mean(kl)),
           "kl_top20_renorm_max": float(np.max(kl)), "vllm_opts": {k: v["opts"].get(k) for k in ("max_num_batched_tokens",)}}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "vllm":
        run_vllm(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "{}")
    else:
        compare(sys.argv[2], sys.argv[3])
