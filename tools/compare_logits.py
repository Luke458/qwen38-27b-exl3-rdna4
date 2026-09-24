#!/usr/bin/env python3
"""Teacher-forced logits comparison (model-level numerical gate).

Compares logits npz files produced by bench/bench_generate.py --teacher-forced-out
between a baseline directory and a candidate directory. Files are matched by
name; comparison metrics per pair: max abs diff, mean KL(baseline || candidate),
top-1 agreement. Emits one metrics JSON for tools/run_candidate.py.

  python tools/compare_logits.py --baseline DIR --candidate DIR --out metrics.json \
      [--budget-logit-max-abs B] [--budget-kl-mean B]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def load_logits(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".npz"):
            out[fn] = os.path.join(d, fn)
    return out


def kl(p_logits, q_logits):
    """KL(p || q) row-wise for logits (softmax inside), mean over rows."""
    def logsoftmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
    lp, lq = logsoftmax(p_logits), logsoftmax(q_logits)
    p = np.exp(lp)
    return float((p * (lp - lq)).sum(axis=-1).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-logit-max-abs", type=float, default=None)
    ap.add_argument("--budget-kl-mean", type=float, default=None)
    ap.add_argument("--budget-delta-ppl", type=float, default=None)
    args = ap.parse_args()

    b, c = load_logits(args.baseline), load_logits(args.candidate)
    metrics = {"pairs": {}, "missing_in_candidate": sorted(set(b) - set(c)),
               "unexpected_in_candidate": sorted(set(c) - set(b))}
    worst_abs, worst_kl, worst_agree = 0.0, 0.0, 1.0
    worst_delta_ppl = -float('inf')
    for fn, baseline_path in b.items():
        if fn not in c:
            continue
        with np.load(baseline_path) as bz, np.load(c[fn]) as cz:
            bl, cl = bz['logits'], cz['logits']
            if ('input_ids' not in bz or 'input_ids' not in cz or
                    not np.array_equal(bz['input_ids'], cz['input_ids'])):
                metrics['pairs'][fn] = {'error': 'missing or different teacher-forced history'}
                continue
            history = bz['input_ids']
        if cl.shape != bl.shape:
            metrics["pairs"][fn] = {"error": f"shape {cl.shape} vs {bl.shape}"}
            continue
        if bl.size == 0 or not np.isfinite(bl).all() or not np.isfinite(cl).all():
            metrics["pairs"][fn] = {"error": "empty or nonfinite logits"}
            continue
        if bl.ndim != 2 or history.size <= bl.shape[0]:
            metrics['pairs'][fn] = {'error': 'invalid logits/history dimensions'}
            continue
        targets = history[-bl.shape[0]:]
        max_abs, kl_total, base_loss, cand_loss = 0.0, 0.0, 0.0, 0.0
        for start in range(0, bl.shape[0], 8):
            br, cr = bl[start:start+8].astype(np.float64), cl[start:start+8].astype(np.float64)
            max_abs = max(max_abs, float(np.max(np.abs(cr-br))))
            def logsoftmax(x):
                x = x - x.max(axis=-1, keepdims=True)
                return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
            bp, cp = logsoftmax(br), logsoftmax(cr)
            kl_total += float((np.exp(bp) * (bp-cp)).sum())
            target = targets[start:start+len(br)]
            if np.any(target < 0) or np.any(target >= br.shape[1]):
                raise ValueError('target token outside vocabulary')
            base_loss -= float(bp[np.arange(len(br)), target].sum())
            cand_loss -= float(cp[np.arange(len(br)), target].sum())
        k = max(0.0, kl_total / len(bl))
        bppl, cppl = float(np.exp(base_loss / len(bl))), float(np.exp(cand_loss / len(bl)))
        delta_ppl = cppl - bppl
        agree = float((cl.argmax(-1) == bl.argmax(-1)).mean())
        metrics["pairs"][fn] = {
            "logit_max_abs": max_abs,
            "kl_mean": k,
            "top1_agreement": agree,
            "baseline_ppl": bppl, "candidate_ppl": cppl, "delta_ppl": delta_ppl,
        }
        worst_abs = max(worst_abs, max_abs)
        worst_kl = max(worst_kl, k)
        worst_agree = min(worst_agree, agree)
        worst_delta_ppl = max(worst_delta_ppl, delta_ppl)

    metrics["logit_max_abs"] = worst_abs
    metrics["kl_mean"] = worst_kl
    metrics["top1_agreement"] = worst_agree
    metrics["delta_ppl"] = worst_delta_ppl if np.isfinite(worst_delta_ppl) else None

    ok = bool(b) and not metrics["unexpected_in_candidate"] and not any(
        "error" in x for x in metrics["pairs"].values())
    if args.budget_logit_max_abs is not None:
        ok &= worst_abs <= args.budget_logit_max_abs
        metrics["budget_logit_max_abs"] = args.budget_logit_max_abs
    if args.budget_kl_mean is not None:
        ok &= worst_kl <= args.budget_kl_mean
        metrics["budget_kl_mean"] = args.budget_kl_mean
    if metrics["missing_in_candidate"]:
        ok = False
    if args.budget_delta_ppl is not None:
        ok &= metrics['delta_ppl'] is not None and metrics['delta_ppl'] <= args.budget_delta_ppl
    ok &= metrics['delta_ppl'] is not None
    metrics["ok"] = ok

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: metrics[k] for k in ("logit_max_abs", "kl_mean", "top1_agreement", "ok")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
