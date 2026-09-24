#!/usr/bin/env python3
"""Compare a candidate against the immutable baseline and apply frozen gates.

  python tools/compare.py --baseline artifacts/baseline --candidate experiments/0001

Reads stage metrics produced by tools/run_candidate.py (raw paired timings,
numerical statistics) plus the frozen acceptance policy, and writes
<candidate>/comparison.json with a verdict:

  PASS            all correctness + performance gates met -> promotion eligible
  NO_GAIN         correct, but performance gates not met
  REGRESSION      correct, but a holdout regression is confirmed
  INCORRECT       correctness gates failed (no performance score is computed)
  INCONCLUSIVE    measurement uncertainty resolved only by more measurement

Missing metrics are missing (never zero) and block a PASS verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def bootstrap_ci(stat, samples, n_boot=10000, alpha=0.05, seed=20260923):
    """Percentile bootstrap CI for `stat` over paired speedup samples."""
    rng = random.Random(seed)
    vals = []
    m = len(samples)
    for _ in range(n_boot):
        resample = [samples[rng.randrange(m)] for _ in range(m)]
        vals.append(stat(resample))
    vals.sort()
    lo = vals[int((alpha / 2) * n_boot)]
    hi = vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def check_float_gates(stats: dict, gates: dict) -> list[str]:
    fails = []
    for key in ("normalized_rms_err", "normalized_max_abs_err", "max_abs_err", "abs_rel_bound_ratio"):
        if key not in gates:
            continue
        lim = gates.get(key)
        if lim is None:
            fails.append(f"{key}: gate not frozen (null)")
        elif stats.get(key) is None:
            fails.append(f"{key}: metric missing")
        elif not isinstance(stats[key], (int, float)) or not math.isfinite(stats[key]):
            fails.append(f"{key}: metric nonfinite or invalid")
        elif stats[key] > lim:
            fails.append(f"{key}: {stats[key]:.6g} > {lim:.6g}")
    cos_lim = gates.get("cosine_similarity_min")
    cos = stats.get("cosine_similarity")
    if cos_lim is None:
        fails.append("cosine_similarity_min: gate not frozen (null)")
    elif cos is None:
        fails.append("cosine_similarity: metric missing")
    elif not isinstance(cos, (int, float)) or not math.isfinite(cos):
        fails.append("cosine_similarity: metric nonfinite or invalid")
    elif cos < cos_lim:
        fails.append(f"cosine_similarity: {cos:.8f} < {cos_lim:.8f}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="artifacts/baseline")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--policy", default="configs/acceptance_policy.frozen.json")
    args = ap.parse_args()

    policy_path = args.policy if os.path.isabs(args.policy) else os.path.join(ROOT, args.policy)
    if not os.path.isfile(policy_path):
        print("BLOCKED: frozen policy missing (tools/freeze_policy.py first)", file=sys.stderr)
        return 2
    policy = load_json(policy_path)
    if policy.get("status") != "FROZEN":
        print("BLOCKED: policy not FROZEN", file=sys.stderr)
        return 2

    cand_dir = args.candidate if os.path.isabs(args.candidate) else os.path.join(ROOT, args.candidate)
    base_dir = args.baseline if os.path.isabs(args.baseline) else os.path.join(ROOT, args.baseline)

    def results(d):
        path = os.path.join(d, "results.jsonl")
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    brec, crec = results(base_dir), results(cand_dir)
    report = {"verdict": None, "correctness": [], "performance": {}, "notes": []}

    def last_metrics(recs, stage):
        out = None
        for r in recs:
            if r.get("stage") == stage:
                out = r.get("metrics") if r.get("ok") is True and r.get("exit_status") == 0 else None
        return out

    def finite_number(value, positive=False):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and (value > 0 if positive else True)

    def policy_digest(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    spec_path = os.path.join(cand_dir, "experiment.json")
    if not os.path.isfile(spec_path):
        print("BLOCKED: candidate experiment.json missing", file=sys.stderr)
        return 2
    spec = load_json(spec_path)
    ids = spec.get("identities", {})
    if ids.get("policy_sha256") != policy_digest(policy_path):
        print("BLOCKED: candidate policy identity differs from compared policy", file=sys.stderr)
        return 2
    for stage in ("build", "correctness", "microbench", "model_correct", "generation"):
        if not any(r.get("stage") == stage and r.get("ok") is True for r in crec):
            report["correctness"].append(f"required stage {stage} did not pass")
    if not crec or crec[-1].get("state") != "GENERATION_COMPLETE":
        report["correctness"].append("candidate has no completed stage chain")

    # ---------------- correctness ----------------
    fails: list[str] = report["correctness"]
    same_arith = last_metrics(crec, "correctness")
    if same_arith is None:
        fails.append("correctness metrics missing")
    else:
        mode = same_arith.get("arithmetic_mode")
        if mode != policy.get("frozen_arithmetic_mode"):
            fails.append(f"arithmetic mode {mode!r} does not match frozen mode")
        if mode == "fp16_same":
            gate = policy["float_gates_same_arithmetic"]
            fails += check_float_gates(same_arith, gate.get("final_gates") or {})
        elif mode == "int8_plain":
            fails += check_float_gates(same_arith, policy["float_gates_int8_plain"]["budget"])
        elif mode == "int8_residual":
            fails += check_float_gates(same_arith, policy["float_gates_int8_residual"]["budget"])
        else:
            fails.append(f"unknown arithmetic_mode {mode!r}")
        if not isinstance(same_arith.get("exactness_failures"), int) or same_arith["exactness_failures"] != 0:
            fails.append(f"exactness failures: {same_arith.get('exactness_failures')}")
    model = last_metrics(crec, "model_correct")
    if not isinstance(model, dict):
        fails.append("model correctness metrics missing")
    else:
        budget = policy["model_level_gates"]
        for metric, limit in (("logit_max_abs", budget["teacher_forced"]["budget_logit_max_abs"]),
                              ("kl_mean", budget["teacher_forced"]["budget_kl_mean"]),
                              ("delta_ppl", budget["held_out_loss"]["budget_delta_ppl"])):
            value = model.get(metric)
            if not finite_number(limit) or not finite_number(value) or value > limit:
                fails.append(f"{metric}: missing, invalid or over frozen budget")
        if model.get("missing_in_candidate") or model.get("ok") is not True:
            fails.append("model correctness reported missing outputs or failure")
    report["correctness"] = fails
    if fails:
        report["verdict"] = "INCORRECT"
        report["notes"].append("correctness failures cannot receive a performance score")
        with open(os.path.join(cand_dir, "comparison.json"), "w") as f:
            json.dump(report, f, indent=1)
        print("INCORRECT")
        for x in fails:
            print(" -", x)
        return 1

    # ---------------- performance ----------------
    perf = policy["performance_gates"]["gates"]
    mb = last_metrics(crec, "microbench") or {}
    gen = last_metrics(crec, "generation") or {}
    gen_b = last_metrics(brec, "generation") or {}
    if gen.get("invalid_samples") or gen_b.get("invalid_samples"):
        report["notes"].append("invalid paired samples require replacement measurements")

    def pairs(dct, workload):
        return dct.get("paired_decode_s", {}).get(workload)

    primary = policy["performance_gates"]["primary_workload"]
    pp = pairs(gen, primary) or []
    bp = gen_b.get("paired_decode_s", {}).get(primary)
    decision_n = policy["performance_gates"]["paired_trials_decision"]
    if (len(pp) < decision_n or not bp or len(bp) < decision_n or
        gen.get("invalid_samples") or gen_b.get("invalid_samples") or
        not all(finite_number(v, True) for v in pp + (bp or []))):
        report["verdict"] = "INCONCLUSIVE"
        report["notes"].append(f"need {decision_n} paired trials on {primary}; have candidate={len(pp)} baseline={len(bp) if bp else 0}")
    else:
        sp = [b / c for b, c in zip(bp[:decision_n], pp[:decision_n])]
        med = statistics.median(sp)
        lo, hi = bootstrap_ci(statistics.median, sp)
        report["performance"]["primary"] = {
            "workload": primary, "n_pairs": len(sp), "median_speedup": med,
            "ci95": [lo, hi], "raw_speedups": sp,
        }
        holdouts = {}
        regressed = []
        for w in [primary] + policy["performance_gates"]["holdout_workloads"]:
            cp = pairs(gen, w) or []
            bpp = gen_b.get("paired_decode_s", {}).get(w) or []
            if len(cp) < decision_n or len(bpp) < decision_n or not all(finite_number(v, True) for v in cp + bpp):
                holdouts[w] = "insufficient pairs"
                regressed.append(w)  # uncertainty -> more measurement, not promotion
                continue
            s = [b / c for b, c in zip(bpp[:decision_n], cp[:decision_n])]
            hm = statistics.median(s)
            hlo, hhi = bootstrap_ci(statistics.median, s)
            holdouts[w] = {"median_speedup": hm, "ci95_lower": hlo, "ci95_upper": hhi}
            if hlo <= 1.0 - perf["holdout_max_regression"]:
                regressed.append(w)
            for label, field in (("prefill", "paired_prefill_s"), ("first_token", "paired_ttft_s")):
                cv = gen.get(field, {}).get(w, [])
                bv = gen_b.get(field, {}).get(w, [])
                if len(cv) < decision_n or len(bv) < decision_n or not all(finite_number(v, True) for v in cv + bv):
                    holdouts[w][label] = "insufficient pairs"
                    regressed.append(w)
                    continue
                slo, shi = bootstrap_ci(statistics.median, [b / c for b, c in zip(bv[:decision_n], cv[:decision_n])])
                holdouts[w][label] = {"ci95_lower": slo, "ci95_upper": shi}
                if slo <= 1.0 - perf["holdout_max_regression"]:
                    regressed.append(w)
            memory = gen.get("paired_peak_vram_bytes", {}).get(w, [])
            fit = policy.get("memory_fit_budget_bytes")
            if not finite_number(fit, True) or len(memory) < decision_n or not all(finite_number(v, True) and v <= fit for v in memory):
                holdouts[w]["memory"] = "missing or exceeds frozen fit budget"
                regressed.append(w)
        report["performance"]["holdouts"] = holdouts

        if regressed:
            report["verdict"] = "REGRESSION" if any(
                isinstance(holdouts.get(w), dict) and any(
                    isinstance(holdouts[w].get(k), dict) and holdouts[w][k]["ci95_upper"] < 1.0 - perf["holdout_max_regression"]
                    for k in ("prefill", "first_token"))
                for w in regressed) else "INCONCLUSIVE"
            report["notes"].append(f"holdout issues: {regressed}")
        elif lo > perf["decode_speedup_ci95_lower_gt"] and med >= perf["decode_speedup_median_ge"]:
            confirm_n = policy["performance_gates"]["paired_trials_confirmation"]
            if len(pp) < decision_n + confirm_n or len(bp) < decision_n + confirm_n:
                report["verdict"] = "INCONCLUSIVE"
                report["notes"].append(f"decision gate passed; {confirm_n} reserved confirmation pairs required")
            else:
                confirmation = [b / c for b, c in zip(
                    bp[decision_n:decision_n + confirm_n], pp[decision_n:decision_n + confirm_n])]
                if not all(finite_number(v, True) for v in confirmation):
                    report["verdict"] = "INCONCLUSIVE"
                    report["notes"].append("invalid confirmation pair")
                else:
                    report["performance"]["confirmation_median_speedup"] = statistics.median(confirmation)
                    report["verdict"] = "PASS" if statistics.median(confirmation) > 1.0 else "INCONCLUSIVE"
        else:
            report["verdict"] = "NO_GAIN"
            report["notes"].append(
                f"primary median {med:.4f} (need >= {perf['decode_speedup_median_ge']}), "
                f"CI lower {lo:.4f} (need > {perf['decode_speedup_ci95_lower_gt']})")

    # weighted layer reporting (never an unweighted average as a tokens/s claim)
    if mb.get("weighted_layer"):
        report["performance"]["weighted_layer"] = mb["weighted_layer"]

    with open(os.path.join(cand_dir, "comparison.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(report["verdict"])
    for k, v in report["performance"].items():
        print(f" {k}: {v}")
    for n in report["notes"]:
        print(" note:", n)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
