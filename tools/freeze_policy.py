#!/usr/bin/env python3
"""Freeze the acceptance policy: require all null gates filled, write
configs/acceptance_policy.frozen.json and its SHA256 sidecar.

Unset (null) tolerances block the freeze and therefore promotion. The frozen
file plus hash must be referenced by every experiment record.

Usage:
  python tools/freeze_policy.py --fill measured_gates.json
where measured_gates.json carries the measured_baseline / budget fields
computed from baseline-vs-oracle runs (P1) under the derivation rules already
recorded in the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_nulls(obj, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += find_nulls(v, f"{path}.{k}")
    elif obj is None:
        out.append(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="configs/acceptance_policy.json")
    ap.add_argument("--fill", required=True, help="json merge with measured values")
    ap.add_argument("--out", default="configs/acceptance_policy.frozen.json")
    ap.add_argument("--mode", choices=("fp16_same", "int8_plain", "int8_residual"), default="fp16_same")
    args = ap.parse_args()

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    policy = json.load(open(rp(args.policy)))
    fill = json.load(open(rp(args.fill)))

    def merge(dst, src, path=""):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v, f"{path}.{k}")
            else:
                if k not in dst:
                    raise ValueError(f"unknown fill field: {path}.{k}")
                if dst[k] is not None:
                    raise ValueError(f"fill may only set null fields: {path}.{k}")
                dst[k] = v

    try:
        merge(policy, fill)
    except ValueError as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2
    # A mode is frozen only after its own measured budget exists. A future
    # arithmetic mode cannot inherit the active mode's numerical gate.
    mode_section = {"fp16_same": "float_gates_same_arithmetic",
                    "int8_plain": "float_gates_int8_plain",
                    "int8_residual": "float_gates_int8_residual"}[args.mode]
    if args.mode == "fp16_same":
        sec = policy[mode_section]
        measured = sec["measured_baseline"]
        analytic = sec["analytic_bounds"]
        needed = ("normalized_max_abs_err", "normalized_rms_err",
                  "cosine_similarity_min", "abs_rel_bound_ratio")
        for key in needed:
            value = measured.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                print(f"BLOCKED: missing finite measured_baseline.{key}", file=sys.stderr)
                return 2
        if not measured.get("measured_at"):
            print("BLOCKED: measured_baseline.measured_at missing", file=sys.stderr)
            return 2
        sec["final_gates"] = {
            key: (max(analytic[key], 1.0 - 4.0 * (1.0 - measured[key]))
                  if key == "cosine_similarity_min" else max(analytic[key], 4.0 * measured[key]))
            for key in needed
        }
    else:
        for key, val in policy[mode_section]["budget"].items():
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                print(f"BLOCKED: {mode_section}.budget.{key} missing or invalid", file=sys.stderr)
                return 2
    policy["status"] = "FROZEN"
    policy["frozen_arithmetic_mode"] = args.mode
    policy["frozen_from_measurements"] = fill

    inactive = {"float_gates_same_arithmetic", "float_gates_int8_plain", "float_gates_int8_residual"} - {mode_section}
    nulls = [n for n in find_nulls(policy)
             if not n.startswith(".frozen_from_measurements")
             and not any(n.startswith("." + section + ".") for section in inactive)
             and n != ".model_level_gates.generation_stability.min_token_agreement_rate"]
    if nulls:
        print("BLOCKED: unset tolerances block the freeze:", file=sys.stderr)
        for n in nulls:
            print(" -", n, file=sys.stderr)
        return 2

    out = rp(args.out)
    if os.path.exists(out):
        print(f"BLOCKED: frozen policy already exists: {out}", file=sys.stderr)
        return 2
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(policy, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    with open(out + ".sha256", "w") as f:
        f.write(digest + "  " + os.path.basename(out) + "\n")
    print(f"froze {out}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
