#!/usr/bin/env python3
"""Experiment controller for the bounded optimization loop.

Implements the runner contract of OPTIMIZATION_LOOP.md:

  python tools/run_candidate.py --experiment experiments/0001/experiment.json

Validates paths and required fields, checks source/binary identity at load,
acquires a host-wide GPU lock, runs each stage in a fresh process with its own
timeout and explicit environment, and writes one append-only result record per
stage to experiments/<id>/results.jsonl.

State machine:
  PROPOSED -> BUILT -> CORRECT -> MICROBENCH_PASSED -> MODEL_CORRECT -> QUALIFIED -> PROMOTED
Terminal: BUILD_FAILED, INCORRECT, NO_GAIN, REGRESSION, INCONCLUSIVE,
          ENVIRONMENT_INVALID, TIMEOUT, GPU_FAULT, BLOCKED

A crash leaves the experiment incomplete; it never updates the champion.
Correctness failures cannot receive a performance score.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import math
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_LOCK = os.environ.get("EXL3_GPU_LOCK", "/tmp/exl3-gfx1201-gpu.lock")

STAGE_ORDER = ["build", "correctness", "microbench", "model_correct", "generation"]
STATE_AFTER = {
    "build": "BUILT",
    "correctness": "CORRECT",
    "microbench": "MICROBENCH_PASSED",
    "model_correct": "MODEL_CORRECT",
    "generation": "GENERATION_COMPLETE",
}
TERMINAL = {
    "BUILD_FAILED", "INCORRECT", "NO_GAIN", "REGRESSION", "INCONCLUSIVE",
    "ENVIRONMENT_INVALID", "TIMEOUT", "GPU_FAULT", "BLOCKED",
}

REQUIRED_FIELDS = [
    "id", "hypothesis", "mechanism", "source", "environment_lock", "identities",
    "dispatch", "stages",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Experiment:
    def __init__(self, spec_path: str):
        self.spec_path = os.path.abspath(spec_path)
        self.dir = os.path.dirname(self.spec_path)
        with open(spec_path) as f:
            self.spec = json.load(f)
        self.results_path = os.path.join(self.dir, "results.jsonl")

    # ---------------- validation ----------------
    def validate(self) -> list[str]:
        errs = []
        for field in REQUIRED_FIELDS:
            if field not in self.spec:
                errs.append(f"missing required field: {field}")
        if errs:
            return errs
        src = self.spec["source"]
        if not isinstance(src, dict) or not isinstance(self.spec.get("stages"), dict):
            return ["source and stages must be objects"]
        for key in ("commit", "binary", "binary_sha256"):
            if key not in src:
                errs.append(f"source.{key} missing")
        if src.get("binary") and not os.path.isfile(src["binary"]):
            errs.append(f"source.binary not found: {src['binary']}")
        if src.get("patch") and not os.path.isfile(os.path.join(self.dir, src["patch"]) if not os.path.isabs(src["patch"]) else src["patch"]):
            errs.append(f"source.patch not found: {src['patch']}")
        for sid, stage in self.spec.get("stages", {}).items():
            if sid not in STAGE_ORDER:
                errs.append(f"unknown stage: {sid}")
            if not isinstance(stage, dict):
                errs.append(f"stages.{sid} must be an object")
                continue
            for key in ("cmd", "cwd", "env", "timeout_s"):
                if key not in stage:
                    errs.append(f"stages.{sid}.{key} missing")
            if "cmd" in stage and (not isinstance(stage["cmd"], list) or not stage["cmd"] or not all(isinstance(x, str) and x for x in stage["cmd"])):
                errs.append(f"stages.{sid}.cmd must be an argument array (list)")
            if "cwd" in stage and not os.path.isdir(stage["cwd"]):
                errs.append(f"stages.{sid}.cwd not found")
            if "env" in stage and (not isinstance(stage["env"], dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in stage["env"].items())):
                errs.append(f"stages.{sid}.env must map strings to strings")
            if "timeout_s" in stage and (not isinstance(stage["timeout_s"], (int, float)) or not math.isfinite(stage["timeout_s"]) or stage["timeout_s"] <= 0):
                errs.append(f"stages.{sid}.timeout_s must be positive and finite")
            if sid != "build" and not stage.get("metrics"):
                errs.append(f"stages.{sid}.metrics required")
            if sid != "build" and stage.get("metrics_required") is False:
                errs.append(f"stages.{sid}.metrics_required cannot be false")
        return errs

    def verify_identities(self) -> list[str]:
        """Check binary + frozen-input identity at load. Any drift is fatal."""
        errs = []
        src = self.spec["source"]
        if src.get("binary") and os.path.isfile(src["binary"]):
            got = sha256_file(src["binary"])
            if got != src["binary_sha256"]:
                errs.append(f"binary hash drift: expected {src['binary_sha256']}, got {got}")
        ids = self.spec["identities"]
        if not isinstance(ids, dict):
            return ["identities must be an object"]
        for key, path in (
            ("policy", ids.get("policy")),
            ("workloads", ids.get("workloads")),
            ("shape_manifest", ids.get("shape_manifest")),
        ):
            if not path:
                errs.append(f"identities.{key} missing")
                continue
            p = path if os.path.isabs(path) else os.path.join(ROOT, path)
            if not os.path.isfile(p):
                errs.append(f"identities.{key} file not found: {path}")
            expected = ids.get(f"{key}_sha256")
            if not expected:
                errs.append(f"identities.{key}_sha256 missing")
            elif os.path.isfile(p) and sha256_file(p) != expected:
                errs.append(f"identities.{key} hash drift")
        for key, path in (("environment_lock", self.spec.get("environment_lock")), ("patch", src.get("patch"))):
            if not path:
                errs.append(f"{key} missing")
                continue
            p = path if os.path.isabs(path) else os.path.join(self.dir if key == "patch" else ROOT, path)
            if not os.path.isfile(p):
                errs.append(f"{key} file not found")
                continue
            expected = src.get("patch_sha256") if key == "patch" else self.spec.get("environment_lock_sha256")
            if not expected:
                errs.append(f"{key}_sha256 missing")
            elif os.path.isfile(p) and sha256_file(p) != expected:
                errs.append(f"{key} hash drift")
        return errs

    # ---------------- records ----------------
    def append_record(self, record: dict) -> None:
        record = dict(record, ts=utcnow())
        with open(self.results_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def set_state(self, state: str, reason: str | None = None) -> None:
        self.append_record({"stage": "state", "state": state, "reason": reason})
        self.state = state

    # ---------------- execution ----------------
    def run_stage(self, sid: str) -> tuple[bool, str | None]:
        stage = self.spec["stages"][sid]
        log_out = os.path.join(self.dir, f"stage_{sid}.stdout.log")
        log_err = os.path.join(self.dir, f"stage_{sid}.stderr.log")
        env = dict(os.environ)
        env.update(stage.get("env", {}))
        metrics_file = stage.get("metrics")
        if metrics_file and os.path.isfile(metrics_file):
            os.unlink(metrics_file)
        pre_binary_hash = sha256_file(self.spec["source"]["binary"])
        t0 = time.monotonic()
        timed_out = False
        with open(log_out, "wb") as fo, open(log_err, "wb") as fe:
            try:
                proc = subprocess.run(
                    stage["cmd"], cwd=stage["cwd"], env=env,
                    stdout=fo, stderr=fe, timeout=stage["timeout_s"],
                )
                exit_status = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_status = -1
            except OSError as e:
                exit_status = -1
                fe.write(str(e).encode())
        dur = time.monotonic() - t0

        metrics = None
        if metrics_file and os.path.isfile(metrics_file):
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                metrics = {"_error": f"metrics unreadable: {e}"}

        ok = exit_status == 0 and not timed_out
        if ok and metrics is None and sid != "build":
            ok = False
            fail_reason = "missing metrics (absent is absent, never zero)"
        elif ok and not isinstance(metrics, dict) and sid != "build":
            ok = False
            fail_reason = "metrics must be a JSON object"
        elif ok and isinstance(metrics, dict) and metrics.get("_error"):
            ok = False
            fail_reason = "unreadable metrics"
        elif ok and isinstance(metrics, dict) and metrics.get("ok") is False:
            ok = False
            fail_reason = "stage metrics report failure"
        elif timed_out:
            fail_reason = f"TIMEOUT after {stage['timeout_s']}s"
        elif exit_status != 0:
            fail_reason = f"exit status {exit_status}"
        else:
            fail_reason = None
        if sha256_file(self.spec["source"]["binary"]) != pre_binary_hash:
            ok = False
            fail_reason = "binary changed during stage"

        self.append_record({
            "stage": sid, "exit_status": exit_status, "duration_s": round(dur, 3),
            "timed_out": timed_out, "cmd": stage["cmd"], "cwd": stage["cwd"],
            "stdout": os.path.relpath(log_out, self.dir),
            "stderr": os.path.relpath(log_err, self.dir),
            "metrics": metrics, "ok": ok, "fail_reason": fail_reason,
        })
        return ok, fail_reason


def acquire_gpu_lock(path: str, budget_s: float = 30.0):
    """Non-blocking GPU lock acquisition with a bounded wait.

    Never blocks forever: a held lock (e.g. an operator's flock wrapper around
    this very process) must classify ENVIRONMENT_INVALID rather than hang --
    a silent hang would stall the campaign and mask the contention.
    """
    f = open(path, "w")
    deadline = time.time() + budget_s
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            if time.time() >= deadline:
                f.close()
                return None
            time.sleep(0.1)


def classify_failure(sid: str, reason: str | None) -> str:
    if reason and reason.startswith("TIMEOUT"):
        return "TIMEOUT"
    if sid == "build":
        return "BUILD_FAILED"
    return "INCORRECT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--from-stage", default=None, help="resume at a stage (fresh processes)")
    ap.add_argument("--dry-run", action="store_true", help="validate only")
    args = ap.parse_args()

    exp = Experiment(args.experiment)
    errs = exp.validate()
    if not errs:
        errs += exp.verify_identities()
    if errs:
        for e in errs:
            print(f"BLOCKED: {e}", file=sys.stderr)
        exp.set_state("BLOCKED", "; ".join(errs))
        return 2
    print(f"experiment {exp.spec['id']}: identity checks passed")
    if args.dry_run:
        return 0

    start = args.from_stage or "build"
    try:
        idx = STAGE_ORDER.index(start)
    except ValueError:
        print(f"unknown stage {start}", file=sys.stderr)
        return 2
    if idx:
        print("BLOCKED: --from-stage cannot bypass earlier gates; resume requires audited prior stages", file=sys.stderr)
        return 2

    lockdir = os.path.dirname(GPU_LOCK)
    if lockdir:
        os.makedirs(lockdir, exist_ok=True)
    lock = acquire_gpu_lock(GPU_LOCK)   # host-wide GPU serialization (bounded)
    if lock is None:
        exp.set_state("ENVIRONMENT_INVALID", f"GPU lock busy after 30s: {GPU_LOCK}")
        print(f"[{exp.spec['id']}] ENVIRONMENT_INVALID (GPU lock busy: {GPU_LOCK})", file=sys.stderr)
        return 2
    with lock:
        exp.set_state("RUNNING")
        for sid in STAGE_ORDER[idx:]:
            if sid not in exp.spec["stages"]:
                exp.set_state("BLOCKED", f"required stage {sid} missing")
                return 2
            # build runs outside GPU time but inside the lock: simpler and safe
            print(f"[{exp.spec['id']}] stage {sid} ...", flush=True)
            ok, reason = exp.run_stage(sid)
            if not ok:
                state = classify_failure(sid, reason)
                exp.set_state(state, f"{sid}: {reason}")
                print(f"[{exp.spec['id']}] {state} ({sid}: {reason})")
                return 1
            exp.set_state(STATE_AFTER[sid])
        # Generation stage produced QUALIFIED in the state map; the promotion
        # decision belongs to tools/compare.py + the controller operator, not
        # to the experiment record itself.
        print(f"[{exp.spec['id']}] stages complete (qualification pending tools/compare.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
