"""Failure-injection tests for the runner and gates (P2 exit gate).

Deliberately wrong outputs, stale binaries, missing metrics and timeouts must
fail or classify correctly. CPU-only: stages are /bin stubs with json metrics.
"""

import json
import os
import subprocess
import sys
import textwrap
import hashlib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "tools", "run_candidate.py")
COMPARE = os.path.join(ROOT, "tools", "compare.py")
POLICY = os.path.join(ROOT, "configs", "acceptance_policy.example.json")


@pytest.fixture(autouse=True)
def private_gpu_lock(tmp_path, monkeypatch):
    """Isolate these tests from the host-wide GPU lock.

    Several tests spawn tools/run_candidate.py as a child; if this pytest
    process were itself wrapped in a hold of the same lock (operator habit of
    `flock /tmp/exl3-gfx1201-gpu.lock ...`), the child's acquisition would
    deadlock against its own parent. A private lock file keeps the suite
    deterministic under any wrapper.
    """
    monkeypatch.setenv("EXL3_GPU_LOCK", str(tmp_path / "gpu.lock"))


def write_exp(tmp_path, stage_cmd, binary=None, metrics_required=True, timeout_s=30):
    exp_dir = tmp_path / "experiments" / "0001"
    exp_dir.mkdir(parents=True)
    metrics = exp_dir / "metrics.json"
    patch = exp_dir / "source.patch"
    patch.write_text("test patch")
    env_lock = exp_dir / "environment.lock.json"
    env_lock.write_text("{}")
    ident = {}
    for key, name in (("policy", "acceptance_policy.json"),
                      ("workloads", "workloads.json"),
                      ("shape_manifest", "shape_manifest.json")):
        path = os.path.join(ROOT, "configs", name)
        if name == "acceptance_policy.json":
            path = POLICY
        elif not os.path.isfile(path):
            # Runner identity tests do not require private checkpoint metadata.
            fixture = tmp_path / name
            fixture.write_text("{}")
            path = str(fixture)
        ident[key] = path
        ident[key + "_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    spec = {
        "id": "0001",
        "hypothesis": "test",
        "mechanism": "test",
        "source": {
            "commit": "311ff5497237a37bea18cf24aa18bc0c573d1d03",
            "binary": str(binary or (exp_dir / "bin.so")),
            "binary_sha256": "",
            "patch": str(patch),
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        },
        "environment_lock": str(env_lock),
        "environment_lock_sha256": hashlib.sha256(env_lock.read_bytes()).hexdigest(),
        "identities": ident,
        "dispatch": {"predicate": "true", "arithmetic_mode": "fp16_same"},
        "stages": {
            "build": {
                "cmd": ["true"], "cwd": str(ROOT), "env": {}, "timeout_s": timeout_s,
                "metrics_required": False,
            },
            "correctness": {
                "cmd": stage_cmd, "cwd": str(ROOT), "env": {},
                "timeout_s": timeout_s, "metrics": str(metrics),
                "metrics_required": metrics_required,
            },
        },
    }
    for sid in ("microbench", "model_correct", "generation"):
        stage_metrics = exp_dir / f"{sid}.json"
        spec["stages"][sid] = {
            "cmd": [sys.executable, "-c", f"import json; json.dump({{'ok': True}}, open({str(stage_metrics)!r}, 'w'))"],
            "cwd": str(ROOT), "env": {}, "timeout_s": timeout_s,
            "metrics": str(stage_metrics),
        }
    bin_path = exp_dir / "bin.so"
    bin_path.write_bytes(b"fake-binary")
    spec["source"]["binary_sha256"] = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    spec_path = exp_dir / "experiment.json"
    spec_path.write_text(json.dumps(spec))
    return spec_path, metrics


def run_runner(spec_path, *extra):
    return subprocess.run(
        [sys.executable, RUNNER, "--experiment", str(spec_path), *extra],
        capture_output=True, text=True, timeout=120,
    )


def test_missing_metrics_fails(tmp_path):
    spec_path, metrics = write_exp(
        tmp_path,
        ["python3", "-c", "print('ran but wrote no metrics')"],
        metrics_required=True,
    )
    r = run_runner(spec_path)
    assert r.returncode == 1
    state = json.loads((tmp_path / "experiments/0001/results.jsonl").read_text().splitlines()[-1])
    assert state["state"] == "INCORRECT"
    assert "missing metrics" in state["reason"]


def test_stale_metrics_are_not_reused(tmp_path):
    spec_path, metrics = write_exp(tmp_path, ["true"])
    metrics.write_text('{"ok": true}')
    r = run_runner(spec_path)
    assert r.returncode == 1
    assert "missing metrics" in r.stdout


def test_stage_reported_failure_stops_chain(tmp_path):
    mfile = tmp_path / "experiments" / "0001" / "metrics.json"
    spec_path, _ = write_exp(
        tmp_path,
        [sys.executable, "-c", f"import json; json.dump({{'ok': False}}, open({str(mfile)!r}, 'w'))"],
    )
    r = run_runner(spec_path)
    assert r.returncode == 1
    assert "stage metrics report failure" in r.stdout


def test_zero_exit_with_metrics_passes_build(tmp_path):
    mfile = tmp_path / "experiments" / "0001" / "metrics.json"
    spec_path, metrics = write_exp(
        tmp_path,
        ["python3", "-c", f"import json; json.dump({{'normalized_rms_err': 0.0}}, open({str(mfile)!r}, 'w'))"],
    )
    r = run_runner(spec_path)
    assert r.returncode == 0, r.stderr
    lines = (tmp_path / "experiments/0001/results.jsonl").read_text().splitlines()
    states = [json.loads(x)["state"] for x in lines if '"stage": "state"' in x or json.loads(x).get("stage") == "state"]
    assert "CORRECT" in states


def test_timeout_classifies_timeout(tmp_path):
    spec_path, metrics = write_exp(
        tmp_path,
        ["python3", "-c", "import time; time.sleep(5)"],
        timeout_s=1,
    )
    r = run_runner(spec_path)
    assert r.returncode == 1
    state = json.loads((tmp_path / "experiments/0001/results.jsonl").read_text().splitlines()[-1])
    assert state["state"] == "TIMEOUT"


def test_stale_binary_blocks(tmp_path):
    spec_path, metrics = write_exp(tmp_path, ["true"], metrics_required=False)
    # corrupt the binary after hashing
    (tmp_path / "experiments/0001/bin.so").write_bytes(b"tampered")
    r = run_runner(spec_path)
    assert r.returncode == 2
    assert "BLOCKED" in r.stderr or "binary hash drift" in r.stderr


def test_missing_fields_blocks(tmp_path):
    exp_dir = tmp_path / "experiments" / "0002"
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(json.dumps({"id": "0002"}))
    r = run_runner(exp_dir / "experiment.json")
    assert r.returncode == 2
    assert "missing required field" in r.stderr


def test_deliberately_wrong_output_classified_incorrect(tmp_path):
    """A candidate producing wrong numbers must be INCORRECT with no score."""
    mfile = tmp_path / "experiments" / "0001" / "metrics.json"
    spec_path, metrics = write_exp(
        tmp_path,
        ["python3", "-c", f"import json; json.dump({{'arithmetic_mode': 'fp16_same', 'normalized_rms_err': 0.5, 'max_abs_err': 9.0, 'cosine_similarity': 0.5, 'exactness_failures': 3}}, open({str(mfile)!r}, 'w'))"],
    )
    r = run_runner(spec_path)
    assert r.returncode == 0  # stages ran; classification happens in compare

    # frozen policy with dummy measured values so compare can run
    policy = json.load(open(POLICY))
    fill = {
        "float_gates_same_arithmetic": {"final_gates": {
            "normalized_rms_err": 0.002, "max_abs_err": 0.02, "cosine_similarity_min": 0.99999,
        }},
    }

    def deepfill(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deepfill(dst[k], v)
            else:
                dst[k] = v

    deepfill(policy, fill)
    # fill every remaining null with a dummy non-null so freeze passes
    def denull(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if v is None:
                    o[k] = 0
                else:
                    denull(v)
    denull(policy)
    policy["status"] = "FROZEN"
    policy["performance_gates"]["primary_workload"] = "1024_prompt_256_decode"
    pol_path = tmp_path / "policy.frozen.json"
    pol_path.write_text(json.dumps(policy))
    spec = json.loads(spec_path.read_text())
    spec["identities"]["policy_sha256"] = hashlib.sha256(pol_path.read_bytes()).hexdigest()
    spec_path.write_text(json.dumps(spec))

    # baseline + candidate result files
    base = tmp_path / "baseline"
    base.mkdir()
    cand = tmp_path / "experiments/0001"
    r2 = subprocess.run(
        [sys.executable, COMPARE, "--baseline", str(base), "--candidate", str(cand),
         "--policy", str(pol_path)],
        capture_output=True, text=True, timeout=120,
    )
    out = json.load(open(cand / "comparison.json"))
    assert out["verdict"] == "INCORRECT", out
    assert out["correctness"], "correctness failures must be listed"
    assert "performance" not in out or not out["performance"].get("primary")
