"""Model quality gates reject different histories and missing evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_logits_identity_and_history(tmp_path):
    baseline, candidate = tmp_path/'baseline', tmp_path/'candidate'
    baseline.mkdir(); candidate.mkdir()
    logits = np.array([[1, 0, -1], [-1, 0, 1]], dtype=np.float16)
    history = np.array([0, 0, 2])
    for d in (baseline, candidate):
        np.savez(d/'sample.npz', logits=logits, input_ids=history)
    command = [sys.executable, str(ROOT/'tools/compare_logits.py'), '--baseline', str(baseline),
               '--candidate', str(candidate), '--out', str(tmp_path/'result.json'),
               '--budget-delta-ppl', '0']
    assert subprocess.run(command, capture_output=True).returncode == 0
    result = json.loads((tmp_path/'result.json').read_text())
    assert result['logit_max_abs'] == result['delta_ppl'] == result['kl_mean'] == 0
    np.savez(candidate/'sample.npz', logits=logits, input_ids=np.array([1, 0, 2]))
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not json.loads((tmp_path/'result.json').read_text())['ok']


def test_empty_logits_directory_cannot_pass(tmp_path):
    d = tmp_path/'empty'; d.mkdir()
    p = subprocess.run([sys.executable, str(ROOT/'tools/compare_logits.py'), '--baseline', str(d),
                        '--candidate', str(d), '--out', str(tmp_path/'result.json')], capture_output=True)
    assert p.returncode != 0
