#!/usr/bin/env python3
"""Run project Python with the verified baseline or an explicit candidate.

  .venv/bin/python tools/run_backend.py bench/bench_generate.py --out result.json
  .venv/bin/python tools/run_backend.py --candidate experiments/0005-geometry/binary -- tests.py

Baseline is always the default. Candidate selection is experimental and does
not imply promotion. The selected extension must match its recorded hash.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', type=Path)
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.candidate:
        directory = args.candidate.resolve()
        manifest = json.loads((directory/'manifest.json').read_text())
        binary = Path(manifest['candidate_binary'])
        expected = manifest['candidate_binary_sha256']
        mode = '2'
    else:
        manifest = json.loads((ROOT/'artifacts/baseline/records.json').read_text())
        binary = ROOT/manifest['binary']
        expected = manifest['binary_sha256']
        directory, mode = binary.parent, '0'
    actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('Selected binary hash differs from recorded artifact')
    env = dict(os.environ,
               PYTHONPATH=f'{directory}:{ROOT / "vendor/rocm_exl3"}',
               EXL3_ROCM_GFX1201_INT8=mode)
    if args.inspect:
        print(json.dumps({'binary': str(binary), 'sha256': actual,
                          'mode': mode, 'pythonpath': env['PYTHONPATH']}, indent=2))
        return
    cmd = args.command
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        ap.error('provide a Python script or -m module after --')
    os.chdir(ROOT)
    os.execve(str(ROOT/'.venv/bin/python'), [str(ROOT/'.venv/bin/python'), *cmd], env)


if __name__ == '__main__':
    main()
