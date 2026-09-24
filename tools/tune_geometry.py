"""Bounded, serialized correctness-first geometry experiment loop.

Three configurations, immutable isolated binary, alternating baseline/candidate
layer trials, raw logs and append-only records. Layer wins are NOT promotion.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary-dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    out, binary_dir = args.out.resolve(), args.binary_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = next(binary_dir.glob('exllamav3_ext*.so'))
    binary_hash = sha(binary)
    env = dict(os.environ, PYTHONPATH=f'{binary_dir}:{ROOT / "vendor/rocm_exl3"}',
               EXL3_ROCM_GFX1201_INT8='2', EXL3_ROCM_GFX1201_INT8_TRACE='0')
    manifest = {'binary': str(binary), 'binary_sha256': binary_hash,
                'policy_sha256': sha(ROOT/'configs/acceptance_policy.frozen.json'),
                'workloads_sha256': sha(ROOT/'configs/workloads.json'),
                'variants': [4, 8, 16], 'max_gpu_seconds': 4*3600,
                'purpose': 'microbenchmark screening only; no model promotion'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    start = time.monotonic()
    def run(command, prefix, overrides=None, timeout=180):
        if sha(binary) != binary_hash:
            raise RuntimeError('candidate binary drift')
        before = time.monotonic()
        e = env | (overrides or {})
        with (out/f'{prefix}.log').open('w') as log:
            result = subprocess.run([str(x) for x in command], cwd=ROOT, env=e,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        record = {'stage': prefix, 'command': [str(x) for x in command],
                  'exit_status': result.returncode, 'duration_s': time.monotonic()-before,
                  'mode': e['EXL3_ROCM_GFX1201_INT8'], 'warps': e.get('EXL3_ROCM_GFX1201_WARPS'),
                  'binary_sha256': binary_hash}
        with (out/'results.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n'); f.flush(); os.fsync(f.fileno())
        if result.returncode:
            raise RuntimeError(f'{prefix} failed; see {out / (prefix + ".log")}')
    summaries = {}
    with open('/tmp/exl3-gfx1201-gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run([PYTHON, 'tools/preflight.py', '--out', out/'preflight.json'], 'preflight')
        for warps in (4, 8, 16):
            env['EXL3_ROCM_GFX1201_WARPS'] = str(warps)
            run([PYTHON, '-m', 'pytest', 'tests/test_gpu_int8.py', 'tests/test_int8_contract.py', '-q'],
                f'w{warps}-correctness', timeout=240)
            print(f'warps={warps}: correctness passed', flush=True)
            times = {'baseline': {}, 'candidate': {}}
            for pair in range(5):
                for side in (('baseline','candidate') if pair % 2 == 0 else ('candidate','baseline')):
                    prefix = f'w{warps}-pair{pair}-{side}'
                    path = out/f'{prefix}.json'
                    run([PYTHON, 'bench/bench_layer.py', '--out', path, '--rotating', '8',
                         '--iters', '100', '--warmup', '25', '--shapes',
                         '17408x5120x3,5120x17408x3'], prefix,
                        {'EXL3_ROCM_GFX1201_INT8': '0' if side == 'baseline' else '2'})
                    data = json.loads(path.read_text())
                    for shape, value in data['shapes'].items():
                        times[side].setdefault(shape, []).append(value['median_ms'])
            summaries[str(warps)] = {shape: {
                'baseline_ms': times['baseline'][shape], 'candidate_ms': times['candidate'][shape],
                'median_speedup': statistics.median([b/c for b,c in zip(times['baseline'][shape],times['candidate'][shape])])}
                for shape in times['baseline']}
            print(json.dumps({'warps': warps, 'results': summaries[str(warps)]}), flush=True)
            (out/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
            if time.monotonic()-start >= manifest['max_gpu_seconds']:
                break
    print('Geometry screening finished; no promotion performed.', flush=True)


if __name__ == '__main__':
    main()
