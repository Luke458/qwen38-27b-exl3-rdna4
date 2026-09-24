"""CPU-only real-checkpoint calibration, independent of candidate outputs.

Checks predeclared int8 approximation limits on one 128-column output block
per eligible shape. Full model quality is a separate mandatory gate.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference'))
import exl3_oracle as O


def main():
    manifest = json.loads((ROOT / 'configs/shape_manifest.json').read_text())
    model = Path(manifest['model'])
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    groups = {}
    for m in manifest['modules']:
        if m['eligible_v1']:
            groups.setdefault((m['K_in'], m['N']), m)
    measurements = []
    for shape, m in groups.items():
        def get(suffix):
            key = m['name'] + '.' + suffix
            with safe_open(model / index[key], framework='np') as f:
                if suffix == 'trellis':
                    return f.get_slice(key)[:, :8, :].astype(np.uint16)
                return f.get_tensor(key).astype(np.float64)
        tr, suh, svh = get('trellis'), get('suh'), get('svh')[:128]
        states = O.trellis_states(tr, 3)
        W = O.decode_trellis(tr, 3)
        for seed in (17, 91):
            rng = np.random.default_rng(seed)
            a = (rng.standard_normal(shape[0]) * 0.2).astype(np.float16)
            ah = O.had_in(a, suh, faithful=True)
            q = max(abs(ah).max(), 1e-8) / 127
            fp = O.had_out(O.to_fp16_vec(ah @ W), svh, faithful=True)
            approx = O.gemv_int8_reference(ah, states, q, svh)
            scale = max(float(np.sqrt(np.mean(fp ** 2))), 1e-12)
            d = approx - fp
            stats = {'shape': list(shape), 'seed': seed, 'tensor': m['name'],
                     'normalized_rms_err': float(np.sqrt(np.mean(d*d)) / scale),
                     'normalized_max_abs_err': float(np.max(abs(d)) / scale),
                     'cosine_similarity': float(approx @ fp / (np.linalg.norm(approx)*np.linalg.norm(fp)))}
            measurements.append(stats)
            print(stats, flush=True)
    limits = {'normalized_rms_err': 0.02, 'normalized_max_abs_err': 0.1,
              'cosine_similarity_min': 0.9998}
    ok = all(s['normalized_rms_err'] <= limits['normalized_rms_err'] and
             s['normalized_max_abs_err'] <= limits['normalized_max_abs_err'] and
             s['cosine_similarity'] >= limits['cosine_similarity_min'] for s in measurements)
    path = ROOT / 'artifacts/baseline/int8-cpu-calibration.json'
    path.write_text(json.dumps({'ok': ok, 'limits': limits, 'measurements': measurements,
        'oracle_sha256': hashlib.sha256((ROOT/'reference/exl3_oracle.py').read_bytes()).hexdigest(),
        'scope': 'real weights, synthetic activations, first 128 output columns per shape'}, indent=2)+'\n')
    if not ok:
        raise SystemExit('Predeclared numerical budget failed CPU calibration; do not freeze')


if __name__ == '__main__':
    main()
