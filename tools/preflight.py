#!/usr/bin/env python3
"""Read telemetry and run checked GPU arithmetic; never mutate device settings.

Run under /tmp/exl3-gfx1201-gpu.lock with the project Python environment.
Unavailable telemetry is explicit rather than silently treated as a zero.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--strict-temp', type=float, default=90.0)
    args = ap.parse_args()
    rec = {'ts': time.time(), 'ok': True, 'reasons': [], 'telemetry': {}}
    rec['rocm_version'] = read('/opt/rocm/.info/version')
    for device in glob.glob('/sys/class/drm/card[0-9]*/device'):
        if read(device + '/vendor') != '0x1002':
            continue
        info = {}
        for name in ('mem_info_vram_used', 'mem_info_vram_total', 'gpu_busy_percent',
                     'pp_dpm_sclk', 'pp_dpm_mclk'):
            info[name] = read(device + '/' + name)
        temps = {}
        for path in glob.glob(device + '/hwmon/hwmon*/temp*_input'):
            raw = read(path)
            label = read(path.replace('_input', '_label')) or Path(path).stem
            if raw is not None:
                temps[label] = int(raw) / 1000.0
        info['temperatures_c'] = temps
        edge = next((v for k, v in temps.items() if k.lower() == 'edge'), None)
        if edge is not None and edge > args.strict_temp:
            rec['reasons'].append(f'edge temperature {edge} exceeds {args.strict_temp}')
        rec['telemetry'][device] = info
    try:
        p = subprocess.run(['fuser', '/dev/kfd'], capture_output=True, text=True, timeout=5)
        rec['kfd_owners'] = p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        rec['kfd_owners'] = None
        rec['kfd_owners_unavailable'] = str(exc)
    try:
        import torch
        rec['torch'] = torch.__version__
        rec['hip'] = torch.version.hip
        prop = torch.cuda.get_device_properties(0)
        rec['arch'] = prop.gcnArchName
        rec['gpu'] = prop.name
        rec['vram_total_bytes'] = prop.total_memory
        if prop.gcnArchName.split(':')[0] != 'gfx1201':
            rec['reasons'].append('expected gfx1201')
        x = torch.arange(32, dtype=torch.int64, device='cuda')
        assert int((x * 2).sum().item()) == 992
        torch.cuda.synchronize()
        rec['probe'] = 'PASS'
        rec['free_total_bytes'] = list(torch.cuda.mem_get_info())
    except Exception as exc:
        rec['probe'] = 'FAIL'
        rec['reasons'].append(repr(exc))
    rec['ok'] = not rec['reasons']
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(rec, indent=2) + '\n')
        os.replace(tmp, path)
    print(json.dumps(rec, indent=2))
    return 0 if rec['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
