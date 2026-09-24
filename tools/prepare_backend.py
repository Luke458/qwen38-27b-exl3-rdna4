#!/usr/bin/env python3
"""Prepare the pinned ROCm fork without installing packages or running builds.

Default baseline uses the frozen compatibility patches. Experimental applies a
separate full patch from the same upstream commit; it never silently changes
the default build. Existing vendor checkouts are verified, never reset.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PIN = "311ff5497237a37bea18cf24aa18bc0c573d1d03"
UPSTREAM = "https://github.com/CarouselAether/rocm_exl3.git"
PATCHES = {
    "baseline": (
        ("0000-smem-budget-clamp.patch", "d0100bff3f76aaf52a274f9b64ca4b177a59fd75b17b9c28d8fcc2a95ab85181"),
        ("0001-int8-port-gfx1201-compat.patch", "440239ea93feb866b2554d7fe93d4ec4487de01168fa2547b42276572253a087"),
    ),
    "experimental": (
        ("experimental-full.patch", "6daa526d22f3cef2c2ccd91b24bf30e0991aaf309e9381270ebe174c2f9a2e3d"),
    ),
}
Q = "exllamav3/exllamav3_ext/rocm/quant/"
FILES = {
    Q + "exl3_gemm_rdna.hip": "0d42d33f7f67ccad350224f310c4a762c6db983e417f29530760a54b85fa1a70",
    Q + "exl3_gemv_int8_rdna.hip": "b17c6884d7257ba6f1ab3ae686d7b5d0ebc35e1a6db888e3f3ef00d3df5be081",
    Q + "exl3_moe_rdna.hip": "af9fa4ff7ec8f2ce9e435b472559e68ab749658a146a4d6d0ba1693ac7fc2fdd",
    "exllamav3/rocm_py/__init__.py": "84bd5bf1c5f889033d4838a531e019959d109fcd126584711444cdfe3c63e52d",
}
EXPERIMENTAL = {
    Q + "exl3_gemv_kernel_rdna.hip.h": "c29b0eff29d64e9b02b0814780a9522f541de869e75920565776a7d321cd3909",
    Q + "exl3_gemv_rdna.hip": "02e5729d2ad8ec6f612790c387abd2d8e226017654be02d2ccb6821e48c8c9dc",
}


def run(*args: str, cwd: Path | None = None) -> str:
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"{' '.join(args)}: {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout.strip()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_files(profile: str) -> dict[str, str]:
    return FILES | (EXPERIMENTAL if profile == "experimental" else {})


def verify(checkout: Path, profile: str) -> None:
    if run("git", "rev-parse", "HEAD", cwd=checkout) != PIN:
        raise RuntimeError(f"{checkout}: checkout is not pinned at {PIN}")
    wanted = expected_files(profile)
    changed = set(run("git", "diff", "--name-only", "HEAD", cwd=checkout).splitlines())
    if changed != set(wanted):
        raise RuntimeError(f"{checkout}: tracked source changes differ from {profile}: {sorted(changed ^ set(wanted))}")
    untracked = run("git", "ls-files", "--others", "--exclude-standard", cwd=checkout)
    if untracked:
        raise RuntimeError(f"{checkout}: untracked files present; preserving checkout")
    for name, expected in wanted.items():
        path = checkout / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"{path}: source hash differs from {profile}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=tuple(PATCHES), default="baseline")
    ap.add_argument("--vendor", type=Path, default=ROOT / "vendor" / "rocm_exl3")
    ap.add_argument("--source", default=UPSTREAM, help="clone source; may be a local Git checkout")
    args = ap.parse_args()
    vendor = args.vendor.resolve()
    try:
        patch_paths = []
        for name, expected in PATCHES[args.profile]:
            path = ROOT / "patches" / name
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"patch missing or changed: {path}")
            patch_paths.append(path)
        if vendor.exists():
            verify(vendor, args.profile)
            print(f"{args.profile} backend already prepared: {vendor}")
            return 0
        vendor.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".exl3-prepare-", dir=vendor.parent) as tmp:
            checkout = Path(tmp) / "checkout"
            run("git", "clone", "--quiet", args.source, str(checkout))
            run("git", "checkout", "--quiet", "--detach", PIN, cwd=checkout)
            run("git", "apply", "--check", *(str(p) for p in patch_paths), cwd=checkout)
            run("git", "apply", *(str(p) for p in patch_paths), cwd=checkout)
            verify(checkout, args.profile)
            if vendor.exists():
                raise RuntimeError(f"destination appeared during preparation: {vendor}")
            checkout.rename(vendor)
        print(f"prepared {args.profile} backend at {vendor} ({PIN})")
        return 0
    except (OSError, RuntimeError) as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
