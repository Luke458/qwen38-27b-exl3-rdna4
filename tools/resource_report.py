#!/usr/bin/env python3
"""Per-kernel resource report from AMDGPU code-object metadata.

Extracts the embedded amdhsa.kernels YAML from a HIP-built .so/.o and reports,
per kernel: vgpr_count, sgpr_count, group_segment_fixed_size (LDS bytes),
private_segment_fixed_size (scratch bytes), kernarg_segment_size,
max_flat_workgroup_size.  This satisfies the candidate record requirement
"compiler VGPR/SGPR/LDS/scratch/spill metadata" (IMPLEMENTATION_PLAN P2) --
spill shows up as private_segment_fixed_size > 0.

Usage:
  python tools/resource_report.py <binary.so|.o> [--out report.json] [--match g1201]

ISA disassembly recipe (for shortlisted candidates):
  hipcc --genco --offload-arch=gfx1201 <tu>.hip -o dev.co
  /opt/rocm/lib/llvm/bin/llvm-objdump -d dev.co > isa.txt
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys


def _msgpack_decode(buf: bytes, pos: int):
    """Minimal MessagePack decoder (code-object v5 metadata subset).
    Note: msgpack multi-byte numbers are BIG-endian."""
    b = buf[pos]
    pos += 1
    if b <= 0x7F:                                   # positive fixint
        return b, pos
    if b >= 0xE0:                                   # negative fixint
        return b - 0x100, pos
    if 0x80 <= b <= 0x8F:                           # fixmap
        n = b & 0x0F
        out = {}
        for _ in range(n):
            k, pos = _msgpack_decode(buf, pos)
            v, pos = _msgpack_decode(buf, pos)
            out[k] = v
        return out, pos
    if 0x90 <= b <= 0x9F:                           # fixarray
        n = b & 0x0F
        out = []
        for _ in range(n):
            v, pos = _msgpack_decode(buf, pos)
            out.append(v)
        return out, pos
    if 0xA0 <= b <= 0xBF:                           # fixstr
        n = b & 0x1F
        s = buf[pos:pos + n].decode("utf8", "replace")
        return s, pos + n
    if b in (0xC0,):                                # nil
        return None, pos
    if b in (0xC2, 0xC3):                           # bool
        return b == 0xC3, pos
    if b in (0xC4, 0xC5, 0xC6):                     # bin8/16/32
        w = {0xC4: 1, 0xC5: 2, 0xC6: 4}[b]
        n = int.from_bytes(buf[pos:pos + w], "big")
        pos += w
        return buf[pos:pos + n], pos + n
    if b in (0xC7, 0xC8, 0xC9):                     # ext (skip)
        w = {0xC7: 1, 0xC8: 2, 0xC9: 4}[b]
        n = int.from_bytes(buf[pos:pos + w], "big")
        pos += w + 1 + n
        return None, pos
    if b in (0xCA,):                                # float32
        v = struct.unpack_from(">f", buf, pos)[0]
        return v, pos + 4
    if b in (0xCB,):                                # float64
        v = struct.unpack_from(">d", buf, pos)[0]
        return v, pos + 8
    if b in (0xCC, 0xCD, 0xCE, 0xCF):               # uint8..64
        w = 1 << (b - 0xCC)
        v = int.from_bytes(buf[pos:pos + w], "big")
        return v, pos + w
    if b in (0xD0, 0xD1, 0xD2, 0xD3):               # int8..64
        w = 1 << (b - 0xD0)
        v = int.from_bytes(buf[pos:pos + w], "big", signed=True)
        return v, pos + w
    if b in (0xD9, 0xDA, 0xDB):                     # str8/16/32
        w = 1 << (b - 0xD9)
        n = int.from_bytes(buf[pos:pos + w], "big")
        pos += w
        s = buf[pos:pos + n].decode("utf8", "replace")
        return s, pos + n
    if b in (0xDC, 0xDD):                           # array16/32
        w = 2 if b == 0xDC else 4
        n = int.from_bytes(buf[pos:pos + w], "big")
        pos += w
        out = []
        for _ in range(n):
            v, pos = _msgpack_decode(buf, pos)
            out.append(v)
        return out, pos
    if b in (0xDE, 0xDF):                           # map16/32
        w = 2 if b == 0xDE else 4
        n = int.from_bytes(buf[pos:pos + w], "big")
        pos += w
        out = {}
        for _ in range(n):
            k, pos = _msgpack_decode(buf, pos)
            v, pos = _msgpack_decode(buf, pos)
            out[k] = v
        return out, pos
    raise ValueError(f"unsupported msgpack byte 0x{b:02x}")


def find_metadata_blocks(data: bytes) -> list[dict]:
    """Locate every AMDGPU code-object metadata note and parse it.

    Code object v5 embeds the metadata as MessagePack after an ELF note named
    "AMDGPU"; older objects carry YAML. Both are handled.
    """
    blocks = []
    # MessagePack (code object v5)
    for m in re.finditer(rb"AMDGPU\x00\x00", data):
        try:
            doc, _ = _msgpack_decode(data, m.end())
        except Exception:
            continue
        if isinstance(doc, dict) and "amdhsa.kernels" in doc:
            blocks.append(doc)
    # YAML (older code objects)
    for m in re.finditer(rb"amdhsa\.kernels:", data):
        start = data.rfind(b"amdhsa.version:", max(0, m.start() - 4096), m.start())
        if start < 0:
            start = m.start()
        end = data.find(b"\x00", m.end())
        stop = end if end > 0 else m.end() + 65536
        text = data[start:stop].decode("utf8", errors="replace")
        try:
            import yaml
            doc = yaml.safe_load(text)
            if isinstance(doc, dict) and "amdhsa.kernels" in doc:
                blocks.append(doc)
        except Exception:
            pass
    return blocks


def summarize(blocks: list[dict], match: str | None) -> list[dict]:
    out = []
    seen = set()
    for b in blocks:
        for k in b.get("amdhsa.kernels", []):
            name = k.get("name") or k.get(".name")
            if not name or name in seen:
                continue
            if match and match not in name:
                continue
            seen.add(name)

            def g(*keys):
                for key in keys:
                    if key in k:
                        return k[key]
                return None

            out.append({
                "kernel": name,
                "vgpr_count": g("vgpr_count", ".vgpr_count"),
                "sgpr_count": g("sgpr_count", ".sgpr_count"),
                "agpr_count": g("agpr_count", ".agpr_count"),
                "lds_bytes": g("group_segment_fixed_size", ".group_segment_fixed_size"),
                "scratch_bytes": g("private_segment_fixed_size", ".private_segment_fixed_size"),
                "kernarg_bytes": g("kernarg_segment_size", ".kernarg_segment_size"),
                "max_flat_workgroup_size": g("max_flat_workgroup_size", ".max_flat_workgroup_size"),
                "wavefront_size": g("wavefront_size", ".wavefront_size"),
                "symbol": g("symbol", ".symbol"),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--match", default=None)
    args = ap.parse_args()

    data = open(args.binary, "rb").read()
    rows = summarize(find_metadata_blocks(data), args.match)
    if not rows:
        print("no amdhsa.kernels metadata found", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r["kernel"])
    rec = {"binary": args.binary, "kernels": rows}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
    for r in rows:
        spill = " SPILL" if r["scratch_bytes"] not in (0, "0", None) else ""
        print(f"{r['kernel'][:64]:64s} vgpr={r['vgpr_count']} sgpr={r['sgpr_count']} "
              f"lds={r['lds_bytes']} scratch={r['scratch_bytes']}{spill}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
