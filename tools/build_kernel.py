"""Build one gfx1201 int8 HIP translation unit into an isolated extension.

Reuses the pinned objects from the existing full build without modifying them.
The candidate shared library is placed directly in --out so that
PYTHONPATH=<out> makes exllamav3.ext import it before the installed extension.

Usage:
  .venv/bin/python tools/build_kernel.py --out experiments/0002/binary
  PYTHONPATH=experiments/0002/binary .venv/bin/python -m pytest tests/test_gpu_int8.py -q

This deliberately does not install the extension or exercise the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import shutil


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/rocm_exl3"
SOURCE = Path("exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_int8_rdna.hip")
OLD_OBJECT = SOURCE.as_posix().replace("/", "_") + ".o"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_logged(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as stream:
        stream.write("command: " + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=VENDOR, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
        stream.write(f"\nexit_code: {result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"build command failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="New, empty candidate binary directory")
    parser.add_argument("--build-temp", type=Path,
                        help="Pinned full-build object directory (auto-detect if unique)")
    parser.add_argument("--rocm-path", type=Path, default=Path(os.environ.get("ROCM_PATH", "/opt/rocm")))
    args = parser.parse_args()

    import torch

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"candidate output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    temps = [args.build_temp.resolve()] if args.build_temp else sorted((VENDOR / "build").glob("temp.*"))
    if len(temps) != 1 or not temps[0].is_dir():
        parser.error(f"expected one pinned build/temp.* directory, found {temps}")
    temp = temps[0]
    objects = sorted(temp.glob("*.o"))
    old = temp / OLD_OBJECT
    if old not in objects or len(objects) < 100:
        parser.error(f"incomplete pinned object set: {len(objects)} objects; missing {old}")
    pinned = {obj.name: digest(obj) for obj in objects}

    source = VENDOR / SOURCE
    rocm_dir = VENDOR / "exllamav3/exllamav3_ext/rocm"
    ext_dir = VENDOR / "exllamav3/exllamav3_ext"
    torch_path = Path(torch.__file__).resolve().parent
    torch_includes = [torch_path / "include",
                      torch_path / "include/torch/csrc/api/include",
                      torch_path / "include/TH", torch_path / "include/THC"]
    includes = [rocm_dir / "cuda_shim", ext_dir, *torch_includes,
                args.rocm_path / "include", Path(sysconfig.get_path("include"))]

    # Mirrors setup.py HIPBuildExtension._build_hip for the pinned gfx1201
    # build. Full-build objects must have used these same defines. The object
    # hashes and complete commands are retained in manifest.json for audit.
    quiet = [] if os.environ.get("EXLLAMA_VERBOSE_BUILD") == "1" else [
        "-Wno-unused-command-line-argument", "-Wno-deprecated-declarations",
        "-Wno-unused-variable", "-Wno-unused-function", "-Wno-unused-value",
        "-Wno-missing-field-initializers", "-Wno-#pragma-messages",
        "-Wno-pass-failed", "-Wno-c++20-extensions",
    ]
    defines = ["-DUSE_ROCM=1", "-DEXL3_RDNA_SMEM_MAX=92160",
               "-D__HIP_PLATFORM_AMD__=1", "-DHIPBLAS_V2",
               "-DHIPBLAS_USE_HIP_HALF", "-DCUDA_HAS_FP16=1",
               "-D__HIP_NO_HALF_OPERATORS__=1", "-D__HIP_NO_HALF_CONVERSIONS__=1",
               "-DHIP_DISABLE_WARP_SYNC_BUILTINS=1",
               "-DTORCH_API_INCLUDE_EXTENSION_H", "-DTORCH_EXTENSION_NAME=exllamav3_ext"]
    include_args = [f"-I{path}" for path in includes]
    candidate_obj = out / OLD_OBJECT
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    candidate_so = out / f"exllamav3_ext{suffix}"
    compile_cmd = ["hipcc", "-c", str(source), "-o", str(candidate_obj),
                   "-fPIC", "-std=c++17", "-O3", "-Wno-register",
                   "-include", str(rocm_dir / "hip_compat.hip.h"), *quiet,
                   "-fgpu-rdc", "--offload-arch=gfx1201", *include_args, *defines]

    link_objects = [candidate_obj if obj == old else obj for obj in objects]
    lib_dirs = [torch_path / "lib", args.rocm_path / "lib"]
    if python_lib := sysconfig.get_config_var("LIBDIR"):
        lib_dirs.append(Path(python_lib))
    link_cmd = ["hipcc", "-shared", "-fgpu-rdc", "--hip-link", "-o", str(candidate_so),
                *(str(obj) for obj in link_objects), *(f"-L{path}" for path in lib_dirs),
                "-lc10", "-ltorch", "-ltorch_cpu", "-ltorch_hip", "-ltorch_python",
                "-lc10_hip", "-lamdhip64", "-lhipblas", "-lrocblas", "-lhiprand", "-fPIC"]

    manifest = {"source": str(source), "source_sha256": digest(source),
                "pinned_object_dir": str(temp), "pinned_objects_sha256": pinned,
                "replaced_object": str(old), "candidate_object": str(candidate_obj),
                "candidate_binary": str(candidate_so), "compile_command": compile_cmd,
                "link_command": link_cmd, "torch_version": torch.__version__}
    manifest_file = out / "manifest.json"
    shutil.copyfile(source, out / "candidate_source.hip")
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        run_logged(compile_cmd, out / "compile.log")
        if digest(source) != manifest["source_sha256"]:
            raise RuntimeError("source changed during candidate compile")
        drifted = [obj.name for obj in objects if digest(obj) != pinned[obj.name]]
        if drifted:
            raise RuntimeError(f"pinned objects changed before link: {drifted}")
        run_logged(link_cmd, out / "link.log")
        manifest["candidate_object_sha256"] = digest(candidate_obj)
        manifest["candidate_binary_sha256"] = digest(candidate_so)
        manifest["status"] = "built"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"binary": str(candidate_so), "sha256": manifest["candidate_binary_sha256"],
                      "PYTHONPATH": str(out)}, indent=2))


if __name__ == "__main__":
    main()
