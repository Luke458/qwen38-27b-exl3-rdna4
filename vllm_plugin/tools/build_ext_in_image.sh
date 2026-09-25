#!/usr/bin/env bash
# Build the exllamav3 ROCm fork's extension inside the rdna4 vLLM image (its torch ABI),
# from a copy of vendor/rocm_exl3 so the vendor tree and its build products are untouched.
#   build_ext_in_image.sh <out-dir>
# The image ships ROCm 7.2.3; the fork's >=7.2.4 guard is skipped deliberately. Validate
# every build with vllm_plugin/tests/xcheck_ext.py against the host control binary before use.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(realpath -m "$1")
WORK=$(mktemp -d "${TMPDIR:-/tmp}/exl3build.XXXXXX")
mkdir -p "$OUT"
rsync -a --exclude build --exclude '*.egg-info' --exclude '*.so' --exclude __pycache__ --exclude .git \
  "$ROOT/vendor/rocm_exl3/" "$WORK/src/"
(cd "$WORK/src" && find exllamav3/exllamav3_ext setup.py -type f | sort | xargs sha256sum | sha256sum) \
  | tee "$OUT/source_tree.sha256"
podman run --rm --security-opt label=disable \
  -v "$WORK":/b -v "$OUT":/out -w /b/src \
  -e EXL3_BACKEND=rocm -e PYTORCH_ROCM_ARCH=gfx1201 -e EXL3_SKIP_ROCM_VERSION_CHECK=1 \
  -e MAX_JOBS="${MAX_JOBS:-16}" \
  --entrypoint python3 docker.io/capicua25x/vllm-rocm-rdna4:0.28.0-rdna4 setup.py build_ext -b /out
sha256sum "$OUT"/exllamav3_ext*.so | tee "$OUT/binary.sha256"
rm -rf "$WORK"
