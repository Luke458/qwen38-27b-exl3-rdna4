#!/usr/bin/env bash
# Build the ROCm fork's kernel extension inside the rdna4 vLLM image (its torch ABI).
#   build_ext_in_image.sh <out-dir>
# Prepares a fresh copy of the pinned fork with the experimental profile patches
# (tools/prepare_backend.py; cloned from upstream, or from vendor/rocm_exl3 when that checkout
# exists), applies the plugin kernel patches, and builds. vendor/ is never modified.
# The image ships ROCm 7.2.3; the fork's >=7.2.4 guard is skipped deliberately.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(realpath -m "$1")
WORK=$(mktemp -d "${TMPDIR:-/tmp}/exl3build.XXXXXX")
mkdir -p "$OUT"
SRC_ARGS=()
[ -d "$ROOT/vendor/rocm_exl3/.git" ] && SRC_ARGS=(--source "$ROOT/vendor/rocm_exl3")
python3 "$ROOT/tools/prepare_backend.py" --profile experimental --vendor "$WORK/src" "${SRC_ARGS[@]}"
# plugin kernel patches (vllm_plugin/patches/exl3-fork)
: > "$OUT/patches.txt"
for p in "$ROOT"/vllm_plugin/patches/exl3-fork/*.patch; do
  [ -e "$p" ] || continue
  (cd "$WORK/src" && git apply "$p") && echo "applied $(basename "$p") $(sha256sum "$p" | cut -c1-16)" \
    | tee -a "$OUT/patches.txt"
done
(cd "$WORK/src" && find exllamav3/exllamav3_ext setup.py -type f | sort | xargs sha256sum | sha256sum) \
  | tee "$OUT/source_tree.sha256"
# source tree of the tested build (2026-09-25); a mismatch means the fork pin or a patch changed
TESTED=386e31a455264406c27f16db332b30d62f02d66bc57457a4f22893d29f2c78a7
grep -q "^$TESTED " "$OUT/source_tree.sha256" && echo "source tree matches the tested build" \
  || echo "WARNING: source tree differs from the tested build ($TESTED)"
podman run --rm --security-opt label=disable \
  -v "$WORK":/b -v "$OUT":/out -w /b/src \
  -e EXL3_BACKEND=rocm -e PYTORCH_ROCM_ARCH=gfx1201 -e EXL3_SKIP_ROCM_VERSION_CHECK=1 \
  -e MAX_JOBS="${MAX_JOBS:-16}" \
  --entrypoint python3 docker.io/capicua25x/vllm-rocm-rdna4:0.28.0-rdna4 setup.py build_ext -b /out
sha256sum "$OUT"/exllamav3_ext*.so | tee "$OUT/binary.sha256"
rm -rf "$WORK"
