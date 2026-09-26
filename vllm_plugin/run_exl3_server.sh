#!/usr/bin/env bash
# Serve an EXL3 checkpoint with the rdna4 vLLM image + the exl3rocm plugin (rootless podman).
#   run_exl3_server.sh <model-dir> <served-name> [extra vllm serve args...]
# EXL3_EXT_DIR: the container-built exllamav3_ext*.so (default ~/models/exl3ext).
# Binds 127.0.0.1:8000 only. fp16 activations. Vision on unless EXL3_TEXT_ONLY=1.
# Defaults (measured, experiments/0019): GPU_MAX_HW_QUEUES=1 (multi-queue cost ~18 ms/step),
# vLLM V2 model runner (EXL3_V2_RUNNER=0 to disable), GDN metadata patch (EXL3_GDN_SHARE=0).
# EXTRA_MOUNTS="-v host:ctr:ro ..." adds mounts (e.g. patched vLLM source files).
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
MODEL_DIR=$(realpath "$1"); NAME=$2; shift 2
# container-built kernel extension (vllm_plugin/tools/build_ext_in_image.sh), next to the models by default
EXL3_EXT_DIR=${EXL3_EXT_DIR:-$HOME/models/exl3ext}
[ -d "$EXL3_EXT_DIR" ] || { echo "kernel extension not found in $EXL3_EXT_DIR (build it with" \
  "vllm_plugin/tools/build_ext_in_image.sh $EXL3_EXT_DIR, or set EXL3_EXT_DIR)" >&2; exit 1; }
IMAGE=docker.io/capicua25x/vllm-rocm-rdna4:0.28.0-rdna4
# torch.compile's cache key does not cover the multimodal settings: a graph compiled text-only fails when
# loaded with vision on ("'NoneType' object has no attribute 'size'" in qwen3_next forward), so text-only
# runs keep their own cache (experiments/0024)
CACHE_DIR="$HOME/.cache/vllm-rdna4-exl3"
[ "${EXL3_TEXT_ONLY:-0}" = "1" ] && CACHE_DIR="$HOME/.cache/vllm-rdna4-exl3-textonly"
mkdir -p "$CACHE_DIR"
ARGS=$(printf ' %q' "$@")
# forward every EXL3_* tuning/debug variable into the container
ENV_FWD=""
for v in $(compgen -e | grep '^EXL3_' | grep -v '^EXL3_EXT_DIR$'); do ENV_FWD="$ENV_FWD -e $v"; done
# EXL3_TEXT_ONLY=1 skips the vision tower (saves ~0.6 GiB for KV cache)
# images are capped at 1 MP by default: an uncapped 2560x1920 image pushed peak VRAM to within
# 104 MiB of the 16 GB card (experiments/0022); later --mm-processor-kwargs arguments override it
MM_ARGS="--mm-processor-kwargs '{\"max_pixels\":${EXL3_MAX_PIXELS:-1048576}}'"
if [ "${EXL3_TEXT_ONLY:-0}" = "1" ]; then MM_ARGS="--limit-mm-per-prompt '{\"image\":0,\"video\":0}'"; fi
# vLLM 0.28 GDN metadata patch: one build per step instead of one per GDN layer (+20% MTP)
GDN_MOUNT=""
if [ "${EXL3_GDN_SHARE:-1}" != "0" ]; then
  GDN_MOUNT="-v $HERE/patches/vllm-0.28.0-rdna4/gdn_attn.py:/build/vllm/vllm/v1/attention/backends/gdn_attn.py:ro"
fi
exec podman run --rm --name vllm-exl3 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$MODEL_DIR":/model:ro \
  -v "$HERE":/plugin:ro \
  -v "$(realpath "$EXL3_EXT_DIR")":/exl3ext:ro \
  -v "$CACHE_DIR":/root/.cache/vllm \
  $GDN_MOUNT ${EXTRA_MOUNTS:-} $ENV_FWD \
  -e PYTHONPATH=/exl3ext -e GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-1} \
  -e VLLM_USE_V2_MODEL_RUNNER=${EXL3_V2_RUNNER:-1} \
  -e EXL3_GEMV_FUSED_HAD=${EXL3_GEMV_FUSED_HAD:-1} \
  --entrypoint bash "$IMAGE" -c "
    cp -r /plugin /tmp/plugin && pip install --no-deps --no-build-isolation -q /tmp/plugin >/dev/null &&
    exec vllm serve /model --served-model-name '$NAME' --host 0.0.0.0 --port 8000 \
      --attention-backend TRITON_ATTN --dtype float16 $MM_ARGS $ARGS"
