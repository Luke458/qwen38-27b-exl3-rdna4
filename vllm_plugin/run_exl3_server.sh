#!/usr/bin/env bash
# Serve an EXL3 checkpoint with the rdna4 vLLM image + the exl3rocm plugin (rootless podman).
#   EXL3_EXT_DIR=<dir with container-built exllamav3_ext*.so> \
#   run_exl3_server.sh <model-dir> <served-name> [extra vllm serve args...]
# Binds 127.0.0.1:8000 only. Text-only (vision tower skipped), fp16 activations.
# EXTRA_MOUNTS="-v host:ctr:ro ..." adds mounts (e.g. patched vLLM source files).
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
MODEL_DIR=$(realpath "$1"); NAME=$2; shift 2
: "${EXL3_EXT_DIR:?set EXL3_EXT_DIR to the directory holding the container-built exllamav3_ext .so}"
IMAGE=docker.io/capicua25x/vllm-rocm-rdna4:0.28.0-rdna4
mkdir -p "$HOME/.cache/vllm-rdna4-exl3"
ARGS=$(printf ' %q' "$@")
exec podman run --rm --name vllm-exl3 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$MODEL_DIR":/model:ro \
  -v "$HERE":/plugin:ro \
  -v "$(realpath "$EXL3_EXT_DIR")":/exl3ext:ro \
  -v "$HOME/.cache/vllm-rdna4-exl3":/root/.cache/vllm \
  ${EXTRA_MOUNTS:-} \
  -e PYTHONPATH=/exl3ext \
  --entrypoint bash "$IMAGE" -c "
    cp -r /plugin /tmp/plugin && pip install --no-deps --no-build-isolation -q /tmp/plugin >/dev/null &&
    exec vllm serve /model --served-model-name '$NAME' --host 0.0.0.0 --port 8000 \
      --attention-backend TRITON_ATTN --dtype float16 \
      --limit-mm-per-prompt '{\"image\":0,\"video\":0}' $ARGS"
