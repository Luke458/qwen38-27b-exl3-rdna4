#!/usr/bin/env bash
# Launch the Capicua25x rdna4 vLLM image (rootless podman) on the local GPU.
#   run_server.sh <model-dir> <served-name> [extra vllm serve args...]
# Model dir is mounted read-only at /model. Binds 127.0.0.1:8000 only.
# EXTRA_MOUNTS="-v host:ctr:ro ..." adds mounts (e.g. patched vLLM source files).
set -euo pipefail
MODEL_DIR=$(realpath "$1"); NAME=$2; shift 2
IMAGE=docker.io/capicua25x/vllm-rocm-rdna4:0.28.0-rdna4
mkdir -p "$HOME/.cache/vllm-rdna4"
exec podman run --rm --name vllm-rdna4 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt label=disable --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$MODEL_DIR":/model:ro \
  -v "$HOME/.cache/vllm-rdna4":/root/.cache/vllm \
  -v "${TRACE_DIR:-/tmp}":/traces \
  ${EXTRA_MOUNTS:-} \
  --entrypoint /usr/local/bin/vllm "$IMAGE" \
  serve /model --served-model-name "$NAME" --host 0.0.0.0 --port 8000 \
  --attention-backend TRITON_ATTN "$@"
