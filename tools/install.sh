#!/usr/bin/env bash
# Install the source-only project on a ROCm 7.2 / gfx1201 host.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-baseline}"
if [[ "$profile" != baseline && "$profile" != experimental ]]; then
  echo "usage: bash tools/install.sh [baseline|experimental]" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/)" >&2
  exit 2
fi
if [[ ! -f /opt/rocm/.info/version ]]; then
  echo "ROCm installation missing at /opt/rocm" >&2
  exit 2
fi

cd "$project_root"
python3 tools/prepare_backend.py --profile "$profile"
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12
fi
uv pip install -p .venv 'setuptools>=77,<81' wheel ninja
uv pip install -p .venv --index-url https://download.pytorch.org/whl/rocm7.2 \
  'torch==2.13.0+rocm7.2' 'triton-rocm==3.7.1'
uv pip install -p .venv -r vendor/rocm_exl3/requirements_rocm.txt \
  -c constraints-rocm.txt
uv pip install -p .venv -r requirements-serve.txt -c constraints-rocm.txt
EXL3_BACKEND=rocm PYTORCH_ROCM_ARCH=gfx1201 uv pip install -p .venv \
  --no-build-isolation --no-deps ./vendor/rocm_exl3

echo "Installed $profile backend in .venv; model weights are not included."
