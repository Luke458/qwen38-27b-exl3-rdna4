#!/usr/bin/env bash
# Serve Qwen3.8-27B EXL3 through the vLLM plugin with a tested profile.
#
#   ./serve.sh            32k context, MTP speculative decoding (~80 tok/s)   [default]
#   ./serve.sh 40k        40k context, MTP
#   ./serve.sh 64k        64k context, no MTP, up to 8 concurrent requests
#   ./serve.sh 32k --generation-config vllm   extra arguments are passed to `vllm serve`
#
# Endpoint: http://127.0.0.1:8000/v1, model "qwen38-27b-exl3". Ctrl-C stops the server.
# Overrides: MODEL_DIR (default ~/models/qwen3.8-27b-exl3-11.5gb), EXL3_EXT_DIR (default ~/exl3ext,
# built by vllm_plugin/tools/build_ext_in_image.sh), EXL3_TEXT_ONLY=1 to skip the vision tower.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
PROFILE=${1:-32k}
[ $# -gt 0 ] && shift
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-exl3-11.5gb}
export EXL3_EXT_DIR=${EXL3_EXT_DIR:-$HOME/exl3ext}

MTP='{"method":"mtp","num_speculative_tokens":3}'
case "$PROFILE" in
  32k) PEAK=14755; ARGS=(--max-model-len 32768 --max-num-seqs 4 --kv-cache-memory-bytes 1760000000 --speculative-config "$MTP") ;;
  40k) PEAK=15122; ARGS=(--max-model-len 40960 --max-num-seqs 4 --kv-cache-memory-bytes 2050000000 --speculative-config "$MTP") ;;
  64k) PEAK=15171; ARGS=(--max-model-len 65536 --max-num-seqs 8 --kv-cache-memory-bytes 2600000000) ;;
  -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown profile '$PROFILE' (use 32k, 40k or 64k)" >&2; exit 2 ;;
esac

[ -d "$MODEL_DIR" ] || { echo "model not found: $MODEL_DIR (set MODEL_DIR)" >&2; exit 1; }
ls "$EXL3_EXT_DIR"/exllamav3_ext*.so >/dev/null 2>&1 || {
  echo "kernel extension not found in $EXL3_EXT_DIR; build it with:" >&2
  echo "  vllm_plugin/tools/build_ext_in_image.sh $EXL3_EXT_DIR" >&2; exit 1; }
if podman ps --format '{{.Names}}' | grep -qx vllm-exl3; then
  echo "a vllm-exl3 server is already running; stop it with: podman stop vllm-exl3" >&2; exit 1
fi

# VRAM check: the server's measured peak plus what is already in use must fit the card
if command -v rocm-smi >/dev/null; then
  read -r USED TOTAL < <(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/ {u=$NF} /Total Memory/ {t=$NF} END {print int(u/1048576), int(t/1048576)}')
  if [ -n "${USED:-}" ] && [ -n "${TOTAL:-}" ] && [ "$TOTAL" -gt 0 ]; then
    FREE_AT_PEAK=$(( TOTAL - USED - PEAK ))
    echo "VRAM: ${USED} MiB in use now; the $PROFILE profile peaks at ~${PEAK} MiB; ~${FREE_AT_PEAK} MiB spare at peak"
    if [ "$FREE_AT_PEAK" -lt 150 ]; then
      echo "warning: too little VRAM headroom for $PROFILE; close GPU-heavy apps or use a smaller profile (32k)" >&2
    fi
  fi
fi

echo "starting $PROFILE profile; first start takes a few minutes (compile + graph capture)"
echo "endpoint: http://127.0.0.1:8000/v1  model: qwen38-27b-exl3  (ready when vLLM logs its startup-complete line)"
exec "$HERE/vllm_plugin/run_exl3_server.sh" "$MODEL_DIR" qwen38-27b-exl3 \
  --max-num-batched-tokens 1024 --kv-cache-dtype int8_per_token_head --mamba-ssm-cache-dtype float16 \
  "${ARGS[@]}" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  "$@"
