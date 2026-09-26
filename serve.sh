#!/usr/bin/env bash
# Serve Qwen3.8-27B EXL3 through the vLLM plugin with one of its profiles (16 GB ones are measured).
#
#   ./serve.sh            32k context, MTP speculative decoding (~80 tok/s)   [default]
#   ./serve.sh 40k        40k context, MTP
#   ./serve.sh 48k-int4   48k context, MTP, 4-bit KV cache (same VRAM as 32k; slightly less accurate)
#   ./serve.sh 64k-int4   64k context, MTP, 4-bit KV cache (VRAM of the 40k profile)
#   ./serve.sh 64k        64k context, no MTP, up to 8 concurrent requests
#   ./serve.sh 128k       128k context, MTP      (32 GB cards, e.g. Radeon AI PRO R9700; untested)
#   ./serve.sh 256k       262k context, MTP      (32 GB cards; the model's maximum; untested)
#   ./serve.sh 32k --generation-config vllm   extra arguments are passed to `vllm serve`
#
# Endpoint: http://127.0.0.1:8000/v1, model "qwen38-27b-exl3". Ctrl-C stops the server.
# Overrides: MODEL_DIR (default ~/models/qwen3.8-27b-exl3-11.5gb), EXL3_EXT_DIR (default ~/models/exl3ext,
# built by vllm_plugin/tools/build_ext_in_image.sh), EXL3_TEXT_ONLY=1 to skip the vision tower.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
PROFILE=${1:-32k}
[ $# -gt 0 ] && shift
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-exl3-11.5gb}
export EXL3_EXT_DIR=${EXL3_EXT_DIR:-$HOME/models/exl3ext}

MTP='{"method":"mtp","num_speculative_tokens":3}'
KV=int8_per_token_head
MIN_TOTAL=0  # MiB of VRAM the card must have
# 16 GB profiles: measured server peaks (experiments/0023). 32 GB profiles: KV sized with the same
# 28,853,760-byte page math (3+ spare pages, see the plugin's KV headroom check); peaks estimated as
# measured non-KV footprint + KV + the long-prefill fp16 KV copy. Not run on a 32 GB card yet.
# int4 profiles (experiments/0024): 1,616-token pages of 29,010,432 bytes, 3+ spare. 48k-int4 peaked at
# 14,665 MiB with a 41k prompt and an image before the exact sink / tail rows (+~0.1 GiB); 64k-int4 at
# 14.5-14.8 GiB text-only at 53-62k, so ~15.3 GiB with vision and exact rows (estimated: a full-length vision
# run did not fit next to a 1.1 GiB desktop).
case "$PROFILE" in
  32k) PEAK=14755; ARGS=(--max-model-len 32768 --max-num-seqs 4 --kv-cache-memory-bytes 1760000000 --speculative-config "$MTP") ;;
  40k) PEAK=15122; ARGS=(--max-model-len 40960 --max-num-seqs 4 --kv-cache-memory-bytes 2050000000 --speculative-config "$MTP") ;;
  48k-int4) PEAK=14800; KV=int4_per_token_head; ARGS=(--max-model-len 49152 --max-num-seqs 4 --kv-cache-memory-bytes 1460000000 --speculative-config "$MTP") ;;
  64k-int4) PEAK=15300; KV=int4_per_token_head; ARGS=(--max-model-len 65536 --max-num-seqs 4 --kv-cache-memory-bytes 1760000000 --speculative-config "$MTP") ;;
  64k) PEAK=15171; ARGS=(--max-model-len 65536 --max-num-seqs 8 --kv-cache-memory-bytes 2600000000) ;;
  128k) PEAK=18700; MIN_TOTAL=30000; ARGS=(--max-model-len 131072 --max-num-seqs 4 --kv-cache-memory-bytes 5250000000 --speculative-config "$MTP") ;;
  256k) PEAK=23800; MIN_TOTAL=30000; ARGS=(--max-model-len 262144 --max-num-seqs 4 --kv-cache-memory-bytes 9900000000 --speculative-config "$MTP") ;;
  -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown profile '$PROFILE' (use 32k, 40k, 48k-int4, 64k-int4, 64k, or on a 32 GB card 128k / 256k)" >&2; exit 2 ;;
esac

vram_mib() {  # prints "<used> <total>" in MiB, or nothing if rocm-smi is unavailable
  command -v rocm-smi >/dev/null || return 0
  rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory/ {u=$NF} /Total Memory/ {t=$NF} END {if (t) print int(u/1048576), int(t/1048576)}'
}
read -r USED TOTAL < <(vram_mib) || true
if [ "$MIN_TOTAL" -gt 0 ] && [ -n "${TOTAL:-}" ] && [ "$TOTAL" -lt "$MIN_TOTAL" ]; then
  echo "the $PROFILE profile needs a 32 GB card (this one has ${TOTAL} MiB); use 32k, 40k or 64k" >&2; exit 1
fi

[ -d "$MODEL_DIR" ] || { echo "model not found: $MODEL_DIR (set MODEL_DIR)" >&2; exit 1; }
ls "$EXL3_EXT_DIR"/exllamav3_ext*.so >/dev/null 2>&1 || {
  echo "kernel extension not found in $EXL3_EXT_DIR; build it with:" >&2
  echo "  vllm_plugin/tools/build_ext_in_image.sh $EXL3_EXT_DIR" >&2; exit 1; }
if podman ps --format '{{.Names}}' | grep -qx vllm-exl3; then
  echo "a vllm-exl3 server is already running; stop it with: podman stop vllm-exl3" >&2; exit 1
fi

# VRAM check: the server's peak plus what is already in use must fit the card
if [ -n "${USED:-}" ] && [ -n "${TOTAL:-}" ]; then
  FREE_AT_PEAK=$(( TOTAL - USED - PEAK ))
  echo "VRAM: ${USED} MiB in use now; the $PROFILE profile peaks at ~${PEAK} MiB; ~${FREE_AT_PEAK} MiB spare at peak"
  if [ "$FREE_AT_PEAK" -lt 150 ]; then
    echo "warning: too little VRAM headroom for $PROFILE; close GPU-heavy apps or use a smaller profile" >&2
  fi
fi

echo "starting $PROFILE profile; first start takes a few minutes (compile + graph capture)"
echo "endpoint: http://127.0.0.1:8000/v1  model: qwen38-27b-exl3  (ready when vLLM logs its startup-complete line)"
exec "$HERE/vllm_plugin/run_exl3_server.sh" "$MODEL_DIR" qwen38-27b-exl3 \
  --max-num-batched-tokens 1024 --kv-cache-dtype "$KV" --mamba-ssm-cache-dtype float16 \
  "${ARGS[@]}" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  "$@"
