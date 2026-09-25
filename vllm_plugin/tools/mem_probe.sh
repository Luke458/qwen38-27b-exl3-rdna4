#!/usr/bin/env bash
# Start the EXL3 server with the given args, then report: model-load memory, KV capacity, VRAM after
# startup, and peak VRAM under a stress load (long prompt, one image, 4 concurrent streams).
#   EXL3_EXT_DIR=... mem_probe.sh NAME [vllm serve args...]
set -uo pipefail
HERE=$(dirname "$(realpath "$0")"); ROOT=$(dirname "$HERE"); REPO=$(dirname "$ROOT")
NAME=$1; shift
LOG=${MEM_PROBE_LOGDIR:-/tmp}/server_$NAME.log
podman rm -f vllm-exl3 >/dev/null 2>&1
("$ROOT/run_exl3_server.sh" "${MODEL_DIR:-$HOME/models/qwen3.8-27b-exl3-11.5gb}" qwen38-27b-exl3 "$@" > "$LOG" 2>&1 &)
sleep 8
for i in $(seq 1 1500); do
  grep -qE "Application startup complete|Traceback|Error:|Memory access fault" "$LOG" && break
  podman ps --format '{{.Names}}' | grep -q vllm-exl3 || break
  sleep 1
done
vram() { rocm-smi --showmeminfo vram 2>/dev/null | grep -oE "Used Memory \(B\): [0-9]+" | grep -oE "[0-9]+$"; }
echo "== $NAME"
grep -oE "Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|ValueError: [^.]*" "$LOG" | head -3
if ! grep -q "Application startup complete" "$LOG"; then echo "STARTUP FAILED"; exit 1; fi
echo "VRAM after startup: $(( $(vram) / 1048576 )) MiB"
PEAK=0
( while true; do v=$(vram); echo "$v"; sleep 0.2; done ) > /tmp/mem_probe_samples.$$ &
SAMPLER=$!
cd "$REPO"
python3 "$HERE/longctx_bench.py" --model qwen38-27b-exl3 --targets ${STRESS_TOKENS:-7000} --decode 64 2>&1 | tail -1
if [ -n "${STRESS_IMAGE:-}" ]; then
  python3 - "$STRESS_IMAGE" <<'EOF' 2>&1 | tail -1
import base64, json, sys, urllib.request
b64 = base64.b64encode(open(sys.argv[1], 'rb').read()).decode()
body = {"model": "qwen38-27b-exl3", "temperature": 0.0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                                  {"type": "text", "text": "Describe the image."}]}]}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
print("image ok:", r["choices"][0]["message"]["content"][:60].replace("\n", " "))
EOF
fi
python3 "$HERE/concurrency_bench.py" --model qwen38-27b-exl3 --conc 4 --max-tokens 128 2>&1 | tail -1
kill $SAMPLER 2>/dev/null
PEAK=$(sort -n /tmp/mem_probe_samples.$$ | tail -1); rm -f /tmp/mem_probe_samples.$$
echo "VRAM peak under stress: $(( PEAK / 1048576 )) MiB (card total 16304 MiB)"
grep -qE "Memory access fault|OutOfMemory|out of memory" "$LOG" && echo "FAULT/OOM IN LOG"
