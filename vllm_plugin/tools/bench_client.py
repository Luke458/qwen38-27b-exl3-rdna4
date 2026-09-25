#!/usr/bin/env python3
"""Single-stream decode probe against an OpenAI-compatible endpoint (stdlib only).

Streams one chat completion per prompt with usage reporting and records TTFT,
completion tokens and decode tok/s (tokens after the first / time after the
first). Greedy (temperature 0) so runs with and without MTP can be compared for
identical text.

  python3 bench_client.py --model qwen3.5-4b --max-tokens 256 --out run.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROMPTS = [
    "Write a detailed, multi-paragraph explanation of how a CPU cache hierarchy works.",
    "Write a Python function that parses an ISO-8601 duration string, with tests.",
    "Tell a long story about a lighthouse keeper who discovers a message in a bottle.",
]


def one(base, model, prompt, max_tokens, key):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    t_first = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text.append(piece)
    t1 = time.perf_counter()
    n = usage["completion_tokens"] if usage else None
    decode_tps = (n - 1) / (t1 - t_first) if n and t_first and n > 1 else None
    return {"ttft_s": (t_first - t0) if t_first else None, "wall_s": t1 - t0,
            "completion_tokens": n, "decode_tok_s": decode_tps, "text": "".join(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--key", default="none")
    ap.add_argument("--out")
    args = ap.parse_args()
    one(args.base, args.model, "Hi", 8, args.key)  # warm
    rows = []
    for rep in range(args.repeats):
        for i, p in enumerate(PROMPTS):
            r = one(args.base, args.model, p, args.max_tokens, args.key)
            r.update(prompt=i, repeat=rep)
            rows.append(r)
            print(f"prompt {i} rep {rep}: ttft {r['ttft_s']:.3f}s  tokens {r['completion_tokens']}  "
                  f"decode {r['decode_tok_s']:.1f} tok/s")
    tps = sorted(r["decode_tok_s"] for r in rows if r["decode_tok_s"])
    print(f"median decode tok/s: {tps[len(tps) // 2]:.1f}")
    if args.out:
        json.dump({"args": vars(args), "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
