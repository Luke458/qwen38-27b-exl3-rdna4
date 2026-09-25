#!/usr/bin/env python3
"""Long-prompt probe (OpenAI API): prefill throughput (prompt tokens / TTFT) and decode tok/s at
several context lengths. Prompt text is local repository prose/code, sized with the usage count.

  python longctx_bench.py --model NAME --targets 2000 6000 12000
"""
import argparse
import glob
import json
import time
import urllib.request


def corpus():
    parts = []
    for pat in ("docs/*.md", "README.md", "vendor/rocm_exl3/README.md", "vendor/rocm_exl3/exllamav3/modules/*.py"):
        for f in sorted(glob.glob(pat)):
            parts.append(open(f, errors="ignore").read())
    return "\n\n".join(parts)


def stream(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; usage = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if (ch.get("delta") or {}).get("content") and t_first is None:
                    t_first = time.perf_counter()
    t1 = time.perf_counter()
    return usage, t_first - t0, t1 - t_first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", type=int, nargs="+", default=[2000, 6000, 12000])
    ap.add_argument("--decode", type=int, default=128)
    args = ap.parse_args()
    text = corpus()
    for target in args.targets:
        chars = int(target * 3.2)
        prompt = "Summarize the following material in a few bullet points.\n\n" + text[:chars]
        usage, ttft, dec = stream(args.base, args.model, prompt, args.decode)
        pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
        print(f"prompt {pt:6d} tok: TTFT {ttft:6.2f} s = {pt / ttft:7.0f} tok/s prefill | "
              f"decode {(ct - 1) / dec:6.1f} tok/s over {ct} tokens", flush=True)


if __name__ == "__main__":
    main()
