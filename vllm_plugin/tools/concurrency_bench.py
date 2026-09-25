#!/usr/bin/env python3
"""Aggregate decode throughput with C concurrent streaming requests (distinct prompts).

  python concurrency_bench.py --model NAME --conc 1 2 4 --max-tokens 256
"""
import argparse
import json
import threading
import time
import urllib.request

TOPICS = ["how CPU caches work", "the history of the printing press", "a Python function that merges "
          "sorted lists, with tests", "how vaccines train the immune system", "a short story about a "
          "lighthouse keeper", "the causes of the French revolution", "how TCP congestion control works",
          "a recipe for sourdough bread"]


def one(base, model, prompt, max_tokens, out, i):
    body = {"model": model, "messages": [{"role": "user", "content": f"Write in detail about {prompt}."}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            if first is None and any((c.get("delta") or {}).get("content") for c in ev.get("choices", [])):
                first = time.perf_counter()
    out[i] = (usage["completion_tokens"], first - t0, time.perf_counter() - first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    for c in args.conc:
        out = [None] * c
        th = [threading.Thread(target=one, args=(args.base, args.model, TOPICS[i % len(TOPICS)], args.max_tokens,
                                                 out, i)) for i in range(c)]
        t0 = time.perf_counter()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.perf_counter() - t0
        toks = sum(o[0] for o in out)
        per = sorted((o[0] - 1) / o[2] for o in out)
        print(f"C={c}: aggregate {toks / wall:6.1f} tok/s (wall incl. prefill), per-stream decode median "
              f"{per[len(per) // 2]:5.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
