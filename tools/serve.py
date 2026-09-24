#!/usr/bin/env python3
"""Launch the existing ExLlamaV3 HTTP server with RX 9070 XT limits.

  .venv/bin/python tools/serve.py --model /path/to/Qwen3.8-27B-EXL3
  .venv/bin/python tools/serve.py --model /path/to/model --inspect

The installed ROCm extension is the default for a fresh build. Research runs
may select a hash-verified --baseline or --candidate binary. The 3-bit int8
experiment stays disabled in every serving mode.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/rocm_exl3"
SERVER_DIR = VENDOR / "rocm_tools/exl3_server"
SERVER_FILE = SERVER_DIR / "server.py"
# Generator.Job.prefill rounds chunks down to PAGE_SIZE=256. Below one page,
# prefill_end stays at the starting position and the request spins indefinitely.
# The earlier 128-token benchmark used direct Model.prefill instead.
PREFILL_CHUNK_SIZE = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def verified_binary(kind: str, candidate: Path | None) -> Path | None:
    if kind == "installed":
        return None
    if kind == "baseline":
        record = ROOT / "artifacts/baseline/records.json"
        if not record.is_file():
            raise ValueError("baseline artifact absent; use the installed backend from a fresh build")
        manifest = json.loads(record.read_text(encoding="utf-8"))
        binary = ROOT / manifest["binary"]
        expected = manifest["binary_sha256"]
    else:
        if candidate is None:
            raise ValueError("--candidate is required with --backend candidate")
        directory = candidate.resolve()
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"candidate manifest absent: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "built":
            raise ValueError("candidate manifest does not record a completed build")
        binary = Path(manifest["candidate_binary"]).resolve()
        expected = manifest["candidate_binary_sha256"]
        if binary.parent != directory:
            raise ValueError("candidate binary is outside its manifest directory")
    binary = binary.resolve()
    if not binary.is_file() or sha256(binary) != expected:
        raise ValueError(f"selected extension is missing or differs from its recorded hash: {binary}")
    if not binary.name.startswith("exllamav3_ext") or binary.suffix != ".so":
        raise ValueError(f"selected file is not an ExLlamaV3 extension: {binary}")
    return binary


def plan(args: argparse.Namespace, environ: dict[str, str]) -> dict:
    model = args.model.resolve()
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError(f"model directory must contain config.json: {model}")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in 1..65535")
    api_key = environ.get("EXL3_API_KEY", "")
    if not is_loopback(args.host) and not api_key:
        raise ValueError("binding beyond loopback requires EXL3_API_KEY in the environment")
    binary = verified_binary(args.backend, args.candidate)
    env = dict(environ)
    env["EXL3_ROCM_GFX1201_INT8"] = "0"
    # EXL3_GEMV=0 routes decode away from GEMV; do not inherit that diagnostic
    # trap into a serving process.
    env.pop("EXL3_GEMV", None)
    return {"model": model, "host": args.host, "port": args.port,
            "served_model_name": args.served_model_name,
            "enable_thinking": args.enable_thinking,
            "api_key": api_key or None, "binary": binary, "backend": args.backend,
            "environment": env}


def server_args(settings: dict) -> argparse.Namespace:
    # The upstream server uses model_init.add_args. Parse only the bounded
    # loader options here, then add its HTTP/sampling defaults to the namespace.
    from exllamav3 import model_init

    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_sampling_args=True,
                        add_draft_model_args=True, default_cache_size=8192,
                        default_autosplit_max_batch_size=1,
                        default_chunk_size=256)
    args = parser.parse_args(["-m", str(settings["model"]), "-cs", "8192",
                              "-ambs", "1", "-chunk_size", "256"])
    args.host = settings["host"]
    args.port = settings["port"]
    args.api_key = settings["api_key"]
    args.served_model_name = settings["served_model_name"]
    args.max_response_tokens = 2048
    args.chat_template_kwargs = json.dumps({"enable_thinking": settings["enable_thinking"]})
    args.loop_window = 0
    args.loop_min_reps = 3
    args.xtc_probability = 0.0
    args.xtc_threshold = 0.1
    args.dry_multiplier = 0.0
    args.dry_base = 1.75
    args.dry_allowed_length = 2
    args.dry_penalty_last_n = -1
    return args


def unsupported_payload(path: str, body: object) -> str | None:
    if not isinstance(body, dict):
        return "request body must be a JSON object"
    if path in ("/v1/chat/completions", "/v1/completions"):
        if body.get("n", 1) != 1:
            return "this launcher supports one completion per request (n=1)"
    if body.get("tools") or body.get("tool_choice"):
        return "tool calls are not supported by this single-request launcher"
    if path == "/v1/chat/completions":
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return "messages must be a list"
        for message in messages:
            if not isinstance(message, dict) or message.get("role") == "tool":
                return "tool messages are not supported"
            if not isinstance(message.get("content"), (str, type(None))):
                return "multimodal message content is not supported"
    elif path in ("/v1/completions", "/completion", "/completions"):
        if not isinstance(body.get("prompt", ""), str):
            return "multimodal or batched prompts are not supported"
    return None


def install_limits(server) -> None:
    from fastapi.responses import JSONResponse

    # Upstream installs unrestricted CORS. This launcher is intended for direct
    # local clients; remove that browser grant before adding its own middleware.
    server.app.user_middleware.clear()
    server.app.middleware_stack = None

    @server.app.middleware("http")
    async def bounded_requests(request, call_next):
        if request.url.path == "/health":
            generator = server.state.generator
            if generator is not None and (generator.error is not None or
                                          generator.iteration_task.done()):
                return JSONResponse({"status": "unhealthy", "reason": "generator stopped"},
                                    status_code=503)
        if request.url.path != "/health" and server.state.args.api_key:
            key = server.state.args.api_key
            bearer = request.headers.get("authorization", "")
            x_key = request.headers.get("x-api-key", "")
            if not (hmac.compare_digest(bearer, f"Bearer {key}") or
                    hmac.compare_digest(x_key, key)):
                return JSONResponse({"detail": "Invalid API key"}, status_code=401)
        if request.method == "POST" and request.url.path in (
            "/v1/chat/completions", "/v1/completions", "/completion", "/completions"
        ):
            try:
                body = await request.json()
            except ValueError:
                return JSONResponse({"detail": "Invalid JSON body"}, status_code=400)
            issue = unsupported_payload(request.url.path, body)
            if issue:
                return JSONResponse({"detail": issue}, status_code=400)
        return await call_next(request)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="EXL3 model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="qwen38-27b-exl3")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen thinking mode (off by default)")
    parser.add_argument("--backend", choices=("installed", "baseline", "candidate"),
                        default="installed")
    parser.add_argument("--candidate", type=Path, help="Candidate binary directory with manifest.json")
    parser.add_argument("--inspect", action="store_true", help="Show settings without importing GPU runtime")
    args = parser.parse_args()
    try:
        settings = plan(args, os.environ)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.inspect:
        print(json.dumps({"model": str(settings["model"]), "host": settings["host"],
                          "port": settings["port"], "backend": settings["backend"],
                          "served_model_name": settings["served_model_name"],
                          "enable_thinking": settings["enable_thinking"],
                          "binary": str(settings["binary"]) if settings["binary"] else "installed",
                          "api_key_configured": bool(settings["api_key"]),
                          "cache_size": 8192, "max_batch_size": 1,
                          "loader_chunk_size": 256, "prefill_chunk_size": PREFILL_CHUNK_SIZE,
                          "max_output_size": 1, "max_response_tokens": 2048,
                          "int8_mode": 0}, indent=2))
        return

    os.environ.clear()
    os.environ.update(settings["environment"])
    if settings["binary"]:
        sys.path.insert(0, str(settings["binary"].parent))
    sys.path.insert(1 if settings["binary"] else 0, str(VENDOR))
    sys.path.insert(0, str(SERVER_DIR))  # dry_sampler is next to server.py

    if settings["binary"]:
        spec = importlib.util.find_spec("exllamav3_ext")
        if spec is None or Path(spec.origin).resolve() != settings["binary"]:
            raise SystemExit("selected extension is not first on Python's import path")
    elif importlib.util.find_spec("exllamav3_ext") is None:
        raise SystemExit("installed exllamav3_ext absent; build the ROCm backend first")

    spec = importlib.util.spec_from_file_location("exl3_server", SERVER_FILE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import upstream server: {SERVER_FILE}")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    # model_init.init passes kwargs into Model.load. The benchmark's memory-safe
    # load used max_output_size=1, while the upstream server omitted that bound.
    original_init = server.model_init.init

    def bounded_init(*a, **kw):
        kw["max_output_size"] = 1
        return original_init(*a, **kw)

    server.model_init.init = bounded_init
    original_generator = server.AsyncGenerator
    from exllamav3.constants import PAGE_SIZE
    if PREFILL_CHUNK_SIZE < PAGE_SIZE:
        raise SystemExit(f"prefill chunk {PREFILL_CHUNK_SIZE} is below generator page size {PAGE_SIZE}")

    class BoundedAsyncGenerator(original_generator):
        def __init__(self, *a, **kw):
            kw["max_batch_size"] = 1
            kw["max_chunk_size"] = PREFILL_CHUNK_SIZE
            super().__init__(*a, **kw)

    server.AsyncGenerator = BoundedAsyncGenerator
    install_limits(server)
    server.main(server_args(settings))


if __name__ == "__main__":
    main()
