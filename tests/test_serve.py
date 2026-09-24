"""CPU-only checks for the bounded server launch plan."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from tools import serve


class ServePlanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}", encoding="utf-8")

    def args(self, **kw):
        defaults = dict(model=self.model, host="127.0.0.1", port=8000,
                        backend="installed", candidate=None,
                        served_model_name="qwen38-27b-exl3", enable_thinking=False)
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def test_loopback_defaults_and_diagnostic_trap_removed(self):
        result = serve.plan(self.args(), {"EXL3_GEMV": "0", "EXL3_ROCM_GFX1201_INT8": "2"})
        self.assertEqual((result["host"], result["port"]), ("127.0.0.1", 8000))
        self.assertEqual(result["environment"]["EXL3_ROCM_GFX1201_INT8"], "0")
        self.assertNotIn("EXL3_GEMV", result["environment"])
        self.assertIsNone(result["binary"])

    def test_network_bind_requires_environment_key(self):
        with self.assertRaisesRegex(ValueError, "EXL3_API_KEY"):
            serve.plan(self.args(host="0.0.0.0"), {})
        result = serve.plan(self.args(host="0.0.0.0"), {"EXL3_API_KEY": "private-test-key"})
        self.assertEqual(result["api_key"], "private-test-key")
        self.assertTrue(serve.is_loopback("::1"))
        self.assertFalse(serve.is_loopback("192.168.1.2"))

    def test_candidate_hash_and_location(self):
        directory = self.root / "candidate"
        directory.mkdir()
        binary = directory / "exllamav3_ext.cpython-312-x86_64-linux-gnu.so"
        binary.write_bytes(b"candidate")
        manifest = {"status": "built", "candidate_binary": str(binary),
                    "candidate_binary_sha256": hashlib.sha256(b"candidate").hexdigest()}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(serve.plan(self.args(backend="candidate", candidate=directory), {})["binary"], binary)
        binary.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            serve.plan(self.args(backend="candidate", candidate=directory), {})

    def test_missing_model_and_invalid_port(self):
        with self.assertRaisesRegex(ValueError, "config.json"):
            serve.plan(self.args(model=self.root), {})
        with self.assertRaisesRegex(ValueError, "port"):
            serve.plan(self.args(port=0), {})

    def test_single_text_request_boundary(self):
        self.assertIsNone(serve.unsupported_payload("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}], "n": 1,
        }))
        self.assertIn("n=1", serve.unsupported_payload("/v1/chat/completions", {"n": 2}))
        self.assertIn("multimodal", serve.unsupported_payload("/v1/chat/completions", {
            "messages": [{"role": "user", "content": [{"type": "image_url"}]}],
        }))
        self.assertIn("tool", serve.unsupported_payload("/v1/chat/completions", {"tools": [{"type": "function"}]}))
        self.assertIn("batched", serve.unsupported_payload("/completion", {"prompt": ["one", "two"]}))

    def test_generator_prefill_has_at_least_one_page(self):
        constants = runpy.run_path(str(serve.VENDOR / "exllamav3/constants.py"))
        self.assertGreaterEqual(serve.PREFILL_CHUNK_SIZE, constants["PAGE_SIZE"])


if __name__ == "__main__":
    unittest.main()
