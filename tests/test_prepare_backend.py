"""CPU-only, offline integrity checks for source-only backend preparation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "prepare_backend.py"
spec = importlib.util.spec_from_file_location("prepare_backend", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prepare)


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], text=True,
                          capture_output=True, timeout=60)


def test_all_published_patch_digests_match():
    assert prepare.PIN == "311ff5497237a37bea18cf24aa18bc0c573d1d03"
    for patches in prepare.PATCHES.values():
        for name, expected in patches:
            assert prepare.digest(ROOT / "patches" / name) == expected
    assert len(prepare.expected_files("baseline")) == 4
    assert len(prepare.expected_files("experimental")) == 6


def test_invalid_existing_directory_is_never_mutated(tmp_path):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    sentinel = vendor / "do-not-touch.txt"
    sentinel.write_text("user work\n")
    result = invoke("--vendor", str(vendor), "--source", str(tmp_path / "absent"))
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr
    assert sentinel.read_text() == "user work\n"
    assert sorted(p.name for p in vendor.iterdir()) == [sentinel.name]


@pytest.mark.skipif(not (ROOT / "vendor" / "rocm_exl3" / ".git").is_dir() or not shutil.which("git"),
                    reason="offline pinned upstream checkout unavailable")
def test_clean_local_clone_profiles_and_mismatch(tmp_path):
    source = ROOT / "vendor" / "rocm_exl3"
    assert subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip() == prepare.PIN
    baseline = tmp_path / "baseline"
    experimental = tmp_path / "experimental"
    for profile, target in (("baseline", baseline), ("experimental", experimental)):
        result = invoke("--profile", profile, "--vendor", str(target), "--source", str(source))
        assert result.returncode == 0, result.stderr
        prepare.verify(target, profile)
        assert invoke("--profile", profile, "--vendor", str(target)).returncode == 0
    before = {name: prepare.digest(baseline / name)
              for name in prepare.expected_files("baseline")}
    mismatch = invoke("--profile", "experimental", "--vendor", str(baseline),
                      "--source", str(source))
    assert mismatch.returncode == 2
    assert "tracked source changes differ" in mismatch.stderr
    assert before == {name: prepare.digest(baseline / name) for name in before}
    prepare.verify(baseline, "baseline")


def test_pin_check_rejects_other_commit_without_mutation(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git unavailable")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "sentinel").write_text("keep")
    subprocess.run(["git", "-C", str(repo), "add", "sentinel"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "test"], check=True)
    result = invoke("--vendor", str(repo))
    assert result.returncode == 2
    assert "not pinned" in result.stderr
    assert (repo / "sentinel").read_text() == "keep"
