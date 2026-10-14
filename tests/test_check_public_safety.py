"""Tests for the public-safety check. Sample identifiers are assembled at runtime so this
file never contains a literal identifier for the check to flag."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("public_safety", Path(__file__).parents[1] / "scripts/ops/check_public_safety.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

UDID = "00008020" + "-" + "001A2B3C4D5E6F70"
TAILSCALE_IP = ".".join(["100", "101", "2", "3"])
PERSONAL_EMAIL = "someone" + "@" + "outlook.com"


@pytest.mark.parametrize("text, problem", [
    (f"udid = {UDID}", "UDID"),
    (f"udid = {UDID.lower()}", "UDID"),
    (f"ssh rig@{TAILSCALE_IP}", "Tailscale"),
    (f"contact {PERSONAL_EMAIL}", "email"),
])
def test_identifiers_are_flagged(text, problem):
    assert any(problem in item for item in check.line_problems("a.md", text))


@pytest.mark.parametrize("text", [
    "Co-Authored-By: Claude <noreply" + "@anthropic.com>",
    "author 123+someone" + "@users.noreply.github.com",
    "git" + "@github.com:owner/repo.git",
    "ssh training-server@training-server",
    "commit " + "a" * 40,
    "address " + ".".join(["100", "0", "0", "1"]),
    "example user" + "@example.com",
])
def test_ordinary_text_passes(text):
    assert check.line_problems("a.md", text) == []


def test_size_and_json_limits():
    assert check.file_problems("a.png", 500 * 1024, None) == []
    assert "500 KB" in check.file_problems("a.png", 500 * 1024 + 1, None)[0]
    assert check.file_problems("a.json", 10, 2000) == []
    assert "2,000-line" in check.file_problems("a.json", 10, 2001)[0]


def test_problems_reads_only_the_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = lambda *args: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t" + "@example.com", *args],
                                       check=True, capture_output=True)
    run("init", "-q")
    (tmp_path / "old.json").write_text("[\n" * 3000)  # pre-existing bulk is not re-flagged
    run("add", "-A"); run("commit", "-q", "-m", "base")
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (tmp_path / "clean.md").write_text("training-server only\n")
    run("add", "-A"); run("commit", "-q", "-m", "clean")
    assert check.problems(base) == []
    (tmp_path / "big.json").write_text("{}\n" * 2001)
    (tmp_path / "note.md").write_text(f"ask {PERSONAL_EMAIL}\n")
    run("add", "-A"); run("commit", "-q", "-m", "bad")
    found = check.problems(base)
    assert any("big.json" in item and "2,000-line" in item for item in found)
    assert any("note.md" in item and "email" in item for item in found)
