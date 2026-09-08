import subprocess
from trueskate_ai.monitoring.deployment import deployment_status


def test_untracked_and_modified_source_are_visible(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/example.py"
    source.write_text("original\n")
    git("add", "src")
    git("commit", "-qm", "baseline")
    sha = git("rev-parse", "HEAD")
    assert deployment_status(tmp_path, sha) == {
        "loaded_revision": sha, "disk_revision": sha,
        "source_dirty": False, "restart_pending": False,
    }
    source.write_text("changed\n")
    assert deployment_status(tmp_path, sha)["source_dirty"]
    git("add", "src")
    git("commit", "-qm", "update")
    assert deployment_status(tmp_path, sha)["restart_pending"]
    (tmp_path / "src/untracked.py").write_text("new\n")
    assert deployment_status(tmp_path, sha)["source_dirty"]


def test_non_checkout_is_unknown(tmp_path):
    state = deployment_status(tmp_path, None)
    assert state["disk_revision"] is None
    assert state["source_dirty"] is None
