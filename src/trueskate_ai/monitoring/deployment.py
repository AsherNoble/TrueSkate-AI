"""Read-only source identity for deployed services; never reports file contents."""
from pathlib import Path
import subprocess


def revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=3, text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def deployment_status(repo: Path, loaded_revision: str | None) -> dict:
    current = revision(repo)
    dirty = None
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal", "--",
             "src", "scripts", "tests", "pyproject.toml", "requirements.txt"],
            stderr=subprocess.DEVNULL, timeout=3, text=True,
        )
        dirty = bool(output.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"loaded_revision": loaded_revision, "disk_revision": current,
            "source_dirty": dirty,
            "restart_pending": bool(current and loaded_revision and current != loaded_revision)}
