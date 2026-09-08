"""Validate the archive directory and enforce byte-for-byte append-only updates."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


def validate_archive(previous: bytes, proposed: bytes) -> None:
    if not proposed.startswith(previous):
        raise ValueError("Archive is append-only: restore existing bytes; append corrections instead.")
    text = proposed.decode("utf-8")
    entries = re.split(r"(?m)^## (ARCH-\d{3,}) — ", text)
    if len(entries) == 1:
        raise ValueError("Archive must contain at least one ARCH entry.")
    seen = set()
    for entry_id, body in zip(entries[1::2], entries[2::2]):
        if entry_id in seen:
            raise ValueError(f"Duplicate archive ID: {entry_id}")
        seen.add(entry_id)
        for field in ("Description", "Commit", "Tag", "Paths", "Recovery", "Artifacts"):
            if not re.search(rf"(?m)^- {field}: .+", body):
                raise ValueError(f"{entry_id}: missing {field}")
        match = re.search(r"(?m)^- Commit: `([a-f0-9]{40})`$", body)
        if not match:
            raise ValueError(f"{entry_id}: Commit must be a full SHA")
        if f"/tree/{match[1]}/" not in body and f"/blob/{match[1]}/" not in body:
            raise ValueError(f"{entry_id}: include a commit-pinned GitHub path link")


def previous_archive(base: str, path: str) -> bytes:
    subprocess.run(["git", "rev-parse", "--verify", f"{base}^{{commit}}"], check=True, capture_output=True)
    paths = subprocess.check_output(["git", "ls-tree", "--name-only", base, "--", path]).decode().splitlines()
    return subprocess.check_output(["git", "show", f"{base}:{path}"]) if path in paths else b""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Target branch commit, not the PR merge base")
    args = parser.parse_args()
    path = "research/ARCHIVE.md"
    validate_archive(previous_archive(args.base, path), Path(path).read_bytes())
    print("Archive integrity passed")


if __name__ == "__main__":
    main()
