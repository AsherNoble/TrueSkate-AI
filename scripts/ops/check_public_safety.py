"""Fail when a change adds bulk files or personal identifiers to the public repository.

Checks every file added or modified between ``--base`` and ``--head``:

- files over 500 KB, and JSON files over 2,000 lines;
- added lines containing a phone UDID, a Tailscale address (100.64/10), or an
  email address other than a GitHub/Anthropic noreply address.

Bulk output belongs in the evidence folder (AGENTS.md, "Writing it down").
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

MAX_BYTES = 500 * 1024
MAX_JSON_LINES = 2000
UDID = re.compile(r"\b[0-9A-F]{8}-[0-9A-F]{16}\b", re.IGNORECASE)
TAILSCALE = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
ALLOWED_EMAIL = re.compile(r"(@users\.noreply\.github\.com|^noreply@anthropic\.com|^git@github\.com"
                           r"|@example\.(com|org|net))$", re.IGNORECASE)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, errors="replace")


def line_problems(path: str, text: str) -> list[str]:
    found = []
    if UDID.search(text):
        found.append(f"{path}: adds a phone UDID")
    if TAILSCALE.search(text):
        found.append(f"{path}: adds a Tailscale address")
    if any(not ALLOWED_EMAIL.search(email) for email in EMAIL.findall(text)):
        found.append(f"{path}: adds an email address")
    return found


def file_problems(path: str, size: int, json_lines: int | None) -> list[str]:
    found = []
    if size > MAX_BYTES:
        found.append(f"{path}: {size // 1024} KB is over the 500 KB limit")
    if json_lines is not None and json_lines > MAX_JSON_LINES:
        found.append(f"{path}: {json_lines} JSON lines is over the 2,000-line limit")
    return found


def problems(base: str, head: str = "HEAD") -> list[str]:
    found: list[str] = []
    for path in git("diff", "--name-only", "--diff-filter=AM", f"{base}...{head}").splitlines():
        size = int(git("cat-file", "-s", f"{head}:{path}"))
        lines = git("show", f"{head}:{path}").count("\n") if path.endswith(".json") else None
        found += file_problems(path, size, lines)
    path, in_header = None, False
    for line in git("diff", "-U0", "--no-color", "--no-ext-diff", f"{base}...{head}").splitlines():
        if line.startswith("diff --git "):
            path, in_header = None, True
        elif in_header and line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            in_header = False
        elif not in_header and path and line.startswith("+"):
            found += line_problems(path, line[1:])
    return list(dict.fromkeys(found))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="Target branch commit, not the PR merge base")
    parser.add_argument("--head", default="HEAD", help="Commit to check (default: HEAD)")
    args = parser.parse_args()
    found = problems(args.base, args.head)
    for item in found:
        print(item)
    if found:
        sys.exit(f"Public-safety check failed: {len(found)} problem(s); see AGENTS.md, 'Writing it down'.")
    print("Public-safety check passed")


if __name__ == "__main__":
    main()
