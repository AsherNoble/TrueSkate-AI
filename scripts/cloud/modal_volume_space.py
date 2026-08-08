"""Measure Modal corpus-volume space, including bytes per top-level session.

This is intentionally a mounted-volume walk instead of ``modal volume ls``:
the CLI only reports direct children and displays directory sizes as 4 KiB.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

CAPACITY_BYTES = 1024 * 1_000_000_000_000  # project free-tier capacity
VOLUME_NAME = "trueskate-corpus"

app = modal.App("trueskate-corpus-space")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/corpus": volume}, timeout=2 * 3600)
def measure() -> dict:
    root = Path("/corpus")
    sessions: list[dict] = []
    total = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        size = 0
        for base, _dirs, files in os.walk(child):
            for filename in files:
                try:
                    size += (Path(base) / filename).stat().st_size
                except OSError:
                    pass
        total += size
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = 0.0
        sessions.append({"name": child.name, "bytes": size, "mtime": mtime})
    return {
        "volume": VOLUME_NAME,
        "used_bytes": total,
        "available_bytes": max(0, CAPACITY_BYTES - total),
        "capacity_bytes": CAPACITY_BYTES,
        "sessions": sessions,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(measure.remote(), sort_keys=True))
