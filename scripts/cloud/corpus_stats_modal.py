"""Corpus stats over the Modal volume, computed IN the cloud (no download).

The trueskate-corpus volume is the corpus of record (the rig offloader deletes
local sessions after verified upload), so readiness numbers must come from the
volume itself. This is the repo's first Modal app: a bare container + the
stdlib-only trueskate_ai.data.corpus_stats module walking the mounted volume.

Run (from the repo root, ~/.modal.toml auth):
    .venv/bin/modal run scripts/cloud/corpus_stats_modal.py
    .venv/bin/modal run scripts/cloud/corpus_stats_modal.py --json-out tmp/corpus_stats_modal.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

VOLUME_NAME = "trueskate-corpus"

app = modal.App("trueskate-corpus-stats")
# Bare image + the package source; corpus_stats is stdlib-only so nothing to pip.
image = modal.Image.debian_slim(python_version="3.11").add_local_dir(
    str(_REPO_ROOT / "src" / "trueskate_ai"), remote_path="/root/trueskate_ai"
)
corpus = modal.Volume.from_name(VOLUME_NAME)


@app.function(image=image, volumes={"/corpus": corpus}, timeout=3600)
def stats() -> dict:
    from trueskate_ai.data.corpus_stats import accumulate, iter_samples
    return accumulate(iter_samples(Path("/corpus")))


@app.local_entrypoint()
def main(json_out: str = "tmp/corpus_stats_modal.json") -> None:
    from trueskate_ai.data.corpus_stats import summarize
    s = stats.remote()
    out = Path(json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2))
    print(f"SLS corpus @ modal volume {VOLUME_NAME}")
    print(summarize(s))
    print(f"JSON -> {out}")
