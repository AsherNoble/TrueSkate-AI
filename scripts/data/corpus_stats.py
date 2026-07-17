"""Corpus stats CLI for local trees (rig data/sls_xctest, or any corpus root).

The cloud twin (scripts/cloud/corpus_stats_modal.py) runs the same aggregation
inside Modal over the trueskate-corpus volume; run THIS one on the rig for the
sessions that haven't offloaded yet, then merge with --merge-json.

Usage:
    python scripts/data/corpus_stats.py [--root data/sls_xctest] \
        [--json-out tmp/corpus_stats_local.json] [--merge-json tmp/corpus_stats_modal.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.data.corpus_stats import accumulate, iter_samples, merge, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SLS corpus stats (local filesystem).")
    ap.add_argument("--root", type=Path, default=_REPO_ROOT / "data" / "sls_xctest")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--merge-json", type=Path, default=None,
                    help="another accumulate() JSON (e.g. the Modal volume's) to merge in")
    args = ap.parse_args()
    if not args.root.exists():
        raise SystemExit(f"corpus root not found: {args.root}")
    stats = accumulate(iter_samples(args.root))
    label = str(args.root)
    if args.merge_json:
        stats = merge(json.loads(args.merge_json.read_text()), stats)
        label += f" + {args.merge_json}"
    print(f"SLS corpus @ {label}")
    print(summarize(stats))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(stats, indent=2))
        print(f"JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
