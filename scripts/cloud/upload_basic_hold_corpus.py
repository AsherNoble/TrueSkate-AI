"""Non-destructively upload a validated basic-hold corpus to a Modal volume.

This deliberately uses ``batch_upload`` rather than the legacy corpus-offload
script: the source corpus remains on the rig, and the requested remote directory
is populated directly (without a duplicated nested directory).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.vision.basic_hold_dataset import BasicHoldClipDataset  # noqa: E402


def validate_corpus(root: Path, *, min_samples: int, require_unique_commands: bool) -> dict:
    """Return upload provenance, rejecting incomplete or replayed corpora."""
    dataset = BasicHoldClipDataset(root)
    command_count = len(set(dataset.command_keys))
    if len(dataset) < min_samples:
        raise ValueError(f"need {min_samples} accepted clips; found {len(dataset)} ({dataset.stats})")
    if require_unique_commands and command_count != len(dataset):
        raise ValueError(
            f"need one exact command per accepted clip; found {command_count} distinct "
            f"commands across {len(dataset)} clips"
        )
    return {
        "accepted": len(dataset),
        "distinct_commands": command_count,
        "dataset_stats": dataset.stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--volume", default="trueskate-mvp")
    parser.add_argument("--remote-subdir", required=True,
                        help="Destination below the volume root, e.g. basic_hold_diverse_xctest")
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--allow-replayed-commands", action="store_true",
                        help="Bypass the one-command-per-clip guard (not for generalisation runs).")
    args = parser.parse_args()
    if args.min_samples < 0:
        parser.error("--min-samples must be non-negative")
    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"--source is not a directory: {source}")
    remote_subdir = args.remote_subdir.strip("/")
    if not remote_subdir or "/../" in f"/{remote_subdir}/" or remote_subdir == "..":
        parser.error("--remote-subdir must be a safe non-root relative path")

    provenance = validate_corpus(
        source,
        min_samples=args.min_samples,
        require_unique_commands=not args.allow_replayed_commands,
    )
    volume = modal.Volume.from_name(args.volume)
    with volume.batch_upload() as upload:
        upload.put_directory(str(source), f"/{remote_subdir}")
    print(json.dumps({
        "source": str(source), "volume": args.volume,
        "remote_subdir": remote_subdir, **provenance,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
