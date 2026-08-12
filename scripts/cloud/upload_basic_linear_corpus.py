"""Non-destructively upload only strict MVP-2 linear clips to Modal."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset  # noqa: E402


def validated_dataset(root: Path, *, min_samples: int,
                      require_unique_commands: bool) -> BasicLinearClipDataset:
    dataset = BasicLinearClipDataset(root)
    command_count = len(set(dataset.command_keys))
    if len(dataset) < min_samples:
        raise ValueError(f"need {min_samples} accepted clips; found {len(dataset)} ({dataset.stats})")
    if require_unique_commands and command_count != len(dataset):
        raise ValueError(
            f"need one exact command per accepted clip; found {command_count} distinct "
            f"commands across {len(dataset)} clips"
        )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--volume", default="trueskate-mvp")
    parser.add_argument("--remote-subdir", required=True)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--allow-replayed-commands", action="store_true")
    args = parser.parse_args()
    if args.min_samples < 0:
        parser.error("--min-samples must be non-negative")
    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"--source is not a directory: {source}")
    remote_subdir = args.remote_subdir.strip("/")
    if not remote_subdir or "/../" in f"/{remote_subdir}/" or remote_subdir == "..":
        parser.error("--remote-subdir must be a safe non-root relative path")
    dataset = validated_dataset(
        source, min_samples=args.min_samples,
        require_unique_commands=not args.allow_replayed_commands,
    )
    volume = modal.Volume.from_name(args.volume)
    with volume.batch_upload() as upload:
        for sample in dataset.sample_paths:
            upload.put_directory(str(sample), f"/{remote_subdir}/{sample.relative_to(source).as_posix()}")
    print(json.dumps({
        "source": str(source), "volume": args.volume, "remote_subdir": remote_subdir,
        "accepted": len(dataset), "distinct_commands": len(set(dataset.command_keys)),
        "dataset_stats": dataset.stats,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
