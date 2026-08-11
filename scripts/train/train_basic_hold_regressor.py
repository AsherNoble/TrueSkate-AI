"""Train the additive basic Model 1 full-clip hold regressor.

Example:
  PYTHONPATH=src .venv/bin/python scripts/train/train_basic_hold_regressor.py \\
    --data data/mvp_xctest --epochs 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from trueskate_ai.vision.basic_hold_dataset import (  # noqa: E402
    BasicHoldClipDataset, split_by_command, split_by_segment,
)
from trueskate_ai.vision.basic_hold_regressor import BasicHoldRegressor  # noqa: E402
from trueskate_ai.vision.basic_hold_training import (  # noqa: E402
    basic_hold_loss, basic_hold_metrics, passes_basic_hold_acceptance,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(b"\n")
    return f"sha256:{len(paths)}:{digest.hexdigest()}"


def train(*, data: Path, out: Path, epochs: int, batch_size: int, lr: float,
          seed: int, base_channels: int, split_strategy: str = "segment") -> dict:
    torch.manual_seed(seed)
    dataset = BasicHoldClipDataset(data)
    splitters = {"segment": split_by_segment, "command": split_by_command}
    if split_strategy not in splitters:
        raise ValueError(f"unknown split strategy {split_strategy!r}; choose from {sorted(splitters)}")
    train_indices, val_indices, test_indices = splitters[split_strategy](dataset, seed=seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size)
    device = _device()
    model = BasicHoldRegressor(base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best: dict | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = basic_hold_loss(model(batch["frames"].to(device)), batch["target"].to(device))
            loss.backward()
            optimizer.step()
        validation = basic_hold_metrics(model, val_loader, device)
        score = validation["coordinate_median"] + validation["duration_mae"]
        print(f"epoch={epoch} val_coord_med={validation['coordinate_median']:.4f} "
              f"val_duration_mae={validation['duration_mae']:.4f}")
        if best is None or score < best["score"]:
            best = {"score": score, "epoch": epoch, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "validation": validation}
    assert best is not None
    model.load_state_dict(best["state_dict"])
    test = basic_hold_metrics(model, test_loader, device)
    payload = {
        "model_type": "basic_hold_regressor_v2_spatiotemporal",
        "uses_pre_touch_difference": True,
        "spatial_map_stride": 4,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_channels": base_channels,
        "sequence_length": dataset.sequence_length,
        "image_height": dataset.image_height,
        "image_width": dataset.image_width,
        "split_seed": seed,
        "split_strategy": split_strategy,
        "dataset_fingerprint": _fingerprint(dataset.sample_paths),
        "dataset_stats": dataset.stats,
        "split_sizes": {"train": len(train_indices), "validation": len(val_indices), "test": len(test_indices)},
        "best_epoch": best["epoch"],
        "validation": best["validation"],
        "test": test,
        "passes_acceptance": passes_basic_hold_acceptance(test),
        "state_dict": best["state_dict"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(out)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train basic full-clip stationary-hold Model 1 experiment.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--split-strategy", choices=("segment", "command"), default="segment",
                        help="segment avoids neighbor leakage; command withholds exact replayed {x,y,dur} values.")
    parser.add_argument("--min-samples", type=int, default=1000,
                        help="Require this many accepted calibrated hold clips (0 disables the milestone gate).")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    dataset_probe = BasicHoldClipDataset(args.data)
    if args.min_samples < 0:
        parser.error("min-samples must be non-negative")
    if args.min_samples and len(dataset_probe) < args.min_samples:
        parser.error(f"need {args.min_samples} accepted basic-hold clips; found {len(dataset_probe)} ({dataset_probe.stats})")
    out = args.out or _ROOT / "notebooks" / "models" / f"basic_hold_regressor_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    result = train(data=args.data, out=out, epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, seed=args.seed, base_channels=args.base_channels,
                   split_strategy=args.split_strategy)
    print(json.dumps({key: value for key, value in result.items() if key != "state_dict"}, indent=2))
    print(f"checkpoint={out}")


if __name__ == "__main__":
    main()
