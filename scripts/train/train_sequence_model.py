"""Train Model 2 — the vision-grounded gesture-sequence policy.

`(n frames + m past strokes) -> next strokes`, trained on Asher's expert play
auto-labeled by Model 1 (the trace extractor). This is the VPT second stage /
the "sequence leap". See experiments/vision_sequence_leap_journal.md and
~/.claude/plans/start-investigating-how-we-jazzy-stroustrup.md.

Run modes:
    --smoke           prove the full pipeline (dataset->model->loss->step) on
                      SYNTHETIC data, no corpus needed. Verifies the architecture
                      runs + a learnable target's loss falls.
    --data <dir>      train on real assembled sequences (SequenceDataset).

The model + tokens are pure torch (trueskate_ai.bc) so this script runs unchanged
on any cloud GPU (Modal first) — the portable-harness target.

Usage:
    python -u scripts/train/train_sequence_model.py --smoke
    python -u scripts/train/train_sequence_model.py --data data/sls_xctest/<sess> --epochs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from trueskate_ai.bc.gesture_tokens import STROKE_DIM  # noqa: E402
from trueskate_ai.bc.model2 import (  # noqa: E402
    SequencePolicy, SequencePolicyConfig, build_policy, config_to_dict, stroke_loss,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SyntheticSequenceDataset(Dataset):
    """Random frames + past strokes; target = mean of past strokes (a learnable
    function of the stroke tokens), so a working model drives loss down — proving
    both the data plumbing and that the network can fit a frame/stroke->stroke map."""

    def __init__(self, n: int, cfg: SequencePolicyConfig, seed: int = 0):
        self.n = n
        self.cfg = cfg
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        c = self.cfg
        frames = rng.random((c.n_frames, c.img_ch, c.img_h, c.img_w), dtype=np.float32)
        past = rng.random((c.m_past, STROKE_DIM)).astype(np.float32)
        target = np.clip(past.mean(axis=0, keepdims=True), 0, 1)  # (1, STROKE_DIM)
        target = np.repeat(target, c.m_out, axis=0).astype(np.float32)  # (m_out, STROKE_DIM)
        mask = np.ones((c.m_past,), dtype=bool)
        return {
            "frames": torch.from_numpy(frames),
            "past_strokes": torch.from_numpy(past),
            "past_mask": torch.from_numpy(mask),
            "target": torch.from_numpy(target),
            "target_mask": torch.ones(c.m_out, dtype=torch.bool),
        }


def train(dataset: Dataset, cfg: SequencePolicyConfig, *, epochs: int, batch_size: int,
          lr: float, out_path: Path, smoke: bool = False) -> float:
    dev = _device()
    print(f"device={dev}  samples={len(dataset)}  epochs={epochs}  batch={batch_size}  "
          f"params: n_frames={cfg.n_frames} m_past={cfg.m_past} m_out={cfg.m_out} d={cfg.d_model}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = build_policy(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    last = float("nan")
    for ep in range(epochs):
        model.train()
        running = 0.0
        for batch in loader:
            frames = batch["frames"].to(dev)
            past = batch["past_strokes"].to(dev)
            mask = batch["past_mask"].to(dev)
            target = batch["target"].to(dev)
            target_mask = batch["target_mask"].to(dev)
            pred, activity_logits = model(frames, past, past_mask=mask)
            loss = stroke_loss(pred, target, target_mask, activity_logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            if smoke:
                break
        last = running / max(1, (1 if smoke else len(loader)))
        print(f"  epoch {ep + 1}/{epochs}  loss={last:.5f}")
        if smoke and ep >= 2:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config_to_dict(cfg),
                "stroke_dim": STROKE_DIM, "checkpoint_version": 2}, out_path)
    print(f"saved checkpoint → {out_path}")
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Model 2 (gesture-sequence policy).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Synthetic pipeline smoke test.")
    mode.add_argument("--data", type=Path, help="Real assembled-sequence dataset dir.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--m-past", type=int, default=4)
    ap.add_argument("--m-out", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--img-h", type=int, default=208,
                    help="Frame height fed to the encoder (portrait; ~2.16:1 with --img-w). Raise to compress less.")
    ap.add_argument("--img-w", type=int, default=96, help="Frame width fed to the encoder.")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "notebooks" / "models" / "sequence_model.pth")
    args = ap.parse_args()

    cfg = SequencePolicyConfig(
        n_frames=args.n_frames, m_past=args.m_past, m_out=args.m_out,
        d_model=args.d_model, img_h=args.img_h, img_w=args.img_w,
    )

    if args.smoke:
        ds: Dataset = SyntheticSequenceDataset(n=64, cfg=cfg)
        final = train(ds, cfg, epochs=max(3, args.epochs), batch_size=args.batch_size,
                      lr=args.lr, out_path=args.out, smoke=True)
        assert np.isfinite(final), "smoke loss is not finite"
        print(f"SMOKE OK — final loss {final:.5f}")
        return

    from trueskate_ai.bc.sequence_dataset import SequenceDataset  # noqa: PLC0415
    ds = SequenceDataset(args.data, cfg=cfg)
    train(ds, cfg, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_path=args.out)


if __name__ == "__main__":
    main()
