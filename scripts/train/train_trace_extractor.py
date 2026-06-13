"""Train Model 1: the learned trace extractor (frame -> touch heatmap).

Consumes the agent self-labeled corpus from collect_self_labeled_traces.py
(sample_*/frame_*.png + meta.json), turns each frame's timestamp into a
ground-truth touch heatmap via vision/self_label, and trains the existing
GaussianBumpPredictor U-Net to predict it. This is the model that replaces the
unreliable hand-tuned trace_extractor — and, once trained, labels Asher's
expert play to feed Model 2 (the sequence leap). See
experiments/vision_sequence_leap_journal.md.

Run modes:
    --smoke              verify the full pipeline (dataset->model->loss->step)
                         on synthetic data, no collected corpus needed.
    --data <dir>         train on a real self_labeled_traces session dir.

Usage:
    python scripts/train/train_trace_extractor.py --smoke
    python scripts/train/train_trace_extractor.py --data data/self_labeled_traces/iPhone_XR_<ts> --epochs 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for p in (_REPO_ROOT / "src", _REPO_ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from PIL import Image  # noqa: E402

from trueskate_ai.vision.self_label import label_frames  # noqa: E402
from gaussian_bump_predictor import GaussianBumpPredictor, GaussianBumpLoss  # noqa: E402

_H, _W = 416, 192          # working resolution (≈ portrait 812:375 aspect)
_HEATMAP_SIGMA = 6.0
_NEG_KEEP_FRAC = 0.2       # keep this fraction of inactive (no-touch) frames as negatives


def make_heatmap(x: float, y: float, H: int, W: int, sigma: float = _HEATMAP_SIGMA) -> np.ndarray:
    """(H, W) float32 Gaussian centred at pixel (x, y). All-zero if x < 0."""
    if x is None or x < 0:
        return np.zeros((H, W), dtype=np.float32)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.exp(-(((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SelfLabeledTraceDataset(Dataset):
    """(color frame -> touch heatmap) from a self_labeled_traces session dir.

    Each sample dir contributes one item per captured frame: the frame image
    paired with a Gaussian heatmap at the ground-truth touch position computed
    from the known gesture + the frame timestamp (latency_s tunable).
    """

    def __init__(self, session_dir: str | Path, *, latency_s: float = 0.0,
                 rng_seed: int = 0):
        self.items: list[tuple[Path, float, float, bool]] = []
        rng = np.random.default_rng(rng_seed)
        for sample_dir in sorted(Path(session_dir).glob("sample_*")):
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            waypoints = [tuple(p) for p in meta["waypoints"]]
            labels = label_frames(
                waypoints, meta["duration"], meta["easing_power"],
                meta["frame_times"], latency_s=latency_s,
                spin_active=meta.get("spin_active", False),
            )
            for fi, lab in enumerate(labels):
                frame_path = sample_dir / f"frame_{fi:03d}.png"
                if not frame_path.exists():
                    continue
                if not lab.active and rng.random() > _NEG_KEEP_FRAC:
                    continue  # subsample no-touch negatives
                self.items.append((frame_path, lab.x, lab.y, lab.active))
        if not self.items:
            raise RuntimeError(f"No labeled frames found under {session_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        frame_path, x, y, active = self.items[idx]
        img = Image.open(frame_path).convert("RGB").resize((_W, _H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        frame = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
        if active and x >= 0:
            hm = make_heatmap(x * _W, y * _H, _H, _W, sigma=_HEATMAP_SIGMA)
        else:
            hm = np.zeros((_H, _W), dtype=np.float32)
        heatmap = torch.from_numpy(hm).unsqueeze(0)  # [1,H,W]
        return {"image": frame, "heatmap": heatmap}


class _SyntheticTraceDataset(Dataset):
    """Tiny synthetic dataset for --smoke: random frame + heatmap at a random point."""

    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(idx)
        frame = torch.from_numpy(rng.random((3, _H, _W), dtype=np.float32))
        x, y = rng.uniform(0, _W), rng.uniform(0, _H)
        hm = make_heatmap(x, y, _H, _W, sigma=_HEATMAP_SIGMA)
        return {"image": frame, "heatmap": torch.from_numpy(hm).unsqueeze(0)}


def train(dataset: Dataset, *, epochs: int, batch_size: int, lr: float,
          out_path: Path, smoke: bool = False) -> None:
    dev = _device()
    print(f"device={dev}  samples={len(dataset)}  epochs={epochs}  batch={batch_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = GaussianBumpPredictor(in_channels=3, base_channels=32).to(dev)
    loss_fn = GaussianBumpLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for batch in loader:
            img = batch["image"].to(dev)
            target = batch["heatmap"].to(dev)
            pred = model(img)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            if smoke:
                break  # one step is enough to prove the pipeline
        print(f"  epoch {ep + 1}/{epochs}  loss={running / max(1, len(loader)):.5f}")
        if smoke:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "h": _H, "w": _W,
                "sigma": _HEATMAP_SIGMA}, out_path)
    print(f"saved checkpoint → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the learned trace extractor (Model 1).")
    ap.add_argument("--data", type=Path, default=None, help="self_labeled_traces session dir")
    ap.add_argument("--smoke", action="store_true", help="synthetic pipeline check, no corpus needed")
    ap.add_argument("--latency-s", type=float, default=0.0, help="frame->touch latency compensation")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "notebooks" / "models" / "trace_extractor_v1.pth")
    args = ap.parse_args()

    if args.smoke:
        ds: Dataset = _SyntheticTraceDataset(n=8)
        train(ds, epochs=1, batch_size=4, lr=args.lr,
              out_path=_REPO_ROOT / "tmp" / "trace_extractor_smoke.pth", smoke=True)
        print("SMOKE OK: dataset → U-Net → GaussianBumpLoss → optimizer step all run.")
        return

    if args.data is None:
        raise SystemExit("Provide --data <session_dir> or --smoke")
    ds = SelfLabeledTraceDataset(args.data, latency_s=args.latency_s)
    train(ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_path=args.out)


if __name__ == "__main__":
    main()
