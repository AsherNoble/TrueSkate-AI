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

import cv2  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from PIL import Image  # noqa: E402

# Validated 2026-06-14: True Skate's orange finger-trace lags the flick — the
# swoosh peaks ~0.4-0.5s AFTER touch release. At latency_s≈0.45 the known-touch
# labels align with the visible trace in ~80% of frames (vs ~1% at 0.0). See
# experiments/vision_sequence_leap_journal.md.
_DEFAULT_LATENCY_S = 0.45
_TRACE_WARM_THRESHOLD = 200  # min warm-orange px near the label to count as trace-aligned

from trueskate_ai.vision.self_label import label_frames  # noqa: E402
from gaussian_bump_predictor import GaussianBumpPredictor, GaussianBumpLoss  # noqa: E402

_H, _W = 288, 128          # working resolution (≈ portrait 2.25:1; still resolves the trace, ~3x faster than 416x192)
_HEATMAP_SIGMA = 6.0
_NEG_KEEP_FRAC = 0.2       # keep this fraction of inactive (no-touch) frames as negatives


def _warm_img(img: np.ndarray, nx: float, ny: float, r: int = 45) -> int:
    """Count warm-orange (trace) pixels within r px of normalised (nx, ny) in a BGR image."""
    H, W = img.shape[:2]
    px, py = int(nx * W), int(ny * H)
    x0, x1, y0, y1 = max(0, px - r), min(W, px + r), max(0, py - r), min(H, py + r)
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    return int(((hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 140)).sum())


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

    def __init__(self, session_dir: str | Path, *, latency_s: float = _DEFAULT_LATENCY_S,
                 require_trace: bool = True, rng_seed: int = 0):
        # Each candidate frame is loaded from disk ONCE here (also needed for the
        # trace gate) and cached resized in memory, so training epochs do no disk
        # I/O — full-res-PNG-per-item was ~6 min/epoch; this is seconds.
        self._frames: list[np.ndarray] = []   # uint8 [H,W,3] RGB
        self._heatmaps: list[np.ndarray] = []  # float16 [H,W]
        rng = np.random.default_rng(rng_seed)
        kept_pos = gated = neg = 0
        root = Path(session_dir)
        sample_dirs = sorted(root.glob("sample_*")) or sorted(root.glob("*/sample_*"))
        for sample_dir in sample_dirs:
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
                if not lab.active:
                    if rng.random() <= _NEG_KEEP_FRAC:
                        img = cv2.imread(str(frame_path))
                        if img is not None:
                            self._cache(img, -1.0, -1.0); neg += 1
                    continue
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                # Trace-presence gate: keep an active frame only if the orange
                # trace actually appears near the (latency-shifted) label.
                if require_trace and _warm_img(img, lab.x, lab.y) < _TRACE_WARM_THRESHOLD:
                    gated += 1
                    continue
                self._cache(img, lab.x, lab.y); kept_pos += 1
        print(f"  dataset: {kept_pos} trace-aligned positives + {neg} negatives kept, "
              f"{gated} gated (latency={latency_s}s, require_trace={require_trace})")
        if not self._frames:
            raise RuntimeError(f"No labeled frames found under {session_dir}")

    def _cache(self, bgr: np.ndarray, nx: float, ny: float) -> None:
        rgb = cv2.cvtColor(cv2.resize(bgr, (_W, _H)), cv2.COLOR_BGR2RGB)
        self._frames.append(rgb.astype(np.uint8))
        hm = make_heatmap(nx * _W, ny * _H, _H, _W) if nx >= 0 else np.zeros((_H, _W), np.float32)
        self._heatmaps.append(hm.astype(np.float16))

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int):
        frame = torch.from_numpy(self._frames[idx].astype(np.float32) / 255.0).permute(2, 0, 1)
        heatmap = torch.from_numpy(self._heatmaps[idx].astype(np.float32)).unsqueeze(0)
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
          out_path: Path, base_channels: int = 32, smoke: bool = False) -> None:
    dev = _device()
    print(f"device={dev}  samples={len(dataset)}  epochs={epochs}  batch={batch_size}  base_ch={base_channels}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = GaussianBumpPredictor(in_channels=3, base_channels=base_channels).to(dev)
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
                "sigma": _HEATMAP_SIGMA, "base_channels": base_channels}, out_path)
    print(f"saved checkpoint → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the learned trace extractor (Model 1).")
    ap.add_argument("--data", type=Path, default=None, help="self_labeled_traces session dir")
    ap.add_argument("--smoke", action="store_true", help="synthetic pipeline check, no corpus needed")
    ap.add_argument("--latency-s", type=float, default=_DEFAULT_LATENCY_S,
                    help="frame->touch trace lag compensation (validated ~0.45s)")
    ap.add_argument("--no-require-trace", action="store_true",
                    help="keep active frames even without a visible trace at the label")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--base-channels", type=int, default=32, help="U-Net width (16 = ~4x faster)")
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
    ds = SelfLabeledTraceDataset(args.data, latency_s=args.latency_s,
                                 require_trace=not args.no_require_trace)
    train(ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_path=args.out,
          base_channels=args.base_channels)


if __name__ == "__main__":
    main()
