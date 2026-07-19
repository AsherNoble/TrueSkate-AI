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
import hashlib
import itertools
import math
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

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


def _build_heatmap(nx: float, ny: float, sx: float, sy: float) -> np.ndarray:
    """Label heatmap: drag bump (nx, ny) ∪ spin-button bump (sx, sy), max-combined.
    A coord pair < 0 = that bump absent; both absent = all-zero (negative frame)."""
    hm = make_heatmap(nx * _W, ny * _H, _H, _W) if nx >= 0 else np.zeros((_H, _W), np.float32)
    if sx >= 0:
        hm = np.maximum(hm, make_heatmap(sx * _W, sy * _H, _H, _W))
    return hm


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _start_relative_frame_times(meta: dict) -> list[float]:
    """frame_times relative to gesture START (what label_frames expects).

    `self_labeled_traces` store times relative to gesture START
    (`gesture_start_monotonic`). The SLS / XCTest collectors store them relative
    to gesture END (`gesture_end_monotonic` / `gesture_video_time_s`, with a
    `capture_offset_s`). For an end-relative frame at fe, the start-relative time
    is fe + duration (start = end − duration). The capture→pixel Δ is already
    folded into the SLS/XCTest frame_times by the aligner; the residual render
    lag stays in `latency_s`.
    """
    ft = meta["frame_times"]
    end_relative = (
        any(k in meta for k in ("gesture_end_monotonic", "gesture_video_time_s"))
        or ("capture_offset_s" in meta and "gesture_start_monotonic" not in meta)
    )
    if end_relative:
        # payload_total_s: a spin_flick payload outlasts its drag (held spin
        # button), and the end anchor is the PAYLOAD end, not the drag end.
        total = float(meta.get("payload_total_s", meta["duration"]))
        return [float(t) + total for t in ft]
    return [float(t) for t in ft]


class SelfLabeledTraceDataset(Dataset):
    """(color frame -> touch heatmap) from a self_labeled_traces session dir.

    Each sample dir contributes one item per captured frame: the frame image
    paired with a Gaussian heatmap at the ground-truth touch position computed
    from the known gesture + the frame timestamp (latency_s tunable).

    spin_flick samples add a SECOND bump at the spin-button coord while the
    rotate button is held (meta: spin_active + spin_hold_start_s/end_s) — at
    inference, spin state = heatmap mass near the button. Hold-window frames
    past the drag are spin-only positives, never negatives.

    By default this dataset does NOT preload full-resolution frames into memory;
    it collects frame paths and label params and loads/processes frames on-the-fly
    in __getitem__. For small datasets or tests set cache_frames=True to preserve
    the legacy behaviour of caching preprocessed frames & heatmaps.
    """

    def __init__(self, session_dir: str | Path, *, latency_s: float = _DEFAULT_LATENCY_S,
                 require_trace: bool = True, rng_seed: int = 0, cache_frames: bool = False,
                 include_path_term: str | None = None, max_samples: int | None = None,
                 negative_keep_frac: float = _NEG_KEEP_FRAC, allow_empty: bool = False,
                 detect_menu_frames: bool = False):
        self.cache_frames = cache_frames
        # Retained for sample-level splitting in both cached and streaming mode.
        self._frame_paths: list[Path] = []
        self._heatmap_params: list[tuple[float, float, float, float]] = []
        if cache_frames:
            # Legacy behaviour: keep preprocessed frames & heatmaps in memory
            self._frames: list[np.ndarray] = []   # uint8 [H,W,3] RGB (preprocessed)
            self._heatmaps: list[np.ndarray] = []  # float16 [H,W]
        else:
            # Memory-efficient mode keeps only paths and label params.
            pass

        rng = np.random.default_rng(rng_seed)
        # lazy imports to avoid heavy deps (e.g., selenium) when running --smoke
        from trueskate_ai.vision.self_label import label_frames  # noqa: E402
        from trueskate_ai.sim.gestures import DEFAULT_SPIN_BUTTON_XY  # noqa: E402
        kept_pos = spin_pos = gated = neg = 0
        root = Path(session_dir)
        if include_path_term:
            needle = "".join(c for c in include_path_term.lower() if c.isalnum())
            matches = lambda p: needle in "".join(c for c in str(p).lower() if c.isalnum())
            # The Modal volume has ~1.8M files.  A root.rglob followed by a
            # string filter scans every inode.  Corpus layout is bounded:
            # root/session/park/sample_* (or legacy session/sample_*), so prune
            # at the session/park directory before enumerating samples.
            park_roots = []
            if matches(root):
                park_roots.append(root)
            for level1 in (p for p in root.iterdir() if p.is_dir()):
                if matches(level1):
                    park_roots.append(level1)
                else:
                    for level2 in (p for p in level1.iterdir() if p.is_dir()):
                        if matches(level2):
                            park_roots.append(level2)
            park_roots.sort(key=lambda p: hashlib.sha256(str(p.relative_to(root)).encode()).digest())
            if max_samples is None:
                sample_dirs = [p for park in park_roots for p in park.glob("sample_*") if p.is_dir()]
            else:
                # Spread the budget across sessions/park dirs and stop each
                # directory iterator at its quota. Use several samples per
                # selected session: sample_00000 alone is often a non-flick
                # warm-up action and is not representative of the normal mix.
                n_parks = min(len(park_roots), max(1, math.ceil(math.sqrt(max_samples))))
                selected_parks = park_roots[:n_parks]
                quota = max(1, math.ceil(max_samples / max(1, len(selected_parks))))
                sample_dirs = [p for park in selected_parks
                               for p in itertools.islice((x for x in park.glob("sample_*") if x.is_dir()), quota)]
                sample_dirs = sample_dirs[:max_samples]
            sample_dirs.sort()
        else:
            # rglob handles flat legacy and nested XCTest corpora uniformly.
            sample_dirs = sorted(p for p in root.rglob("sample_*") if p.is_dir())
        if not include_path_term and max_samples is not None and len(sample_dirs) > max_samples:
            # Hash selection is stable across directory enumeration and avoids
            # taking one contiguous device/session/time slice.
            sample_dirs = sorted(sample_dirs,
                                 key=lambda p: hashlib.sha256(str(p.relative_to(root)).encode()).digest())[:max_samples]
            sample_dirs.sort()
        skipped_nonflick = skipped_menu = skipped_editor = 0
        for sample_dir in sample_dirs:
            # flag_menu_samples.py marks replay/menu-contaminated samples with a
            # `.menu` file; those frames aren't real gameplay and must be excluded.
            if (sample_dir / ".menu").exists():
                skipped_menu += 1
                continue
            # flag_editor_samples.py marks park-editor-contaminated samples with
            # `.editor`; those are not live gameplay and must be excluded too.
            if (sample_dir / ".editor").exists():
                skipped_editor += 1
                continue
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            if "waypoints" not in meta:
                # Model 1 predicts ONE touch heatmap → only single-touch flick
                # samples apply; params/nslot/recipe (multi-gesture) are skipped.
                skipped_nonflick += 1
                continue
            if detect_menu_frames:
                from trueskate_ai.vision.gameplay_filter import is_menu_frame
                candidate_frames = sorted(sample_dir.glob("frame_*.png"))
                mid = len(candidate_frames) // 2
                priority = ([candidate_frames[mid], candidate_frames[0], candidate_frames[-1]]
                            if candidate_frames else [])
                # Training-volume fast path: middle/first/last matched exhaustive
                # detection on 163/163 audited contaminated legacy samples, while
                # avoiding ~10 FUSE reads per gesture. The offline flagger remains
                # exhaustive when permanently marking a corpus.
                if any(is_menu_frame(frame) for frame in dict.fromkeys(priority)):
                    skipped_menu += 1
                    continue
            waypoints = [tuple(p) for p in meta["waypoints"]]
            # spin_flick: the held rotate button is a SECOND labelled touch at
            # the spin-button coord for its hold window — never unlabelled noise.
            spin_hold = None
            if meta.get("spin_active") and meta.get("spin_hold_start_s") is not None:
                spin_hold = (float(meta["spin_hold_start_s"]), float(meta["spin_hold_end_s"]))
            sxy = meta.get("spin_button_xy") or DEFAULT_SPIN_BUTTON_XY
            labels = label_frames(
                waypoints, meta["duration"], meta["easing_power"],
                _start_relative_frame_times(meta), latency_s=latency_s,
                spin_hold=spin_hold,
            )
            for fi, lab in enumerate(labels):
                frame_path = sample_dir / f"frame_{fi:03d}.png"
                if not frame_path.exists():
                    continue
                if not lab.active and not lab.spin_on:
                    if rng.random() <= negative_keep_frac and self._add(frame_path, -1.0, -1.0, -1.0, -1.0):
                        neg += 1
                    continue
                drag_x = drag_y = -1.0
                if lab.active:
                    # The warm/trace gate applies to the DRAG bump only: a held
                    # spin button renders no orange swoosh, so gating it would
                    # erase every spin label.
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        continue
                    if require_trace and _warm_img(img, lab.x, lab.y) < _TRACE_WARM_THRESHOLD:
                        gated += 1
                        if not lab.spin_on:
                            continue  # pure drag positive without a trace — drop, as before
                        # drag bump dropped; the frame stays as a spin-only positive
                    else:
                        drag_x, drag_y = lab.x, lab.y
                sx, sy = (float(sxy[0]), float(sxy[1])) if lab.spin_on else (-1.0, -1.0)
                if not self._add(frame_path, drag_x, drag_y, sx, sy):
                    continue
                if drag_x >= 0:
                    kept_pos += 1
                if lab.spin_on:
                    spin_pos += 1
        print(f"  dataset: {kept_pos} trace-aligned positives + {spin_pos} spin-hold positives "
              f"+ {neg} negatives kept, {gated} gated, {skipped_nonflick} non-flick skipped, "
              f"{skipped_menu} menu/replay skipped, {skipped_editor} editor skipped "
              f"(latency={latency_s}s, require_trace={require_trace}, cache_frames={cache_frames})")
        self.stats = {"trace_positives": kept_pos, "spin_positives": spin_pos,
                      "negatives": neg, "gated": gated, "samples_considered": len(sample_dirs),
                      "nonflick_skipped": skipped_nonflick, "menu_skipped": skipped_menu,
                      "editor_skipped": skipped_editor}
        if cache_frames:
            if not self._frames and not allow_empty:
                raise RuntimeError(f"No labeled frames found under {session_dir}")
        else:
            if not self._frame_paths and not allow_empty:
                raise RuntimeError(f"No labeled frames found under {session_dir}")

    def _add(self, frame_path: Path, nx: float, ny: float, sx: float, sy: float) -> bool:
        """Store one frame + label params; a coord pair < 0 = that bump absent."""
        if self.cache_frames:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                return False
            # preprocess + store small tensors in memory (legacy path)
            from trueskate_ai.bc.frame_prep import prep_frame_rgb  # lazy import
            self._frames.append(prep_frame_rgb(bgr, _H, _W, normalize=False))
            self._heatmaps.append(_build_heatmap(nx, ny, sx, sy).astype(np.float16))
        self._frame_paths.append(frame_path)
        self._heatmap_params.append((nx, ny, sx, sy))
        return True

    def __len__(self) -> int:
        return len(self._frames) if self.cache_frames else len(self._frame_paths)

    def __getitem__(self, idx: int):
        if self.cache_frames:
            frame = torch.from_numpy(self._frames[idx].astype(np.float32) / 255.0).permute(2, 0, 1)
            heatmap = torch.from_numpy(self._heatmaps[idx].astype(np.float32)).unsqueeze(0)
            return {"image": frame, "heatmap": heatmap}
        # load frame on-the-fly and compute heatmap from stored normalized params
        frame_path = self._frame_paths[idx]
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            raise RuntimeError(f"Failed to read frame {frame_path}")
        from trueskate_ai.bc.frame_prep import prep_frame_rgb  # lazy import
        proc = prep_frame_rgb(bgr, _H, _W, normalize=False)
        frame = torch.from_numpy(proc.astype(np.float32) / 255.0).permute(2, 0, 1)
        nx, ny, sx, sy = self._heatmap_params[idx]
        heatmap = torch.from_numpy(_build_heatmap(nx, ny, sx, sy).astype(np.float32)).unsqueeze(0)
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


def localization_metrics(model, loader, device, *, peak_threshold: float = 0.3,
                         tolerance: float = 0.05) -> dict[str, float | int]:
    """Balanced localization metrics for active and inactive frames.

    Distance is measured in normalized screen coordinates, so ``tolerance=.05``
    means within 5% of the screen diagonal and is resolution-independent. The
    checkpoint metric is the mean of active localization accuracy and inactive
    specificity; plain overall accuracy would be dominated by inactive frames.
    """
    model.eval()
    pos_correct = pos_total = neg_correct = neg_total = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["image"].to(device))[:, 0].cpu().numpy()
            target = batch["heatmap"][:, 0].numpy()
            for p, y in zip(pred, target):
                peak = float(p.max())
                if float(y.max()) < 0.5:
                    neg_correct += int(peak < peak_threshold)
                    neg_total += 1
                else:
                    localized = False
                    if peak >= peak_threshold:
                        py, px = np.unravel_index(int(p.argmax()), p.shape)
                        ty, tx = np.unravel_index(int(y.argmax()), y.shape)
                        distance = np.hypot((px - tx) / p.shape[1], (py - ty) / p.shape[0])
                        localized = bool(distance <= tolerance)
                    pos_correct += int(localized)
                    pos_total += 1
    pos_accuracy = pos_correct / max(1, pos_total)
    neg_accuracy = neg_correct / max(1, neg_total)
    present_classes = (int(pos_total > 0) + int(neg_total > 0))
    balanced = ((pos_accuracy if pos_total else 0.0) +
                (neg_accuracy if neg_total else 0.0)) / max(1, present_classes)
    total = pos_total + neg_total
    overall = (pos_correct + neg_correct) / max(1, total)
    return {"balanced_accuracy": float(balanced), "overall_accuracy": float(overall),
            "positive_accuracy": float(pos_accuracy), "negative_accuracy": float(neg_accuracy),
            "positive_frames": pos_total, "negative_frames": neg_total, "frames": total}


def split_by_sample(dataset: SelfLabeledTraceDataset, *, val_fraction: float = 0.15,
                    seed: int = 0, train_negative_keep_frac: float = _NEG_KEEP_FRAC):
    """Deterministic train/validation split with whole gestures kept together."""
    from torch.utils.data import Subset

    groups: dict[Path, list[int]] = {}
    for i, path in enumerate(dataset._frame_paths):
        groups.setdefault(path.parent, []).append(i)
    if len(groups) < 2:
        raise RuntimeError(f"need at least two gesture samples to split, found {len(groups)}")
    ranked = sorted(groups, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).digest())
    n_val = max(1, min(len(ranked) - 1, round(len(ranked) * val_fraction)))
    val_groups = set(ranked[:n_val])
    train_idx = []
    for path, indices in groups.items():
        if path in val_groups:
            continue
        for i in indices:
            params = dataset._heatmap_params[i]
            is_negative = all(v < 0 for v in params)
            keep_key = int.from_bytes(hashlib.sha256(f"train-neg:{seed}:{dataset._frame_paths[i]}".encode()).digest()[:8], "big")
            keep_negative = keep_key / 2**64 < train_negative_keep_frac
            if not is_negative or keep_negative:
                train_idx.append(i)
    val_idx = [i for p, indices in groups.items() if p in val_groups for i in indices]
    return Subset(dataset, train_idx), Subset(dataset, val_idx), len(groups) - n_val, n_val


def train(dataset: Dataset, *, epochs: int, batch_size: int, lr: float,
          out_path: Path, base_channels: int = 32, smoke: bool = False,
          val_dataset: Dataset | None = None, resume_path: Path | None = None,
          peak_threshold: float = 0.3, tolerance: float = 0.05,
          checkpoint_callback=None, target_accuracy: float | None = None) -> None:
    dev = _device()
    print(f"device={dev}  samples={len(dataset)}  epochs={epochs}  batch={batch_size}  base_ch={base_channels}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if smoke:
        # tiny smoke model to avoid importing heavy extras (matplotlib, etc.)
        import torch.nn as nn
        model = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        ).to(dev)
        loss_fn = nn.MSELoss()
    else:
        from trueskate_ai.vision.gaussian_bump_predictor import GaussianBumpPredictor, GaussianBumpLoss  # lazy import
        model = GaussianBumpPredictor(in_channels=3, base_channels=base_channels).to(dev)
        loss_fn = GaussianBumpLoss()
        if resume_path is not None:
            ckpt = torch.load(resume_path, map_location=dev, weights_only=False)
            if int(ckpt.get("base_channels", base_channels)) != base_channels:
                raise ValueError(f"--base-channels {base_channels} does not match {resume_path}")
            model.load_state_dict(ckpt["model_state"])
            print(f"resumed weights from {resume_path}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    val_loader = (DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                  if val_dataset is not None else None)
    best_accuracy = -1.0

    for ep in range(epochs):
        model.train()
        running = 0.0
        step_count = 0
        for batch in loader:
            img = batch["image"].to(dev)
            target = batch["heatmap"].to(dev)
            pred = model(img)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            step_count += 1
            if smoke:
                break  # one step is enough to prove the pipeline
        avg_loss = (running / step_count) if step_count > 0 else 0.0
        suffix = ""
        if val_loader is not None:
            metrics = localization_metrics(
                model, val_loader, dev, peak_threshold=peak_threshold, tolerance=tolerance)
            accuracy = float(metrics["balanced_accuracy"])
            suffix = (f"  val_balanced_accuracy={100 * accuracy:.2f}% "
                      f"(positive={100 * float(metrics['positive_accuracy']):.2f}%/"
                      f"{metrics['positive_frames']}, negative={100 * float(metrics['negative_accuracy']):.2f}%/"
                      f"{metrics['negative_frames']})")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "h": _H, "w": _W,
                            "sigma": _HEATMAP_SIGMA, "base_channels": base_channels,
                            "val_accuracy": float(accuracy), "peak_threshold": peak_threshold,
                            "val_metrics": metrics, "localization_tolerance": tolerance,
                            "epoch": ep + 1}, out_path)
                if checkpoint_callback is not None:
                    checkpoint_callback()
        print(f"  epoch {ep + 1}/{epochs}  loss={avg_loss:.5f}  (steps={step_count}){suffix}")
        if smoke:
            break
        if target_accuracy is not None and val_loader is not None and accuracy >= target_accuracy:
            print(f"target reached: {100 * accuracy:.2f}% >= {100 * target_accuracy:.2f}%")
            break

    if val_loader is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "h": _H, "w": _W,
                    "sigma": _HEATMAP_SIGMA, "base_channels": base_channels}, out_path)
    print(f"saved checkpoint → {out_path}")


def main() -> None:
    global _H, _W
    ap = argparse.ArgumentParser(description="Train the learned trace extractor (Model 1).")
    ap.add_argument("--data", type=Path, default=None, help="self_labeled_traces session dir")
    ap.add_argument("--val-data", type=Path, default=None,
                    help="held-out session dir; best localization-accuracy checkpoint is saved")
    ap.add_argument("--smoke", action="store_true", help="synthetic pipeline check, no corpus needed")
    ap.add_argument("--latency-s", type=float, default=_DEFAULT_LATENCY_S,
                    help="frame->touch trace lag compensation (validated ~0.45s)")
    ap.add_argument("--no-require-trace", action="store_true",
                    help="keep active frames even without a visible trace at the label")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--base-channels", type=int, default=32, help="U-Net width (16 = ~4x faster)")
    ap.add_argument("--img-h", type=int, default=_H, help="working frame height (must be /16)")
    ap.add_argument("--img-w", type=int, default=_W, help="working frame width (must be /16)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", type=Path, default=None, help="fine-tune an existing Model 1 checkpoint")
    ap.add_argument("--peak-threshold", type=float, default=0.3)
    ap.add_argument("--localization-tolerance", type=float, default=0.05,
                    help="normalized screen-diagonal distance for a correct active-frame prediction")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "notebooks" / "models" / "trace_extractor_v1.pth")
    args = ap.parse_args()
    _H, _W = args.img_h, args.img_w

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
    val_ds = (SelfLabeledTraceDataset(args.val_data, latency_s=args.latency_s,
                                      require_trace=not args.no_require_trace, rng_seed=1,
                                      negative_keep_frac=1.0)
              if args.val_data is not None else None)
    train(ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_path=args.out,
          base_channels=args.base_channels, val_dataset=val_ds, resume_path=args.resume,
          peak_threshold=args.peak_threshold, tolerance=args.localization_tolerance)


if __name__ == "__main__":
    main()
