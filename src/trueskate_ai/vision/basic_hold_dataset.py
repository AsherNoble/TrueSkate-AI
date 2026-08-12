"""Strict, clip-level dataset for the additive basic Model 1 hold experiment.

This is intentionally separate from ``temporal_trace_dataset``.  It has one
target per clip -- ``{x, y, dur}`` -- rather than per-frame heatmaps, and admits
only calibrated one-finger stationary holds.  Existing broad Model 1 datasets
remain available to their original trainer.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HOLD_DURATION_MIN_S = 0.30
HOLD_DURATION_MAX_S = 1.50
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_IMAGE_HEIGHT = 288
DEFAULT_IMAGE_WIDTH = 128


def _frame_paths(sample: Path) -> tuple[Path, ...]:
    return tuple(sorted(sample.glob("frame_*.png")))


def _has_frames(sample: Path) -> bool:
    return bool(_frame_paths(sample)) or (sample / "frames.mp4").is_file()


def _decode_frames(sample: Path) -> list[np.ndarray]:
    paths = _frame_paths(sample)
    if paths:
        decoded = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    else:
        capture = cv2.VideoCapture(str(sample / "frames.mp4"))
        decoded = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded.append(frame)
        finally:
            capture.release()
    if not decoded or any(frame is None for frame in decoded):
        raise ValueError(f"{sample}: unreadable frames")
    return decoded


def _decode_even_frames(sample: Path, count: int) -> list[np.ndarray]:
    """Decode only evenly selected frames when the source is a compact video.

    Full video decode is correct but wasteful for clip regressors that always
    resample a fixed-length sequence.  Seeking to selected frame numbers keeps
    the same even temporal coverage while avoiding decode of ~30 unused frames
    per 32-frame aligned clip (and many more for legacy high-fps clips).
    """
    if count < 1:
        raise ValueError("count must be positive")
    paths = _frame_paths(sample)
    if paths:
        indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
        decoded = [cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR) for index in indices]
    else:
        capture = cv2.VideoCapture(str(sample / "frames.mp4"))
        try:
            total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            indices = np.linspace(0, total - 1, count).round().astype(int)
            decoded = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                decoded.append(frame if ok else None)
        finally:
            capture.release()
        # Some H.264 builds report a frame count but cannot reliably random-seek
        # every index.  The clip remains valid, so fall back to sequential decode
        # rather than rejecting a strict sample for decoder behavior alone.
        if any(frame is None for frame in decoded):
            decoded = _decode_frames(sample)
            indices = np.linspace(0, len(decoded) - 1, count).round().astype(int)
            decoded = [decoded[int(index)] for index in indices]
    if not decoded or any(frame is None for frame in decoded):
        raise ValueError(f"{sample}: unreadable selected frames")
    return decoded


def _valid_meta(sample: Path, meta: dict) -> str | None:
    if (sample / ".menu").exists():
        return "menu_marked"
    if str(meta.get("gesture_distribution", "")).casefold() != "hold":
        return "not_hold"
    if bool(meta.get("spin_active", False)):
        return "spin_active"
    if bool(meta.get("use_spin", False)) or meta.get("waypoints") is not None or meta.get("params") is not None:
        return "not_stationary"
    calibration = meta.get("tap_calibration")
    if not isinstance(calibration, dict) or not calibration.get("accepted"):
        return "uncalibrated"
    point = meta.get("point")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return "invalid_point"
    try:
        x, y = (float(point[0]), float(point[1]))
        duration = float(meta.get("hold_duration_s"))
    except (TypeError, ValueError):
        return "invalid_target"
    if not all(math.isfinite(v) for v in (x, y, duration)) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        return "invalid_target"
    if not HOLD_DURATION_MIN_S <= duration <= HOLD_DURATION_MAX_S:
        return "duration_out_of_range"
    if not _has_frames(sample):
        return "no_frames"
    return None


def discover_basic_hold_samples(root: Path) -> tuple[tuple[Path, ...], dict[str, int]]:
    """Return admissible samples and explicit, stable rejection counts."""
    kept: list[Path] = []
    stats: Counter = Counter()
    for meta_path in sorted(root.rglob("meta.json")):
        stats["discovered"] += 1
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            stats["invalid_json"] += 1
            continue
        reason = _valid_meta(meta_path.parent, meta)
        if reason:
            stats[f"rejected_{reason}"] += 1
            continue
        kept.append(meta_path.parent)
        stats["accepted"] += 1
    return tuple(kept), dict(sorted(stats.items()))


class BasicHoldClipDataset(Dataset):
    """One full video clip and its native-unit ``target=[x,y,duration]``."""

    def __init__(self, root: str | Path, *, sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                 image_height: int = DEFAULT_IMAGE_HEIGHT, image_width: int = DEFAULT_IMAGE_WIDTH):
        if sequence_length < 1 or image_height < 1 or image_width < 1:
            raise ValueError("sequence/image dimensions must be positive")
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.sample_paths, self.stats = discover_basic_hold_samples(self.root)
        self.segment_keys = tuple(self._segment_key(path) for path in self.sample_paths)
        self.command_keys = tuple(self._command_key(path) for path in self.sample_paths)

    @staticmethod
    def _segment_key(sample: Path) -> str:
        meta = json.loads((sample / "meta.json").read_text())
        # XCTest samples share one session directory, so directory ancestry alone
        # cannot prevent within-segment leakage.  The aligner stamps both values.
        session = meta.get("session")
        segment = meta.get("segment_index")
        if session is not None and segment is not None:
            return f"{session}:segment_{int(segment):05d}"
        return f"legacy:{sample.parent.parent.name}"

    @staticmethod
    def _command_key(sample: Path) -> str:
        """Stable identity for an exactly replayed stationary hold command."""
        meta = json.loads((sample / "meta.json").read_text())
        return ":".join(
            f"{float(value):.9f}"
            for value in (*meta["point"], float(meta["hold_duration_s"]))
        )

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.sample_paths[index]
        meta = json.loads((sample / "meta.json").read_text())
        source_frames = _decode_frames(sample)
        indices = np.linspace(0, len(source_frames) - 1, self.sequence_length).round().astype(int)
        frames: list[np.ndarray] = []
        for frame_index in indices:
            image = source_frames[int(frame_index)]
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        array = np.stack(frames).astype(np.float32) / 255.0
        return {
            "frames": torch.from_numpy(array).permute(0, 3, 1, 2),
            "target": torch.tensor(
                [float(meta["point"][0]), float(meta["point"][1]), float(meta["hold_duration_s"])],
                dtype=torch.float32,
            ),
        }


def _split_by_key(keys: tuple[str, ...], *, val_fraction: float, test_fraction: float,
                  seed: int) -> tuple[list[int], list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("validation/test fractions must be positive and sum to less than one")
    groups = sorted(set(keys))
    if len(groups) < 3:
        raise ValueError("need at least three independent groups for train/validation/test")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(groups))
    n_test = max(1, round(len(groups) * test_fraction))
    n_val = max(1, round(len(groups) * val_fraction))
    if n_test + n_val >= len(groups):
        n_val = 1
        n_test = 1
    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test:n_test + n_val])
    train = [i for i, key in enumerate(keys) if key not in test_groups | val_groups]
    val = [i for i, key in enumerate(keys) if key in val_groups]
    test = [i for i, key in enumerate(keys) if key in test_groups]
    return train, val, test


def split_by_segment(dataset: BasicHoldClipDataset, *, val_fraction: float = 0.15,
                     test_fraction: float = 0.15, seed: int = 0) -> tuple[list[int], list[int], list[int]]:
    """Deterministically split whole recording segments; never leak neighbors."""
    return _split_by_key(dataset.segment_keys, val_fraction=val_fraction,
                         test_fraction=test_fraction, seed=seed)


def split_by_command(dataset: BasicHoldClipDataset, *, val_fraction: float = 0.15,
                     test_fraction: float = 0.15, seed: int = 0) -> tuple[list[int], list[int], list[int]]:
    """Split by exact ``{x,y,dur}`` command, so replayed gestures never leak."""
    return _split_by_key(dataset.command_keys, val_fraction=val_fraction,
                         test_fraction=test_fraction, seed=seed)
