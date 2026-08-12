"""Strict clip-level dataset for MVP 2 finite-slope linear drags."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from trueskate_ai.data.gesture_sampling import (
    BASIC_LINEAR_MAX_ABS_SLOPE, BASIC_LINEAR_MAX_S, BASIC_LINEAR_MIN_DX,
    BASIC_LINEAR_MIN_S,
)
from trueskate_ai.vision.basic_hold_dataset import (
    DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, DEFAULT_SEQUENCE_LENGTH,
    _decode_even_frames, _has_frames, _split_by_key,
)


def _valid_meta(sample: Path, meta: dict) -> str | None:
    if (sample / ".menu").exists():
        return "menu_marked"
    if (sample / ".trace_mismatch").exists():
        return "trace_mismatch"
    if str(meta.get("gesture_distribution", "")).casefold() != "linear":
        return "not_linear"
    if bool(meta.get("spin_active", False)) or bool(meta.get("use_spin", False)):
        return "spin_active"
    calibration = meta.get("tap_calibration")
    if not isinstance(calibration, dict) or not calibration.get("accepted"):
        return "uncalibrated"
    points = meta.get("waypoints")
    if not isinstance(points, list) or len(points) != 2:
        return "not_two_point"
    try:
        (x0, y0), (x1, y1) = ((float(v) for v in point) for point in points)
        duration = float(meta["duration"])
        easing = float(meta.get("easing_power", 1.0))
    except (TypeError, ValueError, KeyError):
        return "invalid_target"
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1, duration, easing)):
        return "invalid_target"
    if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return "invalid_target"
    dx = x1 - x0
    if abs(dx) < BASIC_LINEAR_MIN_DX:
        return "near_vertical"
    if abs((y1 - y0) / dx) > BASIC_LINEAR_MAX_ABS_SLOPE + 1e-6:
        return "slope_out_of_range"
    if not BASIC_LINEAR_MIN_S <= duration <= BASIC_LINEAR_MAX_S:
        return "duration_out_of_range"
    if abs(easing - 1.0) > 1e-6:
        return "not_constant_velocity"
    if not _has_frames(sample):
        return "no_frames"
    return None


def discover_basic_linear_samples(root: Path) -> tuple[tuple[Path, ...], dict[str, int]]:
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
        else:
            kept.append(meta_path.parent)
            stats["accepted"] += 1
    return tuple(kept), dict(sorted(stats.items()))


class BasicLinearClipDataset(Dataset):
    """One calibrated straight drag clip, target ``[x0,y0,x1,y1,duration]``."""

    def __init__(self, root: str | Path, *, sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                 image_height: int = DEFAULT_IMAGE_HEIGHT, image_width: int = DEFAULT_IMAGE_WIDTH,
                 cache_frames: bool = False):
        if sequence_length < 1 or image_height < 1 or image_width < 1:
            raise ValueError("sequence/image dimensions must be positive")
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.cache_frames = cache_frames
        # A 32×288×128 float clip is ~14 MiB; caching 1,000 of those would
        # consume almost the entire 16 GiB Modal worker before the model or a
        # batch exists. Store decoded RGB uint8 frames (~3.5 GiB for 1,000) and
        # normalize only the selected sample returned to the trainer.
        self._frame_cache: dict[Path, torch.Tensor] = {}
        self.sample_paths, self.stats = discover_basic_linear_samples(self.root)
        self.segment_keys = tuple(self._segment_key(path) for path in self.sample_paths)
        self.command_keys = tuple(self._command_key(path) for path in self.sample_paths)

    @staticmethod
    def _meta(sample: Path) -> dict:
        return json.loads((sample / "meta.json").read_text())

    @classmethod
    def _segment_key(cls, sample: Path) -> str:
        meta = cls._meta(sample)
        if meta.get("session") is not None and meta.get("segment_index") is not None:
            return f"{meta['session']}:segment_{int(meta['segment_index']):05d}"
        return f"legacy:{sample.parent.parent.name}"

    @classmethod
    def _command_key(cls, sample: Path) -> str:
        meta = cls._meta(sample)
        return ":".join(f"{float(v):.9f}" for point in meta["waypoints"] for v in point) + \
            f":{float(meta['duration']):.9f}"

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import cv2
        import numpy as np
        sample = self.sample_paths[index]
        meta = self._meta(sample)
        cached = self._frame_cache.get(sample)
        if cached is None:
            source = _decode_even_frames(sample, self.sequence_length)
            frames = [cv2.cvtColor(
                image if image.shape[:2] == (self.image_height, self.image_width)
                else cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            ) for image in source]
            array = np.stack(frames)
            cached = torch.from_numpy(array).permute(0, 3, 1, 2)
            if self.cache_frames:
                self._frame_cache[sample] = cached
        target = [value for point in meta["waypoints"] for value in point] + [meta["duration"]]
        # The manifest times are touch-start-relative and the command is a
        # constant-velocity two-point drag.  Preserve its exact per-frame
        # trajectory as optional training supervision for score-map ablations;
        # inactive lead-in/post-liftoff frames are explicitly masked.
        raw_times = np.asarray(meta.get("frame_times", []), dtype=np.float32)
        if raw_times.ndim != 1 or len(raw_times) == 0 or not np.isfinite(raw_times).all():
            raise ValueError(f"{sample}: finite frame_times are required for a linear trajectory")
        selected = np.linspace(0, len(raw_times) - 1, self.sequence_length).round().astype(int)
        times = raw_times[selected]
        (x0, y0), (x1, y1) = meta["waypoints"]
        duration = float(meta["duration"])
        fraction = np.clip(times / duration, 0.0, 1.0)
        path = np.stack((float(x0) + fraction * (float(x1) - float(x0)),
                         float(y0) + fraction * (float(y1) - float(y0))), axis=1)
        active = (times >= 0.0) & (times <= duration)
        return {"frames": cached.float().div_(255.0),
                "target": torch.tensor(target, dtype=torch.float32),
                "trajectory_xy": torch.from_numpy(path.astype(np.float32)),
                "trajectory_mask": torch.from_numpy(active)}


def split_by_segment(dataset: BasicLinearClipDataset, *, val_fraction: float = .15,
                     test_fraction: float = .15, seed: int = 0):
    return _split_by_key(dataset.segment_keys, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)


def split_by_command(dataset: BasicLinearClipDataset, *, val_fraction: float = .15,
                     test_fraction: float = .15, seed: int = 0):
    return _split_by_key(dataset.command_keys, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
