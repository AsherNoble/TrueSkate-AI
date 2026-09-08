"""Sequence-preserving supervision for the causal Model 1 touch tracker.

Each corpus ``sample_*`` directory is one independent recurrent sequence.  A
sequence is never joined to another gesture and is never truncated: shorter
samples are padded at the end, while longer samples fail with an actionable
error.  Gesture metadata is decoded back into the touch schedule that was sent
to WDA, including overlapping N-slot drags and the optional held spin finger.

The dataset deliberately keeps unreliable trace frames in chronological order.
When warm-orange gating is enabled, those real frames remain valid recurrent
context but have ``label_mask=False`` so a trainer does not supervise a target
whose rendered trace is absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

DEFAULT_IMAGE_HEIGHT = 288
DEFAULT_IMAGE_WIDTH = 128
DEFAULT_HEATMAP_SIGMA = 6.0
DEFAULT_LATENCY_S = 0.2
DEFAULT_SPIN_BUTTON_XY = (0.0604, 0.4040)
XCTEST_FINGER_STAGGER_S = 0.12

_PARAMS_PER_SLOT = 8
_SPIN_PARAMS = 3
# A tap is instantaneous as a COMMAND but not as PIXELS: measured on-device it
# renders a mark for ~0.2s (~6 frames at 30fps). Give tap intervals that minimum
# width so the frames that actually show the mark are labelled as touch-active.
_TAP_MIN_VISIBLE_S = 0.2
_FRAME_RE = re.compile(r"^frame_(\d+)\.(?:png|jpe?g)$", re.IGNORECASE)


class _UnsupportedSample(ValueError):
    """Internal signal for metadata outside the supported gesture corpus."""


@dataclass(frozen=True)
class _TouchInterval:
    """One physical touch over a closed interval on the payload timeline."""

    start_s: float
    end_s: float
    source_order: int
    kind: str
    waypoints: tuple[tuple[float, float], ...] = ()
    segment_durations: tuple[float, ...] = ()
    constant_xy: tuple[float, float] | None = None
    track: int = -1

    def center_at(self, time_s: float) -> tuple[float, float] | None:
        if not self.start_s <= time_s <= self.end_s:
            return None
        if self.constant_xy is not None:
            return self.constant_xy
        return _position_at(
            list(self.waypoints), list(self.segment_durations), time_s - self.start_s
        )


@dataclass(frozen=True)
class _SequenceRecord:
    sample_path: Path
    frame_paths: tuple[Path, ...]
    frame_times: np.ndarray
    delta_times: np.ndarray
    centers: np.ndarray
    touch_count: np.ndarray
    label_mask: np.ndarray
    kind: str
    required_touches: int
    cached_frames: np.ndarray | None = None


@dataclass(frozen=True)
class _BuildSettings:
    sequence_length: int
    image_height: int
    image_width: int
    max_touches: int
    latency_s: float
    require_trace: bool
    trace_warm_threshold: int
    trace_radius_px: int
    finger_stagger_s: float | None
    cache_frames: bool
    detect_menu_frames: bool


@dataclass(frozen=True)
class _BuildResult:
    record: _SequenceRecord | None
    stats: Counter


def _normalise_term(value: str | os.PathLike[str]) -> str:
    return "".join(c for c in str(value).casefold() if c.isalnum())


def _matches_term(path: str | os.PathLike[str], term: str) -> bool:
    haystack = _normalise_term(path)
    needle = _normalise_term(term)
    if needle in haystack:
        return True
    # Park folders often insert a year: "SLS Super Crown" should match
    # "sls_2016_super_crown" without weakening the match to either word alone.
    without_digits = lambda value: "".join(c for c in value if not c.isdigit())
    digitless_needle = without_digits(needle)
    return bool(digitless_needle) and digitless_needle in without_digits(haystack)


def _segment_durations_s(
    num_waypoints: int, total_duration: float, easing_power: float
) -> list[float]:
    """Pure-data mirror of ``build_curved_drag``'s millisecond schedule.

    Importing the Appium touch module would pull Selenium into Modal dataset
    workers.  Keeping this small arithmetic mirror here makes corpus loading
    device-free while retaining the executor's integer rounding exactly.
    """

    n_segments = max(1, num_waypoints - 1)
    total_ms = int(total_duration * 1000)
    if easing_power == 1.0:
        milliseconds = [max(1, total_ms // n_segments)] * n_segments
    else:
        boundaries = [(index / n_segments) ** easing_power for index in range(n_segments + 1)]
        raw = [boundaries[index + 1] - boundaries[index] for index in range(n_segments)]
        raw_sum = sum(raw)
        milliseconds = [max(1, int(value / raw_sum * total_ms)) for value in raw]
    return [value / 1000.0 for value in milliseconds]


def _position_at(
    waypoints: list[tuple[float, float]], segment_durations: list[float], time_s: float
) -> tuple[float, float]:
    if time_s <= 0.0:
        return waypoints[0]
    elapsed = 0.0
    for index, duration in enumerate(segment_durations):
        if time_s < elapsed + duration or index == len(segment_durations) - 1:
            fraction = min(1.0, max(0.0, (time_s - elapsed) / duration))
            (x0, y0), (x1, y1) = waypoints[index], waypoints[index + 1]
            return x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction
        elapsed += duration
    return waypoints[-1]


def _directory_children(path: Path) -> list[Path]:
    """Return sorted direct child directories without following symlinks.

    Stopping at ``sample_*`` is important on the Modal volume: there may be
    millions of frame files but only directory entries above each sample need
    to be inspected during discovery.
    """

    try:
        with os.scandir(path) as entries:
            children = [
                Path(entry.path)
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
            ]
    except OSError as exc:
        raise RuntimeError(f"Could not scan corpus directory {path}: {exc}") from exc
    return sorted(children, key=lambda child: child.name)


def _candidate_roots(root: Path, include_path_term: str | None) -> list[Path]:
    """Find matching sample-parent roots without enumerating their frame files."""

    roots: list[Path] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        if include_path_term and _matches_term(directory.name, include_path_term):
            roots.append(directory)
            continue
        descendants: list[Path] = []
        saw_sample = False
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    if entry.name.startswith("sample_"):
                        # A corpus leaf can contain hundreds of thousands of
                        # samples.  One is enough to classify the parent; do not
                        # consume the rest of this iterator here.
                        saw_sample = True
                        break
                    descendants.append(Path(entry.path))
        except OSError as exc:
            raise RuntimeError(f"Could not scan corpus directory {directory}: {exc}") from exc
        if saw_sample:
            if include_path_term is None:
                roots.append(directory)
            # If a term was supplied and no ancestor/name matched it, this leaf
            # is deliberately excluded.
            continue
        stack.extend(reversed(sorted(descendants, key=lambda child: child.name)))
    return sorted(set(roots))


def _samples_under(root: Path) -> list[Path]:
    """Enumerate sample directories below one selected corpus root.

    The traversal stops at each sample and therefore never touches its frame
    inode entries.  Only roots selected by the bounded cross-session sampler
    reach this function.
    """

    if root.name.startswith("sample_"):
        return [root]
    samples: list[Path] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        descendants = []
        for child in _directory_children(directory):
            if child.name.startswith("sample_"):
                samples.append(child)
            else:
                descendants.append(child)
        stack.extend(reversed(descendants))
    return samples


def discover_sample_paths(
    root: str | Path,
    *,
    include_path_term: str | None = None,
    max_samples: int | None = None,
) -> list[Path]:
    """Find nested ``sample_*`` directories without descending into frames.

    ``include_path_term`` is punctuation/year-insensitive (``SLS Super Crown``
    also matches ``sls_2016_super_crown``).  A bounded selection combines recent
    lexical session/park roots with a stable hash sample, then draws round-robin
    across up to ``sqrt(max_samples)`` roots.  Including the recent slice is
    intentional: newly requested gesture modes (notably the 0.8 spin mix) must
    enter the next training run instead of waiting for a full-corpus shuffle.
    The hash slice retains older diversity, and discovery never descends into
    the selected samples' frame inodes.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Temporal trace corpus does not exist: {root}")
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    if root.name.startswith("sample_"):
        matches = include_path_term is None or _matches_term(root, include_path_term)
        return [root] if matches else []
    roots = _candidate_roots(root, include_path_term)
    if not roots:
        return []
    if max_samples is None:
        return sorted(sample for candidate in roots for sample in _samples_under(candidate))

    root_budget = min(len(roots), max(1, math.ceil(math.sqrt(max_samples))))
    lexical_roots = sorted(roots)
    recent_count = min(root_budget, max(1, math.ceil(root_budget / 3)))
    recent_roots = lexical_roots[-recent_count:]
    recent_set = set(recent_roots)
    older_hashed = sorted(
        (path for path in roots if path not in recent_set),
        key=lambda path: hashlib.sha256(str(path.relative_to(root)).encode("utf-8")).digest(),
    )
    roots = recent_roots + older_hashed[: root_budget - recent_count]
    per_root: list[list[Path]] = []
    for candidate in roots:
        samples = _samples_under(candidate)
        samples.sort(
            key=lambda path: hashlib.sha256(
                str(path.relative_to(candidate)).encode("utf-8")
            ).digest()
        )
        per_root.append(samples)

    selected: list[Path] = []
    offset = 0
    while len(selected) < max_samples:
        added = False
        for samples in per_root:
            if offset < len(samples):
                selected.append(samples[offset])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        offset += 1
    return sorted(selected)


def _validate_xy(value: Sequence[float], *, identity: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{identity} must contain exactly x,y, got {value!r}")
    x, y = float(value[0]), float(value[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"{identity} contains non-finite coordinates: {value!r}")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"{identity} must be normalized to [0,1], got {(x, y)}")
    return x, y


def _validate_positive(value: object, *, identity: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{identity} must be finite and > 0, got {value!r}")
    return number


def _infer_param_layout(length: int) -> tuple[int, bool]:
    if length >= _PARAMS_PER_SLOT and (length + 1) % 9 == 0:
        return (length + 1) // 9, False
    if length >= _PARAMS_PER_SLOT + _SPIN_PARAMS and (length - 2) % 9 == 0:
        return (length - 2) // 9, True
    raise ValueError(
        f"cannot infer gesture layout from {length} params; expected 9N-1 or 9N+2"
    )


def _spin_button(meta: dict, sample_path: Path) -> tuple[float, float]:
    return _validate_xy(
        meta.get("spin_button_xy", DEFAULT_SPIN_BUTTON_XY),
        identity=f"{sample_path}: spin_button_xy",
    )


def _drag_interval(
    waypoints_value: Sequence[Sequence[float]],
    duration_value: object,
    easing_value: object,
    *,
    start_s: float,
    source_order: int,
    identity: str,
) -> _TouchInterval:
    waypoints = tuple(
        _validate_xy(point, identity=f"{identity} waypoint {index}")
        for index, point in enumerate(waypoints_value)
    )
    if len(waypoints) < 2:
        raise ValueError(f"{identity} needs at least two waypoints")
    duration = _validate_positive(duration_value, identity=f"{identity} duration")
    easing = _validate_positive(easing_value, identity=f"{identity} easing_power")
    segments = tuple(_segment_durations_s(len(waypoints), duration, easing))
    return _TouchInterval(
        start_s=float(start_s),
        end_s=float(start_s + sum(segments)),
        source_order=source_order,
        kind="drag",
        waypoints=waypoints,
        segment_durations=segments,
    )


def _static_schedule(meta: dict, sample_path: Path) -> tuple[list[_TouchInterval], float]:
    """Schedule for a STATIONARY touch sample ("hold" / "tap").

    One finger, one position, held for ``hold_duration_s`` from the payload start.
    A tap has hold_duration_s == 0; it still renders a mark for ~0.2s, so it gets a
    minimum interval rather than a zero-length one that ``center_at`` could never
    match (start_s <= t <= end_s is inclusive, but a zero-width window would only
    ever catch a frame landing exactly on it).
    """
    identity = f"{sample_path}: {meta.get('gesture_distribution')}"
    point = meta.get("point")
    if point is None:
        raise _UnsupportedSample(f"{identity}: stationary sample has no point")
    xy = _validate_xy(point, identity=f"{identity} point")
    hold = float(meta.get("hold_duration_s") or 0.0)
    if not math.isfinite(hold) or hold < 0.0:
        raise ValueError(f"{identity} hold_duration_s must be finite and >= 0, got {hold!r}")
    end = max(hold, _TAP_MIN_VISIBLE_S)
    return (
        [_TouchInterval(start_s=0.0, end_s=end, source_order=0, kind="hold", constant_xy=xy)],
        end,
    )


def _auto_finger_stagger(meta: dict, configured: float | None) -> float:
    if configured is not None:
        stagger = float(configured)
    elif meta.get("min_finger_stagger_s") is not None:
        stagger = float(meta["min_finger_stagger_s"])
    else:
        # collect_sls_xctest.py enables this execution guard before importing
        # touch_actions; the older MJPEG collector did not.
        stagger = XCTEST_FINGER_STAGGER_S if "gesture_video_time_s" in meta else 0.0
    if not math.isfinite(stagger) or stagger < 0.0:
        raise ValueError(f"finger_stagger_s must be finite and >= 0, got {stagger}")
    return stagger


def _flick_schedule(meta: dict, sample_path: Path) -> tuple[list[_TouchInterval], float]:
    identity = str(sample_path)
    try:
        drag = _drag_interval(
            meta["waypoints"], meta["duration"], meta["easing_power"],
            start_s=0.0, source_order=0, identity=f"{identity}: flick",
        )
    except KeyError as exc:
        raise ValueError(f"{identity}: flick metadata is missing {exc.args[0]!r}") from exc
    touches = [drag]
    payload_total = float(meta.get("payload_total_s", meta["duration"]))
    if not math.isfinite(payload_total) or payload_total <= 0.0:
        raise ValueError(f"{identity}: invalid payload_total_s={payload_total!r}")

    if bool(meta.get("spin_active", False)):
        if meta.get("spin_hold_start_s") is None or meta.get("spin_hold_end_s") is None:
            raise ValueError(f"{identity}: active spin flick is missing its hold window")
        start, end = sorted(
            (float(meta["spin_hold_start_s"]), float(meta["spin_hold_end_s"]))
        )
        if not (math.isfinite(start) and math.isfinite(end) and start >= 0.0):
            raise ValueError(f"{identity}: invalid spin hold window {(start, end)}")
        touches.append(
            _TouchInterval(
                start_s=start,
                end_s=end,
                source_order=1,
                kind="spin",
                constant_xy=_spin_button(meta, sample_path),
            )
        )
        payload_total = max(payload_total, end)
    return touches, max(payload_total, drag.end_s)


def _params_schedule(
    meta: dict,
    sample_path: Path,
    *,
    finger_stagger_s: float | None,
) -> tuple[list[_TouchInterval], float]:
    identity = str(sample_path)
    params = [float(value) for value in meta.get("params", [])]
    if not params or not all(math.isfinite(value) for value in params):
        raise ValueError(f"{identity}: params must be a non-empty finite vector")
    inferred_n, inferred_spin = _infer_param_layout(len(params))
    n = int(meta.get("num_gestures", inferred_n))
    use_spin = bool(meta.get("use_spin", inferred_spin))
    expected = 9 * n - 1 + (_SPIN_PARAMS if use_spin else 0)
    if n < 1 or len(params) != expected:
        raise ValueError(
            f"{identity}: {len(params)} params do not match num_gestures={n}, "
            f"use_spin={use_spin} (expected {expected})"
        )

    durations = [
        _validate_positive(params[slot * _PARAMS_PER_SLOT + 6],
                           identity=f"{identity}: slot {slot} duration")
        for slot in range(n)
    ]
    delay_offset = n * _PARAMS_PER_SLOT
    delays = params[delay_offset:delay_offset + max(0, n - 1)]
    raw_starts = [0.0]
    for slot in range(1, n):
        raw_starts.append(raw_starts[-1] + durations[slot - 1] + delays[slot - 1])

    order = sorted(range(n), key=lambda slot: (raw_starts[slot], slot))
    earliest = raw_starts[order[0]]
    starts_by_slot = {slot: raw_starts[slot] - earliest for slot in order}
    stagger = _auto_finger_stagger(meta, finger_stagger_s)
    previous_start: float | None = None
    for slot in order:
        start = starts_by_slot[slot]
        if previous_start is not None:
            start = max(start, previous_start + stagger)
        starts_by_slot[slot] = start
        previous_start = start

    touches: list[_TouchInterval] = []
    for slot in range(n):
        base = slot * _PARAMS_PER_SLOT
        points = [params[base:base + 2], params[base + 2:base + 4], params[base + 4:base + 6]]
        touches.append(
            _drag_interval(
                points, durations[slot], params[base + 7],
                start_s=starts_by_slot[slot], source_order=slot,
                identity=f"{identity}: slot {slot}",
            )
        )

    # WDA's spin fractions are relative to the final (possibly staggered)
    # drag schedule.  Explicit absolute fields in meta were derived before the
    # XCTest stagger existed, so the raw fraction block is authoritative here.
    nominal_total = max(
        starts_by_slot[slot] + durations[slot] for slot in range(n)
    )
    payload_total = max(nominal_total, max(touch.end_s for touch in touches))
    if use_spin:
        spin_offset = delay_offset + max(0, n - 1)
        gate, t0, t1 = params[spin_offset:spin_offset + _SPIN_PARAMS]
        spin_declared = bool(meta.get("spin_active", gate >= 0.0))
        if gate >= 0.0 and spin_declared:
            lo, hi = sorted((min(1.0, max(0.0, t0)), min(1.0, max(0.0, t1))))
            touches.append(
                _TouchInterval(
                    start_s=lo * nominal_total,
                    end_s=hi * nominal_total,
                    source_order=n,
                    kind="spin",
                    constant_xy=_spin_button(meta, sample_path),
                )
            )
    return touches, payload_total


def _assign_tracks(
    touches: list[_TouchInterval], sample_path: Path, max_touches: int
) -> tuple[list[_TouchInterval], int]:
    """Colour touch intervals into stable track slots.

    Sequential strokes can reuse a slot, while overlapping strokes cannot.
    Closed intervals intentionally treat a lift and another down at the exact
    same timestamp as simultaneous for the boundary frame.
    """

    track_ends: list[float] = []
    assigned: dict[int, _TouchInterval] = {}
    for index, touch in sorted(
        enumerate(touches), key=lambda pair: (pair[1].start_s, pair[1].source_order)
    ):
        track = next(
            (candidate for candidate, end in enumerate(track_ends) if end < touch.start_s),
            len(track_ends),
        )
        if track == len(track_ends):
            track_ends.append(touch.end_s)
        else:
            track_ends[track] = touch.end_s
        assigned[index] = replace(touch, track=track)
    required = len(track_ends)
    if required > max_touches:
        raise ValueError(
            f"{sample_path} requires max_touches={required} for its overlapping "
            f"touch schedule, but configured max_touches={max_touches}"
        )
    return [assigned[index] for index in range(len(touches))], required


def _schedule_from_meta(
    meta: dict,
    sample_path: Path,
    *,
    max_touches: int,
    finger_stagger_s: float | None,
) -> tuple[list[_TouchInterval], float, str, int]:
    kind = str(meta.get("gesture_distribution", "")).casefold()
    if kind in {"hold", "tap"}:
        touches, total = _static_schedule(meta, sample_path)
    elif "waypoints" in meta or kind in {"flick", "spin_flick"}:
        touches, total = _flick_schedule(meta, sample_path)
        kind = kind or ("spin_flick" if meta.get("spin_active") else "flick")
    elif "params" in meta or kind in {"nslot", "recipe", "spin"}:
        touches, total = _params_schedule(
            meta, sample_path, finger_stagger_s=finger_stagger_s
        )
        kind = kind or "params"
    else:
        raise _UnsupportedSample(f"{sample_path}: unsupported gesture metadata")
    touches, required = _assign_tracks(touches, sample_path, max_touches)
    return touches, total, kind, required


def _is_end_relative(meta: dict) -> bool:
    if "gesture_start_monotonic" in meta:
        return False
    return any(
        key in meta
        for key in (
            "gesture_end_monotonic", "gesture_video_time_s", "t_call_end_epoch_s",
            "capture_offset_s",
        )
    )


_VIDEO_NAME = "frames.mp4"


def _video_path(sample_path: Path) -> Path | None:
    """The sample's video container, if it is video-backed rather than PNG-backed.

    Storing a sample's frames as one h264 clip instead of 24 PNGs is ~30x smaller
    and — just as important on the corpus volume — 1 inode instead of 24. Frames
    within a sample are the same scene milliseconds apart, so inter-frame
    compression is enormously effective where per-frame PNG can exploit nothing.
    """
    candidate = sample_path / _VIDEO_NAME
    return candidate if candidate.is_file() else None


def _decode_video(path: Path) -> list[np.ndarray]:
    """Decode every frame of a sample clip to BGR, in order.

    Decoded whole rather than seeked per-frame: the clips are ~1-2s, and random
    access into inter-coded video costs far more than one sequential pass.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open sample video {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Sample video decoded zero frames: {path}")
    return frames


def _frame_paths(sample_path: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    try:
        children = sample_path.iterdir()
        for path in children:
            if not path.is_file():
                continue
            match = _FRAME_RE.match(path.name)
            if not match:
                continue
            index = int(match.group(1))
            previous = result.get(index)
            if previous is not None:
                raise ValueError(
                    f"{sample_path}: duplicate frame index {index}: {previous.name}, {path.name}"
                )
            result[index] = path
    except OSError as exc:
        raise RuntimeError(f"Could not enumerate frames under {sample_path}: {exc}") from exc
    return result


def _warm_count(image_bgr: np.ndarray, xy: tuple[float, float], radius_px: int) -> int:
    height, width = image_bgr.shape[:2]
    px = int(round(xy[0] * (width - 1)))
    py = int(round(xy[1] * (height - 1)))
    x0, x1 = max(0, px - radius_px), min(width, px + radius_px + 1)
    y0, y1 = max(0, py - radius_px), min(height, py + radius_px + 1)
    patch = image_bgr[y0:y1, x0:x1]
    if not patch.size:
        return 0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return int(((hsv[..., 0] <= 35) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 140)).sum())


def _build_sequence_record(
    sample_path: Path,
    *,
    settings: _BuildSettings,
) -> _BuildResult:
    """Load and validate one sample without mutating dataset-wide state.

    Keeping all per-sample image work here lets callers overlap independent
    FUSE reads with a bounded thread pool.  The result contains a private
    counter so aggregation can still happen in deterministic candidate order.
    """

    stats: Counter = Counter()
    if (sample_path / ".menu").exists():
        stats["menu_skipped"] += 1
        return _BuildResult(None, stats)
    if (sample_path / ".editor").exists():
        stats["editor_skipped"] += 1
        return _BuildResult(None, stats)
    meta_path = sample_path / "meta.json"
    if not meta_path.is_file():
        stats["missing_meta_skipped"] += 1
        return _BuildResult(None, stats)
    try:
        meta_text = meta_path.read_text()
    except OSError as exc:
        # A storage/FUSE read failure is operational, not a bad sample.  Fail
        # loudly so a broad outage cannot masquerade as a smaller clean corpus.
        raise RuntimeError(f"Could not read {meta_path}: {exc}") from exc
    try:
        meta = json.loads(meta_text)
    except json.JSONDecodeError:
        # Individual interrupted writes exist in the long-running corpus.  They
        # contain no trustworthy supervision, so skip and count them rather
        # than aborting an otherwise valid multi-hour training run.
        stats["bad_meta_skipped"] += 1
        return _BuildResult(None, stats)
    if not isinstance(meta, dict):
        stats["bad_meta_skipped"] += 1
        return _BuildResult(None, stats)

    try:
        touches, schedule_total, kind, required = _schedule_from_meta(
            meta,
            sample_path,
            max_touches=settings.max_touches,
            finger_stagger_s=settings.finger_stagger_s,
        )
    except _UnsupportedSample:
        stats["unsupported_skipped"] += 1
        return _BuildResult(None, stats)
    # Video-backed samples carry no per-frame files, so synthesise the same
    # index->path keys the PNG path produces and seed the loader cache with the
    # decoded frames. Everything downstream (menu detection, trace gating, the
    # uint8 cache) then works identically for both storage formats.
    sample_video = _video_path(sample_path)
    prefetched_bgr: dict[Path, np.ndarray] = {}
    if sample_video is not None:
        decoded = _decode_video(sample_video)
        frames_by_index = {i: sample_path / f"frame_{i:03d}.png" for i in range(len(decoded))}
        prefetched_bgr = {frames_by_index[i]: img for i, img in enumerate(decoded)}
    else:
        frames_by_index = _frame_paths(sample_path)
    raw_times = meta.get("frame_times")
    if not isinstance(raw_times, list) or not raw_times:
        stats["missing_frames_skipped"] += 1
        return _BuildResult(None, stats)
    raw_times_array = np.asarray(raw_times, dtype=np.float64)
    if raw_times_array.ndim != 1 or not np.all(np.isfinite(raw_times_array)):
        raise ValueError(f"{sample_path}: frame_times must be a finite 1-D list")
    if len(raw_times_array) > 1 and np.any(np.diff(raw_times_array) < 0.0):
        raise ValueError(f"{sample_path}: frame_times must be chronological")

    selected_indices = sorted(
        index for index in frames_by_index if index < len(raw_times_array)
    )
    stats["missing_frame_files"] += max(
        0, len(raw_times_array) - len(selected_indices)
    )
    stats["out_of_range_frame_files"] += sum(
        index >= len(raw_times_array) for index in frames_by_index
    )
    if not selected_indices:
        stats["missing_frames_skipped"] += 1
        return _BuildResult(None, stats)
    if len(selected_indices) > settings.sequence_length:
        raise ValueError(
            f"{sample_path} contains {len(selected_indices)} frames, exceeding "
            f"sequence_length={settings.sequence_length}; increase sequence_length because "
            "temporal samples are never truncated"
        )

    selected_paths = tuple(frames_by_index[index] for index in selected_indices)
    # One FUSE read per retained frame at most.  Menu detection, warm-trace
    # gating, and the optional uint8 cache all share this map within a worker.
    loaded_bgr: dict[Path, np.ndarray] = prefetched_bgr

    def load_bgr(frame_path: Path) -> np.ndarray:
        image = loaded_bgr.get(frame_path)
        if image is None:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read frame {frame_path}")
            loaded_bgr[frame_path] = image
        return image

    def load_rgb(frame_path: Path) -> np.ndarray:
        return cv2.cvtColor(load_bgr(frame_path), cv2.COLOR_BGR2RGB)

    if settings.detect_menu_frames:
        from trueskate_ai.vision.gameplay_filter import (
            is_bolt_modal_frame, is_editor_frame, is_menu_frame)

        middle = len(selected_paths) // 2
        priority = tuple(
            dict.fromkeys((selected_paths[0], selected_paths[middle], selected_paths[-1]))
        )
        if any(is_editor_frame(load_rgb(path)) for path in priority):
            stats["detected_editor_skipped"] += 1
            return _BuildResult(None, stats)
        if any(is_menu_frame(load_rgb(path)) for path in priority):
            stats["detected_menu_skipped"] += 1
            return _BuildResult(None, stats)
        # Bolt Challenges center modal: invisible to is_menu_frame (bottom-bar
        # only), it stays open across a whole run and leaks garbage command-position
        # labels into training.  See memory bolt-challenges-modal-contamination.
        if any(is_bolt_modal_frame(load_rgb(path)) for path in priority):
            stats["detected_modal_skipped"] += 1
            return _BuildResult(None, stats)

    capture_times = raw_times_array[selected_indices]
    start_times = (
        capture_times + schedule_total
        if _is_end_relative(meta)
        else capture_times.copy()
    )
    compensated_times = start_times - settings.latency_s
    centers = np.full(
        (len(selected_indices), settings.max_touches, 2), -1.0, dtype=np.float32
    )
    trace_touch = np.zeros(
        (len(selected_indices), settings.max_touches), dtype=np.bool_
    )
    for frame_index, time_s in enumerate(compensated_times):
        for touch in touches:
            center = touch.center_at(float(time_s))
            if center is None:
                continue
            centers[frame_index, touch.track] = center
            trace_touch[frame_index, touch.track] = touch.kind == "drag"
    touch_count = np.sum(centers[..., 0] >= 0.0, axis=1).astype(np.int64)
    label_mask = np.ones(len(selected_indices), dtype=np.bool_)
    gated = 0
    if settings.require_trace:
        for local_index, frame_path in enumerate(selected_paths):
            drag_tracks = np.nonzero(trace_touch[local_index])[0]
            if not len(drag_tracks):
                continue
            image = load_bgr(frame_path)
            reliable = all(
                _warm_count(
                    image,
                    tuple(float(value) for value in centers[local_index, track]),
                    settings.trace_radius_px,
                ) >= settings.trace_warm_threshold
                for track in drag_tracks
            )
            if not reliable:
                label_mask[local_index] = False
                gated += 1

    cached_frames = None
    if settings.cache_frames:
        from trueskate_ai.bc.frame_prep import prep_frame_rgb

        cached_frames = np.stack(
            [
                prep_frame_rgb(
                    load_bgr(frame_path),
                    settings.image_height,
                    settings.image_width,
                    normalize=False,
                )
                for frame_path in selected_paths
            ]
        ).astype(np.uint8, copy=False)
        stats["cached_frame_bytes"] += cached_frames.nbytes

    deltas = np.zeros(len(selected_indices), dtype=np.float32)
    if len(selected_indices) > 1:
        deltas[1:] = np.diff(capture_times).astype(np.float32)
    record = _SequenceRecord(
        sample_path=sample_path,
        frame_paths=selected_paths,
        frame_times=start_times.astype(np.float32),
        delta_times=deltas,
        centers=centers,
        touch_count=touch_count,
        label_mask=label_mask,
        kind=kind,
        required_touches=required,
        cached_frames=cached_frames,
    )
    stats["samples_retained"] += 1
    stats["frames"] += len(selected_indices)
    stats["positive_frames"] += int(np.sum((touch_count > 0) & label_mask))
    stats["negative_frames"] += int(np.sum((touch_count == 0) & label_mask))
    stats["gated_frames"] += gated
    return _BuildResult(record, stats)


def _stable_split_indices(
    length: int,
    val_fraction: float,
    seed: int,
    paths: Sequence[Path],
    multi_touch_frame_counts: Sequence[int] | None = None,
):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if length < 2:
        raise ValueError("At least two sample sequences are required for a train/val split")
    if len(paths) != length:
        raise ValueError(
            f"split paths must contain one entry per sequence, got {len(paths)} for {length}"
        )
    n_val = min(length - 1, max(1, int(round(length * val_fraction))))

    def stable_order(indices: Sequence[int]) -> list[int]:
        return sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}:{paths[index]}".encode("utf-8")
            ).digest(),
        )

    if multi_touch_frame_counts is None:
        order = stable_order(range(length))
        val = sorted(order[:n_val])
        train = sorted(order[n_val:])
        return train, val
    if len(multi_touch_frame_counts) != length:
        raise ValueError(
            "multi_touch_frame_counts must contain one entry per sequence, got "
            f"{len(multi_touch_frame_counts)} for {length}"
        )

    overlap = [
        index for index, count in enumerate(multi_touch_frame_counts) if int(count) > 0
    ]
    ordinary = [
        index for index, count in enumerate(multi_touch_frame_counts) if int(count) <= 0
    ]
    if not overlap or not ordinary:
        order = stable_order(range(length))
        val = sorted(order[:n_val])
        train = sorted(order[n_val:])
        return train, val

    # Preserve the requested global validation size while allocating it between
    # genuine-overlap and ordinary sequences in proportion to corpus exposure.
    # When both strata have enough examples and both global splits have room,
    # retain at least one member of each stratum on each side.
    ideal_overlap_val = n_val * len(overlap) / length
    overlap_val = int(math.floor(ideal_overlap_val + 0.5))
    minimum = max(0, n_val - len(ordinary))
    maximum = min(n_val, len(overlap))
    n_train = length - n_val
    if (
        len(overlap) >= 2
        and len(ordinary) >= 2
        and n_val >= 2
        and n_train >= 2
    ):
        minimum = max(minimum, 1, n_val - (len(ordinary) - 1))
        maximum = min(maximum, len(overlap) - 1, n_val - 1)
    overlap_val = min(maximum, max(minimum, overlap_val))
    ordinary_val = n_val - overlap_val

    overlap_order = stable_order(overlap)
    ordinary_order = stable_order(ordinary)
    val = sorted(overlap_order[:overlap_val] + ordinary_order[:ordinary_val])
    train = sorted(overlap_order[overlap_val:] + ordinary_order[ordinary_val:])
    if not train or not val:
        raise RuntimeError("stratified split unexpectedly produced an empty train or val set")
    return train, val


def split_by_sample(
    dataset: "TemporalTraceSequenceDataset",
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Deterministically split whole gesture sequences, never individual frames."""

    train, val = _stable_split_indices(
        len(dataset),
        val_fraction,
        seed,
        dataset.sample_paths,
        dataset.multi_touch_frame_counts,
    )
    return Subset(dataset, train), Subset(dataset, val)


class TemporalTraceSequenceDataset(Dataset):
    """Fixed-length causal Model 1 sequences reconstructed from gesture metadata.

    Returned tensors use these stable keys:

    ``frames`` [T,3,H,W], ``heatmaps`` [T,1,H,W], ``active`` [T],
    ``centers`` [T,max_touches,2], ``touch_count`` [T], ``delta_times`` [T],
    ``valid_mask`` [T], ``label_mask`` [T], and ``reset_mask`` [T].

    Track columns in ``centers`` remain stable for the lifetime of a touch.
    Non-overlapping later strokes may reuse a free column; inactive columns are
    ``(-1,-1)``.  Padding is always appended and has all masks false.
    """

    def __init__(
        self,
        corpus_root: str | Path,
        *,
        sequence_length: int = 16,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        max_touches: int = 4,
        latency_s: float = DEFAULT_LATENCY_S,
        heatmap_sigma: float = DEFAULT_HEATMAP_SIGMA,
        include_path_term: str | None = None,
        max_samples: int | None = None,
        require_trace: bool = True,
        trace_warm_threshold: int = 200,
        trace_radius_px: int = 45,
        finger_stagger_s: float | None = None,
        cache_frames: bool = False,
        cache_workers: int = 0,
        detect_menu_frames: bool = False,
        allow_empty: bool = False,
    ):
        super().__init__()
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        if image_height < 1 or image_width < 1:
            raise ValueError(f"image dimensions must be positive, got {image_height}x{image_width}")
        if max_touches < 1:
            raise ValueError(f"max_touches must be >= 1, got {max_touches}")
        if not math.isfinite(latency_s):
            raise ValueError(f"latency_s must be finite, got {latency_s}")
        if not math.isfinite(heatmap_sigma) or heatmap_sigma <= 0.0:
            raise ValueError(f"heatmap_sigma must be finite and > 0, got {heatmap_sigma}")
        if trace_warm_threshold < 0 or trace_radius_px < 1:
            raise ValueError("trace gate threshold/radius are invalid")
        if cache_workers < 0:
            raise ValueError(f"cache_workers must be >= 0, got {cache_workers}")

        self.sequence_length = int(sequence_length)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.max_touches = int(max_touches)
        self.latency_s = float(latency_s)
        self.heatmap_sigma = float(heatmap_sigma)
        self.require_trace = bool(require_trace)
        self.cache_frames = bool(cache_frames)
        self._records: list[_SequenceRecord] = []

        candidates = discover_sample_paths(
            corpus_root,
            include_path_term=include_path_term,
            max_samples=max_samples,
        )
        stats: Counter = Counter(samples_discovered=len(candidates))
        kinds: Counter = Counter()
        max_observed_touches = 0
        settings = _BuildSettings(
            sequence_length=self.sequence_length,
            image_height=self.image_height,
            image_width=self.image_width,
            max_touches=self.max_touches,
            latency_s=self.latency_s,
            require_trace=self.require_trace,
            trace_warm_threshold=trace_warm_threshold,
            trace_radius_px=trace_radius_px,
            finger_stagger_s=finger_stagger_s,
            cache_frames=self.cache_frames,
            detect_menu_frames=bool(detect_menu_frames),
        )
        build = partial(_build_sequence_record, settings=settings)
        if cache_workers > 1 and len(candidates) > 1:
            # ThreadPoolExecutor.map yields in input order.  Independent sample
            # reads overlap, while records, counters, and the first raised
            # validation error retain serial candidate semantics.
            with ThreadPoolExecutor(
                max_workers=cache_workers,
                thread_name_prefix="trace-cache",
            ) as executor:
                results = executor.map(build, candidates)
                for result in results:
                    stats.update(result.stats)
                    if result.record is not None:
                        self._records.append(result.record)
                        kinds[result.record.kind] += 1
                        max_observed_touches = max(
                            max_observed_touches, result.record.required_touches
                        )
        else:
            for sample_path in candidates:
                result = build(sample_path)
                stats.update(result.stats)
                if result.record is not None:
                    self._records.append(result.record)
                    kinds[result.record.kind] += 1
                    max_observed_touches = max(
                        max_observed_touches, result.record.required_touches
                    )

        self.sample_paths = tuple(record.sample_path for record in self._records)
        self.positive_frame_counts = [
            int(np.sum((record.touch_count > 0) & record.label_mask))
            for record in self._records
        ]
        self.negative_frame_counts = [
            int(np.sum((record.touch_count == 0) & record.label_mask))
            for record in self._records
        ]
        self.multi_touch_frame_counts = [
            int(np.sum((record.touch_count >= 2) & record.label_mask))
            for record in self._records
        ]
        self.stats = dict(stats)
        self.stats["kinds"] = dict(sorted(kinds.items()))
        self.stats["max_touch_count"] = max_observed_touches
        self.stats["multi_touch_frames"] = int(sum(self.multi_touch_frame_counts))
        self.stats["multi_touch_sequences"] = int(
            sum(count > 0 for count in self.multi_touch_frame_counts)
        )
        self.stats["cached_frame_bytes"] = int(stats["cached_frame_bytes"])
        self.stats["cached_frame_mib"] = self.stats["cached_frame_bytes"] / (1024**2)
        if not self._records and not allow_empty:
            raise RuntimeError(f"No supported temporal trace samples found under {corpus_root}")

        ys, xs = np.mgrid[0:self.image_height, 0:self.image_width]
        self._grid_y = ys.astype(np.float32)
        self._grid_x = xs.astype(np.float32)

    def __len__(self) -> int:
        return len(self._records)

    def sample_frame_times(self, index: int) -> np.ndarray:
        """Return real, start-relative frame times for diagnostics/tests."""

        return self._records[index].frame_times.copy()

    def _heatmaps(self, centers: np.ndarray) -> np.ndarray:
        heatmaps = np.zeros(
            (len(centers), 1, self.image_height, self.image_width), dtype=np.float32
        )
        denominator = 2.0 * self.heatmap_sigma**2
        for frame_index in range(len(centers)):
            for x, y in centers[frame_index]:
                if x < 0.0:
                    continue
                px = x * (self.image_width - 1)
                py = y * (self.image_height - 1)
                bump = np.exp(
                    -((self._grid_x - px) ** 2 + (self._grid_y - py) ** 2) / denominator
                ).astype(np.float32)
                np.maximum(heatmaps[frame_index, 0], bump, out=heatmaps[frame_index, 0])
        return heatmaps

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self._records[index]
        real_length = len(record.frame_paths)
        frames = np.zeros(
            (self.sequence_length, 3, self.image_height, self.image_width), dtype=np.float32
        )
        if record.cached_frames is not None:
            frames[:real_length] = np.transpose(
                record.cached_frames.astype(np.float32) / 255.0, (0, 3, 1, 2)
            )
        else:
            from trueskate_ai.bc.frame_prep import prep_frame_rgb

            for frame_index, frame_path in enumerate(record.frame_paths):
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Failed to read frame {frame_path}")
                frames[frame_index] = np.transpose(
                    prep_frame_rgb(bgr, self.image_height, self.image_width), (2, 0, 1)
                )

        centers = np.full(
            (self.sequence_length, self.max_touches, 2), -1.0, dtype=np.float32
        )
        centers[:real_length] = record.centers
        heatmaps = np.zeros(
            (self.sequence_length, 1, self.image_height, self.image_width), dtype=np.float32
        )
        heatmaps[:real_length] = self._heatmaps(record.centers)
        touch_count = np.zeros(self.sequence_length, dtype=np.int64)
        touch_count[:real_length] = record.touch_count
        active = (touch_count > 0).astype(np.float32)
        delta_times = np.zeros(self.sequence_length, dtype=np.float32)
        delta_times[:real_length] = record.delta_times
        valid_mask = np.zeros(self.sequence_length, dtype=np.bool_)
        valid_mask[:real_length] = True
        label_mask = np.zeros(self.sequence_length, dtype=np.bool_)
        label_mask[:real_length] = record.label_mask
        reset_mask = np.zeros(self.sequence_length, dtype=np.bool_)
        reset_mask[0] = real_length > 0

        return {
            "frames": torch.from_numpy(frames),
            "heatmaps": torch.from_numpy(heatmaps),
            "active": torch.from_numpy(active),
            "centers": torch.from_numpy(centers),
            "touch_count": torch.from_numpy(touch_count),
            "delta_times": torch.from_numpy(delta_times),
            "valid_mask": torch.from_numpy(valid_mask),
            "label_mask": torch.from_numpy(label_mask),
            "reset_mask": torch.from_numpy(reset_mask),
        }
