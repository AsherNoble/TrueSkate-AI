"""Five-touch (die-five) calibration marker detection and A/B scoring.

Detector B of M1-DIE5-COMPARE-20260925: the unchanged single-point
``detect_tap_onset`` runs independently at each of the five marker positions.
A marker is accepted only when at least ``min_votes`` positions detect an onset
within ``tolerance_frames`` of one shared candidate frame (so agreeing positions
may be up to ``2 * tolerance_frames`` apart); the onset is the lower median of
the agreeing frames. This is the frozen preregistered behaviour. Scenery that falsely triggers one position cannot move
the result unless it fools most of the pattern on the same frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence

import numpy as np

from trueskate_ai.collection.tap_timing_calibration import detect_tap_onset

XR_LOGICAL_SIZE = (414, 896)
DIE_FIVE_CENTRE_PT = (207, 448)
DIE_FIVE_OFFSET_PT = 35
DIE_FIVE_POINTS_PT = (
    DIE_FIVE_CENTRE_PT,
    (172, 413), (242, 413), (172, 483), (242, 483),
)
DIE_FIVE_POINTS = tuple(
    (x / XR_LOGICAL_SIZE[0], y / XR_LOGICAL_SIZE[1]) for x, y in DIE_FIVE_POINTS_PT
)


@dataclass(frozen=True)
class DieFiveOnset:
    """Per-position detections and the consensus result (``None`` = rejected)."""

    point_frames: tuple[int | None, ...]
    onset_frame: int | None
    onset_s: float | None
    votes: int

    @property
    def accepted(self) -> bool:
        return self.onset_frame is not None


def die_five_consensus(
    point_frames: Sequence[int | None], *, min_votes: int = 4, tolerance_frames: int = 1,
) -> tuple[int | None, int]:
    """Return ``(onset_frame, votes)``; onset is ``None`` below ``min_votes``."""
    found = [f for f in point_frames if f is not None]
    best: list[int] = []
    for candidate in sorted(set(found)):
        agreeing = sorted(f for f in found if abs(f - candidate) <= tolerance_frames)
        if len(agreeing) > len(best):
            best = agreeing
    if len(best) < min_votes:
        return None, len(best)
    return best[(len(best) - 1) // 2], len(best)


def detect_die_five_onset(
    frames: Sequence[np.ndarray],
    frame_times_s: Sequence[float],
    *,
    command_s: float,
    points: Sequence[tuple[float, float]] = DIE_FIVE_POINTS,
    min_votes: int = 4,
    tolerance_frames: int = 1,
    **detector_kwargs,
) -> DieFiveOnset:
    times = np.asarray(frame_times_s, dtype=np.float64)
    point_frames: list[int | None] = []
    for point in points:
        onset = detect_tap_onset(
            frames, frame_times_s, point_xy=point, command_s=command_s, **detector_kwargs,
        )
        point_frames.append(None if onset is None else int(np.searchsorted(times, onset.onset_s)))
    frame, votes = die_five_consensus(
        point_frames, min_votes=min_votes, tolerance_frames=tolerance_frames,
    )
    return DieFiveOnset(
        point_frames=tuple(point_frames),
        onset_frame=frame,
        onset_s=None if frame is None else float(times[frame]),
        votes=votes,
    )


def classify_error(detected_frame: int | None, human_frame: int, *, tolerance_frames: int = 1) -> str:
    """``no_result``, ``exact``, ``within_tolerance`` or ``gross_early``/``gross_late``."""
    if detected_frame is None:
        return "no_result"
    error = detected_frame - human_frame
    if error == 0:
        return "exact"
    if abs(error) <= tolerance_frames:
        return "within_tolerance"
    return "gross_early" if error < 0 else "gross_late"


def mcnemar_exact_p(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


# --- Exploratory B' (M1-DIE5-COMPARE post-hoc; not the preregistered detector) ---

def tap_onset_candidates(
    frames: Sequence[np.ndarray],
    frame_times_s: Sequence[float],
    *,
    point_xy: tuple[float, float],
    command_s: float,
    reference_window_s: float = 0.5,
    immediate_threshold: float = 4.0,
    confirmation_threshold: float = 6.0,
    history_frames: int = 3,
    lookahead_frames: int = 3,
) -> list[int]:
    """Every frame index passing the production detector's test, not just the first.

    Identical per-frame test to ``detect_tap_onset``. Found in Phase 2 that an
    early scenery trigger at one position hides the marker's real onset there.
    """
    if not frames:
        return []
    times = np.asarray(frame_times_s, dtype=np.float64)
    first = np.asarray(frames[0])
    height, width = first.shape[:2]
    cx, cy = point_xy[0] * (width - 1), point_xy[1] * (height - 1)
    yy, xx = np.ogrid[:height, :width]
    scale = max(width / 828.0, 0.25)
    core_radius = max(2.0, 10.0 * scale)
    ring_inner_radius = max(core_radius + 1.0, 10.0, 20.0 * scale)
    ring_outer_radius = max(ring_inner_radius + 2.0, 20.0, 40.0 * scale)
    squared = (xx - cx) ** 2 + (yy - cy) ** 2
    core = squared <= core_radius**2
    ring = (squared <= ring_outer_radius**2) & (squared > ring_inner_radius**2)
    contrast = []
    for frame in frames:
        brightness = np.asarray(frame).astype(np.float32)
        if brightness.ndim == 3:
            brightness = brightness[..., :3].mean(axis=2)
        contrast.append(float(brightness[core].mean() - brightness[ring].mean()))
    contrast = np.asarray(contrast)
    changes = np.diff(contrast)
    found = []
    for index in np.flatnonzero(times[1:] >= command_s - reference_window_s) + 1:
        if changes[index - 1] < immediate_threshold:
            continue
        before = np.median(contrast[max(0, index - history_frames):index])
        after = np.median(contrast[index:min(len(contrast), index + lookahead_frames)])
        if after - before >= confirmation_threshold:
            found.append(int(index))
    return found


def earliest_candidate_consensus(
    candidates: Sequence[Sequence[int]], *, min_votes: int = 4, tolerance_frames: int = 1,
) -> tuple[int | None, int]:
    """Earliest frame where ``min_votes`` positions each have a candidate within tolerance.

    Returns ``(frame, votes)``; the frame is the lower median of the nearest
    agreeing candidates. ``votes`` is the best support found when rejected.
    """
    best_votes = 0
    for anchor in sorted({f for point in candidates for f in point}):
        nearest = []
        for point in candidates:
            close = [f for f in point if abs(f - anchor) <= tolerance_frames]
            if close:
                nearest.append(min(close, key=lambda f: (abs(f - anchor), f)))
        best_votes = max(best_votes, len(nearest))
        if len(nearest) >= min_votes:
            nearest.sort()
            return nearest[(len(nearest) - 1) // 2], len(nearest)
    return None, best_votes
