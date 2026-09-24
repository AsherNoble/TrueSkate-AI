"""Five-touch (die-five) calibration marker detection and A/B scoring.

Detector B of M1-DIE5-COMPARE-20260925: the unchanged single-point
``detect_tap_onset`` runs independently at each of the five marker positions.
A marker is accepted only when at least ``min_votes`` positions detect an onset
within ``tolerance_frames`` of one another; the onset is the lower median of
the agreeing frames. Scenery that falsely triggers one position cannot move
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
