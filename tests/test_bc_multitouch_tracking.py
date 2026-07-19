from __future__ import annotations

import numpy as np
import pytest

from scripts.data.build_bc_clips import (
    TouchPeak,
    extract_touch_peaks,
    heatmaps_to_touch_tracks,
    touch_tracks_to_strokes,
)
from trueskate_ai.bc.gesture_tokens import strokes_to_param_vector
from trueskate_ai.bc.sequence_dataset import group_overlapping_strokes
from trueskate_ai.rl.cmaes.action_param import unpack_gesture_params
from trueskate_ai.vision.touch_peaks import (
    TouchPeak as CanonicalTouchPeak,
    extract_touch_peaks as canonical_extract_touch_peaks,
)


def _heatmap(centers: list[tuple[float, float]], *, h: int = 96, w: int = 96,
             sigma: float = 2.5) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    result = np.zeros((h, w), dtype=np.float64)
    for x, y in centers:
        bump = np.exp(
            -(
                (xx - x * (w - 1)) ** 2 + (yy - y * (h - 1)) ** 2
            )
            / (2 * sigma ** 2)
        )
        result = np.maximum(result, bump)
    return result.astype(np.float32)


def _heatmap_sequence(frame_centers: list[list[tuple[float, float]]]) -> np.ndarray:
    return np.stack([_heatmap(centers) for centers in frame_centers])


def test_extract_touch_peaks_preserves_two_simultaneous_bumps() -> None:
    peaks = extract_touch_peaks(
        _heatmap([(0.24, 0.31), (0.76, 0.68)]),
        threshold=0.30,
        max_peaks=2,
        nms_radius_px=6,
    )

    assert len(peaks) == 2
    points = sorted((peak.x, peak.y) for peak in peaks)
    assert points[0] == pytest.approx((0.24, 0.31), abs=0.012)
    assert points[1] == pytest.approx((0.76, 0.68), abs=0.012)


def test_clip_builder_reexports_canonical_peak_decoder() -> None:
    assert TouchPeak is CanonicalTouchPeak
    assert extract_touch_peaks is canonical_extract_touch_peaks


def test_extract_touch_peaks_collapses_plateau_and_uses_inclusive_pixel_grid() -> None:
    heatmap = np.zeros((7, 9), dtype=np.float32)
    heatmap[1:3, 2:4] = 0.9
    heatmap[-1, -1] = 0.8

    peaks = extract_touch_peaks(
        heatmap,
        threshold=0.5,
        max_peaks=2,
        nms_radius_px=1,
    )

    assert peaks == [
        TouchPeak(x=2.5 / 8.0, y=1.5 / 6.0, score=pytest.approx(0.9)),
        TouchPeak(x=1.0, y=1.0, score=pytest.approx(0.8)),
    ]


def test_crossing_touches_keep_motion_identity_across_merged_peak() -> None:
    # At the middle frame the two max-combined Gaussians occupy exactly the same
    # location and produce only one observable peak. The other track must stay
    # open and reacquire the correct branch using its causal velocity estimate.
    forward = np.linspace(0.20, 0.80, 9)
    backward = forward[::-1]
    centers = [[(float(a), 0.50), (float(b), 0.50)] if a != b else [(float(a), 0.50)]
               for a, b in zip(forward, backward)]
    times = np.arange(len(centers), dtype=np.float64) / 30.0

    tracks = heatmaps_to_touch_tracks(
        _heatmap_sequence(centers),
        times,
        active_thresh=0.30,
        peak_nms_radius_px=5,
        track_max_gap_s=0.10,
        track_match_distance=0.12,
    )

    assert len(tracks) == 2
    tracks.sort(key=lambda track: track.xs[0])
    assert tracks[0].xs[0] < tracks[0].xs[-1]
    assert tracks[1].xs[0] > tracks[1].xs[-1]
    assert sorted(len(track.times) for track in tracks) == [8, 9]
    assert tracks[0].xs[-1] == pytest.approx(0.80, abs=0.02)
    assert tracks[1].xs[-1] == pytest.approx(0.20, abs=0.02)


def test_self_crossing_spline_remains_one_causal_track() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 25)
    points = [(0.50 + 0.25 * np.sin(t), 0.50 + 0.18 * np.sin(2.0 * t)) for t in theta]
    times = np.arange(len(points), dtype=np.float64) / 30.0

    tracks = heatmaps_to_touch_tracks(
        _heatmap_sequence([[(float(x), float(y))] for x, y in points]),
        times,
        track_match_distance=0.16,
    )

    assert len(tracks) == 1
    assert tracks[0].frame_indices == list(range(len(points)))
    assert list(zip(tracks[0].xs, tracks[0].ys))[12] == pytest.approx(points[12], abs=0.02)
    assert (tracks[0].xs[-1], tracks[0].ys[-1]) == pytest.approx(points[-1], abs=0.02)


def test_concurrent_tracks_assemble_as_overlapping_executable_strokes() -> None:
    times = np.arange(10, dtype=np.float64) / 30.0
    drag = np.linspace(0.20, 0.70, len(times))
    # Moving drag plus a held second finger: both must survive into clip.json's
    # stroke schema, where interval-connected strokes become one action group.
    centers = [[(float(x), 0.65), (0.86, 0.18)] for x in drag]
    tracks = heatmaps_to_touch_tracks(_heatmap_sequence(centers), times)
    strokes = touch_tracks_to_strokes(tracks)

    assert len(strokes) == 2
    assert strokes[0].t_start == pytest.approx(0.0)
    assert strokes[1].t_start == pytest.approx(0.0)
    assert strokes[0].t_end == pytest.approx(times[-1])
    assert strokes[1].t_end == pytest.approx(times[-1])
    assert strokes[1].params[-1] < 0.0

    serialized = [
        {"params": stroke.params.tolist(), "t_start": stroke.t_start, "t_end": stroke.t_end}
        for stroke in strokes
    ]
    groups = group_overlapping_strokes(serialized)
    assert len(groups) == 1
    assert len(groups[0]) == 2

    vector, count = strokes_to_param_vector(np.stack([stroke.params for stroke in strokes]))
    recipe = unpack_gesture_params(np.asarray(vector), num_gestures=count)
    assert count == 2
    assert len(recipe["gestures"]) == 2
    assert recipe["delays"][0] == pytest.approx(strokes[1].params[-1])
    assert recipe["delays"][0] < 0.0
