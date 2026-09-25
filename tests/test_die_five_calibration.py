import cv2
import numpy as np
import pytest

from trueskate_ai.collection.die_five_calibration import (
    DIE_FIVE_POINTS,
    classify_error,
    detect_die_five_onset,
    die_five_consensus,
    mcnemar_exact_p,
)
from trueskate_ai.collection.tap_timing_calibration import detect_tap_onset


def _marker_window(*, onset_s, decoy_s=None, drop=()):
    """XR-proportioned synthetic video with a die-five marker and optional centre-only decoy."""
    times = np.arange(0.0, 1.8, 1 / 30, dtype=np.float64)
    height, width = 448, 207
    frames = []
    for time_s in times:
        image = np.full((height, width, 3), (40, 70, 90), dtype=np.uint8)
        for index, (x, y) in enumerate(DIE_FIVE_POINTS):
            centre = (round(x * (width - 1)), round(y * (height - 1)))
            if index not in drop and onset_s <= time_s < onset_s + 0.20:
                cv2.circle(image, centre, 5, (10, 150, 245), thickness=-1)
            if index == 0 and decoy_s is not None and decoy_s <= time_s < decoy_s + 0.20:
                cv2.circle(image, centre, 5, (200, 200, 200), thickness=-1)
        frames.append(image)
    return frames, times


def test_consensus_requires_four_agreeing_points_and_uses_lower_median():
    assert die_five_consensus([30, 30, 31, 31, None]) == (30, 4)
    assert die_five_consensus([30, 31, 31, 32, 20]) == (31, 4)
    assert die_five_consensus([30, 30, 30, None, None]) == (None, 3)
    assert die_five_consensus([30, 33, 36, 39, 42]) == (None, 1)


def test_die_five_ignores_a_centre_only_early_decoy_that_fools_single_touch():
    frames, times = _marker_window(onset_s=1.0, decoy_s=0.5)
    single = detect_tap_onset(frames, times, point_xy=DIE_FIVE_POINTS[0], command_s=0.9)
    assert single is not None and single.onset_s == pytest.approx(0.5)

    result = detect_die_five_onset(frames, times, command_s=0.9)
    assert result.accepted
    assert result.onset_s == pytest.approx(1.0)
    assert result.votes == 4
    assert result.point_frames[0] == 15


def test_die_five_rejects_when_fewer_than_four_points_are_visible():
    frames, times = _marker_window(onset_s=1.0, drop=(1, 2))
    result = detect_die_five_onset(frames, times, command_s=0.9)
    assert not result.accepted and result.votes == 3 and result.onset_s is None


def test_classify_error_and_exact_mcnemar():
    assert classify_error(None, 10) == "no_result"
    assert classify_error(10, 10) == "exact"
    assert classify_error(9, 10) == "within_tolerance"
    assert classify_error(7, 10) == "gross_early"
    assert classify_error(13, 10) == "gross_late"
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(6, 0) == pytest.approx(2 / 64)
    assert mcnemar_exact_p(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact_p(3, 3) == 1.0


def test_collector_die_five_marker_is_one_request_with_five_simultaneous_fingers():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).parents[1] / "scripts" / "collection" / "collect_sls_xctest.py"
    spec = importlib.util.spec_from_file_location("test_collect_sls_xctest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Driver:
        def __init__(self):
            self.payloads = []

        def execute(self, command, payload):
            self.payloads.append((command, payload))

    for kind, expected in (("die_five", 5), ("single", 1)):
        driver = Driver()
        module._execute_marker(SimpleNamespace(driver=driver, device_w=414, device_h=896), kind, 0.05)
        assert len(driver.payloads) == 1
        sources = driver.payloads[0][1]["actions"]
        assert len(sources) == expected
        downs = [[a["type"] for a in s["actions"]].index("pointerDown") for s in sources]
        assert len(set(downs)) == 1
        xy = sorted((s["actions"][0]["x"], s["actions"][0]["y"]) for s in sources)
        if kind == "die_five":
            assert xy == [(172, 413), (172, 483), (207, 448), (242, 413), (242, 483)]
        else:
            assert xy == [(207, 448)]

    driver = Driver()
    module._execute_marker(SimpleNamespace(driver=driver, device_w=414, device_h=896), "die_five", 0.05, 71)
    xy = sorted((s["actions"][0]["x"], s["actions"][0]["y"]) for s in driver.payloads[0][1]["actions"])
    assert xy == [(136, 377), (136, 519), (207, 448), (278, 377), (278, 519)]


def test_die_five_points_default_is_the_preregistered_geometry_and_scales():
    from trueskate_ai.collection.die_five_calibration import die_five_points, die_five_points_pt
    from trueskate_ai.data.control_hitboxes import point_is_safe

    assert die_five_points() == DIE_FIVE_POINTS
    assert die_five_points_pt() == ((207, 448), (172, 413), (242, 413), (172, 483), (242, 483))
    wide = die_five_points_pt(71)
    assert max(((x - 207) ** 2 + (y - 448) ** 2) ** 0.5 for x, y in wide) == pytest.approx(100.4, abs=0.1)
    assert all(point_is_safe(p) for p in die_five_points(71))


def test_candidate_consensus_recovers_marker_hidden_by_early_corner_triggers():
    from trueskate_ai.collection.die_five_calibration import earliest_candidate_consensus
    # Upper corners fire early on scenery (495/496) but also see the marker at 520.
    candidates = [[521], [496, 520], [495, 521], [520], [520]]
    assert die_five_consensus([521, 496, 495, 520, 520]) == (None, 3)
    assert earliest_candidate_consensus(candidates) == (520, 5)
    assert earliest_candidate_consensus([[10], [30], [50], [70], [90]]) == (None, 1)


def test_candidates_include_every_passing_onset():
    from trueskate_ai.collection.die_five_calibration import tap_onset_candidates
    frames, times = _marker_window(onset_s=1.0, decoy_s=0.5)
    found = tap_onset_candidates(frames, times, point_xy=DIE_FIVE_POINTS[0], command_s=0.9)
    assert 15 in found and 30 in found
