import json

import pytest

from trueskate_ai.data.timing_screen import TimingScreen, passes, predicted_onset_error_s


def _meta(rate: float, gesture_s: float, *, start_s: float = 100.0, end_s: float = 155.0) -> dict:
    return {
        "wda_submitted_monotonic_s": gesture_s,
        "tap_calibration": {
            "method": "wda-submitted-two-centre-controls-v2",
            "rate": rate,
            "detections": [
                {"role": "start", "submitted_to_ios_monotonic_s": start_s},
                {"role": "end", "submitted_to_ios_monotonic_s": end_s},
            ],
        },
    }


def test_predicted_error_follows_the_implicated_anchor():
    assert predicted_onset_error_s(_meta(1.004, 105.0), 0.002) == ("start", pytest.approx(0.2))
    assert predicted_onset_error_s(_meta(0.996, 150.0), 0.002) == ("end", pytest.approx(0.2))
    assert predicted_onset_error_s(_meta(1.0015, 105.0), 0.002) == (None, 0.0)


def test_screen_keeps_late_clips_of_a_high_rate_segment():
    screen = TimingScreen(rate_threshold=0.002, max_error_frames=1.0)
    assert not passes(_meta(1.004, 105.0), screen)   # 200 ms predicted
    assert passes(_meta(1.004, 147.0), screen)       # 32 ms predicted
    assert passes(_meta(1.0001, 101.0), screen)      # ordinary segment


def test_screen_rejects_clips_without_two_anchor_calibration():
    with pytest.raises(ValueError):
        passes({"tap_calibration": {"accepted": True}}, TimingScreen(0.002, 1.0))


def test_cohort_manifest_records_screen_and_exclusions(tmp_path):
    from tests.test_model1_scaling_protocol import _linear_sample
    from trueskate_ai.model1.scaling import build_linear_cohort_manifest

    root = tmp_path / "corpus"
    samples = [_linear_sample(root, i, device="iPhone_XR", park="SLS 2015 Los Angeles") for i in range(4)]
    for sample, (rate, gesture) in zip(samples, ((1.004, 105.0), (1.004, 150.0), (1.0, 110.0), (0.99, 150.0))):
        meta_path = sample / "meta.json"
        meta = json.loads(meta_path.read_text())
        extra = _meta(rate, gesture)
        meta["tap_calibration"] = {"accepted": True, **extra["tap_calibration"]}
        meta["wda_submitted_monotonic_s"] = extra["wda_submitted_monotonic_s"]
        meta_path.write_text(json.dumps(meta))

    plain = build_linear_cohort_manifest(root, cohort="plain", role="training")
    screened = build_linear_cohort_manifest(
        root, cohort="screened", role="training",
        timing_screen=TimingScreen(rate_threshold=0.002, max_error_frames=1.0),
    )
    assert plain["sample_count"] == 4 and "timing_screen" not in plain
    assert screened["sample_count"] == 2
    assert screened["timing_screen"]["excluded"] == 2
    assert screened["timing_screen"]["excluded_by_park"] == {"SLS 2015 Los Angeles": 2}


def test_error_below_the_rate_threshold_is_treated_as_zero():
    # Documents the known limit (M1-SCREEN-DRAFT red-team): at 0.002 a segment at
    # rate 1.0019 keeps its first clip although the formula predicts ~3 frames.
    meta = _meta(1.0019, 100.0)
    assert passes(meta, TimingScreen(rate_threshold=0.002, max_error_frames=1.0))
    assert not passes(meta, TimingScreen(rate_threshold=0.0006, max_error_frames=1.0))


def _multi(gesture: float, *, accepted=True, inliers=(100.0, 120.0, 155.0), outliers=(90.0,)):
    detections = [{"submitted_to_ios_monotonic_s": t, "inlier": True} for t in inliers]
    detections += [{"submitted_to_ios_monotonic_s": t, "inlier": False} for t in outliers]
    return {"wda_submitted_monotonic_s": gesture,
            "tap_calibration": {"method": "wda-submitted-multi-centre-controls-v1",
                                "accepted": accepted, "detections": detections}}


def test_multi_anchor_clips_pass_only_between_accepted_inliers():
    screen = TimingScreen(rate_threshold=0.0008, max_error_frames=1.0)
    assert passes(_multi(130.0), screen)
    assert not passes(_multi(95.0), screen)              # extrapolated before first inlier
    assert not passes(_multi(130.0, accepted=False), screen)
    assert not passes(_multi(130.0, inliers=(100.0, 155.0)), screen)
