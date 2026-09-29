import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "inspect" / "score_multianchor_against_labels.py"
    spec = importlib.util.spec_from_file_location("test_score_multianchor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outlier_matches_gross_error_and_loo_prediction_recovers_bad_start():
    module = _module()
    wda = [3.0, 15.0, 28.0, 40.0, 50.0]
    truth = [int(round(w * 30)) for w in wda]
    a = list(truth)
    a[0] -= 10  # A fires ten frames early on the start
    markers = [{"wda_action_sequence": i, "role": r, "wda_submitted_monotonic_s": w, "a_frame": f}
               for i, (r, w, f) in enumerate(zip(("start", "mid", "mid", "mid", "end"), wda, a))]
    rate = (wda[-1] - a[0] / 30) / (wda[-1] - wda[0])  # A's bent two-anchor fit
    analysis = [{"segment": "s.json", "markers": markers,
                 "fits": {"a": {"rate": rate, "intercept_s": a[0] / 30 - rate * wda[0]}}}]
    human = {("s.json", i): f for i, f in enumerate(truth)}
    result = module.score(analysis, human, frame_times=lambda _: [i / 30 for i in range(2000)])
    assert result["outlier_vs_A_gross"] == {"outlier_and_gross": 1, "outlier_not_gross": 0,
                                            "inlier_and_gross": 0, "inlier_not_gross": 4}
    assert result["prediction_error_frames"]["multi_anchor_loo"]["gross_gt_1"] == 0
    assert result["prediction_error_frames"]["two_anchor"]["gross_gt_1"] >= 1
