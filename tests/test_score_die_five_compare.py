import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "inspect" / "score_die_five_compare.py"
    spec = importlib.util.spec_from_file_location("test_score_die_five", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _segment(name, a_start, b_start, *, mid_a=300, mid_b=300):
    # 30 fps, WDA monotonic == video seconds, so a correct fit has rate 1.
    return {
        "segment": name, "device": "iPhone_XR", "park": "SLS 2015 Los Angeles",
        "markers": [
            {"wda_action_sequence": 0, "role": "start", "marker": "die_five",
             "wda_submitted_monotonic_s": 3.0, "a_frame": a_start, "b_frame": b_start, "a_score": 12.0},
            {"wda_action_sequence": 3, "role": "mid", "marker": "die_five",
             "wda_submitted_monotonic_s": 10.0, "a_frame": mid_a, "b_frame": mid_b, "a_score": 30.0},
            {"wda_action_sequence": 6, "role": "mid", "marker": "single",
             "wda_submitted_monotonic_s": 20.0, "a_frame": 600, "b_frame": None, "a_score": 30.0},
            {"wda_action_sequence": 9, "role": "end", "marker": "die_five",
             "wda_submitted_monotonic_s": 50.0, "a_frame": 1500, "b_frame": 1500, "a_score": 30.0},
        ],
        "fits": {
            "a": {"rate": (50.0 - a_start / 30) / 47.0, "mid_predicted_s": {"3": 10.0, "6": 20.0}},
            "b": {"rate": 1.0, "mid_predicted_s": {"3": 10.0, "6": 20.0}},
        },
        "null_windows": [{"a_frame": 5, "b_frame": None}, {"a_frame": None, "b_frame": None}],
        "null_scanned_s": 3.5,
        "post_reset_null": {"scanned_s": 2.5, "a_frame": 40, "b_frame": None},
    }


def test_scoring_counts_primary_discordance_weights_and_false_alarms():
    module = _module()
    # Truth: start at frame 90. A fires 4 frames early in two recordings.
    analysis = [_segment("s1", 86, 90), _segment("s2", 86, 90), _segment("s3", 90, 90)]
    items, labels = [], {}
    for index, (seg, seq, role, marker, stratum, truth) in enumerate(
        (s, q, r, m, st, t)
        for s in ("s1", "s2", "s3")
        for q, r, m, st, t in ((0, "start", "die_five", "anchor_all_labelled", 90),
                               (9, "end", "die_five", "anchor_all_labelled", 1500),
                               (3, "mid", "die_five", "agree", 300),
                               (6, "mid", "single", "single", 600))
    ):
        item_id = f"{index:03d}"
        items.append({"id": item_id, "first_global_frame": truth - 20, "n_frames": 80, "segment": seg,
                      "wda_action_sequence": seq, "role": role, "marker": marker, "stratum": stratum})
        labels[item_id] = {"frame_index_0based": 20, "uncertain": False}
    result = module.score(analysis, {"seed": 1, "items": items}, labels,
                          frame_times=lambda _: [i / 30 for i in range(2000)])

    primary = result["primary_la_start"]
    assert primary["n_recordings"] == 3
    assert (primary["a_gross"], primary["b_gross"]) == (2, 0)
    assert (primary["discordant_a_only"], primary["discordant_b_only"]) == (2, 0)
    assert primary["mcnemar_exact_p"] == pytest.approx(0.5)
    assert result["b_accepted"]["gross"] == 0
    assert result["b_accepted"]["rule_of_three_upper"] == pytest.approx(3 / 9, abs=1e-4)
    assert result["b_start_rejection"]["iPhone_XR"]["b_no_result_rate"] == 0.0
    assert result["mid_stratum_weights"] == {"agree": 1.0}
    assert result["corner_check"]["pass"] is True
    assert result["fit_replay"]["human"]["anomalous"] == 0
    assert result["fit_replay"]["a"]["anomalous"] == 2
    assert result["fit_replay"]["b"]["mid_residual_within_1"] == 1.0
    fa = result["false_alarms"]
    assert fa["a"]["steady_alarms"] == 3 and fa["b"]["steady_alarms"] == 0
    assert fa["a"]["post_reset_alarms_per_s"] == pytest.approx(3 / 7.5, abs=1e-4)


def test_uncertain_labels_are_excluded():
    module = _module()
    analysis = [_segment("s1", 90, 90)]
    items = [{"id": "000", "first_global_frame": 70, "n_frames": 80, "segment": "s1",
              "wda_action_sequence": 0, "role": "start", "marker": "die_five", "stratum": "anchor_all_labelled"}]
    result = module.score(analysis, {"seed": 1, "items": items},
                          {"000": {"frame_index_0based": None, "uncertain": True}},
                          frame_times=lambda _: [i / 30 for i in range(2000)])
    assert result["uncertain"] == 1 and result["labelled"] == 0
