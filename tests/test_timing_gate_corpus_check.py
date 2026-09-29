import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "timing_gate_corpus_check",
    Path(__file__).parents[1] / "scripts" / "inspect" / "timing_gate_corpus_check.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _segment(root: Path, name: str, rate: float, start_latency: float, end_latency: float) -> None:
    session = root / "iPhone_XR" / "park" / name
    started = 1000.0
    (session / "park" / "sample_000001").mkdir(parents=True)
    (session / "segment_00000.json").write_text(json.dumps({
        "started_at_epoch_s": started,
        "gestures": [
            {"calibration_control": True, "wda_action_sequence": 0, "wda_submitted_epoch_s": started + 2.0},
            {"calibration_control": True, "wda_action_sequence": 9, "wda_submitted_epoch_s": started + 55.0},
        ]}))
    (session / "park" / "sample_000001" / "meta.json").write_text(json.dumps({
        "session": name, "segment_index": 0, "device": "iPhone_XR", "park": "LA",
        "tap_calibration": {"method": "wda-submitted-two-centre-controls-v2", "rate": rate, "detections": [
            {"role": "start", "wda_action_sequence": 0, "onset_video_s": 2.0 + start_latency},
            {"role": "end", "wda_action_sequence": 9, "onset_video_s": 55.0 + end_latency}]}}))


def test_gate_counts_ordinary_passes_and_early_high_rate_starts(tmp_path):
    _segment(tmp_path, "s1", 1.0001, 0.140, 0.150)
    _segment(tmp_path, "s2", 1.004, -0.300, 0.140)
    _segment(tmp_path, "s3", 1.0001, 0.050, 0.140)
    result = gate.analyse(tmp_path)
    assert result["segments"] == 3
    assert result["criterion_1_ordinary_pass"]["start"] == {"n": 2, "k": 1}
    assert result["criterion_2_high_rate"]["start_fails_early"] == {"n": 1, "k": 1}
    assert result["criterion_2_high_rate"]["end_passes"] == {"n": 1, "k": 1}
    assert result["gate_vs_screen"]["gate_fail_segments"] == 2
    assert result["verdict"] == "mapping_not_accurate"
