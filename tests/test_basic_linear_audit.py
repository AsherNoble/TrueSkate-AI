import json
from pathlib import Path
import subprocess
import sys

from trueskate_ai.vision.basic_linear_audit import audit_basic_linear_corpus


def _sample(root: Path, name: str, *, device: str | None, points: list[list[float]],
            duration: float = .6, kind: str = "linear", calibrated: bool = True,
            park: str = "The Workshop") -> Path:
    sample = root / "session" / "the_workshop" / name
    sample.mkdir(parents=True)
    meta = {
        "gesture_distribution": kind, "waypoints": points, "duration": duration,
        "easing_power": 1.0, "spin_active": False,
        "tap_calibration": {"accepted": calibrated}, "park": park,
    }
    if device is not None:
        meta["device"] = device
    (sample / "meta.json").write_text(json.dumps(meta))
    (sample / "frame_000.png").touch()
    return sample


def test_audit_reports_strict_counts_provenance_duplicates_and_coverage(tmp_path):
    command = [[.25, .35], [.65, .55]]
    _sample(tmp_path, "xr1_a", device="iPhone_XR", points=command)
    _sample(tmp_path, "xr2_duplicate", device="iPhone_XR2", points=command)
    _sample(tmp_path, "xr1_b", device="iPhone_XR", points=[[.72, .62], [.40, .35]], duration=.9)
    _sample(tmp_path, "tap", device="iPhone_XR", points=command, kind="tap")
    _sample(tmp_path, "uncalibrated", device="iPhone_XR", points=command, calibrated=False)

    report = audit_basic_linear_corpus(tmp_path, position_bins=2, numeric_bins=3,
                                       sparse_cell_max_count=0)

    assert report["strict_counts"] == {
        "accepted": 3, "discovered": 5, "rejected_not_linear": 1, "rejected_uncalibrated": 1,
    }
    assert report["provenance"] == {
        "per_device": {"iPhone_XR": 2, "iPhone_XR2": 1},
        "per_park": {"The Workshop": 3},
        "parks_by_device": {"iPhone_XR": ["The Workshop"], "iPhone_XR2": ["The Workshop"]},
        "explicit_device_provenanced": 3,
        "missing_device_provenance": 0,
    }
    assert report["duplicates"]["duplicate_samples"] == 1
    assert report["duplicates"]["cross_device_groups"] == 1
    assert report["duplicates"]["duplicate_groups"][0]["count"] == 2
    assert report["coverage"]["nearest_command_spacing"]["unique_commands"] == 2
    assert report["coverage"]["nearest_command_spacing"]["min"] > 0
    assert sum(bin_["count"] for bin_ in report["coverage"]["duration"]) == 3
    assert sum(bin_["count"] for bin_ in report["coverage"]["slope"]) == 3
    assert sum(bin_["count"] for bin_ in report["coverage"]["displacement"]) == 3
    assert report["coverage"]["start_position"]["bins_per_axis"] == 2


def test_audit_cli_enforces_device_park_and_unique_command_gate(tmp_path):
    _sample(tmp_path, "one", device="iPhone_XR", points=[[.25, .35], [.65, .55]])
    command = [sys.executable, "scripts/data/audit_basic_linear_corpus.py", "--data", str(tmp_path),
               "--require-device", "iPhone_XR", "--require-park", "The Workshop",
               "--min-per-device", "1", "--require-unique-commands"]
    passed = subprocess.run(command, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(passed.stdout)["accepted"] == 1
    assert "AUDIT PASSED" in passed.stderr

    failed = subprocess.run(command + ["--require-device", "iPhone_XR2"], capture_output=True, text=True)
    assert failed.returncode == 2
    assert "iPhone_XR2: 0 strict device-provenanced clips < 1" in failed.stderr


def test_audit_cli_enforces_per_device_park_provenance(tmp_path):
    _sample(tmp_path, "xr1", device="iPhone_XR", points=[[.25, .35], [.65, .55]],
            park="SLS 2015 Super Crown")
    _sample(tmp_path, "xr2", device="iPhone_XR2", points=[[.35, .35], [.75, .55]],
            park="SLS 2013 Kansas City")
    command = [sys.executable, "scripts/data/audit_basic_linear_corpus.py", "--data", str(tmp_path),
               "--require-device-park", "iPhone_XR=SLS 2015 Super Crown",
               "--require-device-park", "iPhone_XR2=SLS 2013 Kansas City"]
    passed = subprocess.run(command, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr

    failed = subprocess.run(command[:-1] + ["iPhone_XR2=SLS 2015 Super Crown"],
                             capture_output=True, text=True)
    assert failed.returncode == 2
    assert "iPhone_XR2: accepted parks ['SLS 2013 Kansas City']" in failed.stderr
