from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time


REPO_ROOT = Path(__file__).parents[1]
WATCHDOG = REPO_ROOT / "scripts" / "collection_watchdog.sh"


def _run_watchdog(tmp_path: Path, push_log: Path, *legacy_args: str) -> list[str]:
    env = os.environ | {
        "REPO": str(tmp_path),
        "DATA": str(tmp_path / "data" / "sls_xctest"),
        "TRUESKATE_WATCHDOG_STATE_DIR": str(tmp_path / "state"),
        "WATCHDOG_PUSH_LOG": str(push_log),
        "WATCHDOG_ONCE": "1",
        "NEVER_ARMED_ALERT_SECONDS": "0",
        "WDA_STATUS_TIMEOUT": "0.01",
    }
    result = subprocess.run(
        ["bash", str(WATCHDOG), *legacy_args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return push_log.read_text().splitlines() if push_log.exists() else []


def _fresh_segment(root: Path, device_tag: str) -> None:
    segment = root / "data" / "sls_xctest" / f"{device_tag}_session" / "segment_00000.json"
    segment.parent.mkdir(parents=True, exist_ok=True)
    segment.write_text("{}")
    os.utime(segment, (time.time(), time.time()))


def test_fleet_watchdog_notifies_only_on_persistent_state_transitions(tmp_path):
    push_log = tmp_path / "notifications.log"

    # Two legacy launchd jobs observe the same total outage, but the shared state
    # produces only one fleet-level notification.
    messages = _run_watchdog(tmp_path, push_log, "iPhone_XR", "8100", "XR1")
    messages = _run_watchdog(tmp_path, push_log, "iPhone_XR2", "8103", "XR2")
    assert len(messages) == 1
    assert "Rig down: XR1 and XR2" in messages[0]

    # A repeat check for the same incident is silent, even after a new process.
    assert _run_watchdog(tmp_path, push_log, "iPhone_XR", "8100", "XR1") == messages

    # Partial recovery is a single severity transition; full recovery is one
    # resolution notification, with no duplicates on later healthy checks.
    _fresh_segment(tmp_path, "iPhone_XR")
    messages = _run_watchdog(tmp_path, push_log, "iPhone_XR", "8100", "XR1")
    assert len(messages) == 2
    assert "Rig degraded: XR2" in messages[-1]

    # Recovery is announced only after two consecutive healthy checks.
    _fresh_segment(tmp_path, "iPhone_XR2")
    assert len(_run_watchdog(tmp_path, push_log, "iPhone_XR2", "8103", "XR2")) == 2
    messages = _run_watchdog(tmp_path, push_log, "iPhone_XR2", "8103", "XR2")
    assert len(messages) == 3
    assert "Rig recovered" in messages[-1]
    assert _run_watchdog(tmp_path, push_log, "iPhone_XR", "8100", "XR1") == messages


def test_watchdog_hovering_at_threshold_does_not_alternate_alerts(tmp_path):
    push_log = tmp_path / "notifications.log"
    _run_watchdog(tmp_path, push_log)  # both down
    segments = list((tmp_path / "data" / "sls_xctest").rglob("*.json"))
    assert segments == []
    # Healthy once, stale again, healthy once again: never two healthy checks
    # in a row, so no recovery alert and no repeated "down" alert.
    for _ in range(3):
        _fresh_segment(tmp_path, "iPhone_XR")
        _fresh_segment(tmp_path, "iPhone_XR2")
        _run_watchdog(tmp_path, push_log)
        for segment in (tmp_path / "data" / "sls_xctest").rglob("*.json"):
            os.utime(segment, (time.time() - 3600, time.time() - 3600))
        messages = _run_watchdog(tmp_path, push_log)
    assert len(messages) == 1
    assert "Rig down" in messages[0]


def test_watchdog_finds_segments_in_device_bucketed_stage_layout(tmp_path):
    push_log = tmp_path / "notifications.log"
    root = tmp_path / "data" / "sls_xctest"
    for device in ("iPhone_XR", "iPhone_XR2"):
        segment = root / "basic_linear_sls_stage1" / f"{device}_sls_park" / f"{device}_session" / "segment_00000.json"
        segment.parent.mkdir(parents=True, exist_ok=True)
        segment.write_text("{}")
        os.utime(segment, (time.time(), time.time()))

    assert _run_watchdog(tmp_path, push_log, "iPhone_XR", "8100", "XR1") == []
