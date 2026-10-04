from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from trueskate_ai.utils import notify as notify_mod


@pytest.fixture
def sent(tmp_path, monkeypatch):
    """Capture outgoing ntfy requests; isolate dedupe state in tmp_path."""
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")
    monkeypatch.setenv("NTFY_STATE_DIR", str(tmp_path / "state"))
    bodies: list[str] = []
    monkeypatch.setattr(notify_mod.request, "urlopen",
                        lambda req, **kw: bodies.append(req.data.decode("utf-8")))
    return bodies


def _log(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "state" / "sent.log").read_text().splitlines()]


def test_repeats_are_suppressed_even_when_numbers_change(sent, tmp_path):
    for gb in ("7.9", "7.8", "7.7"):
        notify_mod.notify(f"device free storage low: {gb}GB", title="T", block=True)
    assert sent == ["device free storage low: 7.9GB"]
    assert [e["event"] for e in _log(tmp_path)] == ["sent", "suppressed", "suppressed"]


def test_next_send_after_window_reports_suppressed_count(sent, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(notify_mod.time, "time", lambda: clock[0])
    for _ in range(3):
        notify_mod.notify("XR1 died", title="T", block=True)
    clock[0] += notify_mod.DEDUPE_S + 1
    notify_mod.notify("XR1 died", title="T", block=True)
    assert sent == ["XR1 died", "XR1 died\n(+2 identical alerts suppressed)"]


def test_distinct_alerts_and_disabled_dedupe_both_send(sent):
    notify_mod.notify("XR1 died", title="T", block=True)
    notify_mod.notify("XR2 died", title="T", block=True)
    notify_mod.notify("XR1 died", title="Other", block=True)
    notify_mod.notify("XR1 died", title="T", block=True, dedupe_s=0)
    assert len(sent) == 4


def test_unreadable_state_never_silences_an_alert(sent, tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "state.json").write_text("{not json")
    notify_mod.notify("disk low", block=True)
    assert sent == ["disk low"]


def test_latch_alerts_once_per_incident(sent):
    assert notify_mod.notify_once("storage_XR1", "storage low A", block=True)
    assert not notify_mod.notify_once("storage_XR1", "storage low B", block=True)
    assert notify_mod.clear_latch("storage_XR1")
    assert not notify_mod.clear_latch("storage_XR1")
    # A new incident alerts again (dedupe disabled so only the latch is tested).
    assert notify_mod.notify_once("storage_XR1", "storage low C", block=True, dedupe_s=0)
    assert sent == ["storage low A", "storage low C"]


def _launch_services():
    path = Path(__file__).parents[1] / "scripts" / "launch_services.py"
    spec = importlib.util.spec_from_file_location("launch_services_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_services_flapping_is_one_incident_until_stable():
    ls = _launch_services()
    procs: dict = {}
    alerts = []
    t = 0.0
    # A fault that "recovers" and returns within seconds, many times over.
    for _ in range(20):
        if ls._incident_failure(procs):
            alerts.append("died")
        for _ in range(5):
            t += 2
            if ls._incident_healthy_tick(procs, t):
                alerts.append("recovered")
    assert alerts == ["died"]
    # Stable for the full window: exactly one recovery, then nothing further.
    for _ in range(int(ls._INCIDENT_STABLE_S / 2) + 2):
        t += 2
        if ls._incident_healthy_tick(procs, t):
            alerts.append("recovered")
    assert alerts == ["died", "recovered"]
    # A later, separate fault is a new incident.
    assert ls._incident_failure(procs)
