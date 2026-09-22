"""Foreground checks must recognize iOS overlays that cover True Skate."""
from __future__ import annotations

import requests

from trueskate_ai.sim import device


class _Driver:
    def __init__(self, state: int = 4) -> None:
        self.state = state
        self.activated: list[str] = []

    def query_app_state(self, _bundle_id: str) -> int:
        return self.state

    def activate_app(self, bundle_id: str) -> None:
        self.activated.append(bundle_id)


class _Response:
    def __init__(self, bundle_id: str) -> None:
        self.bundle_id = bundle_id

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"value": {"bundleId": self.bundle_id}}


def _session() -> device.DeviceSession:
    session = device.DeviceSession({
        "name": "iPhone_XR2",
        "logical_w": 414,
        "logical_h": 896,
        "wda_port": 8103,
    })
    session.driver = _Driver()
    return session


def test_control_center_is_not_treated_as_true_skate_foreground(monkeypatch):
    session = _session()
    monkeypatch.setattr(device.requests, "get", lambda *_args, **_kwargs: _Response("com.apple.springboard"))
    monkeypatch.setattr(device.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(device, "skip_loading_screen", lambda *_args, **_kwargs: None)

    assert session.ensure_foreground()
    assert session.driver.activated == [device.BUNDLE_ID]


def test_true_skate_active_bundle_needs_no_relaunch(monkeypatch):
    session = _session()
    monkeypatch.setattr(device.requests, "get", lambda *_args, **_kwargs: _Response(device.BUNDLE_ID))

    assert not session.ensure_foreground()
    assert session.driver.activated == []


def test_active_app_query_failure_preserves_state_fallback(monkeypatch):
    session = _session()

    def unavailable(*_args, **_kwargs):
        raise requests.RequestException("unavailable")

    monkeypatch.setattr(device.requests, "get", unavailable)
    assert not session.ensure_foreground()
    assert session.driver.activated == []
