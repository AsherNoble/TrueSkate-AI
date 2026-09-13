"""Opt-in client and validation for the instrumented WebDriverAgent fork."""

from __future__ import annotations

import json
import math
from urllib.parse import quote
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
BOUNDARIES = (
    "request_entered",
    "preparation_started",
    "preparation_finished",
    "submitted_to_ios",
    "ios_completion_callback",
    "stability_wait_started",
    "stability_wait_finished",
    "request_finished",
)


def _http_json(url: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"unexpected WDA response type: {type(value).__name__}")
    return value


def validate_action_timing_report(
    report: dict,
    *,
    expected_revision: str,
    expected_count: int,
) -> list[dict]:
    """Return ordered records or reject incomplete/misattributed timing data."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected WDA timing schema")
    if report.get("build_revision") != expected_revision:
        raise ValueError("unexpected WDA timing build revision")
    records = report.get("records")
    if report.get("dropped_records") != 0 or not isinstance(records, list):
        raise ValueError("missing or overflowed WDA timing records")
    if len(records) != expected_count:
        raise ValueError(
            f"expected {expected_count} WDA timing records; found {len(records)}"
        )

    session_id = None
    for index, record in enumerate(records):
        if record.get("sequence") != index or record.get("outcome") != "success":
            raise ValueError("unordered or failed WDA action")
        if record.get("missing_ios_callback") or not record.get("ios_callback_result"):
            raise ValueError("missing or unsuccessful iOS callback")
        current_session = record.get("session_id")
        if not current_session or (
            session_id is not None and current_session != session_id
        ):
            raise ValueError("WDA session changed during timing capture")
        session_id = current_session
        monotonic = []
        for boundary in BOUNDARIES:
            stamp = record.get(boundary)
            if not isinstance(stamp, dict):
                raise ValueError(f"missing WDA timing boundary: {boundary}")
            mono = stamp.get("monotonic_s")
            epoch = stamp.get("epoch_s")
            if not isinstance(mono, (int, float)) or not math.isfinite(mono):
                raise ValueError(f"invalid monotonic timestamp: {boundary}")
            if not isinstance(epoch, (int, float)) or not math.isfinite(epoch):
                raise ValueError(f"invalid epoch timestamp: {boundary}")
            monotonic.append(float(mono))
        if any(later < earlier for earlier, later in zip(monotonic, monotonic[1:])):
            raise ValueError("invalid WDA timestamp ordering")
    return records


class WDAActionTimingCapture:
    """Manage one request-scoped timing capture on an existing WDA session."""

    def __init__(
        self,
        *,
        wda_port: int,
        expected_revision: str,
        request_json=_http_json,
    ) -> None:
        self.base_url = f"http://127.0.0.1:{int(wda_port)}"
        self.expected_revision = expected_revision
        self.request_json = request_json
        self.url: str | None = None

    @property
    def active(self) -> bool:
        return self.url is not None

    def start(self) -> None:
        if self.active:
            raise RuntimeError("WDA action timing capture is already active")
        status = self.request_json(self.base_url + "/status")
        session_id = status.get("sessionId")
        if not session_id:
            raise RuntimeError("WDA status has no active session")
        url = (
            self.base_url
            + "/session/"
            + quote(str(session_id), safe="")
            + "/wda/actionTiming"
        )
        current = self.request_json(url).get("value", {})
        if current.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("instrumented WDA timing schema is unavailable")
        if current.get("build_revision") != self.expected_revision:
            raise RuntimeError("instrumented WDA build identity mismatch")
        self.url = url
        try:
            enabled = self.request_json(url, {"enabled": True}).get("value", {})
            if not enabled.get("enabled") or enabled.get("records"):
                raise RuntimeError("WDA timing capture failed to initialize cleanly")
        except Exception:
            self.cleanup()
            raise

    def stop(self) -> dict:
        if not self.active:
            raise RuntimeError("WDA action timing capture is not active")
        url = self.url
        try:
            result = self.request_json(url, {"enabled": False}).get("value", {})
        finally:
            self.url = None
        return result

    def cleanup(self) -> None:
        """Best-effort disable after an interrupted or failed segment."""
        if not self.active:
            return
        try:
            self.stop()
        except Exception:  # noqa: BLE001 - cleanup cannot hide the original error
            self.url = None
