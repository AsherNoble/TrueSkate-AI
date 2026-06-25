"""XCTest-native screen recording — the headless 30fps iOS capture path.

The AVFoundation/CoreMediaIO "DAL" screen-mirror (``vision/dal_capture``) is wedged
on the rig's current macOS/iOS: only QuickTime's private, process-local stream can
read it (see memory ``ios-dal-screen-capture-wedge``). This module instead drives
Apple's own **XCTest screen recording** through Appium
(``mobile: startXCTestScreenRecording``), which on the iPhone XR / iOS 18.7 captures
the FULL screen (828x1792) as **~30fps H.264** and **coexists with the live
WDA/Appium automation** — it IS that XCUITest session, so gestures fire normally
while it records (measured 2026-06-25: 330 frames / 11.03s = 29.9 fps, h264).

KEY DIFFERENCE vs DAL: this is SEGMENT-based, not a real-time frame stream. You
``start()`` a recording, fire gestures (logging each one's host timestamp), then
``stop_and_save()`` which retrieves the whole segment as one ``.mov`` to the host.
Frames are aligned to gestures POST-HOC from the recording's ``startedAt`` anchor +
each frame's PTS (+ a measured command->pixel offset Δ). So the collector records a
bounded segment per period, saves it on the host (the "training-server" Mac), and a
separate aligner slices per-gesture frame windows out of the ``.mov``.

Requirements:
  * Appium launched with ``--allow-insecure xcuitest:xctest_screen_record`` (the
    security gate; see ``scripts/launch_services.py``). Without it, start raises a
    WebDriverException about the insecure feature not being enabled.
  * iOS 18+ real device: the appium-xcuitest driver auto-deletes the on-device
    recording attachment on stop, so device free space stays high across segments.

Segment sizing — CRITICAL: ``stop_and_save`` retrieves the ENTIRE segment as one
base64 blob inside a single Appium/WDA HTTP response (see ``stop_and_save``). Gameplay
motion runs ~76 MB/min at 30fps full-res (measured 2026-06-26, not the ~37 the old
note assumed). Retrieval is reliable to ~114 MB / ~90s; beyond that the WDA test-runner
chokes serializing the payload and aborts the connection (``RemoteDisconnected`` →
the collector crashes and the segment is lost). Keep ``--segment-min`` at 1 (~77 MB).
A 5-min segment (~380 MB) fails every time — that bug produced 121 crash-restarts and
zero saved segments before it was found.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

# appium-xcuitest mobile command names (note the XCTEST capitalisation — the
# lower-case 'Xctest' spelling raises UnknownMethodException on driver v11).
_START_CMD = "mobile: startXCTestScreenRecording"
_STOP_CMD = "mobile: stopXCTestScreenRecording"
_INFO_CMD = "mobile: getXCTestScreenRecordingInfo"


@dataclass
class RecordingResult:
    """Outcome of one recorded segment, with the anchors an aligner needs.

    All times are EPOCH SECONDS on the host/Appium clock. The driver's ``startedAt``
    is epoch seconds (float) and shares the host clock (measured ~0.45s after the
    host ``start()`` call — recording-init latency), so a gesture fired at host
    ``t_epoch_s`` maps to video PTS ``t_epoch_s - started_at_epoch_s`` (+ the
    command->pixel offset Δ).
    """

    mov_path: Path
    started_at_epoch_s: float     # driver-reported epoch-sec when recording began (video t0)
    fps: int                      # fps the driver reports it recorded at
    codec: str
    uuid: str
    n_bytes: int
    host_start_epoch_s: float     # host time.time() right before start returned
    host_stop_epoch_s: float      # host time.time() right after stop returned

    def video_time_for(self, gesture_epoch_s: float, delta_s: float = 0.0) -> float:
        """Video PTS (s into the .mov) showing the effect of a gesture fired at
        ``gesture_epoch_s`` host time, given a measured command->pixel offset Δ."""
        return (gesture_epoch_s - self.started_at_epoch_s) + delta_s

    def summary(self) -> dict:
        return {
            "mov": str(self.mov_path),
            "started_at_epoch_s": self.started_at_epoch_s,
            "fps": self.fps,
            "codec": self.codec,
            "uuid": self.uuid,
            "mb": round(self.n_bytes / 1e6, 2),
            "host_start_epoch_s": self.host_start_epoch_s,
            "host_stop_epoch_s": self.host_stop_epoch_s,
            "init_latency_s": round(self.started_at_epoch_s - self.host_start_epoch_s, 3),
        }


class XCTestScreenRecorder:
    """Drives one XCTest screen-recording segment over an Appium driver.

    Stateless between segments: ``start()`` then ``stop_and_save(path)``. The driver
    must be a live XCUITest session (e.g. ``DeviceWorker.driver``); the recording
    runs on the same session that fires gestures, so they coexist.
    """

    def __init__(self, driver, *, fps: int = 30) -> None:
        self.driver = driver
        self.fps = fps
        self._recording = False
        self._host_start_s = 0.0
        self._start_info: dict = {}

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> dict:
        """Begin a recording segment. Returns the driver's start info dict."""
        if self._recording:
            raise RuntimeError("XCTestScreenRecorder already recording; stop first.")
        self._host_start_s = time.time()
        info = self.driver.execute_script(_START_CMD, {"fps": self.fps}) or {}
        self._start_info = info if isinstance(info, dict) else {}
        self._recording = True
        return self._start_info

    def stop_and_save(self, mov_path: str | Path) -> RecordingResult:
        """Stop, retrieve the segment .mov to ``mov_path`` on the host, return anchors.

        The whole .mov comes back base64 in ONE HTTP response (``res["payload"]``). This
        caps usable segment length: >~114 MB aborts the connection. See module docstring.
        """
        if not self._recording:
            raise RuntimeError("XCTestScreenRecorder is not recording.")
        res = self.driver.execute_script(_STOP_CMD, {})
        host_stop_s = time.time()
        self._recording = False
        if not isinstance(res, dict):
            raise RuntimeError(f"unexpected stop result type: {type(res).__name__}: {str(res)[:120]}")
        payload = res.get("payload")
        if not payload:
            raise RuntimeError(f"stop returned no payload; keys={list(res.keys())}")
        data = base64.b64decode(payload)
        mov_path = Path(mov_path)
        mov_path.parent.mkdir(parents=True, exist_ok=True)
        mov_path.write_bytes(data)
        started_at = res.get("startedAt", self._start_info.get("startedAt", 0)) or 0
        return RecordingResult(
            mov_path=mov_path,
            started_at_epoch_s=float(started_at),
            fps=int(res.get("fps") or self.fps),
            codec=str(res.get("codec") or "?"),
            uuid=str(res.get("uuid") or ""),
            n_bytes=len(data),
            host_start_epoch_s=self._host_start_s,
            host_stop_epoch_s=host_stop_s,
        )

    def abort(self) -> None:
        """Best-effort stop without saving (e.g. on error/shutdown). Frees the device."""
        if not self._recording:
            return
        try:
            self.driver.execute_script(_STOP_CMD, {})
        except Exception:  # noqa: BLE001 — best effort
            pass
        self._recording = False
