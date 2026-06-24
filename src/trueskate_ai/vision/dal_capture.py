"""AVFoundation / CoreMediaIO DAL screen capture — the 60-capable* USB pipe.

This is the same hardware H.264-over-USB path QuickTime uses, NOT WDA's MJPEG
server. It runs on a completely separate subsystem, so it coexists with
WDA + Appium (gestures/OCR) on the same phone.

Plain ffmpeg cannot see iOS devices: only a process that sets the CoreMediaIO
``kCMIOHardwarePropertyAllowScreenCaptureDevices`` flag (QuickTime does; ffmpeg
doesn't) gets them in the AVFoundation device list. We inject ``tools/enable_dal.c``
(built on first run) via ``DYLD_INSERT_LIBRARIES`` — see ``shim_env``. This module
owns that shim logic; ``scripts/view_device.py`` imports it (one source of truth).

*Measured ceiling (iPhone_XR / A12, 2026-06-22): the DAL screen-mirror stream is
hard-capped at a ROCK-STEADY 30fps regardless of the advertised 60fps formats —
see experiments/vision_sequence_leap_journal.md. Steady 30, not 60.

KEY DESIGN — persistent stream, windowed slicing (NOT start/stop per sample):
the DAL device is EXCLUSIVE and FRAGILE. Opening/closing it per sample (the cheap
pattern that works for HTTP MJPEG) both adds ~0.5-1s ffmpeg-launch latency per
sample AND wedges the CMIO session under rapid cycling (device enumerates, ffmpeg
opens it, but zero frames arrive; only a QuickTime re-activate / replug clears it).
So ``DalFrameRecorder`` opens ONE long-lived ffmpeg ``avfoundation -> mjpeg`` stream,
a reader thread timestamps every decoded frame into a rolling buffer, and each
caller ``window(t0, t1)`` slices the frames it needs. The device is opened once per
session, not per sample.

Address devices BY NAME — AVFoundation indices reorder between calls (a phone that
was ``[2]`` becomes ``[1]`` when FaceTime/another device shuffles), so an index can
silently open the wrong camera.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIM_SRC = _REPO_ROOT / "tools" / "enable_dal.c"
_SHIM_DYLIB = _REPO_ROOT / "tools" / "libenable_dal.dylib"

# DAL screen-mirror delivers a steady 30fps on the rig's A12 phones (journal
# 2026-06-22); the advertised 60fps formats are camera descriptors the mirror
# pipeline does not actually fill. Capture at 30 to match what arrives.
DEFAULT_FPS = 30


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")


def shim_env() -> dict:
    """Environment for ffmpeg with the CoreMediaIO DAL shim injected.

    Builds ``tools/libenable_dal.dylib`` from ``tools/enable_dal.c`` on first run
    (or when the source is newer). Needs clang (present via Xcode on these Macs).
    """
    if not _SHIM_DYLIB.exists() or _SHIM_DYLIB.stat().st_mtime < _SHIM_SRC.stat().st_mtime:
        subprocess.run(
            ["clang", "-x", "c", "-dynamiclib", str(_SHIM_SRC),
             "-framework", "CoreMediaIO", "-framework", "CoreFoundation",
             "-o", str(_SHIM_DYLIB)],
            check=True,
        )
    env = os.environ.copy()
    env["DYLD_INSERT_LIBRARIES"] = str(_SHIM_DYLIB)
    return env


def list_dal_devices() -> str:
    """Return ffmpeg's AVFoundation device-list output (printed to stderr)."""
    require_ffmpeg()
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True, env=shim_env(),
    )
    return proc.stderr.rstrip()


def resolve_device_name(substr: str) -> str | None:
    """Find the AVFoundation VIDEO device for the iOS **screen mirror** ``substr``.

    A connected iPhone exposes TWO video devices: the bare screen-mirror device
    (e.g. ``"Test XR_1"``) and a Continuity Camera (``"Test XR_1 Camera"``) — and
    the Camera is usually listed FIRST. A naive "first name containing substr"
    therefore returns the Camera feed (the lens, not the screen), silently
    capturing the wrong thing. So:

      1. exact case-insensitive match wins (the bare screen-mirror name), else
      2. the first containing match that is NOT a ``… Camera`` continuity device, else
      3. the first containing match (last-resort fallback).

    Returns None if no video device matches.
    """
    listing = list_dal_devices()
    in_video = False
    matches: list[str] = []
    for line in listing.splitlines():
        if "AVFoundation video devices:" in line:
            in_video = True
            continue
        if "AVFoundation audio devices:" in line:
            in_video = False
            continue
        if not in_video:
            continue
        m = re.search(r"\]\s*\[\d+\]\s*(.+?)\s*$", line)
        if m and substr.lower() in m.group(1).lower():
            matches.append(m.group(1))
    if not matches:
        return None
    for name in matches:  # 1: exact screen-mirror name
        if name.lower() == substr.lower():
            return name
    for name in matches:  # 2: skip the Continuity Camera variant
        if not name.lower().endswith(" camera"):
            return name
    return matches[0]      # 3: fallback


class DalFrameRecorder:
    """Long-lived AVFoundation→MJPEG capture with a rolling, time-indexed buffer.

    Open once per session with ``open(device_name)``; a reader thread decodes
    every frame and stamps it with ``time.monotonic()`` at decode (the same clock
    the gesture call-returns use, so windows align). ``window(t0, t1)`` returns the
    buffered ``(frames, times)`` in that monotonic interval. ``close()`` kills ffmpeg
    and releases the exclusive device — ALWAYS call it (a leaked ffmpeg wedges the
    next open).

    ffmpeg scales to ``resize_width`` on the GPU/CPU before encoding, so Python only
    decodes small JPEGs. The buffer is pruned to the last ``horizon_s`` seconds.
    """

    def __init__(self, *, fps: int = DEFAULT_FPS, resize_width: int | None = 512,
                 horizon_s: float = 6.0, quality: int = 5) -> None:
        self.fps = fps
        self.resize_width = resize_width
        self.horizon_s = horizon_s
        self.quality = quality
        self.device_name: str | None = None
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr_drain: threading.Thread | None = None
        self._buf: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()
        self._stop = False
        self._frame_count = 0
        self._last_frame_t = 0.0
        self._last_change_t = 0.0      # last time decoded CONTENT actually changed
        self._prev_sig: np.ndarray | None = None
        self._last_err = ""

    # -- lifecycle ----------------------------------------------------------
    def _ffmpeg_cmd(self, device_name: str) -> list[str]:
        scale = f",scale={self.resize_width}:-2" if self.resize_width else ""
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-framerate", str(self.fps),
            "-i", device_name,
            "-an", "-vf", f"fps={self.fps}{scale}",
            "-f", "mjpeg", "-q:v", str(self.quality), "-",
        ]

    def open(self, device_name: str) -> None:
        """Spawn the persistent ffmpeg stream and start buffering frames.

        Pass the EXACT screen-mirror device name (e.g. ``"Test XR_1"``); ffmpeg
        matches by leading substring, so a bare name would otherwise open the
        ``"Test XR_1 Camera"`` Continuity lens, not the screen. Use
        ``resolve_device_name`` to get the right name.
        """
        require_ffmpeg()
        if self._proc is not None:
            raise RuntimeError("DalFrameRecorder already open; call close() first.")
        self.device_name = device_name
        self._stop = False
        self._proc = subprocess.Popen(
            self._ffmpeg_cmd(device_name), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=shim_env(), bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_drain.start()

    def wait_for_frames(self, timeout_s: float = 8.0) -> bool:
        """Block until the first frame arrives (or the stream dies / times out).

        Returns True once frames are flowing. Use after ``open`` to confirm the DAL
        session is live before relying on the stream (a wedged device opens cleanly
        but never delivers frames)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._frame_count > 0:
                return True
            if self._proc is not None and self._proc.poll() is not None:
                return False  # ffmpeg exited (bad device / wedge / permission)
            time.sleep(0.1)
        return False

    def close(self) -> None:
        self._stop = True
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        for t in (self._reader, self._stderr_drain):
            if t is not None:
                t.join(timeout=2.0)
        self._reader = self._stderr_drain = None

    def restart(self) -> bool:
        """Tear down and re-open the same device (stall self-heal). Returns True
        if frames flow again. Re-opens the exclusive device, so only call on a
        confirmed stall, never per sample."""
        name = self.device_name
        if name is None:
            return False
        self.close()
        with self._lock:
            self._buf.clear()
        self._frame_count = 0
        self._prev_sig = None
        self._last_change_t = 0.0
        self.open(name)
        return self.wait_for_frames()

    # -- buffer access ------------------------------------------------------
    def window(self, t_start: float, t_end: float) -> tuple[list[np.ndarray], list[float]]:
        """Frames + monotonic timestamps captured in [t_start, t_end], time-ordered."""
        with self._lock:
            picked = [(t, f) for t, f in self._buf if t_start <= t <= t_end]
        picked.sort(key=lambda tf: tf[0])
        frames = [f for _, f in picked]
        times = [t for t, _ in picked]
        return frames, times

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_frame_age(self) -> float | None:
        """Seconds since the last frame, or None if none yet."""
        return (time.monotonic() - self._last_frame_t) if self._frame_count else None

    @property
    def content_stale_age(self) -> float | None:
        """Seconds since the decoded CONTENT last changed, or None if no frames yet.
        Grows without bound when the DAL session FREEZES even though duplicate frames
        keep 'arriving' — the failure last_frame_age can't see. A consumer whose screen
        changes regularly (e.g. a gesture every ~1-2s) should treat a large value
        (>~8s) as a frozen stream."""
        return (time.monotonic() - self._last_change_t) if self._frame_count else None

    @property
    def last_error(self) -> str:
        return self._last_err

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- internals ----------------------------------------------------------
    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        buf = b""
        try:
            while not self._stop:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    s = buf.find(b"\xff\xd8")
                    if s == -1:
                        buf = b""
                        break
                    e = buf.find(b"\xff\xd9", s + 2)
                    if e == -1:
                        buf = buf[s:]
                        break
                    jpeg = buf[s:e + 2]
                    buf = buf[e + 2:]
                    self._ingest(jpeg)
        except Exception:  # noqa: BLE001 — reader dies quietly; health flags expose it
            pass

    def _ingest(self, jpeg: bytes) -> None:
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        except Exception:  # noqa: BLE001 — drop a torn frame
            return
        t = time.monotonic()
        arr = np.asarray(img, dtype=np.uint8)
        with self._lock:
            self._buf.append((t, arr))
            cutoff = t - self.horizon_s
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()
        self._frame_count += 1
        self._last_frame_t = t
        # Frozen-stream guard: when the CMIO/DAL session stalls, ffmpeg's fps filter
        # emits BITWISE-IDENTICAL duplicate frames — frame_count keeps climbing and
        # last_frame_age stays ~0, fooling a naive liveness check. Track when the
        # decoded CONTENT last changed (subsampled exact-equality) so a consumer can
        # catch a frozen stream that's still "delivering" frames.
        sig = arr[::8, ::8]
        if self._prev_sig is None or not np.array_equal(sig, self._prev_sig):
            self._last_change_t = t
        self._prev_sig = sig

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in iter(proc.stderr.readline, b""):
                if self._stop:
                    break
                txt = line.decode("utf-8", "replace").strip()
                if txt:
                    self._last_err = txt
        except Exception:  # noqa: BLE001
            pass


# --- self-test / manual probe ---------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="DAL capture probe.")
    ap.add_argument("--list", action="store_true", help="List AVFoundation devices and exit.")
    ap.add_argument("--device", help="Device name substring (e.g. 'Test XR_1').")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = ap.parse_args()

    if args.list or not args.device:
        print(list_dal_devices())
        raise SystemExit(0)

    name = resolve_device_name(args.device) or args.device
    print(f"Opening '{name}' at {args.fps}fps for {args.seconds}s...")
    rec = DalFrameRecorder(fps=args.fps)
    t_open = time.monotonic()
    rec.open(name)
    if not rec.wait_for_frames():
        print(f"  no frames — device wedged or busy? last ffmpeg err: {rec.last_error!r}")
        rec.close()
        raise SystemExit(1)
    time.sleep(args.seconds)
    frames, times = rec.window(t_open, time.monotonic())
    rec.close()
    if len(times) >= 2:
        span = times[-1] - times[0]
        eff_fps = (len(times) - 1) / span if span > 0 else 0.0
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        gaps.sort()
        p50 = gaps[len(gaps) // 2]
        p95 = gaps[int(len(gaps) * 0.95)]
        print(f"  {len(times)} frames over {span:.2f}s → {eff_fps:.1f} fps; "
              f"inter-frame p50={p50*1000:.1f}ms p95={p95*1000:.1f}ms max={gaps[-1]*1000:.1f}ms")
    else:
        print(f"  only {len(times)} frame(s) captured.")
