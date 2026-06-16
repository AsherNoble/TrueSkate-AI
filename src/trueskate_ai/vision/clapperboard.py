"""Capture-pipeline latency calibration — the "clapperboard".

Measures Δ, the end-to-end lag between issuing an on-screen command and that
change appearing in a decoded MJPEG frame (game render + MJPEG encode + iproxy
USB tunnel + requests buffering + PIL decode), by timing the board's visible
response to a reset/waypoint tap.

Δ is DISTINCT from the 0.45s orange-trace render lag
(``train_trace_extractor._DEFAULT_LATENCY_S``): that is True Skate's internal
trace animation delay, used only for trace-warmth gating when self-labeling.
Δ is the *pipeline* lag — a frame stamped (monotonic) at ``t`` shows screen state
from ``t − Δ`` — used to pair each frame to the correct point in the gesture
timeline.

Per-frame pairing precision is floored by the MJPEG frame interval (≈1/fps):
averaging many resets tightens the Δ ESTIMATE, never that per-frame quantization.

The detector reuses ``board_localizer.locate_board`` — wiring that previously
standalone CV into the live loop for the first time.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from trueskate_ai.vision.board_localizer import locate_board

# Tuning (normalised board coords; seconds). Validate/adjust on-device.
_MOVE_EPS = 0.04          # board-centroid move that counts as "responded to reset"
_BASELINE_FRAMES = 4      # pre-tap frames used to fix the settled board position
_BASELINE_MAX_STD = 0.02  # reject the sample if the board wasn't still pre-tap
_MAX_WINDOW_S = 1.0       # only look this far past the tap for the response


def board_centroid(rgb: np.ndarray) -> tuple[float, float] | None:
    """locate_board on an RGB frame (converted to the BGR it expects); (cx, cy) or None."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pose = locate_board(bgr)
    return (pose.cx, pose.cy) if pose is not None else None


def offset_from_reset(
    frames: list,
    times: list[float],
    t0: float,
    *,
    centroid_fn=board_centroid,
    move_eps: float = _MOVE_EPS,
    max_window_s: float = _MAX_WINDOW_S,
) -> float | None:
    """Δ for one reset: time of the first post-tap frame whose board centroid
    departs the pre-tap settled position by > ``move_eps``, minus ``t0``.

    ``t0`` is the monotonic time the reset tap was issued. Returns Δ seconds, or
    None when not measurable: no board detected, board not still pre-tap, or the
    board was already at the reset spot (no motion). ``centroid_fn`` is injectable
    so the timing logic is unit-testable without real board frames.
    """
    pre = [(t, f) for t, f in zip(times, frames) if t < t0]
    post = [(t, f) for t, f in zip(times, frames) if t0 <= t <= t0 + max_window_s]
    if len(pre) < 2 or not post:
        return None
    base_cs = [c for _, f in pre[-_BASELINE_FRAMES:] if (c := centroid_fn(f)) is not None]
    if len(base_cs) < 2:
        return None
    base = np.asarray(base_cs, dtype=np.float64)
    if base[:, 0].std() > _BASELINE_MAX_STD or base[:, 1].std() > _BASELINE_MAX_STD:
        return None  # board wasn't settled pre-tap → baseline untrustworthy
    bx, by = float(base[:, 0].mean()), float(base[:, 1].mean())
    for t, f in post:  # post is in capture order
        c = centroid_fn(f)
        if c is None:
            continue
        if abs(c[0] - bx) > move_eps or abs(c[1] - by) > move_eps:
            return float(t - t0)
    return None


@dataclass
class RollingCaptureOffset:
    """Rolling median Δ + MAD jitter from reset-snap samples (most recent ``window``)."""
    window: int = 30
    samples: list[float] = field(default_factory=list)

    def add_reset(self, frames: list, times: list[float], t0: float, **kw) -> float | None:
        d = offset_from_reset(frames, times, t0, **kw)
        if d is not None and d >= 0:
            self.samples.append(d)
            if len(self.samples) > self.window:
                self.samples.pop(0)
        return d

    @property
    def offset_s(self) -> float | None:
        return float(statistics.median(self.samples)) if self.samples else None

    @property
    def jitter_s(self) -> float | None:
        if len(self.samples) < 2:
            return None
        med = statistics.median(self.samples)
        return float(statistics.median([abs(s - med) for s in self.samples]))  # MAD

    @property
    def n(self) -> int:
        return len(self.samples)

    def summary(self) -> dict:
        return {"capture_offset_s": self.offset_s, "capture_offset_jitter_s": self.jitter_s, "n": self.n}


def calibrate_capture_offset(
    driver,
    mjpeg_url: str,
    device_w: float,
    device_h: float,
    *,
    k: int = 8,
    displace: bool = True,
    rng: np.random.Generator | None = None,
    baseline_warmup_s: float = 0.35,
    response_window_s: float = 0.9,
) -> RollingCaptureOffset:
    """Run ``k`` reset cycles on-device, returning a seeded RollingCaptureOffset.

    Each cycle optionally fires a displacing flick (so the board is off-reset and
    the reset produces visible motion), records frames spanning the reset tap, and
    feeds them to the rolling estimator. Imports are local to keep this module
    importable (and the pure functions above unit-testable) without Appium.
    """
    from trueskate_ai.data.gesture_sampling import sample_flick
    from trueskate_ai.sim.gestures import scale_to_device
    from trueskate_ai.sim.touch_actions import curved_drag, reset_position
    from trueskate_ai.vision.color_recorder import TimestampedColorRecorder

    rng = rng or np.random.default_rng(0)
    rec = TimestampedColorRecorder()
    est = RollingCaptureOffset(window=max(k, 30))
    for _ in range(k):
        if displace:
            g = sample_flick(rng)
            pts = [scale_to_device(x, y, device_w, device_h) for x, y in g["waypoints"]]
            easing = None if g["easing_power"] == 1.0 else (lambda t, p=g["easing_power"]: t ** p)
            try:
                curved_drag(driver, pts, total_duration=g["duration"], easing=easing)
                time.sleep(0.3)  # let the board settle off-reset
            except Exception:  # noqa: BLE001
                pass
        rec.start(mjpeg_url)
        time.sleep(baseline_warmup_s)  # capture pre-tap (settled) frames
        t0 = time.monotonic()
        try:
            reset_position(driver, device_w, device_h)
        except Exception:  # noqa: BLE001
            rec.stop()
            continue
        time.sleep(response_window_s)
        frames, times = rec.stop()
        est.add_reset(frames, times, t0)
    return est


# --- self-test (offline timing logic) + optional on-device calibration -----
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Clapperboard capture-offset calibration.")
    ap.add_argument("--device", default=None, help="Device name; if set, run on-device calibration")
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    # Offline: inject a synthetic centroid_fn (board jumps after a known lag) and
    # confirm offset_from_reset recovers the lag at frame quantization.
    fps, lag = 20.0, 0.12
    t0 = 100.0
    times = [t0 - 0.30 + i / fps for i in range(20)]  # ~6 pre, ~14 post
    frames = list(range(len(times)))                  # frame ids; centroid_fn keys off them
    onset_t = t0 + lag

    def fake_centroid(fid):
        t = times[fid]
        return (0.50, 0.55) if t < onset_t else (0.50, 0.65)  # jumps 0.10 in y after lag

    d = offset_from_reset(frames, times, t0, centroid_fn=fake_centroid)
    frame_interval = 1.0 / fps
    assert d is not None and lag <= d < lag + frame_interval + 1e-9, (d, lag)
    print(f"[PASS] offline: recovered Δ={d:.3f}s for injected lag={lag:.3f}s (≤ +1 frame {frame_interval:.3f}s)")

    est = RollingCaptureOffset(window=5)
    for _ in range(5):
        est.add_reset(frames, times, t0, centroid_fn=fake_centroid)
    print(f"[PASS] rolling: offset={est.offset_s:.3f}s jitter={est.jitter_s} n={est.n}")

    if args.device:
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[3]
        if str(_root / "src") not in sys.path:
            sys.path.insert(0, str(_root / "src"))
        from trueskate_ai.rl.device_worker import DEVICES, DeviceWorker

        cfg = next((d for d in DEVICES if d["name"].lower() == args.device.lower()), None)
        if cfg is None:
            raise SystemExit(f"unknown device {args.device}")
        w = DeviceWorker(cfg)
        print(f"Connecting to {cfg['name']} (needs WDA+Appium up)...")
        w.connect()
        try:
            est = calibrate_capture_offset(w.driver, w.mjpeg_url, w.device_w, w.device_h, k=args.k)
            print(f"On-device: {est.summary()}  (jitter floor ≈ 1/fps)")
        finally:
            w.disconnect()
