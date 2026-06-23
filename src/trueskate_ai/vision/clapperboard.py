"""Capture-pipeline latency calibration — the "clapperboard".

Measures Δ, the end-to-end lag between issuing an on-screen command and that
change appearing in a decoded MJPEG frame (render + MJPEG encode + iproxy USB
tunnel + requests buffering + PIL decode).

PRIMARY method — ``calibrate_via_app``: switch to the dedicated Clapperboard app
(a full-screen black/white toggle that flips INSTANTLY, no animation; see the
sibling Clapperboard repo), tap to flip at a known time, and time the flip in the
MJPEG. True Skate animates everything (the board reset, its menus), so no in-game
visual has a timeable onset — hence a dedicated instant target.

FALLBACK method — ``offset_from_reset`` / ``calibrate_capture_offset``: time the
board's snap-back to the reset waypoint via ``board_localizer``. Kept for
reference but less reliable (the reset animates and the localizer glitches mid-
transition), so the app method is preferred.

Anchor to the COMMAND'S CALL-RETURN, not its issue time. Appium's synchronous
``mobile:tap`` (and the W3C gesture calls) return ~0.07s AFTER their visual effect
is captured, with the large, VARIABLE command overhead (~0.8-0.9s, ±0.1s on this
rig) happening BEFORE the effect. Measured on-device: flip-vs-call-issue jitters
±0.1s, but flip-vs-call-return is −0.073s ± 0.015s (sub-frame). So Δ is defined as
``Δ = (flip-frame monotonic time) − (command call-return monotonic time)`` (a
small negative number), and a frame at ``t`` shows the effect of a command whose
call returned at ``t − Δ``. The collector likewise anchors a gesture's frames to
that gesture call's return; the on-screen gesture then spans frame-times
``[Δ − duration, Δ]``.

Δ is DISTINCT from the 0.45s orange-trace render lag
(``train_trace_extractor._DEFAULT_LATENCY_S``), True Skate's internal trace
animation delay used only for trace-warmth gating. Per-frame precision is floored
by the MJPEG frame interval (≈1/fps); averaging tightens the ESTIMATE, not that
quantization.
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
_BASELINE_FRAMES = 4      # pre/post frames used to fix the settled board positions
_BASELINE_MAX_STD = 0.02  # reject the sample if the board wasn't still pre/post
_MAX_WINDOW_S = 1.0       # only look this far past the tap for the response
_MIN_DISPLACEMENT = 0.08  # the reset must move the board home by at least this far
                          # (skips no-op / in-place-flick resets that can't be timed)

# Primary (app-based) calibration constants.
# Re-signed under Asher's own team (free signing requires a team-unique bundle id),
# so the rig's installed id is com.ashernoble.*, not the original com.embodiedstates.*.
DEFAULT_CLAPPERBOARD_BUNDLE = "com.ashernoble.clapperboard"
_TRUESKATE_BUNDLE = "com.trueaxis.skate"
_LUM_THRESHOLD = 25.0      # whole-frame mean-luminance jump that counts as the flip
_LUM_BASELINE_FRAMES = 3   # pre-flip frames used to fix the dark/light baseline


def board_centroid(rgb: np.ndarray) -> tuple[float, float] | None:
    """locate_board on an RGB frame (converted to the BGR it expects); (cx, cy) or None."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pose = locate_board(bgr)
    return (pose.cx, pose.cy) if pose is not None else None


def _settled_mean(centroids: list[tuple[float, float]]) -> np.ndarray | None:
    """Mean of the centroids if they're internally still, else None."""
    if len(centroids) < 2:
        return None
    arr = np.asarray(centroids, dtype=np.float64)
    if arr[:, 0].std() > _BASELINE_MAX_STD or arr[:, 1].std() > _BASELINE_MAX_STD:
        return None
    return arr.mean(axis=0)


def offset_from_reset(
    frames: list,
    times: list[float],
    t0: float,
    *,
    centroid_fn=board_centroid,
    move_eps: float = _MOVE_EPS,
    max_window_s: float = _MAX_WINDOW_S,
    min_displacement: float = _MIN_DISPLACEMENT,
) -> float | None:
    """Δ for one reset: time of the first post-tap frame where the board has left
    its displaced (pre-tap) position AND moved toward the settled reset-home
    position, minus ``t0``.

    Anchoring onset to "closer to home than the start" rejects the spurious
    far-flung detections the localizer emits during the reset transition (a
    one-frame jump away from both endpoints), which a naive "moved by > eps" test
    would mis-time. Returns Δ seconds, or None when not measurable: no/unstable
    board, board not still pre- or post-reset, or the reset didn't actually move
    the board home (no-op / in-place-flick reset). ``centroid_fn`` is injectable
    so the timing logic is unit-testable without real board frames.
    """
    pre = [(t, f) for t, f in zip(times, frames) if t < t0]
    post = [(t, f) for t, f in zip(times, frames) if t0 <= t <= t0 + max_window_s]
    if len(pre) < 2 or len(post) < 3:
        return None
    p_base = _settled_mean([c for _, f in pre[-_BASELINE_FRAMES:] if (c := centroid_fn(f)) is not None])
    p_home = _settled_mean([c for _, f in post[-_BASELINE_FRAMES:] if (c := centroid_fn(f)) is not None])
    if p_base is None or p_home is None:
        return None
    d_bh = float(np.hypot(*(p_base - p_home)))
    if d_bh < min_displacement:
        return None  # the reset didn't move the board home — can't time it
    for t, f in post:  # capture order
        c = centroid_fn(f)
        if c is None:
            continue
        c = np.asarray(c, dtype=np.float64)
        # left the displaced start AND is now closer to home than the start was
        if float(np.hypot(*(c - p_base))) > move_eps and float(np.hypot(*(c - p_home))) < d_bh:
            return float(t - t0)
    return None


@dataclass
class RollingCaptureOffset:
    """Rolling median Δ + MAD jitter from reset-snap samples (most recent ``window``)."""
    window: int = 30
    samples: list[float] = field(default_factory=list)

    def add_reset(self, frames: list, times: list[float], t0: float, **kw) -> float | None:
        return self.add_sample(offset_from_reset(frames, times, t0, **kw))

    def add_sample(self, d: float | None) -> float | None:
        """Add a pre-measured Δ sample. Δ may be NEGATIVE — the app-based offset
        (flip-frame minus command call-return) is ~-0.07s; only None is dropped."""
        if d is not None:
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
    from trueskate_ai.sim.gestures import execute_static_push
    from trueskate_ai.sim.touch_actions import reset_position
    from trueskate_ai.vision.color_recorder import TimestampedColorRecorder

    _ = rng  # kept for signature compatibility; displacement is now a fixed push
    rec = TimestampedColorRecorder()
    est = RollingCaptureOffset(window=max(k, 30))
    for _ in range(k):
        if displace:
            # Roll the board AWAY with a push (NOT an in-place flick) so the reset
            # produces a large, crisp snap-back. Verified on-device: a flick barely
            # moves the centroid and the reset is then untimeable; a push gives a
            # ~0.2-normalised snap with a clean onset.
            try:
                execute_static_push(driver, device_w=device_w, device_h=device_h)
                time.sleep(0.7)  # the board must fully STOP or the baseline is unstable
            except Exception:  # noqa: BLE001
                pass
        rec.start(mjpeg_url, resize_width=512)
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


def _lum_onset(
    frames: list,
    times: list[float],
    anchor: float,
    *,
    threshold: float = _LUM_THRESHOLD,
    baseline_frames: int = _LUM_BASELINE_FRAMES,
) -> float | None:
    """Signed Δ of a full-screen black<->white flip relative to ``anchor``: the
    monotonic time of the first frame whose whole-frame mean luminance differs
    from the pre-flip baseline (mean of the first ``baseline_frames`` frames) by
    > threshold, minus ``anchor``. NEGATIVE when the flip precedes ``anchor`` —
    e.g. the flip is captured ~0.07s before the synchronous tap call returns.
    None if no flip is seen. Scans all frames so the anchor may sit mid-window."""
    if len(frames) < baseline_frames + 1:
        return None
    base = float(np.mean([np.asarray(f, dtype=np.float32).mean() for f in frames[:baseline_frames]]))
    for t, f in zip(times, frames):  # capture order
        if abs(float(np.asarray(f, dtype=np.float32).mean()) - base) > threshold:
            return float(t - anchor)
    return None


def calibrate_via_app(
    driver,
    capture_source: str | None,
    device_w: float,
    device_h: float,
    *,
    recorder=None,
    action_fn=None,
    app_bundle: str = DEFAULT_CLAPPERBOARD_BUNDLE,
    trueskate_bundle: str = _TRUESKATE_BUNDLE,
    k: int = 10,
    lum_threshold: float = _LUM_THRESHOLD,
    return_to_trueskate: bool = True,
) -> RollingCaptureOffset:
    """Measure Δ with the dedicated Clapperboard app (PRIMARY method).

    Switches to the app, taps to flip the full screen black<->white, times the
    flip relative to the tap call's RETURN (the tight anchor — see module docs),
    repeats k times, and switches back. The instant full-frame flip gives a clean
    onset that in-game visuals (which animate) cannot. Δ is ~constant per session,
    so this runs once at startup (and optionally periodically). Requires the
    Clapperboard app installed (bundle ``app_bundle``). Imports are local so the
    module stays importable without Appium.

    ``recorder`` — an ALREADY-OPEN persistent capture (e.g. a
    ``vision.dal_capture.DalFrameRecorder``) whose rolling buffer is sliced per tap
    via ``window(t_pre, now)``; the same Δ method applies (the DAL stream mirrors
    whatever's foregrounded, so it still sees the flip). Pass ``recorder=None`` to use
    a transient per-tap ``TimestampedColorRecorder`` reading ``capture_source`` (the
    WDA MJPEG url) — the legacy path. The flip is timed in whichever pipeline records
    it, so Δ is correct for the capture you actually ship.

    ``action_fn`` — the on-screen action whose CALL-RETURN anchors Δ; defaults to a
    center ``mobile: tap``. Inject a drag (finger-up on the button) to measure Δ for a
    gesture's call-return vs a tap's — see experiments/clapperboard_drag_vs_tap.py.
    """
    from trueskate_ai.vision.color_recorder import TimestampedColorRecorder

    do_action = action_fn or (
        lambda: driver.execute_script("mobile: tap", {"x": 0.5 * device_w, "y": 0.5 * device_h}))
    persistent = recorder is not None
    rec = recorder if persistent else TimestampedColorRecorder()
    est = RollingCaptureOffset(window=max(k, 30))
    driver.activate_app(app_bundle)
    time.sleep(1.2)  # app launch + first-frame settle
    try:
        for _ in range(k):
            t_pre = time.monotonic()
            if not persistent:
                rec.start(capture_source, resize_width=128)
            time.sleep(0.4)  # pre-flip baseline frames (the synchronous tap is slow)
            try:
                do_action()
            except Exception:  # noqa: BLE001
                if not persistent:
                    rec.stop()
                continue
            t_ret = time.monotonic()  # call returned; the flip was captured ~Δ (negative) before this
            time.sleep(0.3)           # ensure the flip frame is in the buffer
            if persistent:
                frames, times = rec.window(t_pre, time.monotonic())
            else:
                frames, times = rec.stop()
            est.add_sample(_lum_onset(frames, times, t_ret, threshold=lum_threshold))
            time.sleep(0.15)  # decorrelate consecutive flips
    finally:
        if return_to_trueskate:
            try:
                driver.activate_app(trueskate_bundle)
                time.sleep(1.0)
            except Exception:  # noqa: BLE001
                pass
    return est


# --- self-test (offline timing logic) + optional on-device calibration -----
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Clapperboard capture-offset calibration.")
    ap.add_argument("--device", default=None, help="Device name; if set, run on-device calibration")
    ap.add_argument("--method", choices=["app", "reset"], default="app",
                    help="app: dedicated Clapperboard app (primary); reset: board-snap fallback")
    ap.add_argument("--bundle", default=DEFAULT_CLAPPERBOARD_BUNDLE, help="Clapperboard app bundle id")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    # Offline: inject a synthetic centroid_fn (board displaced at P_base, snaps to
    # home P_home after a known lag) and confirm offset_from_reset recovers the lag
    # at frame quantization. P_base=(0.50,0.40) -> P_home=(0.50,0.60): d=0.20 > min.
    fps, lag = 20.0, 0.12
    t0 = 100.0
    times = [t0 - 0.30 + i / fps for i in range(20)]  # ~6 pre, ~14 post
    frames = list(range(len(times)))                  # frame ids; centroid_fn keys off them
    onset_t = t0 + lag
    frame_interval = 1.0 / fps

    def fake_centroid(fid):
        return (0.50, 0.40) if times[fid] < onset_t else (0.50, 0.60)

    d = offset_from_reset(frames, times, t0, centroid_fn=fake_centroid)
    assert d is not None and lag <= d < lag + frame_interval + 1e-9, (d, lag)
    print(f"[PASS] offline: recovered Δ={d:.3f}s for injected lag={lag:.3f}s (≤ +1 frame {frame_interval:.3f}s)")

    # Spurious far misdetection at the transition must be REJECTED (not timed).
    spurious_idx = next(i for i, t in enumerate(times) if t >= onset_t)

    def fake_with_spurious(fid):
        if times[fid] < onset_t:
            return (0.50, 0.40)
        if fid == spurious_idx:
            return (0.90, 0.10)   # far from both base and home — localizer glitch
        return (0.50, 0.60)

    d2 = offset_from_reset(frames, times, t0, centroid_fn=fake_with_spurious)
    assert d2 is not None and lag < d2 <= lag + 2 * frame_interval + 1e-9, (d2, lag)
    print(f"[PASS] spurious far-jump rejected: Δ={d2:.3f}s (skipped the glitch frame)")

    # No-op reset (board never displaced) → not measurable.
    assert offset_from_reset(frames, times, t0, centroid_fn=lambda fid: (0.50, 0.55)) is None
    print("[PASS] no-displacement reset → None (correctly skipped)")

    est = RollingCaptureOffset(window=5)
    for _ in range(5):
        est.add_reset(frames, times, t0, centroid_fn=fake_centroid)
    print(f"[PASS] rolling: offset={est.offset_s:.3f}s jitter={est.jitter_s} n={est.n}")

    # _lum_onset is anchor-relative (signed), baseline from the FIRST frames, so
    # the anchor may sit before OR after the flip.
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    lframes = [black if times[i] < onset_t else white for i in range(len(times))]
    flip_t = next(times[i] for i in range(len(times)) if times[i] >= onset_t)
    dl = _lum_onset(lframes, times, t0)                  # anchor before the flip → +lag
    assert dl is not None and lag <= dl < lag + frame_interval + 1e-9, (dl, lag)
    ret_anchor = flip_t + 0.20                           # anchor after the flip (like a call-return)
    dl2 = _lum_onset(lframes, times, ret_anchor)
    assert dl2 is not None and dl2 < 0 and abs(dl2 - (flip_t - ret_anchor)) < 1e-9, dl2
    assert _lum_onset([black] * len(times), times, t0) is None  # no flip → None
    print(f"[PASS] lum-onset signed: +{dl:.3f}s before-anchor, {dl2:.3f}s after-anchor; no-flip → None")

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
            if args.method == "app":
                est = calibrate_via_app(w.driver, w.mjpeg_url, w.device_w, w.device_h,
                                        app_bundle=args.bundle, k=args.k)
            else:
                est = calibrate_capture_offset(w.driver, w.mjpeg_url, w.device_w, w.device_h, k=args.k)
            print(f"On-device ({args.method}): {est.summary()}  (jitter floor ≈ 1/fps)")
        finally:
            w.disconnect()
