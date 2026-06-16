"""Detect a manual park switch from the live frame stream.

The SLS collector keeps firing in the current park until the user physically
loads a new one; this detector tells it WHEN that happened (to advance the park
label, reset the per-park timer, and prompt for the next switch) and WHILE it's
happening (so the collector pauses its taps and doesn't fight the user's menu
navigation).

A park reload is the only event that makes the *whole* frame diverge for a
sustained run AND then re-settle at a DIFFERENT steady state (menu → loading →
new park). Gesture motion, camera pans, the orange trace, and wall-bumps spike
briefly but re-settle at the SAME signature — so the "diverge then settle to
something new" test rejects them. A board-presence confirm (a loaded park has a
board; menus/loading screens don't) gates the final decision.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from trueskate_ai.vision.board_localizer import locate_board

# Tuning (combined signature distance in [0,1]; validate on-device).
_HIGH = 0.35       # distance from the steady ref that counts as "diverged"
_LOW = 0.12        # distance under which frames are "the same scene"
_MIN_SPIKE = 4     # consecutive diverged frames to enter a transition
_SETTLE_FRAMES = 6 # window that must be internally stable to declare re-settled
_MAX_TRANSITION = 240  # frames; failsafe so we never get stuck mid-transition
_THUMB = 24        # thumbnail edge for the structural part of the signature
_HUE_BINS = 16


def _signature(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(structural thumbnail, hue histogram) — a cheap whole-frame fingerprint."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thumb = cv2.resize(gray, (_THUMB, _THUMB)).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0], None, [_HUE_BINS], [0, 180]).flatten().astype(np.float32)
    s = float(hist.sum())
    hist = hist / s if s > 0 else hist
    return thumb, hist


def _dist(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """Combined distance in [0,1]: 50% structural L1 + 50% hue total-variation."""
    thumb_d = float(np.mean(np.abs(a[0] - b[0])) / 255.0)
    hue_d = float(0.5 * np.sum(np.abs(a[1] - b[1])))  # TV distance of normalised hists
    return 0.5 * thumb_d + 0.5 * hue_d


def _mean_sig(sigs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    thumb = np.mean([s[0] for s in sigs], axis=0)
    hist = np.mean([s[1] for s in sigs], axis=0)
    return thumb, hist


def _board_present(rgb: np.ndarray) -> bool:
    return locate_board(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) is not None


class ParkChangeDetector:
    """Stateful diverge-then-resettle detector. Feed frames; it reports switches.

    update(rgb) returns True exactly once per confirmed switch. ``in_transition``
    is True from the moment divergence starts until it re-settles — the collector
    should pause executing gestures while it's True.
    """

    def __init__(
        self,
        *,
        high: float = _HIGH,
        low: float = _LOW,
        min_spike: int = _MIN_SPIKE,
        settle_frames: int = _SETTLE_FRAMES,
        board_fn=_board_present,
    ) -> None:
        self.high = high
        self.low = low
        self.min_spike = min_spike
        self.settle_frames = settle_frames
        self._board_fn = board_fn
        self._ref: tuple[np.ndarray, np.ndarray] | None = None
        self._state = "init"
        self._init_buf: list = []
        self._spike = 0
        self._recent: deque = deque(maxlen=settle_frames)
        self._trans_frames = 0
        self.last_distance = 0.0

    @property
    def in_transition(self) -> bool:
        return self._state == "transition"

    def _settled(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the settled signature if the recent window is internally stable."""
        if len(self._recent) < self.settle_frames:
            return None
        sigs = list(self._recent)
        mean = _mean_sig(sigs)
        if max(_dist(s, mean) for s in sigs) <= self.low:
            return mean
        return None

    def update(self, rgb: np.ndarray) -> bool:
        sig = _signature(rgb)
        self._recent.append(sig)

        if self._state == "init":
            self._init_buf.append(sig)
            if len(self._init_buf) >= self.settle_frames:
                mean = _mean_sig(self._init_buf)
                if max(_dist(s, mean) for s in self._init_buf) <= self.low:
                    self._ref = mean
                    self._state = "steady"
                self._init_buf = self._init_buf[-self.settle_frames:]
            return False

        self.last_distance = _dist(sig, self._ref)

        if self._state == "steady":
            self._spike = self._spike + 1 if self.last_distance > self.high else 0
            if self._spike >= self.min_spike:
                self._state = "transition"
                self._spike = 0
                self._trans_frames = 0
            return False

        # transition
        self._trans_frames += 1
        settled = self._settled()
        if settled is not None:
            d_old = _dist(settled, self._ref)
            if d_old < self.low:
                self._state = "steady"  # transient (gesture/wall-bump) — same scene
                return False
            if d_old > self.high and self._board_fn(rgb):
                self._ref = settled       # confirmed switch → adopt new steady scene
                self._state = "steady"
                return True
        if self._trans_frames >= _MAX_TRANSITION:
            # failsafe: re-baseline on the current frame so we never wedge
            self._ref = sig
            self._state = "steady"
        return False

    def feed(self, frames: list[np.ndarray]) -> bool:
        """Feed a batch of frames; return True if a switch was confirmed in it."""
        switched = False
        for f in frames:
            if self.update(f):
                switched = True
        return switched


# --- offline self-test ----------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def scene(seed, n):  # n frames of a fixed-ish scene (small noise)
        base = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
        base[:] = (seed * 37) % 255
        return [np.clip(base + rng.integers(-3, 3, base.shape), 0, 255).astype(np.uint8) for _ in range(n)]

    det = ParkChangeDetector(board_fn=lambda rgb: True)  # fake board-present for offline

    # 1) steady scene A → no switch
    assert det.feed(scene(1, 12)) is False
    # 2) a brief transient (scene B for 5 frames) then back to A → NO switch (resettles to same)
    fired = det.feed(scene(9, 5)) or det.feed(scene(1, 10))
    assert fired is False, "transient blip must not count as a park switch"
    # 3) sustained change to a NEW scene C → switch fires once
    fired = det.feed(scene(20, 30))
    assert fired is True, "sustained change to a new steady scene must fire"
    print("[PASS] steady→no-fire; transient→no-fire; sustained-new→fire")

    # board-absent gates the decision: same sustained change but board_fn False → no fire
    det2 = ParkChangeDetector(board_fn=lambda rgb: False)
    det2.feed(scene(1, 12))
    assert det2.feed(scene(20, 30)) is False
    print("[PASS] board-absent (menu/loading) suppresses the switch")
    print("ALL PARK-CHANGE TESTS PASS")
