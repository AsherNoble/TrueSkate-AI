"""Trick-speed gesture contract.

One gesture is one touch, from touch-down to lift: 1–4 straight segments of at least 40 ms,
50–600 ms in all, inside the usable gesture area and clear of every protected control along
the whole path. A repeated point is a pause. Each gesture is its own request to WDA's direct
``/wda/perform_trick_gestures`` endpoint, the path the landed kickflip uses
(KICKFLIP-REPEATABILITY-20261009); never bundle gestures (GESTURES.md hard rule).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from trueskate_ai.data.control_hitboxes import segment_is_safe
from trueskate_ai.model1.curved.contract import digest

SCHEMA = "model1_trick_v1"
EXPERIMENT = "TRICK-SPEED-20261015"
SEED = 20261015
WIDTH, HEIGHT = 414, 896
FRAME_COUNT = 120
BOUNDS = (.12, .12, 1., .88)  # GESTURES.md usable area; controls are checked separately
# The simulator stretches moves under ~33 ms to two frames and merges 15–16 ms ones
# (EXEC-TRANSFER-20261013), so segments stay at 40 ms or longer.
MIN_SEGMENT_MS = 40
MAX_SEGMENTS = 4
MIN_GESTURE_MS, MAX_GESTURE_MS = 50, 600
MAX_GESTURES = 4
FAMILIES = ("push", "pop", "flick", "catch", "random")
# model1_trick_v1 named a gesture a "stroke" in its saved keys ("strokes", "stroke_timing", ...)
# and event kinds. They stay, so the recorded trials, frozen splits and checkpoints keep loading
# (GESTURES.md, "Legacy names").
GESTURE_EVENT = "stroke"  # execution.json event kind for one sent gesture


@dataclass(frozen=True)
class Gesture:
    points: tuple[tuple[float, float], ...]
    times_ms: tuple[int, ...]
    family: str

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError("unknown gesture family")
        if not 2 <= len(self.points) <= MAX_SEGMENTS + 1 or len(self.points) != len(self.times_ms):
            raise ValueError("a gesture needs 1–4 segments and one time per point")
        if self.times_ms[0] != 0 or any(type(t) is not int for t in self.times_ms):
            raise ValueError("integer millisecond times starting at zero required")
        if any(b - a < MIN_SEGMENT_MS for a, b in zip(self.times_ms, self.times_ms[1:])):
            raise ValueError(f"every segment must last at least {MIN_SEGMENT_MS} ms")
        if not MIN_GESTURE_MS <= self.times_ms[-1] <= MAX_GESTURE_MS:
            raise ValueError(f"gesture duration outside {MIN_GESTURE_MS}–{MAX_GESTURE_MS} ms")
        x0, y0, x1, y1 = BOUNDS
        if any(len(p) != 2 or not all(math.isfinite(v) for v in p)
               or not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) for p in self.points):
            raise ValueError("points must lie inside the usable gesture area")
        quantized = self.quantized()
        if (self.family == "catch") == any(q != quantized[0] for q in quantized[1:]):
            raise ValueError("a catch holds still and every other gesture moves")
        commands = [tuple(p) for p in self.command_points().tolist()]
        for path in (self.points, commands):
            if any(not segment_is_safe(a, b) for a, b in zip(path, path[1:])):
                raise ValueError("gesture path touches a protected control")

    def quantized(self) -> tuple[tuple[int, int], ...]:
        """Integer logical points, truncated as the WDA payload sends them."""
        return tuple((int(x * WIDTH), int(y * HEIGHT)) for x, y in self.points)

    def command_points(self) -> np.ndarray:
        return np.asarray(self.quantized(), dtype=np.float64) / [WIDTH, HEIGHT]

    def payload(self) -> dict:
        points = self.quantized()
        waypoints = [dict(x=points[0][0], y=points[0][1])]
        waypoints += [dict(x=x, y=y, duration_ms=b - a)
                      for (x, y), a, b in zip(points[1:], self.times_ms, self.times_ms[1:])]
        return {"gestures": [{"waypoints": waypoints}]}

    def positions(self, elapsed_s, *, rate: float = 1.) -> np.ndarray:
        """Commanded position at times since touch-down (video clock rate applied)."""
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("positive finite clock rate required")
        times = np.asarray(self.times_ms, dtype=float) * rate / 1000.
        points = self.command_points()
        elapsed = np.asarray(elapsed_s, dtype=float)
        return np.stack([np.interp(elapsed, times, points[:, axis]) for axis in range(2)], axis=-1)

    def to_dict(self) -> dict:
        value = dict(schema=SCHEMA, **asdict(self))
        value["command_points"] = self.command_points().tolist()
        value["payload_sha256"] = digest(self.payload())
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "Gesture":
        if value.get("schema") != SCHEMA:
            raise ValueError("wrong trick gesture schema")
        gesture = cls(tuple(tuple(p) for p in value["points"]), tuple(value["times_ms"]), value["family"])
        if value.get("payload_sha256") != digest(gesture.payload()):
            raise ValueError("gesture payload fingerprint mismatch")
        return gesture
