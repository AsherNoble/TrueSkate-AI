"""Trick-speed stroke contract.

One stroke is one finger contact: 1–4 straight pieces of at least 40 ms, 50–600 ms in
all, inside the usable gesture area and clear of every protected control along the
whole path. A repeated point is a hold. Each stroke is its own request to WDA's direct
``/wda/perform_trick_gestures`` endpoint, the path the landed kickflip uses
(KICKFLIP-REPEATABILITY-20261009); never bundle strokes (GESTURES.md hard rule).
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
# (EXEC-TRANSFER-20261013), so pieces stay at 40 ms or longer.
MIN_PIECE_MS = 40
MAX_PIECES = 4
MIN_STROKE_MS, MAX_STROKE_MS = 50, 600
MAX_STROKES = 4
FAMILIES = ("push", "pop", "flick", "catch", "random")


@dataclass(frozen=True)
class Stroke:
    points: tuple[tuple[float, float], ...]
    times_ms: tuple[int, ...]
    family: str

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError("unknown stroke family")
        if not 2 <= len(self.points) <= MAX_PIECES + 1 or len(self.points) != len(self.times_ms):
            raise ValueError("a stroke needs 1–4 pieces and one time per point")
        if self.times_ms[0] != 0 or any(type(t) is not int for t in self.times_ms):
            raise ValueError("integer millisecond times starting at zero required")
        if any(b - a < MIN_PIECE_MS for a, b in zip(self.times_ms, self.times_ms[1:])):
            raise ValueError(f"every piece must last at least {MIN_PIECE_MS} ms")
        if not MIN_STROKE_MS <= self.times_ms[-1] <= MAX_STROKE_MS:
            raise ValueError(f"stroke duration outside {MIN_STROKE_MS}–{MAX_STROKE_MS} ms")
        x0, y0, x1, y1 = BOUNDS
        if any(len(p) != 2 or not all(math.isfinite(v) for v in p)
               or not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) for p in self.points):
            raise ValueError("points must lie inside the usable gesture area")
        quantized = self.quantized()
        if (self.family == "catch") == any(q != quantized[0] for q in quantized[1:]):
            raise ValueError("a catch holds still and every other stroke moves")
        commands = [tuple(p) for p in self.command_points().tolist()]
        for path in (self.points, commands):
            if any(not segment_is_safe(a, b) for a, b in zip(path, path[1:])):
                raise ValueError("stroke path touches a protected control")

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
    def from_dict(cls, value: dict) -> "Stroke":
        if value.get("schema") != SCHEMA:
            raise ValueError("wrong trick stroke schema")
        stroke = cls(tuple(tuple(p) for p in value["points"]), tuple(value["times_ms"]), value["family"])
        if value.get("payload_sha256") != digest(stroke.payload()):
            raise ValueError("stroke payload fingerprint mismatch")
        return stroke
