"""Versioned executed-polyline contract and reproducible experiment settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from trueskate_ai.sim.timed_waypoints import TimedWaypoints

SCHEMA = "model1_curved_v1"
SEED = 20261011
WIDTH, HEIGHT = 414, 896
FRAME_COUNT = 120
IMAGE_WIDTH, IMAGE_HEIGHT = 64, 144
MIN_SEGMENT_MS = 100
MAX_SEGMENTS = 10
MIN_DURATION_MS, MAX_DURATION_MS = 300, 1200
POSITION_TOLERANCE = .03
ARC_TOLERANCE_PT = 2.
CHALLENGE_PARK = "SLS 2016 Munich"
SAMPLERS = ("random_chain", "arc")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def save_json(path: Path, value, *, exclusive: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


@dataclass(frozen=True)
class Gesture:
    points: tuple[tuple[float, float], ...]
    times_ms: tuple[int, ...]
    sampler: str
    profile: str = "uniform"
    ideal: dict | None = None

    def __post_init__(self):
        TimedWaypoints(self.points, self.times_ms)
        if not 2 <= len(self.points) <= MAX_SEGMENTS + 1:
            raise ValueError("curved MVP requires 1–10 segments")
        if not MIN_DURATION_MS <= self.times_ms[-1] <= MAX_DURATION_MS:
            raise ValueError("duration outside 300–1200 ms")
        if any(b-a < MIN_SEGMENT_MS for a, b in zip(self.times_ms, self.times_ms[1:])):
            raise ValueError("every movement/pause must last at least 100 ms")
        if self.sampler not in SAMPLERS:
            raise ValueError("unknown sampler")
        if all(p == self.points[0] for p in self.points):
            raise ValueError("a gesture must contain movement")

    def payload(self) -> dict:
        return TimedWaypoints(self.points, self.times_ms).payload(WIDTH, HEIGHT)

    def command_points(self) -> np.ndarray:
        return np.asarray(TimedWaypoints(self.points, self.times_ms).quantized(),
                          dtype=np.float64) / [WIDTH, HEIGHT]

    def positions(self, elapsed_s, *, rate: float = 1.) -> np.ndarray:
        """Evaluate quantized command in calibrated video time; no time warping."""
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("positive finite clock rate required")
        times = np.asarray(self.times_ms, dtype=float) * rate / 1000.
        points = self.command_points()
        elapsed = np.asarray(elapsed_s, dtype=float)
        return np.stack([np.interp(elapsed, times, points[:, axis])
                         for axis in range(2)], axis=-1)

    def to_dict(self) -> dict:
        value = dict(schema=SCHEMA, **asdict(self))
        value["command_points"] = self.command_points().tolist()
        value["payload_sha256"] = digest(self.payload())
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "Gesture":
        if value.get("schema") != SCHEMA:
            raise ValueError("wrong curved gesture schema")
        gesture = cls(tuple(tuple(p) for p in value["points"]),
                      tuple(value["times_ms"]), value["sampler"],
                      value.get("profile", "uniform"), value.get("ideal"))
        if value.get("payload_sha256") != digest(gesture.payload()):
            raise ValueError("gesture payload fingerprint mismatch")
        if value.get("command_points") != gesture.command_points().tolist():
            raise ValueError("quantized command mismatch")
        return gesture


def select_parks(eligible: list[str], seed: int = SEED) -> list[str]:
    import random
    if any(not isinstance(p, str) or not p.strip() for p in eligible):
        raise ValueError("nonempty verified park names required")
    pool = sorted(set(eligible) - {CHALLENGE_PARK})
    if len(pool) < 5:
        raise ValueError("at least five eligible parks excluding Munich required")
    return random.Random(seed).sample(pool, 5)


def settings() -> dict:
    return dict(schema=SCHEMA, seed=SEED, frame_count=FRAME_COUNT,
                image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                min_segment_ms=MIN_SEGMENT_MS, max_segments=MAX_SEGMENTS,
                min_duration_ms=MIN_DURATION_MS, max_duration_ms=MAX_DURATION_MS,
                arc_tolerance_pt=ARC_TOLERANCE_PT, audit_size=100,
                audit_min_pass=95, max_rounds=5, attempt_limit=20000,
                production_per_park=2800, selected_per_park=2500)
