"""Wait for the screen centre to stop moving before a calibration control.

M1-DIE5-COMPARE-20260925 found the false early single-touch calibration
detections coincide with the board still turning or sliding under the screen
centre after the pre-segment reset. This helper compares consecutive
screenshots in a box around the centre and returns once their mean absolute
grey difference falls below a threshold, or when a time limit is reached.
It never touches the device itself; callers pass a screenshot function.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

DECODE_WIDTH = 256
HALF_BOX_FRACTION = (90 / 414, 90 / 896)  # ±90 logical points on a 414×896 screen


def centre_grey(png: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("screenshot could not be decoded")
    height = max(1, round(image.shape[0] * DECODE_WIDTH / image.shape[1]))
    image = cv2.resize(image, (DECODE_WIDTH, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    h, w = image.shape
    ry, rx = round(h * HALF_BOX_FRACTION[1]), round(w * HALF_BOX_FRACTION[0])
    return image[h // 2 - ry:h // 2 + ry, w // 2 - rx:w // 2 + rx]


def centre_difference(previous: np.ndarray, current: np.ndarray) -> float:
    if previous.shape != current.shape:
        raise ValueError(f"screenshot shapes differ: {previous.shape} vs {current.shape}")
    return float(np.abs(current - previous).mean())


@dataclass
class SettleResult:
    settled: bool
    waited_s: float
    differences: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {"settled": self.settled, "waited_s": round(self.waited_s, 3),
                "differences": [round(d, 3) for d in self.differences]}


def wait_for_centre_settle(
    screenshot: Callable[[], bytes],
    *,
    threshold: float,
    max_wait_s: float,
    interval_s: float = 0.25,
    consecutive: int = 2,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SettleResult:
    """Return when ``consecutive`` successive differences are below ``threshold``."""
    if threshold <= 0 or max_wait_s < 0 or interval_s < 0 or consecutive < 1:
        raise ValueError("invalid settle parameters")
    start = clock()
    previous = centre_grey(screenshot())
    differences: list[float] = []
    calm = 0
    while True:
        if clock() - start >= max_wait_s:
            return SettleResult(False, clock() - start, differences)
        sleep(interval_s)
        current = centre_grey(screenshot())
        difference = centre_difference(previous, current)
        differences.append(difference)
        previous = current
        calm = calm + 1 if difference < threshold else 0
        if calm >= consecutive:
            return SettleResult(True, clock() - start, differences)
