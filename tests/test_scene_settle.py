import cv2
import numpy as np
import pytest

from trueskate_ai.collection.scene_settle import (
    centre_difference, centre_grey, wait_for_centre_settle,
)


def _png(offset: int) -> bytes:
    image = np.full((896, 414, 3), 90, dtype=np.uint8)
    cv2.rectangle(image, (150 + offset, 380), (190 + offset, 520), (240, 240, 240), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_centre_difference_sees_motion_near_centre_only():
    still = centre_difference(centre_grey(_png(0)), centre_grey(_png(0)))
    moved = centre_difference(centre_grey(_png(0)), centre_grey(_png(20)))
    assert still == 0.0
    assert moved > 1.0


def test_waits_until_motion_stops():
    offsets = iter([0, 30, 60, 80, 80, 80, 80])
    clock = _Clock()
    result = wait_for_centre_settle(lambda: _png(next(offsets)), threshold=0.5, max_wait_s=5.0,
                                    interval_s=0.25, clock=clock, sleep=clock.sleep)
    assert result.settled
    assert result.waited_s == pytest.approx(1.25)
    assert result.differences[-2:] == [0.0, 0.0]


def test_gives_up_at_the_time_limit():
    counter = iter(range(0, 1000, 25))
    clock = _Clock()
    result = wait_for_centre_settle(lambda: _png(next(counter) % 100), threshold=0.5, max_wait_s=1.0,
                                    interval_s=0.25, clock=clock, sleep=clock.sleep)
    assert not result.settled
    assert result.waited_s == pytest.approx(1.0)
