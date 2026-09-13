import numpy as np

from scripts.inspect.trace_onset_detector import detect_v1


def test_v1_selects_first_warm_colour_increase():
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    frames[3:, 12:20, 12:20] = (60, 20, 10)

    assert detect_v1(frames, 16, 16) == 3


def test_v1_returns_none_without_threshold_crossing():
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)

    assert detect_v1(frames, 16, 16) is None
