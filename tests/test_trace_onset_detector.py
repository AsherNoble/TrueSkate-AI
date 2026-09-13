import numpy as np

from scripts.inspect.trace_onset_detector import detect_v1, detect_v2


def test_v1_selects_first_warm_colour_increase():
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    frames[3:, 12:20, 12:20] = (60, 20, 10)

    assert detect_v1(frames, 16, 16) == 3


def test_v1_returns_none_without_threshold_crossing():
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)

    assert detect_v1(frames, 16, 16) is None


def test_v2_rejects_broad_background_brightening():
    frames = np.zeros((6, 64, 64, 3), dtype=np.uint8)
    frames[2:] = 30

    assert detect_v2(frames, 32, 32) is None


def test_v2_rejects_single_frame_local_flash():
    frames = np.zeros((7, 64, 64, 3), dtype=np.uint8)
    frames[3, 27:38, 27:38] = 60

    assert detect_v2(frames, 32, 32) is None


def test_v2_returns_beginning_of_confirmed_local_glow():
    frames = np.zeros((7, 64, 64, 3), dtype=np.uint8)
    frames[3:, 27:38, 27:38] = 30
    frames[4:, 27:38, 27:38] = 60

    assert detect_v2(frames, 32, 32) == 3
