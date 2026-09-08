import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


def _flagger_module():
    path = Path(__file__).parents[1] / "scripts" / "data" / "flag_menu_samples.py"
    spec = importlib.util.spec_from_file_location("test_flag_menu_samples", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_menu_flagger_inspects_compact_mp4_frames(tmp_path, monkeypatch):
    module = _flagger_module()
    video = tmp_path / "frames.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 64))
    if not writer.isOpened():
        pytest.skip("local OpenCV lacks an MP4 writer")
    for value in (20, 60, 100):
        writer.write(np.full((64, 32, 3), value, dtype=np.uint8))
    writer.release()

    seen = []

    def fake_is_menu_frame(frame):
        seen.append(frame)
        return len(seen) == 2

    monkeypatch.setattr(module, "is_menu_frame", fake_is_menu_frame)
    assert module._sample_has_menu_frame(tmp_path)
    assert len(seen) == 2
    # The flagger converts the OpenCV-decoded BGR array to the RGB array accepted
    # by gameplay_filter before evaluating it.
    assert seen[0].shape == (64, 32, 3)
