"""Offline overlay checks; no research recordings or checkpoints."""
import importlib.util
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prediction_overlay', ROOT / 'scripts/inspect/render_prediction_overlay.py')
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


def test_ffmpeg_preserves_synthetic_frame_sequence(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg tools unavailable')
    frames = np.stack([np.full((32, 24, 3), i * 12, np.uint8) for i in range(7)])
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                    '-s', '24x32', '-r', '10', '-i', '-', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(tmp_path / 'frames.mp4')],
                   input=frames.tobytes(), check=True)
    decoded = overlay._ffmpeg_frames(tmp_path)
    assert len(decoded) == 7
    means = [round(float(f.mean())) for f in decoded]
    # H.264/YUV rounding differs slightly across FFmpeg builds. Sequence length,
    # ordering and approximate levels are the decoder contract this test needs.
    assert all(right > left for left, right in zip(means, means[1:]))
    assert means == pytest.approx(list(range(0, 84, 12)), abs=5)
    selected = overlay._decode(tmp_path, 4, 'ffmpeg')
    assert [float(f.mean()) for f in selected] == [float(decoded[i].mean()) for i in (0, 2, 4, 6)]


def test_error_verdict_requires_all_three_tolerances():
    truth = [.2, .3, .6, .7, .5]
    assert overlay._errors(truth, truth)['recovered']
    for index, delta in ((0, .04), (2, .04), (4, .11)):
        pred = truth.copy()
        pred[index] += delta
        assert not overlay._errors(pred, truth)['recovered']
