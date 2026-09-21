"""EQ-018/EQ-021: the extractor must produce exactly the frames its labels assert.

These drive real ffmpeg against a synthetic source, because the defect lives in
the interaction between `-ss` input-seek quantisation, `-t`, and the `fps`
filter — none of which a mocked subprocess would reproduce.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _aligner():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "collection" / "align_xctest_traces.py"
    spec = importlib.util.spec_from_file_location("align_xctest_traces", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A 30fps clip long enough to slice a 2.3s window out of the middle."""
    out = tmp_path_factory.mktemp("src") / "segment.mov"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x176:rate=30:duration=8", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out


def test_a_clip_is_only_written_when_it_holds_the_asserted_frames(source, tmp_path):
    """The safety property: never emit a clip whose pixels and labels disagree.

    Whether the tail margin is SUFFICIENT cannot be settled here — this ffmpeg
    build produces containers advertising 32 frames of which only 30 decode
    (nb_frames=32, nb_read_frames=30), which the rig's builds do not do (the
    corpus audit found header and decode agreeing at 31).  So the margin must be
    validated on the rig.  What is testable anywhere is the invariant: the
    extractor either produces exactly `max_frames` decodable frames, or it
    writes nothing and says so.
    """
    module = _aligner()
    max_frames, pre_s, window_s, fps = 32, 0.5, 1.8, 30.0
    duration = pre_s + window_s
    output_fps = (max_frames - 1) / max(duration - 1 / fps, 1 / fps)
    sample = tmp_path / "sample"
    accepted = module._extract_sample_video(
        source, sample, start_s=2.0, duration_s=duration,
        resize_width=128, output_fps=output_fps, max_frames=max_frames,
        crf=20, source_fps=fps,
    )
    clip = sample / "frames.mp4"
    if accepted:
        assert module._video_frame_count(clip) == max_frames
    else:
        assert not clip.exists(), "a rejected extract must leave no clip behind"


def test_the_old_no_margin_call_is_what_produced_31_frames(source, tmp_path):
    """Regression witness: without the tail margin ffmpeg flushes a frame short.

    This is the exact defect measured across all 3,040 MVP clips.  If ffmpeg ever
    stops reproducing it the test says so rather than silently passing.
    """
    module = _aligner()
    max_frames, duration, fps = 32, 2.3, 30.0
    output_fps = (max_frames - 1) / max(duration - 1 / fps, 1 / fps)
    out = tmp_path / "old.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "2.000", "-i", str(source),
         "-t", f"{duration:.3f}", "-vf", f"fps={output_fps:.8f},scale=128:-2",
         "-frames:v", str(max_frames), "-c:v", "libx264", "-crf", "20",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    produced = module._video_frame_count(out)
    if produced == max_frames:
        pytest.skip("this ffmpeg build no longer reproduces the short-extract defect")
    # The SHORTFALL MAGNITUDE is source- and build-dependent: the MVP corpus came
    # out uniformly one frame short, this build yields two or three on synthetic
    # input.  The invariant under test is only that the un-margined call comes up
    # short at all -- which is why an unverified frame count was unsafe.
    assert produced < max_frames


def test_a_short_extract_is_rejected_rather_than_written(source, tmp_path):
    """The guard must delete the clip, not hand back a stretched one."""
    module = _aligner()
    sample = tmp_path / "sample"
    # Demand more frames than the requested window can possibly contain.
    assert not module._extract_sample_video(
        source, sample, start_s=2.0, duration_s=0.2,
        resize_width=128, output_fps=5.0, max_frames=32, crf=20, source_fps=30.0,
    )
    assert not (sample / "frames.mp4").exists()


def test_frame_count_decodes_rather_than_trusting_the_header(source, tmp_path):
    module = _aligner()
    out = tmp_path / "probe.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-frames:v", "7",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    assert module._video_frame_count(out) == 7
    assert module._video_frame_count(tmp_path / "missing.mp4") == -1


def test_direct_extract_does_not_put_future_pixels_in_a_pre_onset_frame(tmp_path):
    """A rendered onset must not appear before its declared output time.

    FFmpeg's default nearest-frame rounding moved a source onset at 0.500 s into
    output slot 6 (0.439 s).  The training metadata therefore called a visibly
    active frame ``-0.0613 s`` relative to the gesture.  This fixture has an
    unambiguous black-to-white onset at the same phase as production clips and
    verifies that it first appears in slot 7 (0.512 s).
    """
    module = _aligner()
    source_dir = tmp_path / "onset_source"
    source_dir.mkdir()
    source_fps = 30
    for index in range(150):
        frame = np.zeros((160, 96, 3), dtype=np.uint8)
        if index >= 75:  # 2.500 s in the source, 0.500 s into the sliced window.
            frame[70:90, 38:58] = 255
        assert cv2.imwrite(str(source_dir / f"frame_{index:04d}.png"), frame)
    source = tmp_path / "onset.mov"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", str(source_fps),
         "-i", str(source_dir / "frame_%04d.png"), "-c:v", "libx264",
         "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p", str(source)],
        check=True, capture_output=True,
    )

    max_frames, pre_s, window_s = 32, 0.5, 1.8
    duration = pre_s + window_s
    output_fps = (max_frames - 1) / (duration - 1 / source_fps)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "2.000", "-i", str(source),
         "-t", f"{duration + 2 / source_fps:.3f}",
         "-vf", module._direct_video_filter(output_fps, 96),
         "-frames:v", str(max_frames), str(extracted / "frame_%03d.png")],
        check=True, capture_output=True,
    )
    frames = [cv2.imread(str(path)) for path in sorted(extracted.glob("frame_*.png"))]
    assert len(frames) == max_frames
    assert frames[6][70:90, 38:58].mean() < 30
    assert frames[7][70:90, 38:58].mean() > 220
    assert 6 / output_fps - pre_s < 0.0 <= 7 / output_fps - pre_s
