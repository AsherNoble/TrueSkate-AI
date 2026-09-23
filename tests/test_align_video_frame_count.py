"""EQ-018/EQ-021: the extractor must produce exactly the frames its labels assert.

These drive real ffmpeg against a synthetic source, because the defect lives in
the interaction between `-ss` input-seek quantisation, `-t`, and the `fps`
filter — none of which a mocked subprocess would reproduce.
"""
import json
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


def _decoded_frames(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


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
    selected_times = module._extract_sample_video(
        source, sample, start_s=2.0, duration_s=duration,
        resize_width=128, output_fps=output_fps, max_frames=max_frames,
        crf=20, source_frame_times=module._probe_video_frame_times(source),
    )
    clip = sample / "frames.mp4"
    if selected_times is not None:
        assert len(selected_times) == max_frames
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
    assert module._extract_sample_video(
        source, sample, start_s=2.0, duration_s=0.2,
        resize_width=128, output_fps=5.0, max_frames=32, crf=20,
        source_frame_times=module._probe_video_frame_times(source),
    ) is None
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


def test_direct_extract_preserves_the_selected_source_frame_times(tmp_path):
    """A rendered onset must agree with the exact PTS stored in metadata.

    FFmpeg's default nearest-frame rounding moved a source onset at 0.500 s into
    an earlier synthetic output slot. This fixture has an unambiguous
    black-to-white onset and verifies that the returned metadata is the PTS of
    the exact source frames that were encoded.
    """
    module = _aligner()
    source_dir = tmp_path / "onset_source"
    source_dir.mkdir()
    source_fps = 30
    for index in range(150):
        frame = np.zeros((160, 96, 3), dtype=np.uint8)
        if index >= 75:  # 2.500 s in the source, 0.500 s into the sliced window.
            # Use a full-frame transition: the invariant under test is which
            # source frame was selected, not a codec build's chroma resampling
            # around one small fixed-coordinate patch.
            frame[:] = 255
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
    sample = tmp_path / "sample"
    source_frame_times = module._probe_video_frame_times(source)
    # MOV edit lists can give the first decoded frame a nonzero media PTS on
    # some FFmpeg builds. The extractor's contract is the probed source PTS,
    # rather than an assumed zero-based frame_index / fps clock.
    expected_onset_time = source_frame_times[75]
    selected_times = module._extract_sample_video(
        source, sample, start_s=2.0, duration_s=duration,
        resize_width=96, output_fps=output_fps, max_frames=max_frames,
        crf=20, source_frame_times=source_frame_times,
    )
    assert selected_times is not None
    capture = cv2.VideoCapture(str(sample / "frames.mp4"))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    assert len(frames) == max_frames
    onset_index = next(index for index, frame in enumerate(frames)
                       if frame.mean() > 220)
    assert selected_times[onset_index] == pytest.approx(expected_onset_time, abs=1 / source_fps)
    assert selected_times[onset_index - 1] < expected_onset_time


def test_batch_extract_matches_the_reference_pixels_and_source_times(source, tmp_path):
    module = _aligner()
    source_times = module._probe_video_frame_times(source)
    jobs = [
        module._DirectVideoJob(tmp_path / "batch" / f"sample_{index}", start, 2.3, 13.67647059)
        for index, start in enumerate((0.5, 2.0, 4.0))
    ]
    batch = module._extract_sample_videos_batch(
        source, jobs, resize_width=128, max_frames=32, crf=20,
        source_frame_times=source_times,
    )
    for index, job in enumerate(jobs):
        reference = tmp_path / "reference" / f"sample_{index}"
        expected_times = module._extract_sample_video(
            source, reference, start_s=job.start_s, duration_s=job.duration_s,
            resize_width=128, output_fps=job.output_fps, max_frames=32, crf=20,
            source_frame_times=source_times,
        )
        assert batch[job.sample_dir] == expected_times
        assert expected_times is not None
        actual_frames = _decoded_frames(job.sample_dir / "frames.mp4")
        expected_frames = _decoded_frames(reference / "frames.mp4")
        assert len(actual_frames) == len(expected_frames) == 32
        assert all(np.array_equal(actual, expected)
                   for actual, expected in zip(actual_frames, expected_frames))


def test_batch_segment_writes_the_same_clip_metadata_as_current_path(source, tmp_path):
    module = _aligner()
    manifests = []
    for name in ("reference", "batch"):
        segment = tmp_path / name
        segment.mkdir()
        shutil.copyfile(source, segment / "segment.mov")
        manifest = segment / "segment_00000.json"
        manifest.write_text(json.dumps({
            "mov": "segment.mov",
            "started_at_epoch_s": 1000.0,
            "device": "iPhone_XR",
            "device_logical_w": 414,
            "device_logical_h": 896,
            "segment_index": 0,
            "gestures": [
                {"gesture_index": index, "park": "The Workshop",
                 "gesture_distribution": "linear", "duration": 0.5,
                 "waypoints": [[0.2, 0.4], [0.6, 0.6]],
                 "t_call_start_epoch_s": 1000.0 + onset}
                for index, onset in enumerate((1.0, 3.0, 5.0))
            ],
        }))
        manifests.append(manifest)

    for manifest, batch in zip(manifests, (False, True)):
        assert module.align_segment(
            manifest, pre_s=0.5, window_s=1.8, fps=30,
            resize_width=128, max_frames=32, delta_override=0.0,
            delete_mov=False, direct_video=True,
            batch_direct_video=batch,
        ) == 3

    for index in range(3):
        relative = Path("the_workshop") / f"sample_{index:06d}"
        old = manifests[0].parent / relative
        new = manifests[1].parent / relative
        old_meta = json.loads((old / "meta.json").read_text())
        new_meta = json.loads((new / "meta.json").read_text())
        old_meta.pop("session")
        new_meta.pop("session")
        assert new_meta == old_meta
        assert module._video_frame_count(old / "frames.mp4") == 32
        assert module._video_frame_count(new / "frames.mp4") == 32
        assert all(np.array_equal(actual, expected) for actual, expected in zip(
            _decoded_frames(new / "frames.mp4"), _decoded_frames(old / "frames.mp4")
        ))


def test_batch_failure_retries_the_existing_extractor(source, tmp_path, monkeypatch):
    module = _aligner()
    original_run = module.subprocess.run

    def fail_batch_only(command, *args, **kwargs):
        if "-filter_complex" in command:
            return subprocess.CompletedProcess(command, 1, "", "forced batch failure")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", fail_batch_only)
    times = module._probe_video_frame_times(source)
    jobs = [
        module._DirectVideoJob(tmp_path / f"sample_{index}", start, 2.3, 13.67647059)
        for index, start in enumerate((1.0, 3.0))
    ]
    outputs = module._extract_sample_videos_batch(
        source, jobs, resize_width=128, max_frames=32, crf=20,
        source_frame_times=times,
    )
    assert all(outputs[job.sample_dir] is not None for job in jobs)
    assert all(module._video_frame_count(job.sample_dir / "frames.mp4") == 32
               for job in jobs)
