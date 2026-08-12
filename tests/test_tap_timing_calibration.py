import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from trueskate_ai.vision.tap_timing_calibration import (
    detect_tap_onset,
    fit_tap_offsets,
)


def _tap_window(*, onset_s: float | None, command_s: float = 0.4):
    times = np.arange(0.0, 1.8, 1 / 30, dtype=np.float64)
    height, width = 120, 60
    point = (0.50, 0.65)
    cx = round(point[0] * (width - 1))
    cy = round(point[1] * (height - 1))
    frames = []
    for time_s in times:
        image = np.full((height, width, 3), (40, 70, 90), dtype=np.uint8)
        if onset_s is not None and onset_s <= time_s < onset_s + 0.20:
            cv2.circle(image, (cx, cy), 7, (10, 150, 245), thickness=-1)
        frames.append(image)
    return frames, times, point, command_s


def _aligner_module():
    path = Path(__file__).parents[1] / "scripts" / "data" / "align_xctest_traces.py"
    spec = importlib.util.spec_from_file_location("test_align_xctest_traces", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_tap_onset_uses_local_frame_difference():
    onset_s = 1.1
    frames, times, point, command_s = _tap_window(onset_s=onset_s)

    # A large unrelated UI change arrives earlier, outside the commanded-point ROI.
    unrelated_index = int(np.where(np.isclose(times, 0.7))[0][0])
    frames[unrelated_index][0:30, 0:20] = (255, 255, 255)
    frames[unrelated_index + 1][0:30, 0:20] = (255, 255, 255)

    result = detect_tap_onset(
        frames, times, point_xy=point, command_s=command_s,
    )

    assert result is not None
    assert result.onset_s == pytest.approx(onset_s)
    assert result.score >= result.threshold


def test_detect_tap_onset_rejects_a_one_frame_local_noise_spike():
    onset_s = 1.0
    frames, times, point, command_s = _tap_window(onset_s=onset_s)
    noise_index = int(np.where(np.isclose(times, 0.7))[0][0])
    frames[noise_index][66:88, 20:40] = (255, 255, 255)

    result = detect_tap_onset(
        frames, times, point_xy=point, command_s=command_s,
    )

    assert result is not None
    assert result.onset_s == pytest.approx(onset_s)


def test_detect_tap_onset_abstains_without_a_rendered_mark():
    frames, times, point, command_s = _tap_window(onset_s=None)

    assert detect_tap_onset(
        frames, times, point_xy=point, command_s=command_s,
    ) is None


def test_fit_tap_offsets_rejects_a_detector_outlier():
    result = fit_tap_offsets([1.10, 1.12, 2.50], min_taps=2, max_mad_s=0.10)

    assert result.accepted
    assert result.offset_s == pytest.approx(1.11)
    assert result.mad_s == pytest.approx(0.01)
    assert result.inlier_offsets_s == pytest.approx((1.10, 1.12))


def test_fit_tap_offsets_abstains_when_taps_do_not_agree():
    result = fit_tap_offsets([1.0, 1.3], min_taps=2, max_mad_s=0.10)

    assert not result.accepted
    assert result.offset_s == pytest.approx(1.15)
    assert result.reason == "tap offset MAD 0.150s exceeds allowed 0.100s"


def test_fit_tap_offsets_requires_enough_detected_taps():
    result = fit_tap_offsets([1.1], min_taps=2)

    assert not result.accepted
    assert result.offset_s is None
    assert result.reason == "need at least 2 detected taps; found 1"


def test_segment_tap_calibration_preserves_dispatch_path_relative_offset(monkeypatch, tmp_path):
    aligner = _aligner_module()
    started_at = 1000.0
    point = (0.50, 0.65)
    manifest = {
        "gestures": [
            {"gesture_distribution": "tap", "gesture_index": 1,
             "point": list(point), "t_call_start_epoch_s": started_at + 2.0},
            {"gesture_distribution": "hold", "gesture_index": 2,
             "point": list(point), "t_call_start_epoch_s": started_at + 4.0},
            {"gesture_distribution": "tap", "gesture_index": 3,
             "point": list(point), "t_call_start_epoch_s": started_at + 6.0},
        ]
    }

    def fake_decode(_mov, *, command_video_s, fps, **_kwargs):
        # A real 1.20 s tap offset, comfortably inside the production search window.
        frames, times, _point, _command = _tap_window(
            onset_s=1.70,
            command_s=0.50,
        )
        # _tap_window starts at zero; rebase its timestamps to the requested command.
        return frames, (times - _command + command_video_s).tolist()

    monkeypatch.setattr(aligner, "_decode_calibration_window", fake_decode)
    info, shift = aligner._tap_calibration(
        manifest=manifest,
        mov=tmp_path / "segment.mov",
        started_at=started_at,
        fps=30,
        delta_override=None,
        manifest_delta=None,
        min_taps=2,
        max_mad_s=0.10,
        search_after_s=4.0,
        resize_width=256,
    )

    # The segment shift is relative to the tap fallback (1.11), so adding it to a
    # hold's ActionChains fallback (1.06) preserves the measured -0.05 s difference.
    assert info["accepted"]
    assert info["tap_offset_s"] == pytest.approx(1.20)
    assert shift == pytest.approx(0.09)
    assert aligner._delta_for(manifest["gestures"][1], None, None) + shift == pytest.approx(1.15)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required by the production aligner")
def test_aligner_recovers_a_known_offset_from_a_synthetic_mov(tmp_path):
    aligner = _aligner_module()
    fps = 30
    width, height = 160, 360
    started_at = 1000.0
    taps = [
        # (command video PTS, rendered-mark PTS, normalised point)
        (2.0, 4.3, (0.45, 0.62)),
        (5.0, 7.3, (0.58, 0.72)),
    ]
    source = tmp_path / "source.mp4"
    mov = tmp_path / "segment_00000.mov"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    for frame_index in range(360):
        time_s = frame_index / fps
        image = np.full((height, width, 3), (40, 70, 90), dtype=np.uint8)
        for _command_s, onset_s, point in taps:
            if onset_s <= time_s < onset_s + 0.20:
                cv2.circle(
                    image,
                    (round(point[0] * (width - 1)), round(point[1] * (height - 1))),
                    12,
                    (10, 150, 245),
                    thickness=-1,
                )
        writer.write(image)
    writer.release()
    # XCTest emits h264. Re-encode the synthetic source so calibration sees the
    # same family of codec artefacts as production instead of lossless arrays.
    encoded = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-c:v", "libx264",
         "-crf", "28", "-pix_fmt", "yuv420p", str(mov)],
        capture_output=True,
        text=True,
    )
    assert encoded.returncode == 0, encoded.stderr

    manifest_path = tmp_path / "segment_00000.json"
    manifest_path.write_text(json.dumps({
        "mov": mov.name,
        "started_at_epoch_s": started_at,
        "device_logical_w": 375,
        "device_logical_h": 812,
        "gestures": [
            {"gesture_index": 1, "gesture_distribution": "tap", "point": list(taps[0][2]),
             "hold_duration_s": 0.0, "t_call_start_epoch_s": started_at + taps[0][0],
             "t_call_end_epoch_s": started_at + taps[0][0]},
            {"gesture_index": 2, "gesture_distribution": "hold", "point": [0.50, 0.70],
             "hold_duration_s": 0.5, "t_call_start_epoch_s": started_at + 7.0,
             "t_call_end_epoch_s": started_at + 7.5},
            {"gesture_index": 3, "gesture_distribution": "tap", "point": list(taps[1][2]),
             "hold_duration_s": 0.0, "t_call_start_epoch_s": started_at + taps[1][0],
             "t_call_end_epoch_s": started_at + taps[1][0]},
        ],
    }))

    saved = aligner.align_segment(
        manifest_path,
        pre_s=0.5,
        window_s=1.8,
        fps=fps,
        resize_width=width,
        max_frames=32,
        delta_override=None,
        delete_mov=False,
        tap_calibrate=True,
        tap_calibration_width=width,
    )

    assert saved == 3
    sample = tmp_path / "park" / "sample_000002" / "meta.json"
    meta = json.loads(sample.read_text())
    assert meta["capture_offset_source"] == "tap_self_calibrated"
    # Tap offset = 2.30 (the handover's observed drift range); the hold keeps the
    # ActionChains-relative -0.05 offset instead of inheriting the tap value exactly.
    assert meta["capture_offset_s"] == pytest.approx(2.25, abs=0.04)
    marker = json.loads((tmp_path / "segment_00000.aligned").read_text())
    assert marker["tap_calibration"]["accepted"]
    assert marker["tap_calibration"]["tap_offset_s"] == pytest.approx(2.30, abs=0.04)


def test_aligner_preserves_source_when_requested_calibration_rejects(monkeypatch, tmp_path):
    aligner = _aligner_module()
    mov = tmp_path / "segment_00000.mov"
    mov.write_bytes(b"not decoded because calibration rejects first")
    manifest_path = tmp_path / "segment_00000.json"
    manifest_path.write_text(json.dumps({
        "mov": mov.name,
        "started_at_epoch_s": 1000.0,
        "device_logical_w": 375,
        "device_logical_h": 812,
        "gestures": [],
    }))
    rejected = {
        "accepted": False,
        "reason": "need at least 2 detected taps; found 0",
    }
    monkeypatch.setattr(aligner, "_tap_calibration", lambda **_kwargs: (rejected, None))

    with pytest.raises(aligner.TapCalibrationRejected):
        aligner.align_segment(
            manifest_path,
            pre_s=0.5,
            window_s=1.8,
            fps=30,
            resize_width=160,
            max_frames=32,
            delta_override=None,
            delete_mov=True,
            tap_calibrate=True,
        )

    assert mov.exists()
    assert not (tmp_path / "segment_00000.aligned").exists()
    assert not (tmp_path / "segment_00000.aligning").exists()
    report = json.loads((tmp_path / "segment_00000.calibration_rejected.json").read_text())
    assert not report["accepted"]


def test_direct_video_extracts_a_compact_clip_without_temporary_pngs(tmp_path):
    aligner = _aligner_module()
    source = tmp_path / "segment.mov"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (80, 120))
    assert writer.isOpened()
    for index in range(75):
        writer.write(np.full((120, 80, 3), index, dtype=np.uint8))
    writer.release()
    encoded = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(tmp_path / "segment_h264.mov")],
        capture_output=True, text=True,
    )
    assert encoded.returncode == 0, encoded.stderr
    sample = tmp_path / "sample"
    assert aligner._extract_sample_video(
        tmp_path / "segment_h264.mov", sample, start_s=0.1, duration_s=1.2,
        resize_width=64, output_fps=20.0, max_frames=24, crf=20,
    )
    video = sample / "frames.mp4"
    assert video.is_file() and video.stat().st_size > 0
    cap = cv2.VideoCapture(str(video))
    try:
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == pytest.approx(24, abs=1)
    finally:
        cap.release()
    assert not list(sample.glob("*.png"))


def test_aligner_cli_exits_nonzero_when_requested_calibration_rejects(monkeypatch, tmp_path):
    aligner = _aligner_module()
    mov = tmp_path / "segment_00000.mov"
    mov.write_bytes(b"not decoded because there are no tap events")
    manifest_path = tmp_path / "segment_00000.json"
    manifest_path.write_text(json.dumps({
        "mov": mov.name,
        "started_at_epoch_s": 1000.0,
        "device_logical_w": 375,
        "device_logical_h": 812,
        "gestures": [],
    }))
    monkeypatch.setattr(
        aligner.sys,
        "argv",
        ["align_xctest_traces.py", "--segment", str(manifest_path), "--tap-calibrate"],
    )

    with pytest.raises(SystemExit) as exit_info:
        aligner.main()

    assert exit_info.value.code == 2
    assert mov.exists()
