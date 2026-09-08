from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import cv2
import numpy as np
import pytest
import torch

from scripts.data.build_bc_clips import (
    LoadedTraceExtractor,
    _LEGACY_MODEL_TYPE,
    _TEMPORAL_MODEL_TYPE,
    frames_to_touch_detections,
    load_trace_extractor,
    resolve_inference_config,
    resolve_latency_s,
)
import scripts.data.build_bc_clips as build_bc_clips
from trueskate_ai.vision.temporal_trace_predictor import TemporalTracePredictor


def _write_frames(root: Path, values: list[int], *, h: int = 16, w: int = 12) -> list[Path]:
    paths = []
    for index, value in enumerate(values):
        path = root / f"frame_{index:06d}.png"
        assert cv2.imwrite(str(path), np.full((h, w, 3), value, dtype=np.uint8))
        paths.append(path)
    return paths


class _RecordingTemporalModel:
    """Strict fake: its signature rejects any teacher-feedback argument."""

    config = SimpleNamespace(use_time_deltas=True)

    def __init__(self, *, activity_logits: list[float] | None = None,
                 heatmap_scores: list[float] | None = None):
        self.activity_logits = activity_logits or [10.0, 10.0, 10.0]
        self.heatmap_scores = heatmap_scores or [0.95, 0.95, 0.95]
        self.calls: list[dict] = []

    def step(self, frame, state=None, *, delta_t=None):
        clip_step = 0 if state is None else int(state.clip_step) + 1
        call_index = len(self.calls)
        self.calls.append({
            "cold_start": state is None,
            "clip_step": clip_step,
            "delta_t": float(delta_t),
            "frame_mean": float(frame.mean()),
        })
        heatmap = torch.zeros((1, 1, *frame.shape[-2:]), dtype=frame.dtype,
                              device=frame.device)
        heatmap[0, 0, 5, min(2 + clip_step, frame.shape[-1] - 1)] = self.heatmap_scores[call_index]
        next_state = SimpleNamespace(clip_step=clip_step, previous_heatmap=heatmap)
        return SimpleNamespace(
            heatmap=heatmap,
            active_logits=torch.tensor([self.activity_logits[call_index]],
                                       device=frame.device),
            state=next_state,
        )


def test_loads_canonical_temporal_checkpoint_and_prefers_its_latency(tmp_path: Path) -> None:
    model = TemporalTracePredictor(
        base_channels=4,
        hidden_channels=8,
        downsample_stages=1,
        use_time_deltas=True,
    )
    checkpoint = tmp_path / "temporal.pth"
    torch.save({
        "checkpoint_version": 3,
        "model_type": _TEMPORAL_MODEL_TYPE,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "h": 16,
        "w": 12,
        "sequence_length": 6,
        "latency_s": 0.2,
        "inference_config": {
            "peak_threshold": 0.42,
            "activity_threshold": 0.61,
            "peak_nms_radius_px": 4,
            "max_touches": 3,
        },
    }, checkpoint)

    loaded = load_trace_extractor(checkpoint, torch.device("cpu"))

    assert loaded.is_temporal
    assert isinstance(loaded.model, TemporalTracePredictor)
    assert (loaded.h, loaded.w) == (16, 12)
    assert loaded.latency_s == pytest.approx(0.2)
    assert loaded.peak_threshold == pytest.approx(0.42)
    assert loaded.activity_threshold == pytest.approx(0.61)
    assert loaded.peak_nms_radius_px == 4
    assert loaded.max_touches == 3
    latency, source = resolve_latency_s(loaded, None)
    assert latency == pytest.approx(0.2)
    assert source == "checkpoint metadata"
    latency, source = resolve_latency_s(loaded, 0.125)
    assert latency == pytest.approx(0.125)
    assert source == "CLI override"

    config, sources = resolve_inference_config(
        loaded, peak_threshold=0.75, max_touches=5
    )
    assert config == {
        "peak_threshold": 0.75,
        "activity_threshold": 0.61,
        "peak_nms_radius_px": 4,
        "max_touches": 5,
    }
    assert sources == {
        "peak_threshold": "CLI override",
        "activity_threshold": "checkpoint inference_config",
        "peak_nms_radius_px": "checkpoint inference_config",
        "max_touches": "CLI override",
    }


def test_loads_metadata_free_legacy_checkpoint_without_filename_guessing(
    tmp_path: Path, monkeypatch,
) -> None:
    class LightweightLegacy(torch.nn.Module):
        def __init__(self, in_channels=3, base_channels=32):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(base_channels, in_channels))

    stub = types.ModuleType("trueskate_ai.vision.gaussian_bump_predictor")
    stub.GaussianBumpPredictor = LightweightLegacy
    monkeypatch.setitem(sys.modules, stub.__name__, stub)

    model = LightweightLegacy(base_channels=2)
    checkpoint = tmp_path / "arbitrary_name.pth"
    torch.save({
        "model_state": model.state_dict(),
        "base_channels": 2,
        "h": 16,
        "w": 16,
    }, checkpoint)

    loaded = load_trace_extractor(checkpoint, torch.device("cpu"))

    assert loaded.model_type == _LEGACY_MODEL_TYPE
    assert not loaded.is_temporal
    assert isinstance(loaded.model, LightweightLegacy)
    assert loaded.latency_s is None
    config, sources = resolve_inference_config(loaded)
    assert config == {
        "peak_threshold": 0.30,
        "activity_threshold": 0.50,
        "peak_nms_radius_px": 6,
        "max_touches": 2,
    }
    assert set(sources.values()) == {"historical default"}


@pytest.mark.parametrize(
    ("inference_config", "error"),
    [
        ("bad", "must be a mapping"),
        ({"peak_threshold": -0.1}, "peak_threshold.*\\[0, 1\\]"),
        ({"activity_threshold": float("nan")}, "activity_threshold.*finite"),
        ({"peak_nms_radius_px": 1.5}, "peak_nms_radius_px.*integer >= 0"),
        ({"peak_nms_radius_px": -1}, "peak_nms_radius_px.*integer >= 0"),
        ({"max_touches": 0}, "max_touches.*integer >= 1"),
    ],
)
def test_rejects_malformed_checkpoint_inference_config(
    tmp_path: Path, inference_config, error: str,
) -> None:
    checkpoint = tmp_path / "invalid.pth"
    torch.save({"inference_config": inference_config}, checkpoint)

    with pytest.raises(ValueError, match=error):
        load_trace_extractor(checkpoint, torch.device("cpu"))


def test_cli_decoder_defaults_defer_to_checkpoint(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_bc_clips.py",
            "--model", "model.pth",
            "--clips-root", "clips",
            "--out", "out",
        ],
    )
    monkeypatch.setattr(build_bc_clips, "_run", captured.append)

    build_bc_clips.main()

    assert len(captured) == 1
    args = captured[0]
    assert args.active_thresh is None
    assert args.activity_thresh is None
    assert args.peak_nms_radius_px is None
    assert args.max_touches is None


def test_latency_never_silently_uses_legacy_default_for_temporal() -> None:
    temporal = LoadedTraceExtractor(
        model=object(), h=16, w=12, model_type=_TEMPORAL_MODEL_TYPE, latency_s=None
    )
    with pytest.raises(ValueError, match="no latency_s metadata"):
        resolve_latency_s(temporal, None)

    legacy = LoadedTraceExtractor(
        model=object(), h=16, w=12, model_type=_LEGACY_MODEL_TYPE, latency_s=None
    )
    latency, source = resolve_latency_s(legacy, None)
    assert latency == pytest.approx(0.45)
    assert "legacy" in source


def test_temporal_inference_is_chronological_autoregressive_and_resets_per_clip(
    tmp_path: Path,
) -> None:
    paths = _write_frames(tmp_path, [16, 64, 192])
    times = np.array([1.2, 1.24, 1.31], dtype=np.float64)
    model = _RecordingTemporalModel(
        activity_logits=[10.0] * 6,
        heatmap_scores=[0.95] * 6,
    )

    first = frames_to_touch_detections(
        model,
        paths,
        times,
        16,
        12,
        torch.device("cpu"),
        model_type=_TEMPORAL_MODEL_TYPE,
        active_thresh=0.3,
    )
    second = frames_to_touch_detections(
        model,
        paths,
        times,
        16,
        12,
        torch.device("cpu"),
        model_type=_TEMPORAL_MODEL_TYPE,
        active_thresh=0.3,
    )

    assert [len(peaks) for peaks in first] == [1, 1, 1]
    assert [len(peaks) for peaks in second] == [1, 1, 1]
    assert [call["cold_start"] for call in model.calls] == [True, False, False] * 2
    assert [call["clip_step"] for call in model.calls] == [0, 1, 2] * 2
    assert [call["delta_t"] for call in model.calls[:3]] == pytest.approx([0.0, 0.04, 0.07])
    assert [call["frame_mean"] for call in model.calls[:3]] == pytest.approx(
        [16 / 255, 64 / 255, 192 / 255]
    )
    # The peak's state-dependent motion proves each prediction consumed the
    # causal rollout; resetting reproduces the same path in the second clip.
    assert [peak[0].x for peak in first] == pytest.approx([2 / 11, 3 / 11, 4 / 11])
    assert [peak[0].x for peak in second] == pytest.approx([2 / 11, 3 / 11, 4 / 11])


def test_temporal_activity_and_spatial_thresholds_are_both_required(tmp_path: Path) -> None:
    paths = _write_frames(tmp_path, [32, 64, 96])
    model = _RecordingTemporalModel(
        activity_logits=[10.0, -10.0, 10.0],
        heatmap_scores=[0.95, 0.95, 0.20],
    )

    detections = frames_to_touch_detections(
        model,
        paths,
        np.arange(3, dtype=np.float64) / 30.0,
        16,
        12,
        torch.device("cpu"),
        model_type=_TEMPORAL_MODEL_TYPE,
        activity_thresh=0.5,
        active_thresh=0.3,
    )

    assert [len(peaks) for peaks in detections] == [1, 0, 0]
