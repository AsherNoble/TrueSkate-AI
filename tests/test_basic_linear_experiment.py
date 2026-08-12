import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from trueskate_ai.data.gesture_sampling import (
    BASIC_LINEAR_MAX_ABS_SLOPE,
    BASIC_LINEAR_MAX_S,
    BASIC_LINEAR_MIN_DX,
    BASIC_LINEAR_MIN_S,
    sample_basic_linear_mixture,
)
from trueskate_ai.vision.basic_linear_dataset import (
    BasicLinearClipDataset,
    split_by_command,
    split_by_segment,
)
from trueskate_ai.vision.basic_linear_regressor import BasicLinearRegressor
from trueskate_ai.vision.basic_linear_training import (
    basic_linear_endpoint_map_loss, basic_linear_loss, basic_linear_metrics,
    passes_basic_linear_acceptance,
)
from scripts.train.train_basic_linear_regressor import _IndexedSubset


def _write_sample(root: Path, segment: str, name: str, *, kind: str = "linear",
                  points: list[list[float]] | None = None, duration: float = 0.6,
                  easing: float = 1.0, calibrated: bool = True, menu: bool = False) -> Path:
    sample = root / segment / "the_workshop" / name
    sample.mkdir(parents=True)
    points = points or [[0.30, 0.40], [0.60, 0.55]]
    meta = {
        "gesture_distribution": kind,
        "waypoints": points,
        "duration": duration,
        "easing_power": easing,
        "spin_active": False,
        "tap_calibration": {"accepted": calibrated},
        "session": "test_session",
        "segment_index": int(segment.rsplit("_", 1)[-1]),
    }
    (sample / "meta.json").write_text(json.dumps(meta))
    if menu:
        (sample / ".menu").touch()
    for index in range(3):
        assert cv2.imwrite(str(sample / f"frame_{index:03d}.png"), np.full((10, 8, 3), 30 * index, np.uint8))
    return sample


def test_linear_dataset_only_admits_calibrated_two_point_constant_velocity_lines(tmp_path):
    accepted = _write_sample(tmp_path, "segment_1", "sample_ok")
    _write_sample(tmp_path, "segment_1", "sample_tap", kind="tap")
    _write_sample(tmp_path, "segment_2", "sample_uncalibrated", calibrated=False)
    _write_sample(tmp_path, "segment_2", "sample_near_vertical", points=[[.4, .3], [.45, .5]])
    _write_sample(tmp_path, "segment_3", "sample_curve", easing=2.0)
    _write_sample(tmp_path, "segment_3", "sample_menu", menu=True)
    mismatch = _write_sample(tmp_path, "segment_4", "sample_mismatch")
    (mismatch / ".trace_mismatch").write_text("mismatched rendered trace\n")

    dataset = BasicLinearClipDataset(tmp_path, sequence_length=4, image_height=20, image_width=12)
    assert dataset.sample_paths == (accepted,)
    assert dataset.stats == {
        "accepted": 1, "discovered": 7, "rejected_menu_marked": 1,
        "rejected_near_vertical": 1, "rejected_not_constant_velocity": 1,
        "rejected_not_linear": 1, "rejected_trace_mismatch": 1, "rejected_uncalibrated": 1,
    }
    item = dataset[0]
    assert item["frames"].shape == (4, 3, 20, 12)
    assert item["target"].tolist() == pytest.approx([.3, .4, .6, .55, .6])

    cached = BasicLinearClipDataset(tmp_path, sequence_length=4, image_height=20, image_width=12,
                                    cache_frames=True)
    first = cached[0]["frames"]
    second = cached[0]["frames"]
    assert len(cached._frame_cache) == 1
    assert next(iter(cached._frame_cache.values())).dtype == torch.uint8
    assert torch.equal(first, second)


def test_linear_segment_and_command_splits_are_disjoint(tmp_path):
    for segment in ("segment_1", "segment_2", "segment_3", "segment_4", "segment_5"):
        _write_sample(tmp_path, segment, f"a_{segment}")
        _write_sample(tmp_path, segment, f"b_{segment}", points=[[.25, .65], [.62, .52]])
        _write_sample(tmp_path, segment, f"c_{segment}", points=[[.72, .30], [.42, .62]])
    dataset = BasicLinearClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    for splitter, keys in ((split_by_segment, dataset.segment_keys), (split_by_command, dataset.command_keys)):
        partitions = splitter(dataset, seed=9)
        grouped = [set(keys[index] for index in indices) for indices in partitions]
        assert all(grouped)
        assert not (grouped[0] & grouped[1] or grouped[0] & grouped[2] or grouped[1] & grouped[2])


def test_linear_regressor_returns_native_bounded_quintuplets():
    model = BasicLinearRegressor(base_channels=4)
    frames = torch.rand(2, 5, 3, 30, 18)
    output = model(frames)
    assert output.shape == (2, 5)
    assert torch.all((output[:, :4] >= 0.0) & (output[:, :4] <= 1.0))
    assert torch.all((output[:, 4] >= BASIC_LINEAR_MIN_S) & (output[:, 4] <= BASIC_LINEAR_MAX_S))
    gesture = model.predict_linear(frames)
    assert set(gesture) == {"x0", "y0", "x1", "y1", "dur"}
    assert gesture["dur"].shape == (2,)


def test_linear_regressor_accepts_explicit_start_time_prior():
    model = BasicLinearRegressor(start_onset=-.24, start_sigma=.08)
    assert model.start_onset == pytest.approx(-.24)
    assert model.start_sigma == pytest.approx(.08)
    with pytest.raises(ValueError, match="start_sigma"):
        BasicLinearRegressor(start_sigma=0.)


def test_linear_regressor_retains_stride_two_spatial_endpoint_evidence():
    model = BasicLinearRegressor(base_channels=4)
    assert model.start_score is not model.end_score
    encoded = model.encoder(torch.rand(2, 6, 30, 18))
    assert encoded.shape[-2:] == (15, 9)


def test_linear_sampler_and_acceptance_contract():
    rng = np.random.default_rng(8)
    samples = [sample_basic_linear_mixture(rng) for _ in range(200)]
    linears = [sample for sample in samples if sample.kind == "linear"]
    assert linears and all(sample.kind in {"linear", "tap"} for sample in samples)
    for sample in linears:
        (x0, y0), (x1, y1) = sample.waypoints
        dx = x1 - x0
        assert abs(dx) >= BASIC_LINEAR_MIN_DX
        assert abs((y1 - y0) / dx) <= BASIC_LINEAR_MAX_ABS_SLOPE
        assert BASIC_LINEAR_MIN_S <= sample.duration <= BASIC_LINEAR_MAX_S
        assert sample.easing_power == 1.0
        assert sample.meta()["payload_total_s"] == sample.duration
    assert all(sample_basic_linear_mixture(rng, tap_fraction=0.0).kind == "linear" for _ in range(20))
    with pytest.raises(ValueError, match="tap_fraction"):
        sample_basic_linear_mixture(rng, tap_fraction=1.0)
    metrics = {
        "start_coordinate_median": .03, "end_coordinate_median": .03,
        "duration_mae": .10, "gesture_recovery_accuracy": .95,
    }
    assert passes_basic_linear_acceptance(metrics)
    assert not passes_basic_linear_acceptance({**metrics, "end_coordinate_median": .031})
    assert not passes_basic_linear_acceptance({**metrics, "gesture_recovery_accuracy": .949})


def test_recovery_metric_requires_every_component_to_be_within_tolerance():
    class ExactThenNearMiss(torch.nn.Module):
        def forward(self, frames):
            # First is exactly recoverable; second exceeds only the end tolerance.
            return torch.tensor([[.3, .4, .6, .55, .6], [.3, .4, .631, .55, .6]],
                                dtype=frames.dtype, device=frames.device)

    loader = DataLoader([
        {"frames": torch.zeros(3, 3, 10, 8), "target": torch.tensor([.3, .4, .6, .55, .6])},
        {"frames": torch.zeros(3, 3, 10, 8), "target": torch.tensor([.3, .4, .6, .55, .6])},
    ], batch_size=2)
    metrics = basic_linear_metrics(ExactThenNearMiss(), loader, torch.device("cpu"))
    assert metrics["gesture_recovery_accuracy"] == pytest.approx(.5)
    assert metrics["start_recovery_accuracy"] == pytest.approx(1.0)
    assert metrics["end_recovery_accuracy"] == pytest.approx(.5)
    assert metrics["duration_recovery_accuracy"] == pytest.approx(1.0)


def test_linear_endpoint_map_auxiliary_requires_maps_and_is_differentiable():
    prediction = torch.tensor([[.3, .4, .6, .55, .6]], requires_grad=True)
    target = torch.tensor([[.3, .4, .6, .55, .6]])
    start = torch.rand(1, 4, 5, 6, requires_grad=True)
    end = torch.rand(1, 4, 5, 6, requires_grad=True)
    with pytest.raises(ValueError, match="score maps"):
        basic_linear_loss(prediction, target, map_weight=.01)
    loss = basic_linear_loss(prediction, target, start_scores=start, end_scores=end, map_weight=.01)
    loss.backward()
    assert start.grad is not None and end.grad is not None
    direct = basic_linear_endpoint_map_loss(start.detach(), target[:, :2], torch.tensor([.24]))
    assert torch.isfinite(direct)


def test_indexed_subset_preserves_original_dataset_indices(tmp_path):
    _write_sample(tmp_path, "segment_1", "one")
    _write_sample(tmp_path, "segment_2", "two", points=[[.25, .65], [.62, .52]])
    dataset = BasicLinearClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    item = _IndexedSubset(dataset, [1])[0]
    assert item["sample_index"] == 1


def test_linear_sample_meta_carries_device_provenance(tmp_path):
    # The dataset preserves extra metadata emitted by the aligner; this fixture
    # protects the field used by the held-out device recovery audit.
    sample = _write_sample(tmp_path, "segment_1", "one")
    meta = json.loads((sample / "meta.json").read_text())
    meta["device"] = "iPhone_XR2"
    (sample / "meta.json").write_text(json.dumps(meta))
    dataset = BasicLinearClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    assert dataset._meta(dataset.sample_paths[0])["device"] == "iPhone_XR2"


def test_linear_collector_exposes_native_resolution_option():
    result = subprocess.run(
        [sys.executable, "scripts/data/collect_sls_xctest.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--align-resize-width" in result.stdout


def test_linear_collector_uses_a_device_specific_seed_file():
    source = Path("scripts/ops/mvp_collect_linear.sh").read_text()
    assert ".basic_linear_next_seed_${DEVICE}" in source
    assert "device_seed=$(printf '%s' \"$DEVICE\" | cksum" in source
    assert 'printf \'%s\\n\' "$next_seed" > "$SEED_FILE"' in source


def test_linear_finalizer_can_target_a_fresh_modal_volume():
    source = Path("scripts/ops/finalize_basic_linear_run.sh").read_text()
    assert 'VOLUME="${MODAL_CORPUS_VOLUME:-trueskate-mvp}"' in source
    assert '--volume "$VOLUME"' in source


def test_linear_dataset_decodes_only_selected_video_frames(monkeypatch, tmp_path):
    accepted = _write_sample(tmp_path, "segment_1", "sample")
    for frame in accepted.glob("frame_*.png"):
        frame.unlink()
    (accepted / "frames.mp4").write_bytes(b"placeholder")
    calls: list[int] = []

    def fake_decode(sample, count):
        assert sample == accepted and count == 4
        calls.append(count)
        return [np.full((20, 12, 3), index, np.uint8) for index in range(count)]

    monkeypatch.setattr("trueskate_ai.vision.basic_linear_dataset._decode_even_frames", fake_decode)
    dataset = BasicLinearClipDataset(tmp_path, sequence_length=4, image_height=20, image_width=12)
    assert dataset[0]["frames"].shape == (4, 3, 20, 12)
    assert calls == [4]


def test_selected_video_decode_falls_back_when_random_seek_is_unreliable(monkeypatch, tmp_path):
    import trueskate_ai.vision.basic_hold_dataset as holds

    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "frames.mp4").write_bytes(b"placeholder")

    class BrokenSeekCapture:
        def __init__(self, *_args):
            pass
        def get(self, _property):
            return 4
        def set(self, *_args):
            pass
        def read(self):
            return False, None
        def release(self):
            pass

    frames = [np.full((3, 2, 3), index, np.uint8) for index in range(4)]
    monkeypatch.setattr(holds.cv2, "VideoCapture", BrokenSeekCapture)
    monkeypatch.setattr(holds, "_decode_frames", lambda _sample: frames)
    selected = holds._decode_even_frames(sample, 2)
    assert [int(frame[0, 0, 0]) for frame in selected] == [0, 3]


def test_modal_linear_training_reserves_cache_headroom():
    source = Path("scripts/cloud/train_basic_linear_modal.py").read_text()
    assert 'memory=32768' in source
