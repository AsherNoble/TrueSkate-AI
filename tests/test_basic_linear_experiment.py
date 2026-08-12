import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

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
from trueskate_ai.vision.basic_linear_training import passes_basic_linear_acceptance


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

    dataset = BasicLinearClipDataset(tmp_path, sequence_length=4, image_height=20, image_width=12)
    assert dataset.sample_paths == (accepted,)
    assert dataset.stats == {
        "accepted": 1, "discovered": 6, "rejected_menu_marked": 1,
        "rejected_near_vertical": 1, "rejected_not_constant_velocity": 1,
        "rejected_not_linear": 1, "rejected_uncalibrated": 1,
    }
    item = dataset[0]
    assert item["frames"].shape == (4, 3, 20, 12)
    assert item["target"].tolist() == pytest.approx([.3, .4, .6, .55, .6])


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
    metrics = {"start_coordinate_median": .03, "end_coordinate_median": .03, "duration_mae": .10}
    assert passes_basic_linear_acceptance(metrics)
    assert not passes_basic_linear_acceptance({**metrics, "end_coordinate_median": .031})
