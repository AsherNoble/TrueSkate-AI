import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from trueskate_ai.data.gesture_sampling import (
    BASIC_HOLD_MAX_S,
    BASIC_HOLD_MIN_S,
    sample_basic_hold_mixture,
)
from trueskate_ai.vision.basic_hold_dataset import (
    BasicHoldClipDataset,
    split_by_command,
    split_by_segment,
)
from trueskate_ai.vision.basic_hold_regressor import BasicHoldRegressor
from trueskate_ai.vision.basic_hold_training import passes_basic_hold_acceptance


def _write_sample(root: Path, segment: str, name: str, *, kind: str = "hold",
                  duration: float = 0.6, calibrated: bool = True, menu: bool = False) -> Path:
    sample = root / segment / "the_workshop" / name
    sample.mkdir(parents=True)
    meta = {
        "gesture_distribution": kind,
        "point": [0.4, 0.6],
        "hold_duration_s": duration,
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


def test_basic_hold_dataset_only_admits_calibrated_positive_duration_holds(tmp_path):
    accepted = _write_sample(tmp_path, "segment_1", "sample_ok")
    _write_sample(tmp_path, "segment_1", "sample_tap", kind="tap", duration=0.0)
    _write_sample(tmp_path, "segment_2", "sample_short", duration=0.2)
    _write_sample(tmp_path, "segment_2", "sample_uncalibrated", calibrated=False)
    _write_sample(tmp_path, "segment_3", "sample_menu", menu=True)

    dataset = BasicHoldClipDataset(tmp_path, sequence_length=4, image_height=20, image_width=12)
    assert dataset.sample_paths == (accepted,)
    assert dataset.stats == {
        "accepted": 1, "discovered": 5, "rejected_duration_out_of_range": 1,
        "rejected_menu_marked": 1, "rejected_not_hold": 1, "rejected_uncalibrated": 1,
    }
    item = dataset[0]
    assert item["frames"].shape == (4, 3, 20, 12)
    assert item["target"].tolist() == pytest.approx([0.4, 0.6, 0.6])


def test_segment_split_is_disjoint(tmp_path):
    for segment in ("segment_1", "segment_2", "segment_3", "segment_4", "segment_5"):
        _write_sample(tmp_path, segment, f"sample_{segment}")
    dataset = BasicHoldClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    train, validation, test = split_by_segment(dataset, seed=9)
    grouped = [set(dataset.segment_keys[index] for index in indices) for indices in (train, validation, test)]
    assert all(grouped)
    assert not (grouped[0] & grouped[1] or grouped[0] & grouped[2] or grouped[1] & grouped[2])


def test_command_split_keeps_replayed_holds_in_one_partition(tmp_path):
    # The old per-segment loop restarted the RNG at seed zero.  A command split
    # must never let an exactly replayed hold appear on both sides.
    for segment in ("segment_1", "segment_2", "segment_3", "segment_4", "segment_5"):
        _write_sample(tmp_path, segment, f"sample_a_{segment}")
        _write_sample(tmp_path, segment, f"sample_b_{segment}", duration=0.9)
        _write_sample(tmp_path, segment, f"sample_c_{segment}", duration=1.2)
    dataset = BasicHoldClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    train, validation, test = split_by_command(dataset, seed=9)
    grouped = [set(dataset.command_keys[index] for index in indices) for indices in (train, validation, test)]
    assert all(grouped)
    assert not (grouped[0] & grouped[1] or grouped[0] & grouped[2] or grouped[1] & grouped[2])


def test_regressor_returns_native_bounded_triplets():
    model = BasicHoldRegressor(base_channels=4)
    frames = torch.rand(2, 5, 3, 30, 18)
    output = model(frames)
    assert output.shape == (2, 3)
    assert torch.all((output[:, :2] >= 0.0) & (output[:, :2] <= 1.0))
    assert torch.all((output[:, 2] >= BASIC_HOLD_MIN_S) & (output[:, 2] <= BASIC_HOLD_MAX_S))
    gesture = model.predict_hold(frames)
    assert set(gesture) == {"x", "y", "dur"}
    assert gesture["dur"].shape == (2,)


def test_basic_hold_sampler_and_acceptance_contract():
    rng = np.random.default_rng(8)
    samples = [sample_basic_hold_mixture(rng) for _ in range(100)]
    holds = [sample for sample in samples if sample.kind == "hold"]
    assert holds and all(BASIC_HOLD_MIN_S <= sample.hold_duration_s <= BASIC_HOLD_MAX_S for sample in holds)
    assert all(sample.kind in {"hold", "tap"} for sample in samples)
    assert passes_basic_hold_acceptance({"coordinate_median": 0.03, "duration_mae": 0.10})
    assert not passes_basic_hold_acceptance({"coordinate_median": 0.031, "duration_mae": 0.10})
