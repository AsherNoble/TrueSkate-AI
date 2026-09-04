import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from trueskate_ai.data.cohort_manifest import (
    assert_zero_cohort_leakage, read_manifest, write_manifest,
)
from trueskate_ai.data.sequential_shards import (
    build_sequential_shards, materialize_sequential_shards,
)
from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset
from trueskate_ai.vision.model1_certification import (
    TouchTrack, certification_report, complete_gesture_recovered,
    one_sided_binomial_lower_bound,
)
from trueskate_ai.vision.model1_scaling import (
    assert_deterministic_nesting, build_experiment_manifest,
    build_linear_cohort_manifest, build_nested_subset_manifests,
    estimate_modal_rungs, fit_error_scaling_law, gradient_clipping_decision,
    scaling_status,
)


def _linear_sample(root: Path, index: int, *, device: str, park: str,
                   x_offset: float = 0.0, command_index: int | None = None) -> Path:
    command_index = index if command_index is None else command_index
    sample = root / f"session_2026090{1 + index % 3}_{device}" / park.replace(" ", "_") / f"sample_{index:04d}"
    sample.mkdir(parents=True)
    x0 = .20 + x_offset + command_index * .002
    duration = .40 + command_index * .001
    meta = {
        "gesture_distribution": "linear",
        "waypoints": [[x0, .35], [x0 + .30, .50]],
        "duration": duration,
        "easing_power": 1.0,
        "spin_active": False,
        "tap_calibration": {"accepted": True},
        "session": f"session_2026090{1 + index % 3}_{device}",
        "segment_index": index,
        "device": device,
        "park": park,
        "frame_times": [-.2, 0.0, duration, duration + .2],
    }
    (sample / "meta.json").write_text(json.dumps(meta))
    for frame in range(3):
        assert cv2.imwrite(str(sample / f"frame_{frame:03d}.png"),
                           np.full((10, 8, 3), frame * 30, np.uint8))
    return sample


def _cohort(root: Path, selection: str, *, role: str, start: int, count: int):
    selection_root = root / selection
    parks = ("SLS 2015 Super Crown", "SLS 2013 Kansas City")
    devices = ("iPhone_XR", "iPhone_XR2")
    for offset in range(count):
        _linear_sample(
            selection_root, offset, device=devices[offset % 2], park=parks[(offset // 2) % 2],
            x_offset=start * .001, command_index=start + offset,
        )
    return build_linear_cohort_manifest(
        selection_root, corpus_root=root, cohort=selection, role=role,
    )


def test_nested_manifests_are_deterministic_balanced_and_prefix_preserving(tmp_path):
    training = _cohort(tmp_path, "training", role="training", start=0, count=12)
    first = build_nested_subset_manifests(training, [4, 8, 12], seed=17)
    second = build_nested_subset_manifests(training, [4, 8, 12], seed=17)

    assert [item["fingerprint"] for item in first] == [item["fingerprint"] for item in second]
    assert_deterministic_nesting(first)
    assert [item["sample_count"] for item in first] == [4, 8, 12]
    for subset in first:
        assert max(subset["coverage"]["device"].values()) - min(
            subset["coverage"]["device"].values()
        ) <= 1


def test_experiment_manifest_drives_dataset_and_refuses_changed_content(tmp_path):
    training = _cohort(tmp_path, "training", role="training", start=0, count=6)
    validation = _cohort(tmp_path, "validation", role="validation", start=100, count=4)
    subset = build_nested_subset_manifests(training, [4], seed=0)[0]
    experiment = build_experiment_manifest(subset, validation, name="linear_n4")
    path = tmp_path / "manifests" / "linear_n4.json"
    write_manifest(path, experiment)

    dataset = BasicLinearClipDataset(
        tmp_path, manifest=path, manifest_partition="train",
        sequence_length=2, image_height=10, image_width=8,
    )
    assert len(dataset) == 4
    assert dataset.manifest_fingerprint == read_manifest(path)["fingerprint"]

    shard_dir = tmp_path / "shards"
    shard_payload = build_sequential_shards(
        tmp_path, path, shard_dir, max_samples=2, max_bytes=1024**2,
    )
    assert len(shard_payload["shards"]) == 4  # 4 train + 4 validation, two per shard
    staged_root = materialize_sequential_shards(shard_dir / "shards.json", tmp_path / "staged")
    staged = BasicLinearClipDataset(
        staged_root, manifest=shard_dir / "experiment.json", manifest_partition="train",
        sequence_length=2, image_height=10, image_width=8,
    )
    for key in ("frames", "target", "trajectory_xy", "trajectory_mask"):
        assert np.array_equal(dataset[0][key].numpy(), staged[0][key].numpy())

    changed = dataset.sample_paths[0] / "frame_000.png"
    assert cv2.imwrite(str(changed), np.full((10, 8, 3), 255, np.uint8))
    with pytest.raises(ValueError, match="content changed"):
        BasicLinearClipDataset(tmp_path, manifest=path, manifest_partition="train")


def test_cross_cohort_exact_command_collision_is_rejected(tmp_path):
    training = _cohort(tmp_path, "training", role="training", start=0, count=4)
    validation_root = tmp_path / "validation"
    _linear_sample(validation_root, 0, device="iPhone_XR", park="SLS 2013 Kansas City",
                   command_index=0)
    validation = build_linear_cohort_manifest(
        validation_root, corpus_root=tmp_path, cohort="validation", role="validation",
    )
    with pytest.raises(ValueError, match="cohort leakage by command"):
        assert_zero_cohort_leakage([training, validation])


def test_trainer_records_manifest_bound_train_and_validation_curves(tmp_path, monkeypatch):
    import scripts.train.train_basic_linear_regressor as trainer
    import torch

    training = _cohort(tmp_path, "training", role="training", start=0, count=6)
    validation = _cohort(tmp_path, "validation", role="validation", start=100, count=4)
    subset = build_nested_subset_manifests(training, [4], seed=0)[0]
    experiment = build_experiment_manifest(subset, validation, name="linear_n4")
    path = tmp_path / "experiment.json"
    write_manifest(path, experiment)
    monkeypatch.setattr(trainer, "_device", lambda: torch.device("cpu"))

    result = trainer.train(
        data=tmp_path, out=tmp_path / "model.pth", experiment_manifest=path,
        epochs=1, batch_size=2, lr=1e-3, seed=0, base_channels=2,
        image_width=16, image_height=36, evaluate_test=False,
        record_train_metrics=True,
    )

    assert result["experiment_manifest_fingerprint"] == experiment["fingerprint"]
    assert result["split_sizes"] == {"train": 4, "validation": 4, "test": 0}
    assert len(result["training_curve"]) == len(result["validation_curve"]) == 1
    assert result["epoch_history"][0]["training_samples_per_second"] > 0
    assert result["accelerator"]["type"] == "cpu"


def test_scaling_and_clipping_decisions_apply_predeclared_rules():
    observations = []
    for size, recoveries in ((100, [.80, .81, .79]), (200, [.83, .82, .84]),
                             (400, [.85, .84, .86])):
        observations.extend({"training_samples": size, "late_validation_recovery": value}
                            for value in recoveries)
    status = scaling_status(observations)
    assert status["plateau"] is True
    assert status["status"] == "plateau_diagnosis_required"

    selected = gradient_clipping_decision([.75, .85, .80], [.795, .805, .800])
    assert selected["selected"] is True
    rejected = gradient_clipping_decision([.75, .85, .80], [.77, .78, .76])
    assert rejected["selected"] is False


def test_scaling_law_fit_needs_residual_freedom_and_retains_uncertainty():
    observations = []
    for size in (100, 200, 400, 800):
        expected_error = .02 + 4.0 * size ** -.7
        for offset in (-.001, 0.0, .001):
            observations.append({
                "training_samples": size,
                "late_validation_recovery": 1.0 - expected_error + offset,
                "validation_samples": 12_000,
            })
    fit = fit_error_scaling_law(observations, bootstrap_samples=20, seed=4)
    assert fit["degrees_of_freedom"] == 1
    assert fit["parameters"]["e_floor"] == pytest.approx(.02, abs=.002)
    assert fit["parameters"]["alpha"] == pytest.approx(.7, abs=.03)
    assert fit["bootstrap_successes"] > 0
    assert set(fit["bootstrap_95_interval"]) == {"e_floor", "a", "alpha"}
    with pytest.raises(ValueError, match="at least four sizes"):
        fit_error_scaling_law(observations[:9], bootstrap_samples=0)


def test_modal_cost_estimate_uses_measured_base_and_additive_resources():
    estimate = estimate_modal_rungs([13_100, 26_200, 52_400])
    assert estimate["hourly_rate_usd"] == pytest.approx(1.316583)
    assert estimate["rungs"][0]["estimated_cost_usd"] == pytest.approx(33.25575, rel=.001)
    assert estimate["rungs"][1]["estimated_cost_usd"] == pytest.approx(
        2 * estimate["rungs"][0]["estimated_cost_usd"]
    )
    assert estimate["total_gpu_only_cost_usd"] == pytest.approx(
        sum(row["gpu_only_cost_usd"] for row in estimate["rungs"])
    )


def _drag(points, start=0.0, end=1.0, easing=1.0):
    return TouchTrack("drag", start, end, tuple(points), easing)


def _spin(start=.2, end=.8):
    return TouchTrack("spin", start, end, ((.06, .40),))


def test_complete_gesture_contract_scores_curves_spin_and_extra_touches():
    target = _drag(((.2, .3), (.5, .7), (.8, .4)))
    predicted = _drag(((.205, .3), (.5, .69), (.795, .4)), end=1.05)
    assert complete_gesture_recovered([predicted], [target], subtype="curved")

    bad_curve = _drag(((.2, .3), (.5, .55), (.8, .4)))
    assert not complete_gesture_recovered([bad_curve], [target], subtype="curved")
    assert complete_gesture_recovered(
        [predicted, _spin(.2 + 2 / 30, .8 - 2 / 30)],
        [target, _spin()], subtype="curved_spin",
    )
    assert not complete_gesture_recovered(
        [predicted, _spin(), _drag(((.1, .1), (.2, .2)))],
        [target, _spin()], subtype="curved_spin",
    )


def test_certification_bound_allows_at_most_twenty_failures_at_30000():
    assert one_sided_binomial_lower_bound(29_980, 30_000) > .999
    assert one_sided_binomial_lower_bound(29_979, 30_000) < .999
    passing = [True] * 29_980 + [False] * 20
    report = certification_report({
        "linear": passing, "curved": passing, "curved_spin": passing,
    })
    assert report["passes"] is True
    too_small = certification_report({
        "linear": [True] * 2_999, "curved": [True] * 2_999,
        "curved_spin": [True] * 2_999,
    })
    assert too_small["passes"] is False
