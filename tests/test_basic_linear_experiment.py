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
    basic_linear_endpoint_map_loss, basic_linear_trajectory_map_loss, basic_linear_loss, basic_linear_metrics,
    basic_linear_recovery_records, passes_basic_linear_acceptance, RECOVERY_ENDPOINT_TOLERANCE,
)
from scripts.train.train_basic_linear_regressor import _IndexedSubset, split_with_fresh_command_holdout


def _write_sample(root: Path, segment: str, name: str, *, kind: str = "linear",
                  points: list[list[float]] | None = None, duration: float = 0.6,
                  easing: float = 1.0, calibrated: bool = True, menu: bool = False,
                  device: str | None = None) -> Path:
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
        "frame_times": [-.5, 0., duration / 2, duration, duration + .3],
    }
    if device is not None:
        meta["device"] = device
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
    assert item["trajectory_xy"].shape == (4, 2)
    assert item["trajectory_mask"].dtype == torch.bool

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


def test_linear_command_identity_includes_both_endpoints_and_duration(tmp_path):
    _write_sample(tmp_path, "segment_1", "a", points=[[.30, .40], [.60, .55]], duration=.60)
    _write_sample(tmp_path, "segment_2", "b", points=[[.30, .40], [.61, .55]], duration=.60)
    _write_sample(tmp_path, "segment_3", "c", points=[[.30, .40], [.60, .55]], duration=.61)
    dataset = BasicLinearClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    assert len(set(dataset.command_keys)) == 3


def test_fresh_command_holdout_keeps_legacy_training_only_and_rejects_overlap(tmp_path):
    for index in range(4):
        _write_sample(tmp_path / "legacy", f"segment_{index}", f"legacy_{index}",
                      points=[[.20 + index * .01, .40], [.60 + index * .01, .55]])
    for index in range(8):
        _write_sample(tmp_path / "fresh", f"segment_{index}", f"fresh_{index}",
                      points=[[.30 + index * .01, .35], [.70 + index * .01, .50]])
    dataset = BasicLinearClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    train, val, test = split_with_fresh_command_holdout(dataset, fresh_source="fresh", seed=3)
    fresh = {index for index, path in enumerate(dataset.sample_paths)
             if "fresh" in path.relative_to(tmp_path).parts}
    legacy = set(range(len(dataset))) - fresh
    assert set(val).issubset(fresh) and set(test).issubset(fresh)
    assert legacy.issubset(train)
    assert not ({dataset.command_keys[index] for index in train}
                & ({dataset.command_keys[index] for index in val} | {dataset.command_keys[index] for index in test}))
    _write_sample(tmp_path / "fresh", "segment_99", "dup", points=[[.20, .40], [.60, .55]])
    overlapping = BasicLinearClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    with pytest.raises(ValueError, match="overlap"):
        split_with_fresh_command_holdout(overlapping, fresh_source="fresh", seed=3)


def test_device_stratified_fresh_holdout_represents_each_phone_in_both_evaluations(tmp_path):
    for device, x_offset in (("iPhone_XR", 0.0), ("iPhone_XR2", .10)):
        for index in range(10):
            _write_sample(tmp_path / "fresh", f"{device}_segment_{index}", f"{device}_{index}",
                          points=[[.20 + x_offset + index * .01, .35],
                                  [.55 + x_offset + index * .01, .50]],
                          device=device)
    dataset = BasicLinearClipDataset(tmp_path, sequence_length=2, image_height=10, image_width=8)
    train, validation, test = split_with_fresh_command_holdout(
        dataset, fresh_source="fresh", seed=3, stratify_by_device=True,
    )
    assert train
    for partition in (validation, test):
        devices = {dataset._meta(dataset.sample_paths[index])["device"] for index in partition}
        assert devices == {"iPhone_XR", "iPhone_XR2"}


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


def test_train_records_preclip_gradient_norms_only_when_clipping_is_enabled(tmp_path, monkeypatch):
    """Gradient clipping must be measurable, while the control remains unchanged."""
    import scripts.train.train_basic_linear_regressor as trainer

    for index in range(12):
        _write_sample(tmp_path, f"segment_{index}", f"sample_{index}",
                      points=[[.20 + index * .01, .35], [.58 + index * .01, .55]])
    monkeypatch.setattr(trainer, "_device", lambda: torch.device("cpu"))
    kwargs = dict(data=tmp_path, epochs=1, batch_size=2, lr=1e-3, seed=7,
                  base_channels=2, image_width=16, image_height=36, evaluate_test=False)
    control = trainer.train(out=tmp_path / "control.pth", **kwargs)
    clipped = trainer.train(out=tmp_path / "clipped.pth", max_grad_norm=.01, **kwargs)

    assert control["max_grad_norm"] is None
    assert len(control["gradient_norm_history"]) == 1
    assert control["gradient_norm_history"][0]["steps"] > 0
    assert clipped["max_grad_norm"] == pytest.approx(.01)
    assert len(clipped["gradient_norm_history"]) == 1
    gradient = clipped["gradient_norm_history"][0]
    assert gradient["steps"] > 0
    assert gradient["max"] >= gradient["p95"] >= gradient["mean"] > 0
    assert 0 <= gradient["clipped_steps"] <= gradient["steps"]


def test_linear_regressor_accepts_explicit_start_time_prior():
    model = BasicLinearRegressor(start_onset=-.24, start_sigma=.08, end_onset=.24)
    assert model.start_onset == pytest.approx(-.24)
    assert model.start_sigma == pytest.approx(.08)
    assert model.end_onset == pytest.approx(.24)
    with pytest.raises(ValueError, match="start_sigma"):
        BasicLinearRegressor(start_sigma=0.)


def test_linear_regressor_optional_temporal_mixer_preserves_output_contract():
    model = BasicLinearRegressor(base_channels=4, temporal_mixer=True)
    assert model.temporal_mixer is not None
    output = model(torch.rand(2, 5, 3, 30, 18))
    assert output.shape == (2, 5)
    assert torch.all((output[:, :4] >= 0.0) & (output[:, :4] <= 1.0))


def test_linear_regressor_can_expose_separate_trajectory_track_scores():
    model = BasicLinearRegressor(base_channels=4, trajectory_track=True)
    prediction, start, end, track = model.forward_with_track_scores(torch.rand(2, 5, 3, 30, 18))
    assert prediction.shape == (2, 5)
    assert track.shape == start.shape == end.shape == (2, 5, 15, 9)
    assert model.trajectory_fusion is not None
    assert model.trajectory_fusion.item() == pytest.approx(-4.0)


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


def test_linear_trajectory_map_auxiliary_uses_active_frame_targets_and_is_differentiable():
    scores = torch.rand(1, 4, 5, 6, requires_grad=True)
    path = torch.tensor([[[.2, .3], [.3, .4], [.4, .5], [.5, .6]]])
    mask = torch.tensor([[False, True, True, False]])
    loss = basic_linear_trajectory_map_loss(scores, path, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad is not None
    prediction = torch.tensor([[.3, .4, .6, .55, .6]], requires_grad=True)
    target = prediction.detach().clone()
    with pytest.raises(ValueError, match="trajectory targets"):
        basic_linear_loss(prediction, target, start_scores=scores.detach(), end_scores=scores.detach(),
                          trajectory_weight=.01)
    track = torch.rand(1, 4, 5, 6, requires_grad=True)
    loss = basic_linear_loss(prediction, target, start_scores=scores.detach(), end_scores=scores.detach(),
                             trajectory_xy=path, trajectory_mask=mask, trajectory_weight=.01,
                             trajectory_scores=track)
    loss.backward()
    assert track.grad is not None


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


def test_linear_collector_supports_clean_segment_boundary_resets():
    result = subprocess.run(
        [sys.executable, "scripts/data/collect_sls_xctest.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--reset-before-segment" in result.stdout
    assert "--segment-reset-settle-s" in result.stdout
    assert "--no-menu-guard" in result.stdout
    assert "--no-run-notifications" in result.stdout
    assert "--heartbeat-path" in result.stdout
    source = Path("scripts/data/collect_sls_xctest.py").read_text()
    assert "--segment-reset-settle-s must be >= 1.5" in source
    assert source.index("if args.reset_before_segment:") < source.index("rec.start()")
    assert source.index("if not args.no_gameplay_guard:", source.index("post-gesture foreground")) < \
        source.index("if not args.no_menu_guard:", source.index("post-gesture menu/editor"))


def test_linear_collector_uses_a_device_specific_seed_file():
    source = Path("scripts/ops/mvp_collect_linear.sh").read_text()
    assert ".basic_linear_next_seed_${DEVICE}" in source
    assert "device_seed=$(printf '%s' \"$DEVICE\" | cksum" in source
    assert 'printf \'%s\\n\' "$next_seed" > "$SEED_FILE"' in source
    assert 'BASIC_LINEAR_CALIBRATION_TAPS_PER_SEGMENT' in source
    assert '--calibration-taps-per-segment "$CALIBRATION_TAPS_PER_SEGMENT"' in source
    assert 'BASIC_LINEAR_CALIBRATION_TAP_HOLD_S' in source
    assert '--calibration-tap-hold-s "$CALIBRATION_TAP_HOLD_S"' in source
    assert '--reset-before-segment' in source
    assert 'BASIC_LINEAR_NO_MENU_GUARD' in source
    assert 'MENU_GUARD_ARGS=(--no-menu-guard)' in source
    assert '--no-run-notifications' in source
    assert 'HEARTBEAT_FILE=' in source
    assert '--heartbeat-path "$HEARTBEAT_FILE"' in source


def test_linear_finalizer_can_target_a_fresh_modal_volume():
    source = Path("scripts/ops/finalize_basic_linear_run.sh").read_text()
    assert 'VOLUME="${MODAL_CORPUS_VOLUME:-trueskate-mvp}"' in source
    assert '--volume "$VOLUME"' in source
    assert 'BASIC_LINEAR_TEMPORAL_MIXER' in source
    assert 'TRAIN_ARGS+=(--temporal-mixer)' in source


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
    assert 'gpu="any", timeout=3 * 3600, memory=16384' in source


def test_modal_linear_cpu_fallback_is_separate_and_labelled():
    source = Path("scripts/cloud/train_basic_linear_modal.py").read_text()
    assert "def train_remote_cpu(" in source
    assert 'execution_hardware"] = "cpu"' in source
    assert "cpu=8.0" in source
    assert "cpu=8.0, timeout=12 * 3600, memory=16384" in source
    assert "temporal_mixer=temporal_mixer" in source
    assert "trajectory_track=trajectory_track" in source


def test_line_fit_recovers_exact_endpoints_from_a_noise_free_track():
    # The decoder's whole premise is that a constant-velocity command is
    # over-determined by ~30 contact observations, so the noise-free solve must
    # be exact up to the conditioning ridge.
    fraction = torch.linspace(0., 1., 30)[None, :]
    start, end = torch.tensor([[.2, .7]]), torch.tensor([[.8, .3]])
    positions = start[:, None, :] + (end - start)[:, None, :] * fraction[:, :, None]
    fitted_start, fitted_end = BasicLinearRegressor._fit_constant_velocity(
        positions, fraction, torch.ones_like(fraction),
    )
    assert torch.allclose(fitted_start, start, atol=1e-3)
    assert torch.allclose(fitted_end, end, atol=1e-3)


def test_line_fit_irls_downweights_a_single_outlier_frame():
    # This is the tail mechanism: one occluded or mis-detected frame must not
    # be able to drag an endpoint outside the 0.03 recovery tolerance, which is
    # exactly what an unweighted least-squares fit allows.
    fraction = torch.linspace(0., 1., 30)[None, :]
    start, end = torch.tensor([[.2, .7]]), torch.tensor([[.8, .3]])
    positions = start[:, None, :] + (end - start)[:, None, :] * fraction[:, :, None]
    positions[0, 7] = torch.tensor([.05, .05])
    weights = torch.ones_like(fraction)

    plain_start, _plain_end = BasicLinearRegressor._fit_constant_velocity(positions, fraction, weights)
    assert torch.linalg.vector_norm(plain_start - start) > RECOVERY_ENDPOINT_TOLERANCE

    for _ in range(3):
        fitted_start, fitted_end = BasicLinearRegressor._fit_constant_velocity(positions, fraction, weights)
        path = fitted_start[:, None, :] + (fitted_end - fitted_start)[:, None, :] * fraction[:, :, None]
        residual = torch.linalg.vector_norm(positions - path, dim=2)
        weights = weights * (.02 / residual.clamp_min(1e-6)).clamp(max=1.)
    robust_start, robust_end = BasicLinearRegressor._fit_constant_velocity(positions, fraction, weights)
    assert torch.linalg.vector_norm(robust_start - start) < 1e-2
    assert torch.linalg.vector_norm(robust_end - end) < 1e-2


def test_line_fit_ignores_zero_weighted_frames():
    fraction = torch.linspace(0., 1., 12)[None, :]
    start, end = torch.tensor([[.25, .6]]), torch.tensor([[.75, .35]])
    positions = start[:, None, :] + (end - start)[:, None, :] * fraction[:, :, None]
    weights = torch.ones_like(fraction)
    weights[0, 3] = 0.
    reference = BasicLinearRegressor._fit_constant_velocity(positions, fraction, weights)
    # A zero-weighted frame must contribute nothing, so corrupting its position
    # cannot move the solution by any amount.  (Comparing against a fit that
    # keeps the frame would differ slightly and legitimately: the conditioning
    # ridge is fixed while the total weight mass is not.)
    positions[0, 3] = torch.tensor([.99, .01])
    masked = BasicLinearRegressor._fit_constant_velocity(positions, fraction, weights)
    assert torch.allclose(reference[0], masked[0], atol=1e-9)
    assert torch.allclose(reference[1], masked[1], atol=1e-9)


def test_line_fit_regressor_preserves_the_output_contract_and_drops_the_cold_gate():
    model = BasicLinearRegressor(base_channels=4, line_fit=True, temporal_mixer=True)
    # The fit needs the moving-contact map, so enabling it implies the track.
    assert model.line_fit_enabled and model.trajectory_score is not None
    assert model.onset_head is not None
    # The cold sigmoid(-4) blend is what kept the earlier trajectory control
    # from ever earning influence; the fit replaces it rather than blending past it.
    assert model.trajectory_fusion is None
    frames = torch.rand(2, 8, 3, 30, 18)
    output = model(frames)
    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()
    assert torch.all((output[:, 4] >= BASIC_LINEAR_MIN_S) & (output[:, 4] <= BASIC_LINEAR_MAX_S))
    gesture = model.predict_linear(frames)
    assert torch.all((gesture["x0"] >= 0.) & (gesture["x1"] <= 1.))
    output.sum().backward()
    assert model.trajectory_score.weight.grad is not None
    assert model.onset_head[0].weight.grad is not None


def test_line_fit_rejects_invalid_robustness_settings():
    with pytest.raises(ValueError, match="irls_iterations"):
        BasicLinearRegressor(line_fit=True, irls_iterations=-1)
    with pytest.raises(ValueError, match="huber_delta"):
        BasicLinearRegressor(line_fit=True, huber_delta=0.)


def test_zero_irls_iterations_is_plain_least_squares():
    # A declared control: the same decoder without reweighting, so the robust
    # gain can be attributed to IRLS rather than to the line fit alone.
    model = BasicLinearRegressor(base_channels=4, line_fit=True, irls_iterations=0)
    assert model.irls_iterations == 0
    assert model(torch.rand(1, 8, 3, 30, 18)).shape == (1, 5)


def test_linear_trainer_exposes_line_fit_and_resolution_controls():
    result = subprocess.run(
        [sys.executable, "scripts/train/train_basic_linear_regressor.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    for flag in ("--line-fit", "--irls-iterations", "--huber-delta", "--image-width", "--image-height"):
        assert flag in result.stdout


def test_line_fit_requires_supervised_trajectory_evidence():
    # The fit reads endpoints off the contact map, so an unsupervised map would
    # make the run silently meaningless rather than merely worse.
    result = subprocess.run(
        [sys.executable, "scripts/train/train_basic_linear_regressor.py",
         "--data", "/nonexistent", "--line-fit"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--trajectory-weight" in result.stderr


def test_recovery_records_carry_predicted_and_target_pairs():
    class _Fixed(torch.nn.Module):
        def forward(self, frames):
            return torch.tensor([[.30, .40, .60, .55, .60]]).expand(len(frames), 5).clone()

    target = torch.tensor([[.30, .40, .90, .55, .60]])
    loader = [{"frames": torch.rand(1, 2, 3, 6, 6), "target": target}]
    records = basic_linear_recovery_records(_Fixed(), loader, torch.device("cpu"))
    assert len(records) == 1
    assert records[0]["recovered"] == 0.0
    assert records[0]["predicted"][2] == pytest.approx(.60)
    assert records[0]["target"][2] == pytest.approx(.90)


def _modal_linear_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_modal_linear", Path("scripts/cloud/train_basic_linear_modal.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_evaluation_honours_the_trained_dataset_shape():
    # Evaluators used to build every dataset at the library default, so a
    # checkpoint trained at another width -- or with another knot count -- would
    # have been scored on inputs it never saw, silently for resolution and with a
    # post-corpus-load shape error for knots.
    module = _modal_linear_module()
    assert module._payload_dataset_kwargs([{"image_width": 256, "image_height": 576}]) == {
        "image_width": 256, "image_height": 576, "knots": 2,
    }
    assert module._payload_dataset_kwargs([{"knots": 3}]) == {
        "image_width": 128, "image_height": 288, "knots": 3,
    }
    assert module._payload_dataset_kwargs([{}]) == {
        "image_width": 128, "image_height": 288, "knots": 2,
    }
    with pytest.raises(ValueError, match="disagree on decode resolution"):
        module._payload_dataset_kwargs([{"image_width": 128, "image_height": 288},
                                        {"image_width": 256, "image_height": 576}])
    with pytest.raises(ValueError, match="disagree on knot count"):
        module._payload_dataset_kwargs([{"knots": 2}, {"knots": 3}])
    # A generator argument must not be consumed by the first check.
    assert module._payload_dataset_kwargs(iter([{"knots": 3}, {"knots": 3}]))["knots"] == 3
    # A literal call-site COUNT is the wrong guard and was inverted: an evaluator
    # that skips the helper leaves the count unchanged and passes.  Check the
    # property instead -- every dataset construction inside a Modal function must
    # take its shape from the helper.
    source = Path("scripts/cloud/train_basic_linear_modal.py").read_text()
    blocks = source.split("@app.function")[1:]
    constructions = 0
    for block in blocks:
        name = block.split("def ", 1)[1].split("(", 1)[0]
        cursor = 0
        while (found := block.find("BasicLinearClipDataset(", cursor)) != -1:
            constructions += 1
            call = block[found:found + 400]
            if name == "audit_orange_endpoint_cue":
                # The one deliberate exception: it scores no checkpoint, so there
                # is no payload to take a shape from.
                assert "_payload_dataset_kwargs" not in call
            else:
                assert "_payload_dataset_kwargs" in call, f"{name} builds a dataset without the helper"
            cursor = found + 1
    # 18 = 17 checkpoint-backed evaluators plus the one orange-cue exception above.
    # Bumping this deliberately is the point: a new evaluator cannot land without
    # being seen here.  Last bumped for `evaluate_test_once` (EQ-004, 2026-08-21).
    assert constructions == 18, f"expected 18 dataset constructions, found {constructions}"

    # Resolving the shape is not the same as decoding it.  Evaluators whose
    # bodies hardcode the 5-wide start/end/duration layout must refuse a k>2
    # checkpoint rather than emit a plausible, mislabelled artefact.
    # evaluate_refinement still cannot: refine_linear_endpoints hard-requires a
    # [batch,5] prediction, so it refuses rather than misreporting.
    assert '_require_two_knots(_payload_dataset_kwargs([payload]), "evaluate_refinement")' in source
    assert source.count("_require_two_knots(_payload_dataset_kwargs") == 1
    # The other two were made knot-general (EQ-012) and must not reintroduce a
    # hardcoded start/end/duration read.
    for evaluator in ("audit_endpoint_residuals", "autopsy_failures"):
        body = source[source.index(f"def {evaluator}("):]
        body = body[:body.index("\n@app.") if "\n@app." in body else len(body)]
        assert "_require_two_knots" not in body
        for banned in ("[item, :2]", "[item, 2:4]", "[item, 4]", "[:, :4].reshape(-1, 2, 2)",
                       '"x0", "y0", "x1", "y1", "duration"'):
            assert banned not in body, f"{evaluator} still hardcodes the 5-wide layout: {banned}"
        # A banned-substring list only encodes today's spellings.  The positive
        # invariant is the stable one: the knot layout must come from the shared
        # helpers, whatever the surrounding code looks like.
        assert any(helper in body for helper in
                   ("knot_columns", "knot_component_labels", "knot_errors")), \
            f"{evaluator} does not resolve its knot layout through the shared helpers"

    # A line-fit checkpoint decodes its knots from the trajectory map, so the
    # endpoint score maps describe no coordinate -- and knots>2 REQUIRES
    # line_fit, making this reachable for every k=3 autopsy.
    autopsy = source[source.index("def autopsy_failures("):]
    assert 'line_fit = bool(payload.get("line_fit"))' in autopsy
    assert "trajectory_score_peak_frame" in autopsy
    assert "forward_with_track_scores" in autopsy
    # recovered gates every knot, so every knot needs its own trail evidence --
    # otherwise a clip can fail on a knot the report says nothing about.
    assert 'f"trail_gap_knot{knot}"' in autopsy
    assert 'f"trail_frame_knot{knot}"' in autopsy
    # The k=2 keys stay, so existing artefacts and render_linear_failures keep working.
    for retained in ("trail_gap_start", "trail_gap_end", "trail_frame_start", "trail_frame_end"):
        assert f'"{retained}"' in autopsy
    assert "commanded_start, commanded_end = per_knot_trail[0], per_knot_trail[-1]" in autopsy
    # The trail arithmetic lives in a unit-tested helper, not inline in a Modal
    # body where only its source text can be asserted.
    assert "nearest_trail_gaps(grid, strong[item], knot_points)" in autopsy
    assert "def nearest(" not in autopsy
    # The loop bound comes from the target width, so trail_gap_end cannot quietly
    # become an interior knot if prediction and target widths ever disagree.
    assert "target_knots(target.shape[1])" in autopsy
    # Summary-level evidence covers every knot, not just the endpoints.
    assert "failed_knot_trail_gaps" in autopsy
    assert "median_trail_gap_by_knot" in autopsy
    with pytest.raises(ValueError, match="knot-general"):
        module._require_two_knots({"knots": 3}, "audit_endpoint_residuals")
    assert module._require_two_knots({"knots": 2}, "x") == {"knots": 2}

def test_modal_linear_entry_points_expose_the_line_fit_decoder():
    source = Path("scripts/cloud/train_basic_linear_modal.py").read_text()
    assert source.count("line_fit=line_fit") == 3
    assert 'line_fit=bool(payload.get("line_fit", False))' in source


def test_bias_correction_evaluator_takes_its_split_from_the_checkpoint():
    # A corpus that has gained samples since training reshuffles the split, so a
    # re-derived "test" set can quietly contain trained-on commands -- inflating
    # baseline and corrected numbers together, which the paired test cannot
    # reveal.  Split identity must come from the payload, not from arguments.
    source = Path("scripts/cloud/train_basic_linear_modal.py").read_text()
    body = source[source.index("def evaluate_bias_correction("):]
    body = body[:body.index("@app.local_entrypoint()")]
    assert 'payload.get("split_seed")' in body
    assert 'payload.get("dataset_fingerprint")' in body
    assert 'payload.get("split_sizes")' in body
    # Dataset shape -- resolution AND knot count -- comes from the one helper.
    assert "_payload_dataset_kwargs([payload])" in body
    # The fit must never be handed test records.
    assert "fit_along_path_bias(validation_records" in body
    assert "fit_along_path_bias(corrected_records" not in body


def test_training_payload_records_the_whole_validation_curve_not_just_its_argmax():
    """The reported figure is a best-of-N order statistic (EQ-049).

    Within-run validation sd across late epochs is ~6.3 points, so the argmax is
    biased upward by selection.  The payload must therefore carry the curve and a
    plateau mean beside the headline, and must say how many epochs the headline
    was maximised over -- otherwise a reader cannot tell selection bias from model
    quality, and re-analysis needs the runs repeated.
    """
    import ast

    source = Path("scripts/train/train_basic_linear_regressor.py").read_text()
    tree = ast.parse(source)
    train_fn = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "train")

    epoch_loops = [node for node in ast.walk(train_fn)
                   if isinstance(node, ast.For) and getattr(node.target, "id", None) == "epoch"]
    assert len(epoch_loops) == 1, "expected exactly one epoch loop"
    appended_in_loop = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and getattr(node.func.value, "id", None) == "validation_curve"
        for node in ast.walk(epoch_loops[0]))
    assert appended_in_loop, "validation_curve must be appended once per epoch, inside the loop"

    payload = next(node.value for node in ast.walk(train_fn)
                   if isinstance(node, ast.Assign)
                   and getattr(node.targets[0], "id", None) == "payload")
    keys = {key.value for key in payload.keys if isinstance(key, ast.Constant)}
    for required in ("validation_curve", "validation_plateau_mean_last10",
                     "validation_is_best_of_n_epochs"):
        assert required in keys, f"payload must record {required}"


@pytest.mark.parametrize("optional", [
    {"trajectory_track": True},
    {"line_fit": True},
    {"temporal_mixer": False},
    {"trajectory_track": True, "temporal_mixer": False},
    {"line_fit": True, "knots": 3},
])
def test_seed_matches_shared_weights_across_arms_that_differ_only_in_optional_modules(optional):
    """`--seed` must mean the same initialisation in every arm (EQ-048).

    Module construction draws from the global RNG, so an optional module built
    before another module shifts the stream for everything after it.  That made
    `duration_head` differ between arms at the same seed -- and, because the
    training DataLoader took its shuffle from the same global stream, changed the
    minibatch order for every epoch too.  A paired per-seed comparison across arms
    is meaningless while that is true.

    Ordering alone is insufficient: whichever optional is built first still shifts
    the stream for the later optionals.  So optionals draw one seed each,
    unconditionally, and build inside a forked RNG -- which pins three things at
    once: unconditional weights, optional weights where both arms have them, and
    the global stream position after construction.
    """
    shared = dict(base_channels=4, temporal_mixer=True)

    def build(**overrides):
        torch.manual_seed(11)
        model = BasicLinearRegressor(**{**shared, **overrides})
        return model, torch.randn(3)          # global RNG position after construction

    baseline, baseline_rng = build()
    variant, variant_rng = build(**optional)

    baseline_state = dict(baseline.named_parameters())
    variant_state = dict(variant.named_parameters())
    for name in ("encoder", "start_score", "end_score", "duration_head"):
        for key, value in baseline_state.items():
            if key.startswith(name + "."):
                assert torch.equal(value, variant_state[key]), (
                    f"{key} differs between arms at the same seed; an optional module "
                    "is consuming RNG that an unconditional one depends on")

    # Optional modules present in BOTH arms must also match: whether some other
    # optional is enabled must not perturb them.
    for key, value in baseline_state.items():
        if key in variant_state and not key.split(".")[0] in ("encoder", "start_score",
                                                              "end_score", "duration_head"):
            assert torch.equal(value, variant_state[key]), (
                f"optional weight {key} differs although both arms build it")

    # And the global stream must be left in the same place, so anything drawn
    # after construction -- the DataLoader shuffle above all -- is arm-independent.
    assert torch.equal(baseline_rng, variant_rng), (
        "optional modules must not advance the global RNG differently between arms")

    source = Path("scripts/train/train_basic_linear_regressor.py").read_text()
    assert "generator=shuffle_generator" in source, (
        "the training DataLoader must take an explicit seed-derived generator")
