from types import SimpleNamespace

import pytest
import torch

from trueskate_ai.vision import temporal_trace_training as trace_training
from trueskate_ai.vision.temporal_trace_predictor import TemporalTraceState
from trueskate_ai.vision.temporal_trace_training import (
    BalancedTemporalTraceLoss,
    TeacherForcingSchedule,
    autoregressive_trace_rollout,
    corrupt_teacher_heatmaps,
    evaluate_temporal_trace_model,
    evaluate_temporal_trace_threshold_grid,
    extract_touch_peaks,
    sample_teacher_forcing_mask,
    temporal_trace_metrics,
)


def test_balanced_loss_normalises_positive_background_and_activity_classes():
    criterion = BalancedTemporalTraceLoss(
        focal_gamma=0.0, background_target_power=1.0
    )
    prediction = torch.full((1, 3, 1, 2, 2), 0.1, requires_grad=True)
    with torch.no_grad():
        prediction[0, 0, 0, 0, 0] = 0.8
        prediction[0, 2] = 0.99  # masked padding must have no effect
    target = torch.zeros_like(prediction)
    target[0, 0, 0, 0, 0] = 1.0
    active_logits = torch.tensor([[2.0, -1.0, 20.0]], requires_grad=True)
    target_active = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, False]])

    losses = criterion(
        prediction,
        active_logits,
        target,
        target_active,
        valid_mask=valid,
    )

    torch.testing.assert_close(losses.heatmap_positive, -torch.log(torch.tensor(0.8)))
    # Seven labeled background pixels all predict 0.1.
    torch.testing.assert_close(losses.heatmap_background, -torch.log(torch.tensor(0.9)))
    torch.testing.assert_close(losses.activity_positive, torch.nn.functional.softplus(torch.tensor(-2.0)))
    torch.testing.assert_close(losses.activity_negative, torch.nn.functional.softplus(torch.tensor(-1.0)))
    torch.testing.assert_close(
        losses.heatmap,
        (losses.heatmap_positive + losses.heatmap_background) / 2,
    )
    torch.testing.assert_close(
        losses.activity,
        (losses.activity_positive + losses.activity_negative) / 2,
    )
    losses.total.backward()
    assert prediction.grad is not None
    assert active_logits.grad is not None
    assert prediction.grad[0, 2].abs().sum() == 0
    assert active_logits.grad[0, 2] == 0


def test_balanced_loss_applies_configured_positive_class_fractions():
    criterion = BalancedTemporalTraceLoss(
        heatmap_positive_fraction=0.8,
        activity_positive_fraction=0.25,
        focal_gamma=0.0,
        background_target_power=1.0,
    )
    prediction = torch.tensor([[[[[0.8, 0.1]]]]])
    target = torch.tensor([[[[[1.0, 0.0]]]]])
    active_logits = torch.tensor([[2.0, -1.0]])
    target_active = torch.tensor([[1.0, 0.0]])
    prediction = prediction.expand(1, 2, 1, 1, 2).clone()
    target = target.expand_as(prediction).clone()

    losses = criterion(prediction, active_logits, target, target_active)

    torch.testing.assert_close(
        losses.heatmap,
        0.8 * losses.heatmap_positive + 0.2 * losses.heatmap_background,
    )
    torch.testing.assert_close(
        losses.activity,
        0.25 * losses.activity_positive + 0.75 * losses.activity_negative,
    )


def test_balanced_loss_custom_fractions_do_not_discount_one_class_batches():
    criterion = BalancedTemporalTraceLoss(
        heatmap_positive_fraction=0.8,
        activity_positive_fraction=0.25,
        focal_gamma=0.0,
    )
    prediction = torch.full((1, 2, 1, 2, 2), 0.2)
    target = torch.zeros_like(prediction)
    logits = torch.full((1, 2), -2.0)
    inactive = torch.zeros(1, 2)

    negative_only = criterion(prediction, logits, target, inactive)

    torch.testing.assert_close(
        negative_only.heatmap, negative_only.heatmap_background
    )
    torch.testing.assert_close(
        negative_only.activity, negative_only.activity_negative
    )

    positive_only = criterion(
        prediction,
        logits,
        torch.ones_like(target),
        torch.ones(1, 2),
    )
    torch.testing.assert_close(positive_only.heatmap, positive_only.heatmap_positive)
    torch.testing.assert_close(positive_only.activity, positive_only.activity_positive)


def test_balanced_loss_handles_one_class_and_fully_masked_batch():
    criterion = BalancedTemporalTraceLoss(focal_gamma=0.0)
    prediction = torch.full((1, 2, 1, 3, 3), 0.2, requires_grad=True)
    target = torch.zeros_like(prediction)
    logits = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    inactive = torch.zeros(1, 2)

    one_class = criterion(prediction, logits, target, inactive)
    assert one_class.heatmap_positive == 0
    torch.testing.assert_close(one_class.heatmap, one_class.heatmap_background)
    torch.testing.assert_close(one_class.activity, one_class.activity_negative)
    assert torch.isfinite(one_class.total)

    none = criterion(
        prediction,
        logits,
        target,
        inactive,
        label_mask=torch.zeros(1, 2, dtype=torch.bool),
    )
    assert none.total == 0
    none.total.backward()
    assert prediction.grad is not None


def test_balanced_loss_stays_finite_for_saturated_float16_probabilities():
    criterion = BalancedTemporalTraceLoss(
        hard_negative_weight=0.5,
        hard_negative_top_k=2,
    )
    prediction = torch.tensor(
        [[[[[1.0, 0.0], [1.0, 0.0]]]]],
        dtype=torch.float16,
        requires_grad=True,
    )
    active_logits = torch.tensor([[20.0]], dtype=torch.float16, requires_grad=True)
    target = torch.tensor(
        [[[[[1.0, 0.0], [0.0, 0.0]]]]], dtype=torch.float16
    )

    losses = criterion(prediction, active_logits, target, torch.ones(1, 1))

    assert losses.total.dtype == torch.float32
    assert torch.isfinite(losses.total)
    assert losses.heatmap_hard_negative.dtype == torch.float32
    assert torch.isfinite(losses.heatmap_hard_negative)
    losses.total.backward()
    assert prediction.grad is not None
    assert active_logits.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.isfinite(active_logits.grad).all()


def test_hard_negative_loss_selects_top_pixel_per_labeled_frame():
    criterion = BalancedTemporalTraceLoss(
        heatmap_weight=0.0,
        activity_weight=0.0,
        focal_gamma=0.0,
        hard_negative_weight=2.0,
        hard_negative_top_k=1,
        hard_negative_target_exclusion_threshold=0.05,
    )
    prediction = torch.tensor(
        [
            [
                [[[0.1, 0.2, 0.8, 0.7]]],
                [[[0.1, 0.9, 0.3, 0.2]]],
                [[[0.999, 0.1, 0.1, 0.1]]],
                [[[0.999, 0.1, 0.1, 0.1]]],
            ]
        ],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    # Even though its prediction is high, this target-tail pixel is not a
    # negative candidate because it is above the exclusion threshold.
    target[0, 0, 0, 0, 2] = 1.0
    target[0, 0, 0, 0, 3] = 0.2
    valid_mask = torch.tensor([[True, True, False, True]])
    label_mask = torch.tensor([[True, True, True, False]])

    losses = criterion(
        prediction,
        torch.zeros(1, 4),
        target,
        torch.zeros(1, 4),
        valid_mask=valid_mask,
        label_mask=label_mask,
    )

    # Frame zero contributes p=.2 and frame one p=.9. Invalid and unlabeled
    # frames do not contribute despite their p=.999 false positives.
    expected = (-torch.log(torch.tensor(0.8)) - torch.log(torch.tensor(0.1))) / 2
    torch.testing.assert_close(losses.heatmap_hard_negative, expected)
    torch.testing.assert_close(losses.total, 2.0 * expected)
    losses.total.backward()
    assert prediction.grad[0, 0, 0, 0, 1] > 0
    assert prediction.grad[0, 1, 0, 0, 1] > 0
    assert prediction.grad[0, 0, 0, 0, 3] == 0
    assert prediction.grad[0, 2].abs().sum() == 0
    assert prediction.grad[0, 3].abs().sum() == 0


def test_zero_weight_preserves_original_loss_and_reports_zero_hard_negative():
    criterion = BalancedTemporalTraceLoss(
        focal_gamma=0.0,
        hard_negative_weight=0.0,
        hard_negative_top_k=1,
    )
    prediction = torch.full((1, 1, 1, 2, 2), 0.25, requires_grad=True)
    target = torch.zeros_like(prediction)
    losses = criterion(
        prediction,
        torch.zeros(1, 1, requires_grad=True),
        target,
        torch.zeros(1, 1),
    )

    assert losses.heatmap_hard_negative == 0
    torch.testing.assert_close(
        losses.total,
        criterion.heatmap_weight * losses.heatmap
        + criterion.activity_weight * losses.activity,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hard_negative_weight": -0.1}, "hard_negative_weight"),
        ({"hard_negative_weight": float("nan")}, "hard_negative_weight"),
        ({"hard_negative_top_k": 0}, "hard_negative_top_k"),
        ({"hard_negative_top_k": 1.5}, "hard_negative_top_k"),
        ({"hard_negative_top_k": True}, "hard_negative_top_k"),
        (
            {"hard_negative_target_exclusion_threshold": -0.1},
            "hard_negative_target_exclusion_threshold",
        ),
        (
            {"hard_negative_target_exclusion_threshold": 1.1},
            "hard_negative_target_exclusion_threshold",
        ),
    ],
)
def test_hard_negative_loss_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BalancedTemporalTraceLoss(**kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("heatmap_positive_fraction", -0.1),
        ("heatmap_positive_fraction", 1.1),
        ("heatmap_positive_fraction", float("nan")),
        ("activity_positive_fraction", float("inf")),
    ],
)
def test_balanced_loss_validates_positive_class_fractions(name, value):
    with pytest.raises(ValueError, match=name):
        BalancedTemporalTraceLoss(**{name: value})


def test_teacher_forcing_schedule_has_exact_endpoints_and_validation():
    linear = TeacherForcingSchedule(
        start_probability=0.8,
        end_probability=0.2,
        warmup_epochs=2,
        decay_epochs=3,
    )
    assert linear(0) == pytest.approx(0.8)
    assert linear(1) == pytest.approx(0.8)
    assert linear(2) == pytest.approx(0.8)
    assert linear(3) == pytest.approx(0.5)
    assert linear(4) == pytest.approx(0.2)
    assert linear(99) == pytest.approx(0.2)

    cosine = TeacherForcingSchedule(decay_epochs=3, curve="cosine")
    assert cosine(0) == pytest.approx(1.0)
    assert cosine(1) == pytest.approx(0.5)
    assert cosine(2) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="epoch"):
        cosine(-1)
    with pytest.raises(ValueError, match="curve"):
        TeacherForcingSchedule(curve="quadratic")


def test_teacher_forcing_mask_never_crosses_unknown_padding_or_reset():
    valid = torch.tensor([[True, True, True, True, False]])
    labeled = torch.tensor([[True, False, True, True, False]])
    resets = torch.tensor([[True, False, False, True, False]])

    mask = sample_teacher_forcing_mask(
        1.0, valid, label_mask=labeled, reset_mask=resets
    )

    # t1 may consume labeled t0. t2 follows an unlabeled target, t3 resets,
    # and t4 is padding.
    assert mask.tolist() == [[False, True, False, False, False]]
    assert not sample_teacher_forcing_mask(0.0, valid).any()


def test_teacher_heatmap_corruption_is_non_mutating_masked_and_mapwise():
    teachers = torch.ones(1, 3, 1, 4, 4)
    original = teachers.clone()
    label_mask = torch.tensor([[True, False, True]])

    dropped = corrupt_teacher_heatmaps(
        teachers,
        dropout_probability=1.0,
        label_mask=label_mask,
    )
    assert not dropped.any()
    torch.testing.assert_close(teachers, original)

    generator = torch.Generator().manual_seed(17)
    noisy = corrupt_teacher_heatmaps(
        torch.full_like(teachers, 0.5),
        noise_std=0.2,
        label_mask=label_mask,
        generator=generator,
    )
    assert noisy[0, 0].std() > 0
    assert not noisy[0, 1].any()
    assert torch.all((noisy >= 0) & (noisy <= 1))


def _peaked_heatmap(points, *, height=20, width=20):
    heatmap = torch.zeros(height, width)
    for x, y, score in points:
        heatmap[round(y * (height - 1)), round(x * (width - 1))] = score
    return heatmap


def test_local_max_nms_keeps_distinct_peaks_and_collapses_neighbors():
    heatmap = _peaked_heatmap(
        [(0.25, 0.25, 0.9), (0.30, 0.25, 0.8), (0.75, 0.75, 0.7)]
    )
    peaks = extract_touch_peaks(
        heatmap, threshold=0.5, nms_radius_px=2, max_peaks=4
    )
    assert [(round(p.x, 2), round(p.y, 2)) for p in peaks] == [
        (0.26, 0.26),
        (0.74, 0.74),
    ]


def test_metrics_use_sparse_exact_centers_and_score_multi_peak_precision_recall():
    predicted = torch.zeros(1, 3, 1, 20, 20)
    predicted[0, 0, 0] = _peaked_heatmap([(0.25, 0.25, 0.95)])
    predicted[0, 1, 0] = _peaked_heatmap(
        [(0.25, 0.75, 0.95), (0.75, 0.75, 0.9), (0.50, 0.20, 0.8)]
    )
    logits = torch.tensor([[8.0, 8.0, -8.0]])
    active = torch.tensor([[True, True, False]])
    centers = torch.full((1, 3, 3, 2), -1.0)
    centers[0, 0, 0] = torch.tensor([0.25, 0.25])
    # Sparse stable track columns: slot zero has lifted while slots one/two live.
    centers[0, 1, 1] = torch.tensor([0.25, 0.75])
    centers[0, 1, 2] = torch.tensor([0.75, 0.75])
    counts = torch.tensor([[1, 2, 0]])

    metrics = temporal_trace_metrics(
        predicted,
        logits,
        active,
        centers,
        counts,
        peak_threshold=0.5,
        localization_tolerance=0.03,
        nms_radius_px=1,
    )

    assert metrics["positive_accuracy"] == pytest.approx(1.0)
    assert metrics["negative_accuracy"] == pytest.approx(1.0)
    # The extra active-frame peak must now prevent strict acceptance from
    # reporting a perfect localizer despite perfect recall/specificity.
    assert metrics["acceptance_score"] == pytest.approx(3 / 4)
    assert metrics["peak_precision"] == pytest.approx(3 / 4)
    assert metrics["peak_recall"] == pytest.approx(1.0)
    assert metrics["multi_peak_precision"] == pytest.approx(2 / 3)
    assert metrics["multi_peak_recall"] == pytest.approx(1.0)
    assert metrics["target_touches"] == 3
    assert metrics["multi_touch_frames"] == 1


def test_negative_specificity_rejects_contradictory_heatmap_peak():
    predicted = torch.zeros(1, 1, 1, 12, 12)
    predicted[0, 0, 0, 4, 5] = 0.9
    metrics = temporal_trace_metrics(
        predicted,
        torch.tensor([[-8.0]]),
        torch.tensor([[False]]),
        torch.full((1, 1, 2, 2), -1.0),
        torch.zeros(1, 1, dtype=torch.long),
    )
    assert metrics["negative_accuracy"] == 0.0
    # No positive validation class may never accidentally pass acceptance.
    assert metrics["positive_accuracy"] == 0.0
    assert metrics["acceptance_score"] == 0.0


class _FeedbackSpyModel(torch.nn.Module):
    """Tiny step-only model whose output exposes previous-feedback leakage."""

    def __init__(self):
        super().__init__()
        self.previous_mass = []

    def step(self, frame, state=None, *, delta_t=None):
        batch, _, height, width = frame.shape
        previous = (
            frame.new_zeros((batch, 1, height, width))
            if state is None
            else state.previous_heatmap
        )
        self.previous_mass.append(previous.flatten(1).sum(dim=1).detach().clone())
        heatmap = torch.maximum(frame[:, :1], previous)
        hidden = frame.new_zeros((batch, 1, height, width))
        logits = torch.where(
            heatmap.flatten(1).amax(dim=1) >= 0.5,
            frame.new_full((batch,), 8.0),
            frame.new_full((batch,), -8.0),
        )
        return SimpleNamespace(
            heatmap=heatmap,
            active_logits=logits,
            state=TemporalTraceState(hidden=hidden, previous_heatmap=heatmap),
        )


class _ThresholdGridSpyModel(torch.nn.Module):
    """Expose scripted heatmaps/activity probabilities and count causal steps."""

    def __init__(self):
        super().__init__()
        self.step_calls = 0

    def step(self, frame, state=None, *, delta_t=None):
        self.step_calls += 1
        batch, _, height, width = frame.shape
        heatmap = frame[:, :1].clone()
        probability = frame[:, 1, 0, 0].clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(probability)
        hidden = frame.new_zeros((batch, 1, height, width))
        return SimpleNamespace(
            heatmap=heatmap,
            active_logits=logits,
            state=TemporalTraceState(hidden=hidden, previous_heatmap=heatmap),
        )


def _threshold_grid_batch():
    frames = torch.zeros(1, 4, 3, 12, 12)
    frames[0, 0, 0, 3, 4] = 0.9  # labeled touch
    frames[0, 1, 0, 4, 5] = 0.6  # raw-only negative contradiction
    frames[0, 2, 0, 5, 6] = 0.4  # active-only above activity threshold
    frames[0, :, 1, 0, 0] = torch.tensor([0.8, 0.2, 0.8, 0.2])
    centers = torch.full((1, 4, 1, 2), -1.0)
    centers[0, 0, 0] = torch.tensor([4 / 11, 3 / 11])
    return {
        "frames": frames,
        "active": torch.tensor([[True, False, False, False]]),
        "centers": centers,
        "touch_count": torch.tensor([[1, 0, 0, 0]]),
        "valid_mask": torch.ones(1, 4, dtype=torch.bool),
        "label_mask": torch.ones(1, 4, dtype=torch.bool),
        "reset_mask": torch.tensor([[True, False, False, False]]),
    }


def test_threshold_grid_uses_one_rollout_and_decode_per_frame(monkeypatch):
    model = _ThresholdGridSpyModel()
    batch = _threshold_grid_batch()
    decode_thresholds = []
    original_extract = trace_training.extract_touch_peaks

    def counted_extract(*args, **kwargs):
        decode_thresholds.append(kwargs["threshold"])
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(trace_training, "extract_touch_peaks", counted_extract)
    grid = evaluate_temporal_trace_threshold_grid(
        model,
        [batch],
        "cpu",
        peak_thresholds=[0.7, 0.3, 0.5, 0.3],
        activity_thresholds=[0.9, 0.5, 0.9],
        localization_tolerance=0.02,
        nms_radius_px=1,
        max_peaks=2,
    )

    assert list(grid) == [
        (0.3, 0.5),
        (0.3, 0.9),
        (0.5, 0.5),
        (0.5, 0.9),
        (0.7, 0.5),
        (0.7, 0.9),
    ]
    assert model.step_calls == 4
    assert decode_thresholds == [0.3] * 4

    low = grid[(0.3, 0.5)]
    assert low["positive_accuracy"] == pytest.approx(1.0)
    assert low["peak_precision"] == pytest.approx(1 / 2)
    # Existing strict specificity remains the intersection of independently
    # clean activity and raw-heatmap predictions.
    assert low["negative_accuracy"] == pytest.approx(1 / 3)
    assert low["raw_heatmap_negative_specificity"] == pytest.approx(1 / 3)
    assert low["activity_negative_specificity"] == pytest.approx(2 / 3)
    assert low["emitted_negative_specificity"] == pytest.approx(2 / 3)

    middle = grid[(0.5, 0.5)]
    assert middle["positive_accuracy"] == pytest.approx(1.0)
    assert middle["peak_precision"] == pytest.approx(1.0)
    assert middle["negative_accuracy"] == pytest.approx(1 / 3)
    assert middle["raw_heatmap_negative_specificity"] == pytest.approx(2 / 3)
    assert middle["activity_negative_specificity"] == pytest.approx(2 / 3)
    assert middle["emitted_negative_specificity"] == pytest.approx(1.0)

    # Every pre-existing metric agrees exactly with the existing evaluator at
    # the same fixed thresholds; the grid only adds decomposed diagnostics.
    single = evaluate_temporal_trace_model(
        _ThresholdGridSpyModel(),
        [batch],
        "cpu",
        peak_threshold=0.5,
        activity_threshold=0.5,
        localization_tolerance=0.02,
        nms_radius_px=1,
        max_peaks=2,
    )
    for name, expected in single.items():
        assert middle[name] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("peak_thresholds", "activity_thresholds", "message"),
    [
        ([], [0.5], "peak_thresholds"),
        ([0.3], [], "activity_thresholds"),
        ([-0.1], [0.5], "peak_thresholds"),
        ([0.3], [float("nan")], "activity_thresholds"),
    ],
)
def test_threshold_grid_validates_nonempty_probability_inputs(
    peak_thresholds, activity_thresholds, message
):
    with pytest.raises(ValueError, match=message):
        evaluate_temporal_trace_threshold_grid(
            _ThresholdGridSpyModel(),
            [],
            "cpu",
            peak_thresholds=peak_thresholds,
            activity_thresholds=activity_thresholds,
        )


def test_autoregressive_rollout_uses_predictions_and_clears_state_on_reset():
    model = _FeedbackSpyModel()
    frames = torch.zeros(1, 3, 3, 8, 8)
    frames[0, 0, 0, 2, 2] = 1.0
    reset_mask = torch.tensor([[True, False, True]])

    output = autoregressive_trace_rollout(model, frames, reset_mask=reset_mask)

    assert output.heatmaps[0, 1, 0, 2, 2] == 1.0
    assert model.previous_mass[0].item() == 0.0
    assert model.previous_mass[1].item() == 1.0
    assert model.previous_mass[2].item() == 0.0


def test_evaluator_uses_canonical_batch_and_never_reads_teacher_heatmaps():
    model = _FeedbackSpyModel()
    frames = torch.zeros(1, 2, 3, 10, 10)
    frames[0, 0, 0, 2, 3] = 1.0
    centers = torch.full((1, 2, 2, 2), -1.0)
    centers[0, 0, 0] = torch.tensor([3 / 9, 2 / 9])
    centers[0, 1, 0] = torch.tensor([3 / 9, 2 / 9])
    batch = {
        "frames": frames,
        # Deliberately wrong teachers prove evaluation cannot consume them.
        "heatmaps": torch.zeros(1, 2, 1, 10, 10),
        "active": torch.ones(1, 2, dtype=torch.bool),
        "centers": centers,
        "touch_count": torch.ones(1, 2, dtype=torch.long),
        "delta_times": torch.tensor([[0.0, 0.03]]),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "label_mask": torch.ones(1, 2, dtype=torch.bool),
        "reset_mask": torch.tensor([[True, False]]),
    }

    metrics = evaluate_temporal_trace_model(
        model,
        [batch],
        "cpu",
        peak_threshold=0.5,
        localization_tolerance=0.02,
        nms_radius_px=1,
    )

    assert metrics["positive_accuracy"] == pytest.approx(1.0)
    assert metrics["positive_frames"] == 2
    assert model.previous_mass[1].item() == 1.0
