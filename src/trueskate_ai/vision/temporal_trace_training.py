"""Training and validation utilities for causal temporal Model 1.

The touch heatmap is extremely sparse, so averaging a pixel loss over the
whole image makes an all-background prediction look deceptively good.  The
loss below normalises target mass and background mass independently.  The
activity head receives the same treatment at frame level: active and inactive
examples contribute equally whenever both classes are present.

Validation is deliberately autoregressive.  It calls :meth:`step` without
teacher heatmaps and feeds the model's own previous prediction back through
its explicit state.  Metrics use exact normalised target centres supplied by
the sequence dataset, rather than reducing a multi-touch target heatmap to one
global argmax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_trace_predictor import TemporalTraceState


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _normalise_probability_grid(
    name: str, values: Iterable[float]
) -> tuple[float, ...]:
    """Validate, sort, and exactly deduplicate one threshold dimension."""

    try:
        validated = [
            _validate_probability(f"{name}[{index}]", value)
            for index, value in enumerate(values)
        ]
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty iterable") from exc
    if not validated:
        raise ValueError(f"{name} must contain at least one threshold")
    # Canonicalise negative zero as well as ordinary duplicates so keys and
    # iteration order do not depend on the caller's input representation.
    canonical = (0.0 if value == 0.0 else value for value in validated)
    return tuple(sorted(set(canonical)))


def _frame_mask(
    reference: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    label_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return a boolean ``[B,T]`` mask on ``reference``'s device."""

    expected = reference.shape[:2]
    mask = torch.ones(expected, dtype=torch.bool, device=reference.device)
    for name, supplied in (("valid_mask", valid_mask), ("label_mask", label_mask)):
        if supplied is None:
            continue
        if tuple(supplied.shape) != tuple(expected):
            raise ValueError(
                f"{name} must have shape {tuple(expected)}, got {tuple(supplied.shape)}"
            )
        mask &= supplied.to(device=reference.device, dtype=torch.bool)
    return mask


def _normalised_component(loss_sum: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Safely normalise one loss component, returning differentiable zero."""

    return loss_sum / mass.clamp_min(torch.finfo(loss_sum.dtype).eps)


def _mean_present(
    first: torch.Tensor,
    first_mass: torch.Tensor,
    second: torch.Tensor,
    second_mass: torch.Tensor,
    *,
    first_fraction: float = 0.5,
) -> torch.Tensor:
    """Weight represented components without discounting one-class batches."""

    present = torch.stack((first_mass > 0, second_mass > 0)).to(first.dtype)
    values = torch.stack((first, second))
    if first_fraction == 0.5:
        # Preserve the original scalar and gradient path for the defaults.
        return (values * present).sum() / present.sum().clamp_min(1.0)

    both_present = present.prod()
    configured_weights = values.new_tensor(
        (first_fraction, 1.0 - first_fraction)
    )
    weights = present * (1.0 - both_present) + both_present * configured_weights
    return (values * weights).sum()


@dataclass
class TemporalTraceLossOutput:
    """Differentiable temporal Model-1 loss and its balanced components."""

    total: torch.Tensor
    heatmap: torch.Tensor
    heatmap_positive: torch.Tensor
    heatmap_background: torch.Tensor
    heatmap_hard_negative: torch.Tensor
    activity: torch.Tensor
    activity_positive: torch.Tensor
    activity_negative: torch.Tensor
    positive_target_mass: torch.Tensor
    background_target_mass: torch.Tensor
    positive_activity_mass: torch.Tensor
    negative_activity_mass: torch.Tensor
    labeled_frames: torch.Tensor

    def detached(self) -> dict[str, float]:
        """Return scalar logging values without retaining the autograd graph."""

        return {
            name: float(value.detach().cpu())
            for name, value in vars(self).items()
        }


class BalancedTemporalTraceLoss(nn.Module):
    """Separately normalised heatmap-mass and frame-activity loss.

    ``target_heatmaps`` may contain max-combined Gaussian bumps.  Each target
    value contributes soft positive mass ``target`` and background mass
    ``(1-target)**background_target_power``.  These two losses are normalised
    and balanced independently, preventing the millions of background pixels
    from drowning the touch centres.  ``focal_gamma`` may be set to zero for
    ordinary balanced BCE.

    Activity labels are likewise split into positive and negative BCE terms.
    ``heatmap_positive_fraction`` and ``activity_positive_fraction`` control
    their respective positive-class share when both classes are represented;
    both default to an equal 0.5 balance.  If a masked batch contains only one
    side of either task, that side receives its full task weight rather than
    being scaled by the configured fraction.  If no labeled frames exist,
    every component is a differentiable zero.

    An optional hard-negative term selects the ``hard_negative_top_k`` largest
    negative focal losses independently in every labeled frame.  Pixels whose
    target is above ``hard_negative_target_exclusion_threshold`` are excluded
    so Gaussian touch targets cannot be mined as negatives.  Its default
    weight is zero, preserving the original objective and computation path.
    """

    def __init__(
        self,
        *,
        heatmap_weight: float = 1.0,
        activity_weight: float = 1.0,
        heatmap_positive_fraction: float = 0.5,
        activity_positive_fraction: float = 0.5,
        focal_gamma: float = 2.0,
        background_target_power: float = 4.0,
        epsilon: float = 1e-6,
        hard_negative_weight: float = 0.0,
        hard_negative_top_k: int = 64,
        hard_negative_target_exclusion_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        if heatmap_weight < 0 or activity_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if background_target_power <= 0:
            raise ValueError("background_target_power must be positive")
        if not 0 < epsilon < 0.5:
            raise ValueError("epsilon must be between 0 and 0.5")
        heatmap_positive_fraction = _validate_probability(
            "heatmap_positive_fraction", heatmap_positive_fraction
        )
        activity_positive_fraction = _validate_probability(
            "activity_positive_fraction", activity_positive_fraction
        )
        if not math.isfinite(hard_negative_weight) or hard_negative_weight < 0:
            raise ValueError("hard_negative_weight must be finite and non-negative")
        if (
            isinstance(hard_negative_top_k, bool)
            or not isinstance(hard_negative_top_k, int)
            or hard_negative_top_k <= 0
        ):
            raise ValueError("hard_negative_top_k must be a positive integer")
        hard_negative_target_exclusion_threshold = _validate_probability(
            "hard_negative_target_exclusion_threshold",
            hard_negative_target_exclusion_threshold,
        )
        self.heatmap_weight = float(heatmap_weight)
        self.activity_weight = float(activity_weight)
        self.heatmap_positive_fraction = heatmap_positive_fraction
        self.activity_positive_fraction = activity_positive_fraction
        self.focal_gamma = float(focal_gamma)
        self.background_target_power = float(background_target_power)
        self.epsilon = float(epsilon)
        self.hard_negative_weight = float(hard_negative_weight)
        self.hard_negative_top_k = hard_negative_top_k
        self.hard_negative_target_exclusion_threshold = (
            hard_negative_target_exclusion_threshold
        )

    def forward(
        self,
        predicted_heatmaps: torch.Tensor,
        active_logits: torch.Tensor,
        target_heatmaps: torch.Tensor,
        target_active: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> TemporalTraceLossOutput:
        if predicted_heatmaps.ndim != 5 or predicted_heatmaps.shape[2] != 1:
            raise ValueError(
                "predicted_heatmaps must have shape [B,T,1,H,W], got "
                f"{tuple(predicted_heatmaps.shape)}"
            )
        if tuple(target_heatmaps.shape) != tuple(predicted_heatmaps.shape):
            raise ValueError(
                "target_heatmaps must match predicted_heatmaps, got "
                f"{tuple(target_heatmaps.shape)} and {tuple(predicted_heatmaps.shape)}"
            )
        batch_steps = predicted_heatmaps.shape[:2]
        if tuple(active_logits.shape) != tuple(batch_steps):
            raise ValueError(
                f"active_logits must have shape {tuple(batch_steps)}, got "
                f"{tuple(active_logits.shape)}"
            )
        if tuple(target_active.shape) != tuple(batch_steps):
            raise ValueError(
                f"target_active must have shape {tuple(batch_steps)}, got "
                f"{tuple(target_active.shape)}"
            )

        # Keep probability/log arithmetic in float32 even when the predictor is
        # running under CUDA autocast.  In float16, ``1 - 1e-6`` rounds back to
        # exactly one, so a saturated sigmoid can survive the clamp below and
        # make ``log1p(-prediction)`` infinite.  Autograd still propagates
        # through this cast while the convolutional model retains AMP's speed.
        loss_dtype = (
            torch.float32
            if predicted_heatmaps.dtype in (torch.float16, torch.bfloat16)
            else predicted_heatmaps.dtype
        )
        target = target_heatmaps.to(
            device=predicted_heatmaps.device, dtype=loss_dtype
        ).clamp(0.0, 1.0)
        prediction = predicted_heatmaps.to(dtype=loss_dtype).clamp(
            self.epsilon, 1.0 - self.epsilon
        )
        loss_active_logits = active_logits.to(dtype=loss_dtype)
        activity_target = target_active.to(
            device=active_logits.device, dtype=loss_dtype
        ).clamp(0.0, 1.0)
        frame_mask = _frame_mask(predicted_heatmaps, valid_mask, label_mask)
        pixel_mask = frame_mask[:, :, None, None, None].to(loss_dtype)

        positive_mass = target * pixel_mask
        background_mass = (1.0 - target).pow(self.background_target_power) * pixel_mask
        positive_element = -positive_mass * (1.0 - prediction).pow(
            self.focal_gamma
        ) * prediction.log()
        background_element = -background_mass * prediction.pow(
            self.focal_gamma
        ) * torch.log1p(-prediction)

        positive_target_mass = positive_mass.sum()
        background_target_mass = background_mass.sum()
        heatmap_positive = _normalised_component(
            positive_element.sum(), positive_target_mass
        )
        heatmap_background = _normalised_component(
            background_element.sum(), background_target_mass
        )
        heatmap_loss = _mean_present(
            heatmap_positive,
            positive_target_mass,
            heatmap_background,
            background_target_mass,
            first_fraction=self.heatmap_positive_fraction,
        )

        # Mine false-positive pixels independently in each labeled frame so a
        # frame with many confident mistakes cannot drown all other frames.
        # Keep this branch out of the default objective entirely: besides
        # avoiding top-k overhead, this makes weight=0 reproduce the prior
        # scalar and gradient path exactly.
        if self.hard_negative_weight > 0.0:
            hard_negative_element = -prediction.pow(
                self.focal_gamma
            ) * torch.log1p(-prediction)
            hard_negative_candidates = (
                target <= self.hard_negative_target_exclusion_threshold
            ) & frame_mask[:, :, None, None, None]
            flattened_loss = hard_negative_element.flatten(start_dim=2)
            flattened_candidates = hard_negative_candidates.flatten(start_dim=2)
            selected_count = min(self.hard_negative_top_k, flattened_loss.shape[-1])
            selected_loss = flattened_loss.masked_fill(
                ~flattened_candidates, -torch.inf
            ).topk(selected_count, dim=-1).values
            selected_mask = torch.isfinite(selected_loss)
            per_frame_count = selected_mask.sum(dim=-1)
            per_frame_loss = torch.where(selected_mask, selected_loss, 0.0).sum(
                dim=-1
            ) / per_frame_count.clamp_min(1).to(loss_dtype)
            represented_frames = (per_frame_count > 0).to(loss_dtype)
            heatmap_hard_negative = (
                per_frame_loss * represented_frames
            ).sum() / represented_frames.sum().clamp_min(1.0)
        else:
            heatmap_hard_negative = prediction.sum() * 0.0

        activity_frame_mask = frame_mask.to(loss_dtype)
        positive_activity_mass = (activity_target * activity_frame_mask).sum()
        negative_activity_mass = ((1.0 - activity_target) * activity_frame_mask).sum()
        # softplus forms are stable BCE-with-logits for their respective labels.
        activity_positive = _normalised_component(
            (
                F.softplus(-loss_active_logits)
                * activity_target
                * activity_frame_mask
            ).sum(),
            positive_activity_mass,
        )
        activity_negative = _normalised_component(
            (
                F.softplus(loss_active_logits)
                * (1.0 - activity_target)
                * activity_frame_mask
            ).sum(),
            negative_activity_mass,
        )
        activity_loss = _mean_present(
            activity_positive,
            positive_activity_mass,
            activity_negative,
            negative_activity_mass,
            first_fraction=self.activity_positive_fraction,
        )
        total = self.heatmap_weight * heatmap_loss + self.activity_weight * activity_loss
        if self.hard_negative_weight > 0.0:
            total = total + self.hard_negative_weight * heatmap_hard_negative
        return TemporalTraceLossOutput(
            total=total,
            heatmap=heatmap_loss,
            heatmap_positive=heatmap_positive,
            heatmap_background=heatmap_background,
            heatmap_hard_negative=heatmap_hard_negative,
            activity=activity_loss,
            activity_positive=activity_positive,
            activity_negative=activity_negative,
            positive_target_mass=positive_target_mass,
            background_target_mass=background_target_mass,
            positive_activity_mass=positive_activity_mass,
            negative_activity_mass=negative_activity_mass,
            labeled_frames=activity_frame_mask.sum(),
        )


@dataclass(frozen=True)
class TeacherForcingSchedule:
    """Epoch-based scheduled-sampling probability.

    ``warmup_epochs`` retain ``start_probability``.  The following
    ``decay_epochs`` include both endpoints (for example, a 3-epoch linear
    decay is ``start, midpoint, end``).  Validation should always use zero,
    independent of this training schedule.
    """

    start_probability: float = 1.0
    end_probability: float = 0.0
    warmup_epochs: int = 0
    decay_epochs: int = 20
    curve: str = "linear"

    def __post_init__(self) -> None:
        _validate_probability("start_probability", self.start_probability)
        _validate_probability("end_probability", self.end_probability)
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.decay_epochs < 1:
            raise ValueError("decay_epochs must be at least 1")
        if self.curve not in {"linear", "cosine"}:
            raise ValueError("curve must be 'linear' or 'cosine'")

    def probability(self, epoch: int) -> float:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if epoch < self.warmup_epochs:
            return float(self.start_probability)
        if self.decay_epochs == 1:
            return float(self.end_probability)
        progress = min(
            1.0, (epoch - self.warmup_epochs) / float(self.decay_epochs - 1)
        )
        if self.curve == "cosine":
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        return float(
            self.start_probability
            + progress * (self.end_probability - self.start_probability)
        )

    def __call__(self, epoch: int) -> float:
        return self.probability(epoch)


def sample_teacher_forcing_mask(
    probability: float,
    valid_mask: torch.Tensor,
    *,
    label_mask: Optional[torch.Tensor] = None,
    reset_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample causal teacher feedback transitions as a boolean ``[B,T]`` mask.

    Mask element ``t`` means target heatmap ``t-1`` may be supplied while
    predicting frame ``t``.  Consequently the first timestep, transitions
    after padding/unlabeled history, and transitions across a reset are always
    false.
    """

    probability = _validate_probability("probability", probability)
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [B,T]")
    valid = valid_mask.to(dtype=torch.bool)
    labeled = (
        torch.ones_like(valid)
        if label_mask is None
        else label_mask.to(device=valid.device, dtype=torch.bool)
    )
    if tuple(labeled.shape) != tuple(valid.shape):
        raise ValueError("label_mask must match valid_mask")
    resets = (
        torch.zeros_like(valid)
        if reset_mask is None
        else reset_mask.to(device=valid.device, dtype=torch.bool)
    )
    if tuple(resets.shape) != tuple(valid.shape):
        raise ValueError("reset_mask must match valid_mask")

    eligible = torch.zeros_like(valid)
    if valid.shape[1] > 1:
        eligible[:, 1:] = (
            valid[:, 1:] & valid[:, :-1] & labeled[:, :-1] & ~resets[:, 1:]
        )
    if probability == 0.0:
        return torch.zeros_like(valid)
    if probability == 1.0:
        return eligible
    sampled = torch.rand(
        valid.shape, device=valid.device, generator=generator
    ) < probability
    return sampled & eligible


def corrupt_teacher_heatmaps(
    heatmaps: torch.Tensor,
    *,
    dropout_probability: float = 0.0,
    noise_std: float = 0.0,
    valid_mask: Optional[torch.Tensor] = None,
    label_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return noisy scheduled-sampling targets without modifying the input.

    Dropout is sampled once per frame/heatmap, not once per pixel.  Gaussian
    noise is intentionally applied to foreground and background so the model
    learns to recover from both weakened tracks and spurious feedback.  Invalid
    or unlabeled maps are always zeroed and must also be excluded from the
    teacher-forcing mask.
    """

    dropout_probability = _validate_probability(
        "dropout_probability", dropout_probability
    )
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if heatmaps.ndim != 5 or heatmaps.shape[2] != 1:
        raise ValueError("heatmaps must have shape [B,T,1,H,W]")
    frame_mask = _frame_mask(heatmaps, valid_mask, label_mask)
    map_mask = frame_mask[:, :, None, None, None]

    result = heatmaps.clone()
    if noise_std:
        noise = torch.randn(
            result.shape,
            device=result.device,
            dtype=result.dtype,
            generator=generator,
        )
        result = result + float(noise_std) * noise * map_mask.to(result.dtype)
    if dropout_probability:
        keep = torch.rand(
            frame_mask.shape, device=result.device, generator=generator
        ) >= dropout_probability
        map_mask = map_mask & keep[:, :, None, None, None]
    return result.clamp_(0.0, 1.0) * map_mask.to(result.dtype)


@dataclass(frozen=True)
class DetectedTouchPeak:
    """One local heatmap maximum in normalised screen coordinates."""

    x: float
    y: float
    score: float


@dataclass(frozen=True)
class _TemporalTraceMetricFrame:
    """One validated, labeled frame prepared for thresholded metrics."""

    heatmap: torch.Tensor
    activity_probability: torch.Tensor
    active: bool
    exact_centers: list[tuple[float, float]]


def extract_touch_peaks(
    heatmap: torch.Tensor,
    *,
    threshold: float = 0.3,
    nms_radius_px: int = 6,
    max_peaks: int = 8,
) -> list[DetectedTouchPeak]:
    """Decode one map with the same plateau/NMS semantics as deployment."""

    from trueskate_ai.vision.touch_peaks import (
        extract_touch_peaks as extract_canonical_touch_peaks,
    )

    threshold = _validate_probability("threshold", threshold)
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must have shape [H,W], got {tuple(heatmap.shape)}")
    if nms_radius_px < 0:
        raise ValueError("nms_radius_px must be non-negative")
    if max_peaks < 1:
        raise ValueError("max_peaks must be at least 1")
    return [
        DetectedTouchPeak(x=peak.x, y=peak.y, score=peak.score)
        for peak in extract_canonical_touch_peaks(
            heatmap.detach().float().cpu().numpy(),
            threshold=threshold,
            nms_radius_px=nms_radius_px,
            max_peaks=max_peaks,
        )
    ]


def _maximum_matches(
    predictions: list[DetectedTouchPeak],
    targets: list[tuple[float, float]],
    tolerance: float,
) -> int:
    """Maximum-cardinality bipartite matching within a distance tolerance."""

    neighbours: list[list[int]] = []
    for target_x, target_y in targets:
        ranked = sorted(
            (
                (math.hypot(peak.x - target_x, peak.y - target_y), index)
                for index, peak in enumerate(predictions)
            ),
            key=lambda pair: pair[0],
        )
        neighbours.append([index for distance, index in ranked if distance <= tolerance])

    prediction_owner = [-1] * len(predictions)

    def assign(target_index: int, visited: set[int]) -> bool:
        for prediction_index in neighbours[target_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous_target = prediction_owner[prediction_index]
            if previous_target < 0 or assign(previous_target, visited):
                prediction_owner[prediction_index] = target_index
                return True
        return False

    return sum(assign(target_index, set()) for target_index in range(len(targets)))


def _iter_temporal_trace_metric_frames(
    predicted_heatmaps: torch.Tensor,
    active_logits: torch.Tensor,
    target_active: torch.Tensor,
    centers: torch.Tensor,
    touch_count: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    label_mask: Optional[torch.Tensor] = None,
) -> Iterable[_TemporalTraceMetricFrame]:
    """Yield validated CPU frames shared by single and grid evaluation."""

    if predicted_heatmaps.ndim != 5 or predicted_heatmaps.shape[2] != 1:
        raise ValueError("predicted_heatmaps must have shape [B,T,1,H,W]")
    batch, steps = predicted_heatmaps.shape[:2]
    expected_frames = (batch, steps)
    for name, tensor in (
        ("active_logits", active_logits),
        ("target_active", target_active),
        ("touch_count", touch_count),
    ):
        if tuple(tensor.shape) != expected_frames:
            raise ValueError(
                f"{name} must have shape {expected_frames}, got {tuple(tensor.shape)}"
            )
    if (
        centers.ndim != 4
        or tuple(centers.shape[:2]) != expected_frames
        or centers.shape[-1] != 2
    ):
        raise ValueError("centers must have shape [B,T,K,2]")

    max_targets = centers.shape[2]
    frame_mask = _frame_mask(predicted_heatmaps, valid_mask, label_mask).cpu()
    heatmaps = predicted_heatmaps.detach().cpu()
    activity_probabilities = torch.sigmoid(active_logits.detach()).cpu()
    target_activity = target_active.detach().cpu().to(torch.bool)
    target_centers = centers.detach().cpu().float()
    counts = touch_count.detach().cpu().to(torch.int64)

    for batch_index in range(batch):
        for time_index in range(steps):
            if not bool(frame_mask[batch_index, time_index]):
                continue
            count = int(counts[batch_index, time_index])
            if not 0 <= count <= max_targets:
                raise ValueError(
                    f"touch_count {count} exceeds centers capacity {max_targets}"
                )
            active = bool(target_activity[batch_index, time_index])
            if active != (count > 0):
                raise ValueError(
                    "labeled target_active must agree with touch_count > 0"
                )

            # Track columns are stable across time, so valid slots can be
            # sparse when one touch lifts while another continues.  The
            # negative x sentinel, not a contiguous prefix assumption, is
            # authoritative.
            center_row = target_centers[batch_index, time_index]
            valid_center_slots = torch.nonzero(
                center_row[:, 0] >= 0, as_tuple=False
            ).flatten()
            if len(valid_center_slots) != count:
                raise ValueError(
                    f"touch_count {count} disagrees with "
                    f"{len(valid_center_slots)} non-sentinel centers"
                )
            exact_centers = [
                (float(center_row[i, 0]), float(center_row[i, 1]))
                for i in valid_center_slots.tolist()
            ]
            if any(
                not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
                for x, y in exact_centers
            ):
                raise ValueError("valid center coordinates must lie in [0,1]")

            yield _TemporalTraceMetricFrame(
                heatmap=heatmaps[batch_index, time_index, 0],
                activity_probability=activity_probabilities[
                    batch_index, time_index
                ],
                active=active,
                exact_centers=exact_centers,
            )


@dataclass
class TemporalTraceMetricAccumulator:
    """Streaming exact-centre metrics for autoregressive validation."""

    peak_threshold: float = 0.3
    activity_threshold: float = 0.5
    localization_tolerance: float = 0.05
    nms_radius_px: int = 6
    max_peaks: int = 8

    def __post_init__(self) -> None:
        _validate_probability("peak_threshold", self.peak_threshold)
        _validate_probability("activity_threshold", self.activity_threshold)
        if self.localization_tolerance < 0:
            raise ValueError("localization_tolerance must be non-negative")
        if self.nms_radius_px < 0:
            raise ValueError("nms_radius_px must be non-negative")
        if self.max_peaks < 1:
            raise ValueError("max_peaks must be at least 1")
        self.labeled_frames = 0
        self.positive_frames = 0
        self.negative_frames = 0
        self.negative_correct = 0
        self.target_touches = 0
        self.predicted_peaks = 0
        self.matched_touches = 0
        self.multi_touch_frames = 0
        self.multi_target_touches = 0
        self.multi_predicted_peaks = 0
        self.multi_matched_touches = 0

    def update(
        self,
        predicted_heatmaps: torch.Tensor,
        active_logits: torch.Tensor,
        target_active: torch.Tensor,
        centers: torch.Tensor,
        touch_count: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> None:
        for frame in _iter_temporal_trace_metric_frames(
            predicted_heatmaps,
            active_logits,
            target_active,
            centers,
            touch_count,
            valid_mask=valid_mask,
            label_mask=label_mask,
        ):
            raw_peaks = extract_touch_peaks(
                frame.heatmap,
                threshold=self.peak_threshold,
                nms_radius_px=self.nms_radius_px,
                max_peaks=self.max_peaks,
            )
            predicts_active = bool(
                frame.activity_probability >= self.activity_threshold
            )
            self._update_decoded_frame(frame, raw_peaks, predicts_active)

    def _update_decoded_frame(
        self,
        frame: _TemporalTraceMetricFrame,
        raw_peaks: list[DetectedTouchPeak],
        predicts_active: bool,
    ) -> None:
        """Update counters from peaks already decoded at this threshold."""

        # The activity head gates emitted touch centres.  For negative
        # specificity, however, contradictory raw heatmap peaks are also false
        # positives so neither head can hide the other.
        emitted_peaks = raw_peaks if predicts_active else []
        matches = _maximum_matches(
            emitted_peaks, frame.exact_centers, self.localization_tolerance
        )
        count = len(frame.exact_centers)

        self.labeled_frames += 1
        self.predicted_peaks += len(emitted_peaks)
        self.target_touches += count
        self.matched_touches += matches
        if frame.active:
            self.positive_frames += 1
        else:
            self.negative_frames += 1
            self.negative_correct += int(not predicts_active and not raw_peaks)
        if count >= 2:
            self.multi_touch_frames += 1
            self.multi_target_touches += count
            self.multi_predicted_peaks += len(emitted_peaks)
            self.multi_matched_touches += matches

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        # Missing validation classes must never satisfy the 90% acceptance gate.
        return float(numerator / denominator) if denominator else 0.0

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    def compute(self) -> dict[str, float | int]:
        positive_recall = self._ratio(self.matched_touches, self.target_touches)
        negative_specificity = self._ratio(self.negative_correct, self.negative_frames)
        peak_precision = self._ratio(self.matched_touches, self.predicted_peaks)
        peak_recall = positive_recall
        multi_precision = self._ratio(
            self.multi_matched_touches, self.multi_predicted_peaks
        )
        multi_recall = self._ratio(
            self.multi_matched_touches, self.multi_target_touches
        )
        acceptance_score = min(
            positive_recall,
            negative_specificity,
            peak_precision,
        )
        return {
            "positive_accuracy": positive_recall,
            "positive_touch_recall": positive_recall,
            "negative_accuracy": negative_specificity,
            "negative_frame_specificity": negative_specificity,
            # Recall alone lets a noisy localizer pass by emitting one correct
            # peak plus arbitrary extras on every active frame.  Those extras
            # become malformed touch tracks downstream, so strict acceptance
            # also requires peak precision.  Multi-touch F1 remains diagnostic:
            # small validation splits may contain too few overlap frames for a
            # stable hard gate.
            "acceptance_score": acceptance_score,
            "peak_precision": peak_precision,
            "peak_recall": peak_recall,
            "peak_f1": self._f1(peak_precision, peak_recall),
            "multi_peak_precision": multi_precision,
            "multi_peak_recall": multi_recall,
            "multi_peak_f1": self._f1(multi_precision, multi_recall),
            "frames": self.labeled_frames,
            "labeled_frames": self.labeled_frames,
            "positive_frames": self.positive_frames,
            "negative_frames": self.negative_frames,
            "negative_correct_frames": self.negative_correct,
            "target_touches": self.target_touches,
            "predicted_peaks": self.predicted_peaks,
            "matched_touches": self.matched_touches,
            "multi_touch_frames": self.multi_touch_frames,
            "multi_target_touches": self.multi_target_touches,
            "multi_predicted_peaks": self.multi_predicted_peaks,
            "multi_matched_touches": self.multi_matched_touches,
        }


@dataclass
class _ThresholdGridMetricAccumulator:
    """Existing metrics plus decomposed negative-frame diagnostics."""

    metrics: TemporalTraceMetricAccumulator
    raw_heatmap_negative_correct: int = 0
    activity_negative_correct: int = 0
    emitted_negative_correct: int = 0

    def update(
        self,
        frame: _TemporalTraceMetricFrame,
        raw_peaks: list[DetectedTouchPeak],
        predicts_active: bool,
    ) -> None:
        self.metrics._update_decoded_frame(frame, raw_peaks, predicts_active)
        if frame.active:
            return
        self.raw_heatmap_negative_correct += int(not raw_peaks)
        self.activity_negative_correct += int(not predicts_active)
        self.emitted_negative_correct += int(not (predicts_active and raw_peaks))

    def compute(self) -> dict[str, float | int]:
        result = self.metrics.compute()
        negative_frames = self.metrics.negative_frames
        ratio = self.metrics._ratio
        result.update(
            raw_heatmap_negative_specificity=ratio(
                self.raw_heatmap_negative_correct, negative_frames
            ),
            activity_negative_specificity=ratio(
                self.activity_negative_correct, negative_frames
            ),
            emitted_negative_specificity=ratio(
                self.emitted_negative_correct, negative_frames
            ),
        )
        return result


def temporal_trace_metrics(
    predicted_heatmaps: torch.Tensor,
    active_logits: torch.Tensor,
    target_active: torch.Tensor,
    centers: torch.Tensor,
    touch_count: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    label_mask: Optional[torch.Tensor] = None,
    peak_threshold: float = 0.3,
    activity_threshold: float = 0.5,
    localization_tolerance: float = 0.05,
    nms_radius_px: int = 6,
    max_peaks: int = 8,
) -> dict[str, float | int]:
    """Compute exact-centre metrics for an already-autoregressive rollout."""

    accumulator = TemporalTraceMetricAccumulator(
        peak_threshold=peak_threshold,
        activity_threshold=activity_threshold,
        localization_tolerance=localization_tolerance,
        nms_radius_px=nms_radius_px,
        max_peaks=max_peaks,
    )
    accumulator.update(
        predicted_heatmaps,
        active_logits,
        target_active,
        centers,
        touch_count,
        valid_mask=valid_mask,
        label_mask=label_mask,
    )
    return accumulator.compute()


@dataclass
class AutoregressiveTraceOutput:
    """Minimal output from validation's teacher-free causal rollout."""

    heatmaps: torch.Tensor
    active_logits: torch.Tensor
    state: TemporalTraceState


def autoregressive_trace_rollout(
    model: nn.Module,
    frames: torch.Tensor,
    *,
    delta_times: Optional[torch.Tensor] = None,
    reset_mask: Optional[torch.Tensor] = None,
) -> AutoregressiveTraceOutput:
    """Roll out ``model.step`` with only predicted feedback and causal resets."""

    if frames.ndim != 5:
        raise ValueError("frames must have shape [B,T,C,H,W]")
    batch, steps = frames.shape[:2]
    if steps < 1:
        raise ValueError("frames must contain at least one timestep")
    if delta_times is not None and tuple(delta_times.shape) != (batch, steps):
        raise ValueError("delta_times must have shape [B,T]")
    resets = (
        torch.zeros((batch, steps), dtype=torch.bool, device=frames.device)
        if reset_mask is None
        else reset_mask.to(device=frames.device, dtype=torch.bool)
    )
    if tuple(resets.shape) != (batch, steps):
        raise ValueError("reset_mask must have shape [B,T]")

    state: Optional[TemporalTraceState] = None
    heatmaps = []
    active_logits = []
    for time_index in range(steps):
        reset_rows = resets[:, time_index]
        if state is not None and bool(reset_rows.any()):
            keep_hidden = (~reset_rows).reshape(batch, 1, 1, 1).to(state.hidden.dtype)
            keep_heatmap = (~reset_rows).reshape(batch, 1, 1, 1).to(
                state.previous_heatmap.dtype
            )
            state = TemporalTraceState(
                hidden=state.hidden * keep_hidden,
                previous_heatmap=state.previous_heatmap * keep_heatmap,
            )
        delta_t = delta_times[:, time_index] if delta_times is not None else None
        output = model.step(frames[:, time_index], state, delta_t=delta_t)
        state = output.state
        heatmaps.append(output.heatmap)
        active_logits.append(output.active_logits)
    assert state is not None
    return AutoregressiveTraceOutput(
        heatmaps=torch.stack(heatmaps, dim=1),
        active_logits=torch.stack(active_logits, dim=1),
        state=state,
    )


def evaluate_temporal_trace_model(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    *,
    peak_threshold: float = 0.3,
    activity_threshold: float = 0.5,
    localization_tolerance: float = 0.05,
    nms_radius_px: int = 6,
    max_peaks: int = 8,
) -> dict[str, float | int]:
    """Evaluate canonical temporal batches with no teacher forcing.

    Required batch keys are ``frames``, ``active``, ``centers`` and
    ``touch_count``.  ``delta_times``, ``valid_mask``, ``label_mask`` and
    ``reset_mask`` are optional but honored when supplied.  The ``heatmaps``
    targets are intentionally unused: exact centres are the localization
    ground truth, and validation must never feed target heatmaps back.
    """

    accumulator = TemporalTraceMetricAccumulator(
        peak_threshold=peak_threshold,
        activity_threshold=activity_threshold,
        localization_tolerance=localization_tolerance,
        nms_radius_px=nms_radius_px,
        max_peaks=max_peaks,
    )
    required = ("frames", "active", "centers", "touch_count")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError(f"temporal validation batch missing keys: {missing}")
            frames = batch["frames"].to(device)
            delta_times = batch.get("delta_times")
            reset_mask = batch.get("reset_mask")
            output = autoregressive_trace_rollout(
                model,
                frames,
                delta_times=(delta_times.to(device) if delta_times is not None else None),
                reset_mask=(reset_mask.to(device) if reset_mask is not None else None),
            )
            accumulator.update(
                output.heatmaps,
                output.active_logits,
                batch["active"],
                batch["centers"],
                batch["touch_count"],
                valid_mask=batch.get("valid_mask"),
                label_mask=batch.get("label_mask"),
            )
    return accumulator.compute()


def evaluate_temporal_trace_threshold_grid(
    model: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device | str,
    *,
    peak_thresholds: Iterable[float],
    activity_thresholds: Iterable[float],
    localization_tolerance: float = 0.05,
    nms_radius_px: int = 6,
    max_peaks: int = 8,
) -> dict[tuple[float, float], dict[str, float | int]]:
    """Evaluate a threshold grid with one teacher-free model rollout.

    Result keys are sorted ``(peak_threshold, activity_threshold)`` pairs.
    Every value contains the complete metrics returned by
    :func:`evaluate_temporal_trace_model`, whose negative specificity retains
    its strict ``inactive activity AND no raw peak`` meaning, plus three
    decomposed negative-frame specificities.

    Each labeled frame's local maxima are decoded exactly once at the lowest
    requested peak threshold.  Higher peak thresholds only filter those peaks
    by score; the fixed NMS radius and maximum peak count therefore remain
    identical across the grid.
    """

    peak_values = _normalise_probability_grid("peak_thresholds", peak_thresholds)
    activity_values = _normalise_probability_grid(
        "activity_thresholds", activity_thresholds
    )
    minimum_peak_threshold = peak_values[0]
    # ``extract_touch_peaks`` decodes float32 maps.  Apply higher thresholds at
    # that same precision so filtering exactly matches an independent decode.
    peak_score_cutoffs = {
        threshold: float(torch.tensor(threshold, dtype=torch.float32))
        for threshold in peak_values
    }
    accumulators: dict[
        tuple[float, float], _ThresholdGridMetricAccumulator
    ] = {}
    for peak_threshold in peak_values:
        for activity_threshold in activity_values:
            accumulators[(peak_threshold, activity_threshold)] = (
                _ThresholdGridMetricAccumulator(
                    TemporalTraceMetricAccumulator(
                        peak_threshold=peak_threshold,
                        activity_threshold=activity_threshold,
                        localization_tolerance=localization_tolerance,
                        nms_radius_px=nms_radius_px,
                        max_peaks=max_peaks,
                    )
                )
            )

    required = ("frames", "active", "centers", "touch_count")
    model.eval()
    with torch.no_grad():
        for batch in loader:
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError(f"temporal validation batch missing keys: {missing}")
            frames = batch["frames"].to(device)
            delta_times = batch.get("delta_times")
            reset_mask = batch.get("reset_mask")
            output = autoregressive_trace_rollout(
                model,
                frames,
                delta_times=(
                    delta_times.to(device) if delta_times is not None else None
                ),
                reset_mask=(
                    reset_mask.to(device) if reset_mask is not None else None
                ),
            )
            for frame in _iter_temporal_trace_metric_frames(
                output.heatmaps,
                output.active_logits,
                batch["active"],
                batch["centers"],
                batch["touch_count"],
                valid_mask=batch.get("valid_mask"),
                label_mask=batch.get("label_mask"),
            ):
                minimum_peaks = extract_touch_peaks(
                    frame.heatmap,
                    threshold=minimum_peak_threshold,
                    nms_radius_px=nms_radius_px,
                    max_peaks=max_peaks,
                )
                peaks_by_threshold = {
                    threshold: [
                        peak
                        for peak in minimum_peaks
                        if peak.score >= peak_score_cutoffs[threshold]
                    ]
                    for threshold in peak_values
                }
                activity_by_threshold = {
                    threshold: bool(frame.activity_probability >= threshold)
                    for threshold in activity_values
                }
                for peak_threshold in peak_values:
                    raw_peaks = peaks_by_threshold[peak_threshold]
                    for activity_threshold in activity_values:
                        accumulators[(peak_threshold, activity_threshold)].update(
                            frame,
                            raw_peaks,
                            activity_by_threshold[activity_threshold],
                        )

    return {key: accumulator.compute() for key, accumulator in accumulators.items()}


__all__ = [
    "AutoregressiveTraceOutput",
    "BalancedTemporalTraceLoss",
    "DetectedTouchPeak",
    "TeacherForcingSchedule",
    "TemporalTraceLossOutput",
    "TemporalTraceMetricAccumulator",
    "autoregressive_trace_rollout",
    "corrupt_teacher_heatmaps",
    "evaluate_temporal_trace_model",
    "evaluate_temporal_trace_threshold_grid",
    "extract_touch_peaks",
    "sample_teacher_forcing_mask",
    "temporal_trace_metrics",
]
