"""Train temporal Model 1: causal RGB + previous-heatmap touch tracking.

Unlike the legacy single-frame trace extractor, this trainer keeps every
gesture sample as one chronological recurrent sequence.  Training gradually
replaces noisy teacher heatmaps with the model's own previous prediction;
validation is fully autoregressive and resets state only at sample boundaries.

The 90% target is deliberately strict: positive touch-localisation recall,
negative-frame specificity, and emitted-peak precision must each independently
reach the threshold.

Examples:
    python scripts/train/train_temporal_trace_extractor.py --smoke
    python scripts/train/train_temporal_trace_extractor.py \
        --data data/sls_xctest --include-path-term supercrown --epochs 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.vision.temporal_trace_dataset import (  # noqa: E402
    DEFAULT_HEATMAP_SIGMA,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_LATENCY_S,
    TemporalTraceSequenceDataset,
    split_by_sample,
)
from trueskate_ai.vision.temporal_trace_predictor import (  # noqa: E402
    TemporalTracePredictor,
)
from trueskate_ai.vision.temporal_trace_training import (  # noqa: E402
    BalancedTemporalTraceLoss,
    TeacherForcingSchedule,
    corrupt_teacher_heatmaps,
    evaluate_temporal_trace_model,
    evaluate_temporal_trace_threshold_grid,
    sample_teacher_forcing_mask,
)


CHECKPOINT_VERSION = 3
MODEL_TYPE = "temporal_trace_predictor_v1"
ACCEPTANCE_SEMANTICS = "min_touch_recall_negative_specificity_peak_precision"


def _parse_probability_grid(value: str, *, name: str) -> tuple[float, ...]:
    """Parse a deterministic comma-delimited inference-threshold grid."""

    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-delimited probabilities") from exc
    if not parsed or any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in parsed):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return tuple(sorted(set(parsed)))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _base_dataset_and_indices(dataset: Dataset) -> tuple[Dataset, list[int]]:
    if isinstance(dataset, Subset):
        if isinstance(dataset.dataset, Subset):
            raise ValueError("nested Subset datasets are not supported")
        return dataset.dataset, [int(index) for index in dataset.indices]
    return dataset, list(range(len(dataset)))


def _dataset_split_fingerprint(dataset: Dataset | None) -> str | None:
    """Stable identity for the exact gesture sequences behind one split.

    A true resume may restore a stored best validation score only when it is
    evaluated against the same train/validation sequences.  In particular,
    adding new spin samples changes this fingerprint and cannot silently retain
    the old no-spin best checkpoint.
    """

    if dataset is None:
        return None
    base, indices = _base_dataset_and_indices(dataset)
    sample_paths = getattr(base, "sample_paths", None)
    if sample_paths is None or len(sample_paths) != len(base):
        raise ValueError(
            "temporal training datasets must expose one sample_path per sequence "
            "so checkpoint resume can validate the split identity"
        )
    positive = getattr(base, "positive_frame_counts", None)
    negative = getattr(base, "negative_frame_counts", None)
    multi_touch = getattr(base, "multi_touch_frame_counts", None)
    digest = hashlib.sha256()
    for index in indices:
        digest.update(str(index).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(sample_paths[index]).encode("utf-8"))
        if positive is not None and negative is not None:
            multi_count = int(multi_touch[index]) if multi_touch is not None else 0
            digest.update(
                f"\0{int(positive[index])}:{int(negative[index])}:{multi_count}".encode(
                    "ascii"
                )
            )
        digest.update(b"\n")
    return f"sha256_split_v2:{len(indices)}:{digest.hexdigest()}"


def balanced_sequence_sampler(
    dataset: Dataset,
    *,
    seed: int = 0,
    overlap_sampling_fraction: float = 0.25,
):
    """Oversample sequences according to their rare-class frame contribution.

    Full sequences remain intact.  The sampler only changes which gesture is
    drawn, while :class:`BalancedTemporalTraceLoss` independently normalises
    positive and negative frames/pixels inside each batch.
    """

    if (
        not math.isfinite(overlap_sampling_fraction)
        or not 0.0 <= overlap_sampling_fraction <= 1.0
    ):
        raise ValueError("overlap_sampling_fraction must be finite and in [0,1]")
    base, indices = _base_dataset_and_indices(dataset)
    positive = getattr(base, "positive_frame_counts", None)
    negative = getattr(base, "negative_frame_counts", None)
    if positive is None or negative is None or not indices:
        return None
    positive = np.asarray([positive[index] for index in indices], dtype=np.float64)
    negative = np.asarray([negative[index] for index in indices], dtype=np.float64)
    total_positive = float(positive.sum())
    total_negative = float(negative.sum())
    if total_positive <= 0.0 or total_negative <= 0.0:
        return None
    # Expected sampling mass is split equally between the two frame classes.
    weights = positive / total_positive + negative / total_negative
    multi_touch = getattr(base, "multi_touch_frame_counts", None)
    if multi_touch is not None and overlap_sampling_fraction > 0.0:
        if len(multi_touch) != len(base):
            raise ValueError(
                "multi_touch_frame_counts must contain one entry per sequence"
            )
        overlap_mass = np.asarray(
            [multi_touch[index] for index in indices], dtype=np.float64
        )
        total_overlap = float(overlap_mass.sum())
        if total_overlap > 0.0:
            # Reserve a bounded share of draw probability for sequences in
            # proportion to their genuine labeled overlap frames.  Sampling
            # still returns whole causal sequences and keeps the epoch length
            # unchanged.  With no usable overlap mass, retain the exact legacy
            # positive/negative weights above.
            weights = (
                (1.0 - overlap_sampling_fraction) * weights / weights.sum()
                + overlap_sampling_fraction * overlap_mass / total_overlap
            )
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        return None
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    model: TemporalTracePredictor,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    h: int,
    w: int,
    sequence_length: int,
    latency_s: float,
    heatmap_sigma: float,
    metrics: dict | None,
    best_score: float,
    training_config: dict,
    split_fingerprints: dict[str, str | None] | None = None,
) -> dict:
    inference_config = None
    if metrics is not None and all(
        key in metrics
        for key in (
            "peak_threshold",
            "activity_threshold",
            "peak_nms_radius_px",
            "max_touches",
        )
    ):
        inference_config = {
            "peak_threshold": float(metrics["peak_threshold"]),
            "activity_threshold": float(metrics["activity_threshold"]),
            "peak_nms_radius_px": int(metrics["peak_nms_radius_px"]),
            "max_touches": int(metrics["max_touches"]),
        }
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_type": MODEL_TYPE,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "h": int(h),
        "w": int(w),
        "sequence_length": int(sequence_length),
        "latency_s": float(latency_s),
        "heatmap_sigma": float(heatmap_sigma),
        "val_metrics": metrics,
        "val_acceptance_score": float(best_score),
        "acceptance_semantics": ACCEPTANCE_SEMANTICS,
        "training_config": training_config,
        "inference_config": inference_config,
        "split_fingerprints": split_fingerprints,
        "timing_semantics": "causal_rgb_previous_predicted_heatmap",
    }


def _load_resume(
    path: Path,
    model: TemporalTracePredictor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    image_height: int,
    image_width: int,
    sequence_length: int,
    latency_s: float,
    heatmap_sigma: float,
    expected_split_fingerprints: dict[str, str | None] | None = None,
    load_optimizer: bool = True,
) -> tuple[int, float, dict | None]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type") if isinstance(checkpoint, dict) else None
    if model_type != MODEL_TYPE:
        raise RuntimeError(
            f"{path} is not a {MODEL_TYPE} checkpoint. Legacy Model 1 weights are "
            "incompatible with the recurrent state, activity head, and timing semantics; "
            "retrain temporal Model 1 instead of migrating them."
        )
    version = checkpoint.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"{path} has temporal checkpoint_version={version!r}; expected "
            f"{CHECKPOINT_VERSION}. Retrain or resume with a checkpoint produced "
            "by this trainer so recurrent/timing semantics cannot be mixed."
        )
    timing_semantics = checkpoint.get("timing_semantics")
    if timing_semantics != "causal_rgb_previous_predicted_heatmap":
        raise RuntimeError(
            f"{path} has incompatible timing_semantics={timing_semantics!r}"
        )
    acceptance_semantics = checkpoint.get("acceptance_semantics")
    if acceptance_semantics != ACCEPTANCE_SEMANTICS:
        raise RuntimeError(
            f"{path} has incompatible acceptance_semantics="
            f"{acceptance_semantics!r}; expected {ACCEPTANCE_SEMANTICS!r}. "
            "Its stored best score did not enforce peak precision, so start a "
            "fresh run rather than reusing stale validation state."
        )
    saved_config = checkpoint.get("model_config")
    if saved_config != asdict(model.config):
        raise ValueError(
            f"resume model_config {saved_config!r} does not match requested "
            f"{asdict(model.config)!r}"
        )
    expected_data_config = {
        "h": int(image_height),
        "w": int(image_width),
        "sequence_length": int(sequence_length),
    }
    for key, expected in expected_data_config.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"resume {key}={checkpoint.get(key)!r} does not match requested "
                f"{expected!r}"
            )
    for key, expected in (
        ("latency_s", float(latency_s)),
        ("heatmap_sigma", float(heatmap_sigma)),
    ):
        actual = checkpoint.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"resume {key}={actual!r} does not match requested {expected!r}"
            )
    if expected_split_fingerprints is not None:
        saved_fingerprints = checkpoint.get("split_fingerprints")
        if saved_fingerprints != expected_split_fingerprints:
            raise ValueError(
                "resume train/validation split fingerprint does not match the "
                "current corpus. Use a fresh run (or a future explicit weights-only "
                "fine-tune mode) so stale best metrics cannot discard new training. "
                f"saved={saved_fingerprints!r}, current={expected_split_fingerprints!r}"
            )
    if "model_state" not in checkpoint:
        raise RuntimeError(f"{path} is missing model_state")
    model.load_state_dict(checkpoint["model_state"])
    if load_optimizer and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best = float(checkpoint.get("val_acceptance_score", -1.0))
    if start_epoch < 0 or not math.isfinite(best) or not -1.0 <= best <= 1.0:
        raise RuntimeError(
            f"{path} has invalid resume progress epoch={start_epoch}, best={best}"
        )
    metrics = checkpoint.get("val_metrics")
    if metrics is not None:
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{path} val_metrics must be a mapping or null")
        try:
            metric_score = min(
                float(metrics["positive_accuracy"]),
                float(metrics["negative_accuracy"]),
                float(metrics["peak_precision"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{path} val_metrics lacks valid recall/specificity/precision"
            ) from exc
        if not math.isclose(best, metric_score, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"{path} val_acceptance_score={best} disagrees with strict "
                f"min(positive, negative, peak_precision)={metric_score}"
            )
    print(f"resumed temporal checkpoint {path} at epoch {start_epoch} (best={best:.4f})")
    return start_epoch, best, metrics


def train_temporal(
    dataset: Dataset,
    *,
    val_dataset: Dataset | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    out_path: Path,
    image_height: int,
    image_width: int,
    sequence_length: int,
    latency_s: float,
    heatmap_sigma: float = DEFAULT_HEATMAP_SIGMA,
    base_channels: int = 16,
    hidden_channels: int = 32,
    downsample_stages: int = 2,
    num_workers: int = 0,
    sampling_seed: int = 0,
    overlap_sampling_fraction: float = 0.25,
    model_seed: int = 0,
    resume_path: Path | None = None,
    resume_weights_only: bool = False,
    target_accuracy: float = 0.90,
    peak_threshold: float = 0.30,
    activity_threshold: float = 0.50,
    peak_thresholds: tuple[float, ...] | None = None,
    activity_thresholds: tuple[float, ...] | None = None,
    peak_nms_radius_px: int = 6,
    localization_tolerance: float = 0.05,
    teacher_start: float = 0.90,
    teacher_end: float = 0.05,
    teacher_warmup_epochs: int = 1,
    teacher_decay_epochs: int | None = None,
    feedback_dropout: float = 0.10,
    feedback_noise_std: float = 0.03,
    heatmap_loss_weight: float = 1.0,
    activity_loss_weight: float = 1.0,
    heatmap_positive_fraction: float = 0.5,
    activity_positive_fraction: float = 0.5,
    hard_negative_weight: float = 0.0,
    hard_negative_top_k: int = 64,
    hard_negative_target_exclusion_threshold: float = 0.05,
    lr_plateau_patience: int | None = 3,
    cosine_min_lr: float | None = None,
    gradient_clip_norm: float = 2.0,
    checkpoint_callback: Callable[[], None] | None = None,
    smoke: bool = False,
    device: torch.device | None = None,
) -> dict | None:
    """Train and save the best fully-autoregressive temporal checkpoint."""

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0,1]")
    if len(dataset) < 1:
        raise ValueError("training dataset is empty")
    if (
        not math.isfinite(overlap_sampling_fraction)
        or not 0.0 <= overlap_sampling_fraction <= 1.0
    ):
        raise ValueError("overlap_sampling_fraction must be finite and in [0,1]")
    if lr_plateau_patience is not None and lr_plateau_patience < 0:
        raise ValueError("lr_plateau_patience must be non-negative or None")
    if cosine_min_lr is not None and not 0.0 <= cosine_min_lr <= learning_rate:
        raise ValueError("cosine_min_lr must be in [0, learning_rate] or None")
    if cosine_min_lr is not None and lr_plateau_patience not in (None, 0):
        raise ValueError("cosine and plateau LR schedulers are mutually exclusive")
    if resume_weights_only and resume_path is None:
        raise ValueError("resume_weights_only requires resume_path")
    split_fingerprints = {
        "train": _dataset_split_fingerprint(dataset),
        "validation": _dataset_split_fingerprint(val_dataset),
    }
    device = device or _device()
    torch.manual_seed(model_seed)
    model = TemporalTracePredictor(
        in_channels=3,
        base_channels=base_channels,
        hidden_channels=hidden_channels,
        downsample_stages=downsample_stages,
        use_time_deltas=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = BalancedTemporalTraceLoss(
        heatmap_weight=heatmap_loss_weight,
        activity_weight=activity_loss_weight,
        heatmap_positive_fraction=heatmap_positive_fraction,
        activity_positive_fraction=activity_positive_fraction,
        hard_negative_weight=hard_negative_weight,
        hard_negative_top_k=hard_negative_top_k,
        hard_negative_target_exclusion_threshold=(
            hard_negative_target_exclusion_threshold
        ),
    )
    start_epoch = 0
    best_score = -1.0
    best_metrics: dict | None = None
    if resume_path is not None:
        start_epoch, best_score, best_metrics = _load_resume(
            resume_path,
            model,
            optimizer,
            device,
            image_height=image_height,
            image_width=image_width,
            sequence_length=sequence_length,
            latency_s=latency_s,
            heatmap_sigma=heatmap_sigma,
            expected_split_fingerprints=(
                None if resume_weights_only else split_fingerprints
            ),
            load_optimizer=not resume_weights_only,
        )
        if resume_weights_only:
            start_epoch = 0
            best_score = -1.0
            best_metrics = None
            print(
                "using checkpoint weights as fresh initialization; optimizer, "
                "epoch, best metric, and split identity were reset"
            )
        if start_epoch >= epochs:
            raise ValueError(
                f"resume checkpoint already reached epoch {start_epoch}; request epochs>{start_epoch}"
            )
        # A resumed run may intentionally write to a new model-volume name.  Keep
        # the previously best checkpoint there immediately; otherwise a run with
        # no validation improvement would return a path that was never created.
        if not resume_weights_only and resume_path.resolve() != out_path.resolve():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resume_path, out_path)
            if checkpoint_callback is not None:
                checkpoint_callback()

    decay_epochs = teacher_decay_epochs or max(2, epochs - teacher_warmup_epochs)
    teacher_schedule = TeacherForcingSchedule(
        start_probability=teacher_start,
        end_probability=teacher_end,
        warmup_epochs=teacher_warmup_epochs,
        decay_epochs=decay_epochs,
        curve="cosine",
    )
    training_config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "target_accuracy": target_accuracy,
        "peak_threshold": peak_threshold,
        "activity_threshold": activity_threshold,
        "peak_thresholds": list(peak_thresholds or (peak_threshold,)),
        "activity_thresholds": list(activity_thresholds or (activity_threshold,)),
        "peak_nms_radius_px": peak_nms_radius_px,
        "localization_tolerance": localization_tolerance,
        "teacher_forcing": asdict(teacher_schedule),
        "feedback_dropout": feedback_dropout,
        "feedback_noise_std": feedback_noise_std,
        "heatmap_loss_weight": heatmap_loss_weight,
        "activity_loss_weight": activity_loss_weight,
        "heatmap_positive_fraction": heatmap_positive_fraction,
        "activity_positive_fraction": activity_positive_fraction,
        "hard_negative_weight": hard_negative_weight,
        "hard_negative_top_k": hard_negative_top_k,
        "hard_negative_target_exclusion_threshold": (
            hard_negative_target_exclusion_threshold
        ),
        "sampling_seed": sampling_seed,
        "overlap_sampling_fraction": overlap_sampling_fraction,
        "model_seed": model_seed,
        "lr_plateau_patience": lr_plateau_patience,
        "cosine_min_lr": cosine_min_lr,
        "resume_weights_only": resume_weights_only,
    }
    sampler = balanced_sequence_sampler(
        dataset,
        seed=sampling_seed,
        overlap_sampling_fraction=overlap_sampling_fraction,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        if val_dataset is not None
        else None
    )
    validation_max_peaks = int(
        getattr(getattr(val_dataset, "dataset", val_dataset), "max_touches", 8)
    )
    use_amp = device.type == "cuda" and not smoke
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    plateau_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=lr_plateau_patience,
            min_lr=1e-6,
        )
        if lr_plateau_patience not in (None, 0)
        else None
    )
    cosine_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - start_epoch),
            eta_min=cosine_min_lr,
        )
        if cosine_min_lr is not None
        else None
    )
    print(
        f"device={device} sequences={len(dataset)} val={len(val_dataset) if val_dataset else 0} "
        f"T={sequence_length} image={image_height}x{image_width} batch={batch_size} "
        f"params={sum(parameter.numel() for parameter in model.parameters()):,} "
        f"balanced_sampler={sampler is not None}"
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        teacher_probability = teacher_schedule(epoch)
        running = {
            "total": 0.0,
            "heatmap": 0.0,
            "heatmap_positive": 0.0,
            "heatmap_background": 0.0,
            "hard_negative": 0.0,
            "activity": 0.0,
            "activity_positive": 0.0,
            "activity_negative": 0.0,
        }
        steps = 0
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            target_heatmaps = batch["heatmaps"].to(device, non_blocking=True)
            target_active = batch["active"].to(device, non_blocking=True)
            delta_times = batch["delta_times"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            label_mask = batch["label_mask"].to(device, non_blocking=True)
            reset_mask = batch["reset_mask"].to(device, non_blocking=True)
            if bool(reset_mask[:, 1:].any()):
                raise ValueError("dataset emitted a mid-sequence reset; split it into separate items")
            teacher_mask = sample_teacher_forcing_mask(
                teacher_probability,
                valid_mask,
                label_mask=label_mask,
                reset_mask=reset_mask,
            )
            teacher_heatmaps = corrupt_teacher_heatmaps(
                target_heatmaps,
                dropout_probability=feedback_dropout,
                noise_std=feedback_noise_std,
                valid_mask=valid_mask,
                label_mask=label_mask,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                output = model(
                    frames,
                    teacher_heatmaps=teacher_heatmaps,
                    teacher_forcing_mask=teacher_mask,
                    delta_times=delta_times,
                    detach_feedback=False,
                )
                losses = criterion(
                    output.heatmaps,
                    output.active_logits,
                    target_heatmaps,
                    target_active,
                    valid_mask=valid_mask,
                    label_mask=label_mask,
                )
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running["total"] += float(losses.total.detach())
            running["heatmap"] += float(losses.heatmap.detach())
            running["heatmap_positive"] += float(
                losses.heatmap_positive.detach()
            )
            running["heatmap_background"] += float(
                losses.heatmap_background.detach()
            )
            running["hard_negative"] += float(
                losses.heatmap_hard_negative.detach()
            )
            running["activity"] += float(losses.activity.detach())
            running["activity_positive"] += float(
                losses.activity_positive.detach()
            )
            running["activity_negative"] += float(
                losses.activity_negative.detach()
            )
            steps += 1
            if smoke:
                break
        if not steps:
            raise RuntimeError("training DataLoader produced no batches")

        suffix = ""
        metrics = None
        if val_loader is not None:
            threshold_grid = evaluate_temporal_trace_threshold_grid(
                model,
                val_loader,
                device,
                peak_thresholds=peak_thresholds or (peak_threshold,),
                activity_thresholds=activity_thresholds or (activity_threshold,),
                localization_tolerance=localization_tolerance,
                nms_radius_px=peak_nms_radius_px,
                max_peaks=validation_max_peaks,
            )
            (selected_peak_threshold, selected_activity_threshold), metrics = max(
                threshold_grid.items(),
                key=lambda item: (
                    float(item[1]["acceptance_score"]),
                    float(item[1]["peak_f1"]),
                    float(item[1]["positive_accuracy"]),
                ),
            )
            metrics = dict(metrics)
            metrics.update(
                peak_threshold=selected_peak_threshold,
                activity_threshold=selected_activity_threshold,
                peak_nms_radius_px=peak_nms_radius_px,
                max_touches=validation_max_peaks,
            )
            score = float(metrics["acceptance_score"])
            if plateau_scheduler is not None:
                plateau_scheduler.step(score)
            suffix = (
                f" val_accept={100 * score:.2f}%"
                f" pos={100 * float(metrics['positive_accuracy']):.2f}%/{metrics['target_touches']}"
                f" neg={100 * float(metrics['negative_accuracy']):.2f}%/{metrics['negative_frames']}"
                f" prec={100 * float(metrics['peak_precision']):.2f}%/{metrics['predicted_peaks']}"
                f" peakF1={100 * float(metrics['peak_f1']):.2f}%"
                f" thr={selected_peak_threshold:.2f}/{selected_activity_threshold:.2f}"
                f" rawneg={100 * float(metrics['raw_heatmap_negative_specificity']):.2f}%"
                f" actneg={100 * float(metrics['activity_negative_specificity']):.2f}%"
                f" emitneg={100 * float(metrics['emitted_negative_specificity']):.2f}%"
                f" multiF1={100 * float(metrics['multi_peak_f1']):.2f}%/"
                f"{metrics['multi_touch_frames']}"
            )
            if score > best_score:
                best_score = score
                best_metrics = dict(metrics)
                _save_checkpoint(
                    out_path,
                    _checkpoint_payload(
                        model,
                        optimizer,
                        epoch=epoch + 1,
                        h=image_height,
                        w=image_width,
                        sequence_length=sequence_length,
                        latency_s=latency_s,
                        heatmap_sigma=heatmap_sigma,
                        metrics=best_metrics,
                        best_score=best_score,
                        training_config=training_config,
                        split_fingerprints=split_fingerprints,
                    ),
                )
                if checkpoint_callback is not None:
                    checkpoint_callback()
        else:
            if plateau_scheduler is not None:
                plateau_scheduler.step(-running["total"] / steps)

        print(
            f"epoch {epoch + 1}/{epochs} loss={running['total'] / steps:.5f} "
            f"heat={running['heatmap'] / steps:.5f} "
            f"heat+/-={running['heatmap_positive'] / steps:.5f}/"
            f"{running['heatmap_background'] / steps:.5f} "
            f"hardneg={running['hard_negative'] / steps:.5f} "
            f"act={running['activity'] / steps:.5f} "
            f"act+/-={running['activity_positive'] / steps:.5f}/"
            f"{running['activity_negative'] / steps:.5f} "
            f"teacher={teacher_probability:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} steps={steps}{suffix}"
        )
        if cosine_scheduler is not None:
            cosine_scheduler.step()
        if smoke:
            break
        if metrics is not None and (
            float(metrics["positive_accuracy"]) >= target_accuracy
            and float(metrics["negative_accuracy"]) >= target_accuracy
            and float(metrics["peak_precision"]) >= target_accuracy
        ):
            print(
                f"strict target reached: positive={100 * float(metrics['positive_accuracy']):.2f}% "
                f"negative={100 * float(metrics['negative_accuracy']):.2f}% and "
                f"precision={100 * float(metrics['peak_precision']):.2f}% "
                f">= {100 * target_accuracy:.2f}%"
            )
            break

    if val_loader is None:
        _save_checkpoint(
            out_path,
            _checkpoint_payload(
                model,
                optimizer,
                epoch=epoch + 1,
                h=image_height,
                w=image_width,
                sequence_length=sequence_length,
                latency_s=latency_s,
                heatmap_sigma=heatmap_sigma,
                metrics=None,
                best_score=-1.0,
                training_config=training_config,
                split_fingerprints=split_fingerprints,
            ),
        )
        if checkpoint_callback is not None:
            checkpoint_callback()
    print(f"saved best temporal checkpoint -> {out_path}")
    return best_metrics


class _SyntheticTemporalDataset(Dataset):
    """Small deterministic sequence corpus for a one-step pipeline smoke."""

    def __init__(self, count: int = 6, *, steps: int = 6, height: int = 48, width: int = 24):
        self.count = count
        self.steps = steps
        self.height = height
        self.width = width
        self.max_touches = 2
        self.sample_paths = tuple(Path(f"synthetic_{index}") for index in range(count))
        self.positive_frame_counts = [3] * count
        self.negative_frame_counts = [steps - 3] * count
        self.multi_touch_frame_counts = [2 if index % 2 == 0 else 0 for index in range(count)]
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        self._ys, self._xs = ys, xs

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = np.random.default_rng(index)
        frames = generator.random((self.steps, 3, self.height, self.width), dtype=np.float32) * 0.1
        centers = np.full((self.steps, 2, 2), -1.0, np.float32)
        for t in range(1, 4):
            centers[t, 0] = (0.25 + 0.15 * t, 0.72 - 0.12 * t)
        if index % 2 == 0:
            centers[2:4, 1] = (0.08, 0.40)
        heatmaps = np.zeros((self.steps, 1, self.height, self.width), np.float32)
        for t in range(self.steps):
            for x, y in centers[t]:
                if x < 0:
                    continue
                px, py = x * (self.width - 1), y * (self.height - 1)
                bump = np.exp(-((self._xs - px) ** 2 + (self._ys - py) ** 2) / 8.0)
                heatmaps[t, 0] = np.maximum(heatmaps[t, 0], bump)
            # Give the RGB stream a learnable warm trace cue.
            frames[t, 0] = np.maximum(frames[t, 0], heatmaps[t, 0])
        touch_count = (centers[..., 0] >= 0).sum(axis=1).astype(np.int64)
        valid = np.ones(self.steps, np.bool_)
        reset = np.zeros(self.steps, np.bool_)
        reset[0] = True
        return {
            "frames": torch.from_numpy(frames),
            "heatmaps": torch.from_numpy(heatmaps),
            "active": torch.from_numpy((touch_count > 0).astype(np.float32)),
            "centers": torch.from_numpy(centers),
            "touch_count": torch.from_numpy(touch_count),
            "delta_times": torch.tensor([0.0] + [1 / 30] * (self.steps - 1)),
            "valid_mask": torch.from_numpy(valid),
            "label_mask": torch.from_numpy(valid.copy()),
            "reset_mask": torch.from_numpy(reset),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-path-term", default="supercrown")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--img-h", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--img-w", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--max-touches", type=int, default=4)
    parser.add_argument("--latency-s", type=float, default=DEFAULT_LATENCY_S)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--downsample-stages", type=int, default=2)
    parser.add_argument(
        "--peak-threshold-grid",
        default="0.30,0.50,0.65,0.75,0.85,0.90,0.95",
    )
    parser.add_argument(
        "--activity-threshold-grid",
        default="0.50,0.70,0.80,0.90,0.95,0.98",
    )
    parser.add_argument("--peak-nms-radius-px", type=int, default=6)
    parser.add_argument("--teacher-start", type=float, default=0.90)
    parser.add_argument("--teacher-end", type=float, default=0.05)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=1)
    parser.add_argument("--teacher-decay-epochs", type=int, default=0)
    parser.add_argument("--feedback-dropout", type=float, default=0.10)
    parser.add_argument("--feedback-noise-std", type=float, default=0.03)
    parser.add_argument("--heatmap-loss-weight", type=float, default=1.0)
    parser.add_argument("--activity-loss-weight", type=float, default=1.0)
    parser.add_argument("--heatmap-positive-fraction", type=float, default=0.5)
    parser.add_argument("--activity-positive-fraction", type=float, default=0.5)
    parser.add_argument("--overlap-sampling-fraction", type=float, default=0.25)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-top-k", type=int, default=64)
    parser.add_argument(
        "--hard-negative-target-exclusion-threshold", type=float, default=0.05
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=3,
        help="epochs without strict-metric improvement before halving LR; 0 disables",
    )
    parser.add_argument("--no-require-trace", action="store_true")
    parser.add_argument("--cache-frames", action="store_true")
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=0,
        help="parallel sample/frame cache workers (0 or 1 keeps serial loading)",
    )
    parser.add_argument("--detect-menu-frames", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-weights-only", action="store_true")
    parser.add_argument(
        "--cosine-min-lr",
        type=float,
        default=-1.0,
        help="enable cosine LR decay to this value; negative disables",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        peak_thresholds = _parse_probability_grid(
            args.peak_threshold_grid, name="--peak-threshold-grid"
        )
        activity_thresholds = _parse_probability_grid(
            args.activity_threshold_grid, name="--activity-threshold-grid"
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.smoke:
        dataset = _SyntheticTemporalDataset()
        train_set, val_set = split_by_sample(dataset, val_fraction=0.33, seed=0)
        out = args.out or (_REPO_ROOT / "tmp" / "temporal_trace_smoke.pth")
        train_temporal(
            train_set,
            val_dataset=val_set,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            out_path=out,
            image_height=dataset.height,
            image_width=dataset.width,
            sequence_length=dataset.steps,
            latency_s=DEFAULT_LATENCY_S,
            base_channels=4,
            hidden_channels=8,
            downsample_stages=2,
            peak_thresholds=(0.3, 0.7),
            activity_thresholds=(0.5, 0.9),
            hard_negative_weight=0.10,
            hard_negative_top_k=4,
            heatmap_positive_fraction=0.70,
            activity_positive_fraction=0.50,
            overlap_sampling_fraction=0.25,
            lr_plateau_patience=None,
            smoke=True,
        )
        checkpoint = torch.load(out, map_location="cpu", weights_only=False)
        assert checkpoint["model_type"] == MODEL_TYPE
        assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
        assert checkpoint["training_config"]["heatmap_positive_fraction"] == 0.70
        assert checkpoint["training_config"]["overlap_sampling_fraction"] == 0.25
        print("SMOKE OK: sequence -> scheduled rollout -> balanced loss -> autoregressive val -> v3 checkpoint")
        return

    if args.data is None:
        parser.error("--data is required unless --smoke is used")
    dataset = TemporalTraceSequenceDataset(
        args.data,
        sequence_length=args.sequence_length,
        image_height=args.img_h,
        image_width=args.img_w,
        max_touches=args.max_touches,
        latency_s=args.latency_s,
        include_path_term=args.include_path_term or None,
        max_samples=args.max_samples or None,
        require_trace=not args.no_require_trace,
        cache_frames=args.cache_frames,
        cache_workers=args.cache_workers,
        detect_menu_frames=args.detect_menu_frames,
    )
    print(json.dumps(dataset.stats, indent=2, sort_keys=True))
    train_set, val_set = split_by_sample(
        dataset, val_fraction=args.val_fraction, seed=args.split_seed
    )
    out = args.out or (
        _REPO_ROOT
        / "notebooks"
        / "models"
        / f"trace_extractor_temporal_v1_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    )
    train_temporal(
        train_set,
        val_dataset=val_set,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        out_path=out,
        image_height=args.img_h,
        image_width=args.img_w,
        sequence_length=args.sequence_length,
        latency_s=args.latency_s,
        base_channels=args.base_channels,
        hidden_channels=args.hidden_channels,
        downsample_stages=args.downsample_stages,
        resume_path=args.resume,
        resume_weights_only=args.resume_weights_only,
        sampling_seed=args.split_seed,
        overlap_sampling_fraction=args.overlap_sampling_fraction,
        model_seed=args.model_seed,
        peak_thresholds=peak_thresholds,
        activity_thresholds=activity_thresholds,
        peak_nms_radius_px=args.peak_nms_radius_px,
        teacher_start=args.teacher_start,
        teacher_end=args.teacher_end,
        teacher_warmup_epochs=args.teacher_warmup_epochs,
        teacher_decay_epochs=args.teacher_decay_epochs or None,
        feedback_dropout=args.feedback_dropout,
        feedback_noise_std=args.feedback_noise_std,
        heatmap_loss_weight=args.heatmap_loss_weight,
        activity_loss_weight=args.activity_loss_weight,
        heatmap_positive_fraction=args.heatmap_positive_fraction,
        activity_positive_fraction=args.activity_positive_fraction,
        hard_negative_weight=args.hard_negative_weight,
        hard_negative_top_k=args.hard_negative_top_k,
        hard_negative_target_exclusion_threshold=(
            args.hard_negative_target_exclusion_threshold
        ),
        lr_plateau_patience=args.lr_plateau_patience or None,
        cosine_min_lr=(args.cosine_min_lr if args.cosine_min_lr >= 0.0 else None),
    )


if __name__ == "__main__":
    main()
