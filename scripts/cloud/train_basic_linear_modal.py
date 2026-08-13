"""Train the strict MVP-2 linear-drag regressor on Modal.

The corpus is expected to already live below ``MODAL_CORPUS_VOLUME``.  This
module only mounts and reads it; checkpoints and immutable metric summaries are
written to ``trueskate-models``.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import modal

_SCRIPT_PATH = Path(__file__).resolve()
_ROOT = _SCRIPT_PATH.parents[2] if len(_SCRIPT_PATH.parents) > 2 else _SCRIPT_PATH.parent
CORPUS_VOLUME = os.environ.get("MODAL_CORPUS_VOLUME", "trueskate-corpus")
MODELS_VOLUME = "trueskate-models"

app = modal.App("trueskate-basic-linear")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0")
    # gesture_sampling imports the shared CMA-ES bounds, which transitively
    # imports the device gesture module.  The trainer never opens a WebDriver
    # session, but that module declares Selenium classes at import time.
    .pip_install("torch", "opencv-python-headless", "numpy", "selenium")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(_ROOT / "src" / "trueskate_ai"), remote_path="/root/src/trueskate_ai")
    .add_local_file(str(_ROOT / "scripts" / "train" / "train_basic_linear_regressor.py"),
                    remote_path="/root/scripts/train/train_basic_linear_regressor.py")
)
corpus = modal.Volume.from_name(CORPUS_VOLUME)
models = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_basic_linear_regressor", "/root/scripts/train/train_basic_linear_regressor.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _model_from_payload(payload, torch):
    """Reconstruct a checkpoint using its recorded inference architecture."""
    from trueskate_ai.vision.basic_linear_regressor import BasicLinearRegressor
    return BasicLinearRegressor(
        base_channels=int(payload["base_channels"]),
        start_onset=float(payload.get("start_onset", .24)),
        start_sigma=float(payload.get("start_sigma", .05)),
        end_onset=float(payload.get("end_onset", .24)),
        temporal_mixer=bool(payload.get("temporal_mixer", False)),
        trajectory_track=bool(payload.get("trajectory_track", False)),
    )


# This compact clip regressor does not need A10G-class throughput.  The
# dedicated 2k run spent hours queued without a container on A10G, while its
# fixed 32 GiB memory headroom is what protects decoded-frame caching.  Accept
# any compatible accelerator from Modal's pool so a scarce named type cannot
# indefinitely block the deterministic protocol.
@app.function(image=image, gpu="any", timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def train_remote(data_subdir: str, run_label: str, *, epochs: int = 40,
                 batch_size: int = 8, lr: float = 1e-3, seed: int = 0,
                 base_channels: int = 16, split_strategy: str = "command",
                 cache_frames: bool = True, split_seed: int | None = None,
                 map_weight: float = 0.0, start_onset: float = .24,
                 start_sigma: float = .05, end_onset: float = .24,
                 temporal_mixer: bool = False, trajectory_weight: float = 0.0,
                 trajectory_track: bool = False, fresh_holdout_source: str | None = None,
                 evaluate_test: bool = True, fresh_stratify_by_device: bool = False) -> dict:
    trainer = _trainer()
    checkpoint = Path("/models") / f"basic_linear_{run_label}.pth"
    payload = trainer.train(
        data=Path("/corpus") / data_subdir,
        out=checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        split_seed=split_seed,
        map_weight=map_weight,
        start_onset=start_onset,
        start_sigma=start_sigma,
        end_onset=end_onset,
        temporal_mixer=temporal_mixer,
        trajectory_weight=trajectory_weight,
        trajectory_track=trajectory_track,
        fresh_holdout_source=fresh_holdout_source,
        evaluate_test=evaluate_test,
        fresh_stratify_by_device=fresh_stratify_by_device,
        base_channels=base_channels,
        split_strategy=split_strategy,
        cache_frames=cache_frames,
    )
    result = {key: value for key, value in payload.items() if key != "state_dict"}
    result["checkpoint"] = checkpoint.name
    result["run_label"] = run_label
    (Path("/models") / f"basic_linear_{run_label}.json").write_text(json.dumps(result, indent=2))
    models.commit()
    return result


@app.function(image=image, cpu=8.0, timeout=12 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def train_remote_cpu(data_subdir: str, run_label: str, *, epochs: int = 40,
                     batch_size: int = 8, lr: float = 1e-3, seed: int = 0,
                     base_channels: int = 16, split_strategy: str = "command",
                     cache_frames: bool = True, split_seed: int | None = None,
                     map_weight: float = 0.0, start_onset: float = .24,
                     start_sigma: float = .05, end_onset: float = .24,
                     temporal_mixer: bool = False, trajectory_weight: float = 0.0,
                     trajectory_track: bool = False, fresh_holdout_source: str | None = None,
                     evaluate_test: bool = True, fresh_stratify_by_device: bool = False) -> dict:
    """Scheduler-independent execution fallback for the same compact protocol.

    This is intentionally a separate function rather than silently removing a
    GPU request.  The data split, model, optimiser and acceptance metric remain
    identical; only the hardware differs, and the result is separately labelled.
    """
    trainer = _trainer()
    checkpoint = Path("/models") / f"basic_linear_{run_label}.pth"
    payload = trainer.train(
        data=Path("/corpus") / data_subdir,
        out=checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        split_seed=split_seed,
        map_weight=map_weight,
        start_onset=start_onset,
        start_sigma=start_sigma,
        end_onset=end_onset,
        temporal_mixer=temporal_mixer,
        trajectory_weight=trajectory_weight,
        trajectory_track=trajectory_track,
        fresh_holdout_source=fresh_holdout_source,
        evaluate_test=evaluate_test,
        fresh_stratify_by_device=fresh_stratify_by_device,
        base_channels=base_channels,
        split_strategy=split_strategy,
        cache_frames=cache_frames,
    )
    result = {key: value for key, value in payload.items() if key != "state_dict"}
    result["checkpoint"] = checkpoint.name
    result["run_label"] = run_label
    result["execution_hardware"] = "cpu"
    (Path("/models") / f"basic_linear_{run_label}.json").write_text(json.dumps(result, indent=2))
    models.commit()
    return result


@app.function(image=image, gpu="any", timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def evaluate_refinement(data_subdir: str, checkpoint_name: str, *, seed: int = 0,
                        batch_size: int = 8) -> dict:
    """Grid-evaluate post-inference orange refinement on the held-out commands."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_refinement import refine_linear_endpoints
    from trueskate_ai.vision.basic_linear_training import basic_linear_metrics

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, _val, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    model = _model_from_payload(payload, torch).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    batches = [(batch["frames"].to(device), batch["target"].to(device))
               for batch in DataLoader(Subset(data, test_indices), batch_size=batch_size)]
    grid = [(0., .08, .10), (.02, .06, .10), (.05, .06, .10),
            (.10, .06, .10), (.05, .08, .13)]
    results = {}
    for blend, spatial_sigma, time_sigma in grid:
        class Refined(torch.nn.Module):
            def forward(self, frames):
                base = model(frames)
                return refine_linear_endpoints(frames, base, blend=blend,
                                               spatial_sigma=spatial_sigma, time_sigma=time_sigma)
        results[f"blend={blend}:space={spatial_sigma}:time={time_sigma}"] = basic_linear_metrics(
            Refined(), [{"frames": frames, "target": target} for frames, target in batches], device)
    class Base(torch.nn.Module):
        def forward(self, frames):
            return model(frames)
    output = {"checkpoint": checkpoint_name, "test_samples": len(test_indices),
              "baseline": basic_linear_metrics(Base(), [{"frames": frames, "target": target}
                                                          for frames, target in batches], device),
              "results": results}
    stem = Path(checkpoint_name).stem
    (Path("/models") / f"{stem}_refinement_grid.json").write_text(json.dumps(output, indent=2))
    (Path("/models") / f"{stem}_component_audit.json").write_text(json.dumps(output["baseline"], indent=2))
    models.commit()
    return output


@app.function(image=image, gpu="any", timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def evaluate_start_timing(data_subdir: str, checkpoint_name: str, *, seed: int = 0,
                          batch_size: int = 8) -> dict:
    """Ablate the fixed start-time prior on a frozen held-out checkpoint.

    This is intentionally post-training: it establishes whether the known
    alignment window, rather than the learned spatial map, is limiting start
    recovery before consuming another training run.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_training import basic_linear_metrics

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, _val, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    model = _model_from_payload(payload, torch).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    batches = [(batch["frames"].to(device), batch["target"].to(device))
               for batch in DataLoader(Subset(data, test_indices), batch_size=batch_size)]

    def timed_start(frames, *, onset: float, sigma: float):
        base, start_scores, _end_scores = model.forward_with_scores(frames)
        steps = frames.shape[1]
        time = torch.linspace(0., 1., steps, dtype=frames.dtype, device=frames.device)
        active = torch.where(time < .18, torch.full_like(time, -12.0), torch.zeros_like(time))
        prior = active - ((time - onset) / sigma).square()
        x0, y0 = model._read_xy(start_scores, prior)
        return torch.cat((x0[:, None], y0[:, None], base[:, 2:]), dim=1)

    results = {}
    # The aligned clips retain 0.5s of lead-in.  Do not assume the rendered
    # trail starts at the command timestamp: validate the actual temporal
    # location on held-out commands, including the pre-command portion of the
    # window.  This is an evaluation-only sweep over a frozen checkpoint.
    for onset in (-.30, -.24, -.18, -.12, -.06, .00, .06, .12, .18, .24, .30):
        for sigma in (.03, .05, .08, .12, .17):
            class TimedStart(torch.nn.Module):
                def forward(self, frames):
                    return timed_start(frames, onset=onset, sigma=sigma)
            results[f"onset={onset:.2f}:sigma={sigma:.2f}"] = basic_linear_metrics(
                TimedStart(), [{"frames": frames, "target": target} for frames, target in batches], device,
            )
    output = {"checkpoint": checkpoint_name, "test_samples": len(test_indices), "results": results}
    stem = Path(checkpoint_name).stem
    (Path("/models") / f"{stem}_start_timing_grid.json").write_text(json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, gpu="any", timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def evaluate_start_timing_validation_selected(data_subdir: str, checkpoint_name: str, *, seed: int = 0,
                                              batch_size: int = 8) -> dict:
    """Select a frozen start-time prior on validation commands, then test once."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_training import basic_linear_metrics

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, val_indices, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    model = _model_from_payload(payload, torch).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    def batches(indices):
        return [(batch["frames"].to(device), batch["target"].to(device))
                for batch in DataLoader(Subset(data, indices), batch_size=batch_size)]
    val_batches, test_batches = batches(val_indices), batches(test_indices)

    def metric_for(onset: float, sigma: float, prepared):
        class TimedStart(torch.nn.Module):
            def forward(self, frames):
                base, start_scores, _end_scores = model.forward_with_scores(frames)
                time = torch.linspace(0., 1., frames.shape[1], dtype=frames.dtype, device=frames.device)
                active = torch.where(time < .18, torch.full_like(time, -12.0), torch.zeros_like(time))
                prior = active - ((time - onset) / sigma).square()
                x0, y0 = model._read_xy(start_scores, prior)
                return torch.cat((x0[:, None], y0[:, None], base[:, 2:]), dim=1)
        return basic_linear_metrics(TimedStart(), [{"frames": frames, "target": target}
                                                    for frames, target in prepared], device)

    candidates = [(onset, sigma) for onset in (-.24, -.12, -.06, .00, .06, .12, .24)
                  for sigma in (.05, .08, .12)]
    ranked = []
    for onset, sigma in candidates:
        metric = metric_for(onset, sigma, val_batches)
        rank = (-metric["gesture_recovery_accuracy"], metric["start_coordinate_median"])
        ranked.append((rank, onset, sigma, metric))
    ranked.sort(key=lambda item: item[0])
    _rank, onset, sigma, validation = ranked[0]
    test = metric_for(onset, sigma, test_batches)
    output = {
        "checkpoint": checkpoint_name,
        "selected_onset": onset,
        "selected_sigma": sigma,
        "validation": validation,
        "test": test,
        "validation_top5": [
            {"onset": candidate_onset, "sigma": candidate_sigma, "metrics": metric}
            for _rank, candidate_onset, candidate_sigma, metric in ranked[:5]
        ],
    }
    stem = Path(checkpoint_name).stem
    (Path("/models") / f"{stem}_start_timing_validation_selected.json").write_text(
        json.dumps(output, indent=2),
    )
    models.commit()
    return output


@app.function(image=image, timeout=3 * 3600, memory=8192,
              volumes={"/corpus": corpus, "/models": models})
def audit_orange_endpoint_cue(data_subdir: str, *, seed: int = 0,
                             batch_size: int = 8) -> dict:
    """Measure raw endpoint-cue observability on held-out commands.

    This diagnostic is deliberately not an inference method: it uses each
    target's duration to centre fixed temporal windows, then asks whether the
    rendered warm-pixel evidence is spatially close enough to the labelled
    endpoint.  It distinguishes an alignment/rendering ceiling from a learned
    endpoint-reader failure without tuning a model on test commands.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command

    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, _val, test_indices = split_by_command(data, seed=seed)
    totals = {key: 0 for key in ("samples", "start", "end", "both")}
    time = torch.linspace(0., 1., data.sequence_length)
    xa = torch.linspace(0., 1., data.image_width)
    ya = torch.linspace(0., 1., data.image_height)

    for batch in DataLoader(Subset(data, test_indices), batch_size=batch_size):
        frames, target = batch["frames"], batch["target"]
        reference = frames[:, :max(1, round(frames.shape[1] * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        warm = ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

        def closest_error(xy, centre):
            temporal = torch.exp(-.5 * ((time[None, :, None, None] - centre[:, None, None, None]) / .06).square())
            index = (warm * temporal).flatten(1).argmax(dim=1)
            plane = data.image_height * data.image_width
            spatial = index.remainder(plane)
            y = spatial.div(data.image_width, rounding_mode="floor")
            x = spatial.remainder(data.image_width)
            point = torch.stack((xa[x], ya[y]), dim=1)
            return torch.linalg.vector_norm(point - xy, dim=1)

        onset = target.new_full((len(target),), .24)
        liftoff = (onset + target[:, 4] / 2.27).clamp(max=.88)
        start = closest_error(target[:, :2], onset)
        end = closest_error(target[:, 2:4], liftoff)
        passed_start, passed_end = start <= .03, end <= .03
        totals["samples"] += len(target)
        totals["start"] += int(passed_start.sum())
        totals["end"] += int(passed_end.sum())
        totals["both"] += int((passed_start & passed_end).sum())
    output = {
        "diagnostic": "target-timed warm-pixel argmax; not an inference result",
        "test_samples": totals["samples"],
        "start_within_0.03": totals["start"] / totals["samples"],
        "end_within_0.03": totals["end"] / totals["samples"],
        "both_within_0.03": totals["both"] / totals["samples"],
    }
    (Path("/models") / "basic_linear_orange_cue_audit_20260813.json").write_text(json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, gpu="any", timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_endpoint_residuals(data_subdir: str, checkpoint_name: str, *, seed: int = 0,
                             batch_size: int = 8) -> dict:
    """Describe signed endpoint residuals for an already-evaluated checkpoint."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, _val, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    model = _model_from_payload(payload, torch).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    residuals = []
    with torch.no_grad():
        for batch in DataLoader(Subset(data, test_indices), batch_size=batch_size):
            predicted = model(batch["frames"].to(device)).cpu().numpy()
            target = batch["target"].numpy()
            residuals.append(predicted - target)
    values = np.concatenate(residuals)
    errors = np.linalg.norm(values[:, :4].reshape(-1, 2, 2), axis=2)
    output = {
        "checkpoint": checkpoint_name,
        "test_samples": int(len(values)),
        "mean_signed_residual": dict(zip(("x0", "y0", "x1", "y1", "duration"), values.mean(axis=0).tolist())),
        "median_signed_residual": dict(zip(("x0", "y0", "x1", "y1", "duration"), np.median(values, axis=0).tolist())),
        "start_end_error_correlation": float(np.corrcoef(errors[:, 0], errors[:, 1])[0, 1]),
        "start_end_both_fail": float(np.mean((errors[:, 0] > .03) & (errors[:, 1] > .03))),
        "start_only_fail": float(np.mean((errors[:, 0] > .03) & (errors[:, 1] <= .03))),
        "end_only_fail": float(np.mean((errors[:, 0] <= .03) & (errors[:, 1] > .03))),
    }
    stem = Path(checkpoint_name).stem
    (Path("/models") / f"{stem}_endpoint_residual_audit.json").write_text(json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, gpu="any", timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def evaluate_checkpoint_ensemble(data_subdir: str, checkpoint_names: str, *, seed: int = 0,
                                 batch_size: int = 8,
                                 fresh_holdout_source: str | None = None,
                                 fresh_stratify_by_device: bool = False) -> dict:
    """Validation-select a convex checkpoint ensemble, then test it once.

    Every candidate has been trained on the same corpus/split.  We enumerate
    fixed convex weights using validation commands only, so the test commands
    remain untouched until the one selected evaluation.
    """
    import itertools
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_training import basic_linear_metrics

    checkpoint_names = [name.strip() for name in checkpoint_names.split(",") if name.strip()]
    if len(checkpoint_names) < 2:
        raise ValueError("need at least two checkpoints")
    payloads = [torch.load(Path("/models") / name, map_location="cpu", weights_only=False)
                for name in checkpoint_names]
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    if fresh_holdout_source is None:
        _train, val_indices, test_indices = split_by_command(data, seed=seed)
    else:
        _train, val_indices, test_indices = _trainer().split_with_fresh_command_holdout(
            data, fresh_source=fresh_holdout_source, seed=seed,
            stratify_by_device=fresh_stratify_by_device,
        )
    device = torch.device("cuda")
    models_local = []
    for payload in payloads:
        model = _model_from_payload(payload, torch).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models_local.append(model)

    def batches(indices):
        return [(batch["frames"].to(device), batch["target"].to(device))
                for batch in DataLoader(Subset(data, indices), batch_size=batch_size)]

    val_batches, test_batches = batches(val_indices), batches(test_indices)

    # The grid below has O(n_models^10) candidate weights.  Rerunning every
    # neural model for each candidate needlessly turns a validation-only model
    # selection into thousands of identical forward passes.  Cache only the
    # validation predictions once; test frames are deliberately not evaluated
    # until after selection, preserving the one-shot held-out protocol.
    with torch.no_grad():
        validation_predictions = [
            [model(frames) for frames, _target in val_batches]
            for model in models_local
        ]
    candidate_weights = []
    # 0.1 granularity is deliberately pre-declared and compact.  It covers
    # individual checkpoints and two/three-way averages without test tuning.
    for units in itertools.product(range(11), repeat=len(models_local)):
        if sum(units) == 10:
            candidate_weights.append(tuple(value / 10 for value in units))

    def metrics_for_validation(weights):
        class CachedEnsemble(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.index = 0

            def forward(self, _frames):
                value = sum(
                    weight * validation_predictions[model_index][self.index]
                    for model_index, weight in enumerate(weights)
                )
                self.index += 1
                return value

        return basic_linear_metrics(
            CachedEnsemble(), [{"frames": frames, "target": target} for frames, target in val_batches], device,
        )

    def metric_for_indices(weights, indices):
        class Ensemble(torch.nn.Module):
            def forward(self, frames):
                return sum(weight * model(frames) for weight, model in zip(weights, models_local))
        return basic_linear_metrics(
            Ensemble(), [{"frames": frames, "target": target}
                         for frames, target in batches(indices)], device,
        )

    ranked = []
    for weights in candidate_weights:
        metric = metrics_for_validation(weights)
        rank = (-metric["gesture_recovery_accuracy"],
                metric["start_coordinate_median"] + metric["end_coordinate_median"] + metric["duration_mae"])
        ranked.append((rank, weights, metric))
    ranked.sort(key=lambda item: item[0])
    _rank, selected_weights, validation = ranked[0]
    test = metric_for_indices(selected_weights, test_indices)
    test_by_device = None
    if fresh_stratify_by_device:
        # The partition is device-balanced by construction, but retain the
        # individual scores in the same one-shot post-selection test phase.
        # This makes a high pooled score falsifiable rather than allowing one
        # phone to conceal a device-specific endpoint failure.
        groups: dict[str, list[int]] = {}
        for index in test_indices:
            device_name = data._meta(data.sample_paths[index]).get("device")
            if not isinstance(device_name, str) or not device_name:
                raise ValueError("stratified test sample is missing explicit device provenance")
            groups.setdefault(device_name, []).append(index)
        test_by_device = {
            device_name: metric_for_indices(selected_weights, indices)
            for device_name, indices in sorted(groups.items())
        }
    output = {
        "checkpoints": checkpoint_names,
        "fresh_holdout_source": fresh_holdout_source,
        "fresh_stratify_by_device": fresh_stratify_by_device,
        "selected_weights": dict(zip(checkpoint_names, selected_weights)),
        "validation": validation,
        "test": test,
        "test_by_device": test_by_device,
        "validation_top5": [
            {"weights": dict(zip(checkpoint_names, weights)), "metrics": metric}
            for _rank, weights, metric in ranked[:5]
        ],
    }
    split_label = "command" if fresh_holdout_source is None else f"fresh_{fresh_holdout_source}"
    if fresh_stratify_by_device:
        split_label += "_device_stratified"
    (Path("/models") / f"basic_linear_checkpoint_ensemble_{split_label}.json").write_text(
        json.dumps(output, indent=2),
    )
    models.commit()
    return output


@app.local_entrypoint()
def main(data_subdir: str, run_label: str = "baseline", epochs: int = 40,
         batch_size: int = 8, lr: float = 1e-3, seed: int = 0,
         base_channels: int = 16, split_strategy: str = "command",
         cache_frames: bool = True, split_seed: int | None = None,
         map_weight: float = 0.0, start_onset: float = .24,
         start_sigma: float = .05, end_onset: float = .24,
         temporal_mixer: bool = False, trajectory_weight: float = 0.0,
         trajectory_track: bool = False, fresh_holdout_source: str | None = None,
         evaluate_test: bool = True, fresh_stratify_by_device: bool = False) -> None:
    result = train_remote.remote(
        data_subdir, run_label, epochs=epochs, batch_size=batch_size, lr=lr,
        seed=seed, base_channels=base_channels, split_strategy=split_strategy,
        cache_frames=cache_frames, split_seed=split_seed, map_weight=map_weight,
        start_onset=start_onset, start_sigma=start_sigma, end_onset=end_onset,
        temporal_mixer=temporal_mixer,
        trajectory_weight=trajectory_weight,
        trajectory_track=trajectory_track,
        fresh_holdout_source=fresh_holdout_source,
        evaluate_test=evaluate_test,
        fresh_stratify_by_device=fresh_stratify_by_device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
