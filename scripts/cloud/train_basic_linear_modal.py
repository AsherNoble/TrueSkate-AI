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
        line_fit=bool(payload.get("line_fit", False)),
        irls_iterations=int(payload.get("irls_iterations") or 3),
        huber_delta=float(payload.get("huber_delta") or .02),
        knots=int(payload.get("knots") or 2),
    )


def _payload_dataset_kwargs(payloads) -> dict:
    """Resolve the dataset shape a set of checkpoints was trained against.

    Evaluation used to build every dataset at the library default, which would
    silently feed a checkpoint trained at another width the wrong input rather
    than failing.  Checkpoints combined into one ensemble must agree.

    ``knots`` is resolved here for the same reason and in the same place: the
    model is rebuilt from ``payload["knots"]`` while the dataset used to default
    to 2, so a k=3 checkpoint was scored against a 2-knot target and threw a
    shape error only after the whole corpus had loaded.  One helper for both
    means a new evaluator cannot pick up half the contract.
    """
    payloads = list(payloads)
    sizes = {(int(payload.get("image_width") or 128), int(payload.get("image_height") or 288))
             for payload in payloads}
    if len(sizes) != 1:
        raise ValueError(f"checkpoints disagree on decode resolution: {sorted(sizes)}")
    knots = {int(payload.get("knots") or 2) for payload in payloads}
    if len(knots) != 1:
        raise ValueError(f"checkpoints disagree on knot count: {sorted(knots)}")
    width, height = sizes.pop()
    return {"image_width": width, "image_height": height, "knots": knots.pop()}


def _require_two_knots(kwargs: dict, evaluator: str) -> dict:
    """Fail loudly in evaluators whose bodies hardcode the 5-wide layout.

    Resolving ``knots`` from the payload made k=3 datasets *constructible*, which
    is not the same as making every evaluator k=3-correct.  Bodies that slice
    ``[:, :2]`` / ``[:, 2:4]`` / ``[:, 4]`` as start/end/duration read the
    interior knot and a coordinate as "duration" under k=3 — previously a shape
    error, and after the sweep a complete, plausible, mislabelled artefact.  A
    silent misreport is far worse than the crash it replaced, so these raise
    until they are made knot-general.
    """
    if kwargs["knots"] != 2:
        raise ValueError(
            f"{evaluator} decodes a fixed start/end/duration layout and cannot score a "
            f"{kwargs['knots']}-knot checkpoint; make it knot-general before using it here")
    return kwargs


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
                 evaluate_test: bool = True, fresh_stratify_by_device: bool = False,
                 line_fit: bool = False, irls_iterations: int = 3, huber_delta: float = .02,
                 image_width: int = 128, image_height: int = 288, knots: int = 2) -> dict:
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
        line_fit=line_fit,
        irls_iterations=irls_iterations,
        huber_delta=huber_delta,
        image_width=image_width,
        image_height=image_height,
        knots=knots,
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
                     evaluate_test: bool = True, fresh_stratify_by_device: bool = False,
                     line_fit: bool = False, irls_iterations: int = 3, huber_delta: float = .02,
                     image_width: int = 128, image_height: int = 288, knots: int = 2) -> dict:
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
        line_fit=line_fit,
        irls_iterations=irls_iterations,
        huber_delta=huber_delta,
        image_width=image_width,
        image_height=image_height,
        knots=knots,
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
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_require_two_knots(_payload_dataset_kwargs([payload]), "evaluate_refinement"))
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
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
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
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
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
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_require_two_knots(_payload_dataset_kwargs([payload]), "audit_endpoint_residuals"))
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


@app.function(image=image, cpu=8.0, timeout=3 * 3600, memory=16384,
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
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs(payloads))
    if fresh_holdout_source is None:
        _train, val_indices, test_indices = split_by_command(data, seed=seed)
    else:
        _train, val_indices, test_indices = _trainer().split_with_fresh_command_holdout(
            data, fresh_source=fresh_holdout_source, seed=seed,
            stratify_by_device=fresh_stratify_by_device,
        )
    # Validation selection and one final ensemble forward pass are small
    # compared with training.  Keeping this CPU-backed prevents the verdict
    # from waiting behind scarce GPU capacity after all checkpoints are ready.
    device = torch.device("cpu")
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


@app.function(image=image, cpu=8.0, timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def evaluate_bias_correction(data_subdir: str, checkpoint_name: str, *,
                             batch_size: int = 8, statistic: str = "mean",
                             seed: int | None = None,
                             fresh_holdout_source: str | None = None,
                             fresh_stratify_by_device: bool | None = None) -> dict:
    """Fit the along-path end bias on validation, apply it to test, once.

    The correction is a scalar, so the headline is deliberately NOT the accuracy
    delta: at n~153 a 3-clip change sits inside its own confidence interval
    (EQ-001 red team).  What this reports instead is the paired evidence — which
    clips flipped, in which direction, and the exact McNemar p — plus the
    perpendicular error distribution, because EQ-008 showed the predicted-chord
    operator only stands in for the autopsy's commanded-chord one while that
    distribution stays near the sd it was measured at.

    **Read this before believing the output.** Exact two-sided McNemar needs
    b>=6 with c=0 to clear p<0.05.  The journal's own estimate for this
    correction is ~2.6 gained / ~0.9 lost, so this run is expected to land near
    p=0.375 and *cannot* resolve whether the correction helps on a 153-clip
    split.  It establishes the operator's behaviour and the direction of the
    flips; significance needs EQ-007's >=3,000-command holdout.

    Every split parameter defaults from the checkpoint payload rather than from
    an argument, and the corpus is fingerprint-matched against the one the
    checkpoint trained on.  Without that, a corpus that has gained samples since
    training silently reshuffles the split, and "test" quietly fills with
    commands the checkpoint was fit on — inflating baseline and corrected
    numbers together, in a way the paired test cannot reveal.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_bias import (
        along_path_fit_key, discordant_pairs, fit_along_path_bias, mcnemar_exact_p,
        perpendicular_error, signed_along_path_error,
    )
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_training import (
        RECOVERY_DURATION_TOLERANCE_S, RECOVERY_ENDPOINT_TOLERANCE,
        basic_linear_metrics, basic_linear_recovery_records,
    )

    trainer = _trainer()
    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    # Split identity belongs to the checkpoint, not to this call.  Arguments may
    # only override where the payload is silent.
    seed = payload.get("split_seed") if seed is None else seed
    if seed is None:
        raise ValueError("checkpoint records no split_seed and none was supplied")
    if fresh_holdout_source is None:
        fresh_holdout_source = payload.get("fresh_holdout_source")
    if fresh_stratify_by_device is None:
        fresh_stratify_by_device = bool(payload.get("fresh_stratify_by_device"))
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    fingerprint = trainer._fingerprint(data.sample_paths)
    trained_on = payload.get("dataset_fingerprint")
    if trained_on and trained_on != fingerprint:
        raise ValueError(
            "corpus does not match the one this checkpoint was split on "
            f"({fingerprint} vs {trained_on}); the split would silently differ and 'test' could "
            "contain trained-on commands")
    if fresh_holdout_source is None:
        _train, val_indices, test_indices = split_by_command(data, seed=seed)
    else:
        _train, val_indices, test_indices = trainer.split_with_fresh_command_holdout(
            data, fresh_source=fresh_holdout_source, seed=seed,
            stratify_by_device=fresh_stratify_by_device,
        )
    recorded_sizes = payload.get("split_sizes") or {}
    for name, indices in (("validation", val_indices), ("test", test_indices)):
        if name in recorded_sizes and recorded_sizes[name] != len(indices):
            raise ValueError(
                f"re-derived {name} split has {len(indices)} clips but the checkpoint recorded "
                f"{recorded_sizes[name]}; the split is not the one that was trained against")
    device = torch.device("cpu")
    model = _model_from_payload(payload, torch).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    def loader(indices):
        return DataLoader(Subset(data, indices), batch_size=batch_size)

    validation_records = basic_linear_recovery_records(model, loader(val_indices), device)
    # fit_on is derived from the actual index set, so an artefact cannot claim a
    # provenance it does not have.
    correction = fit_along_path_bias(validation_records, statistic=statistic, axis="predicted",
                                     fit_on=along_path_fit_key("validation", val_indices))
    # The autopsy's operator, for comparison only.  apply() refuses a
    # commanded-axis fit, so this exists purely to answer EQ-002's question of
    # whether the two estimators agree on real records.
    commanded = fit_along_path_bias(validation_records, statistic=statistic, axis="commanded",
                                    fit_on=along_path_fit_key("validation", val_indices))

    # The dataset is deterministic, shuffle is off and Subset preserves order, so
    # these two passes are element-wise the same clips: the pairing is real.
    uncorrected_records = basic_linear_recovery_records(model, loader(test_indices), device)
    corrected_records = basic_linear_recovery_records(model, loader(test_indices), device,
                                                      correction=correction)
    before = [record["recovered"] for record in uncorrected_records]
    after = [record["recovered"] for record in corrected_records]
    gained, lost = discordant_pairs(before, after)
    baseline = basic_linear_metrics(model, loader(test_indices), device)
    corrected = basic_linear_metrics(model, loader(test_indices), device, correction=correction)

    def distribution(values):
        values = [value for value in values if value is not None]
        if not values:
            return {"samples": 0}
        return {"samples": len(values), "sd": float(np.std(values)),
                "median": float(np.median(values)), "p90": float(np.quantile(values, .90)),
                "p99": float(np.quantile(values, .99))}

    output = {
        "checkpoint": checkpoint_name,
        "data_subdir": data_subdir,
        "seed": seed,
        "statistic": statistic,
        "fresh_holdout_source": fresh_holdout_source,
        "fresh_stratify_by_device": fresh_stratify_by_device,
        "dataset_fingerprint": fingerprint,
        "endpoint_tolerance": RECOVERY_ENDPOINT_TOLERANCE,
        "duration_tolerance_s": RECOVERY_DURATION_TOLERANCE_S,
        "correction": {"shift": correction.shift, "samples": correction.samples,
                       "axis": correction.axis, "fit_on": correction.fit_on},
        "commanded_axis_shift": commanded.shift,
        "axis_disagreement": abs(correction.shift - commanded.shift),
        "baseline_recovery": baseline["gesture_recovery_accuracy"],
        "corrected_recovery": corrected["gesture_recovery_accuracy"],
        "test_samples": len(before),
        # The headline pair.  A net gain with a large p is a coin flip, not a win.
        "gained": gained, "lost": lost, "mcnemar_p": mcnemar_exact_p(gained, lost),
        "mcnemar_note": "b>=6 with c=0 is required for p<0.05; this split cannot show significance",
        # EQ-010: the axis transfer scales with the SQUARE of this.  It is a
        # folded (nonnegative) magnitude, so its sd is not a signed sd.
        "test_perpendicular_magnitude": distribution(
            [perpendicular_error(record) for record in uncorrected_records]),
        "validation_along": distribution(
            [signed_along_path_error(record) for record in validation_records]),
        "test_along_uncorrected": distribution(
            [signed_along_path_error(record) for record in uncorrected_records]),
        "baseline_metrics": baseline,
        "corrected_metrics": corrected,
    }
    label = f"{Path(checkpoint_name).stem}_{statistic}_seed{seed}"
    (Path("/models") / f"basic_linear_bias_correction_{label}.json").write_text(
        json.dumps(output, indent=2),
    )
    models.commit()
    return {key: value for key, value in output.items()
            if key not in {"baseline_metrics", "corrected_metrics"}}


@app.local_entrypoint()
def main(data_subdir: str, run_label: str = "baseline", epochs: int = 40,
         batch_size: int = 8, lr: float = 1e-3, seed: int = 0,
         base_channels: int = 16, split_strategy: str = "command",
         cache_frames: bool = True, split_seed: int | None = None,
         map_weight: float = 0.0, start_onset: float = .24,
         start_sigma: float = .05, end_onset: float = .24,
         temporal_mixer: bool = False, trajectory_weight: float = 0.0,
         trajectory_track: bool = False, fresh_holdout_source: str | None = None,
         evaluate_test: bool = True, fresh_stratify_by_device: bool = False,
         line_fit: bool = False, irls_iterations: int = 3, huber_delta: float = .02,
         image_width: int = 128, image_height: int = 288, knots: int = 2) -> None:
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
        line_fit=line_fit,
        irls_iterations=irls_iterations,
        huber_delta=huber_delta,
        image_width=image_width,
        image_height=image_height,
        knots=knots,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.function(image=image, gpu="any", timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def autopsy_failures(data_subdir: str, checkpoint_name: str, *, seed: int = 0,
                     batch_size: int = 8, fresh_holdout_source: str | None = None,
                     fresh_stratify_by_device: bool = False, label: str = "autopsy",
                     partition: str = "test") -> dict:
    """Classify why individual held-out clips fail, not merely how often.

    Recovery percentage cannot distinguish a model that missed visible evidence
    from a label the pixels never supported, yet the two demand opposite fixes.
    For every failing endpoint this measures the distance from the *commanded*
    point to the nearest rendered trail pixel anywhere in the clip:

    * small  -> the trail is where the label says; the model misread it.
    * large  -> no rendered evidence there at all; robust decoding cannot help.

    Positions come only from the command manifest and the model, so this adds
    no label leakage; the colour mask is used to locate evidence, never to
    define a target.
    """
    import json as _json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_require_two_knots(_payload_dataset_kwargs([payload]), "autopsy_failures"))
    if fresh_holdout_source is None:
        _train, val_indices, evaluated_indices = split_by_command(data, seed=seed)
    else:
        _train, val_indices, evaluated_indices = _trainer().split_with_fresh_command_holdout(
            data, fresh_source=fresh_holdout_source, seed=seed,
            stratify_by_device=fresh_stratify_by_device,
        )
    # Any correction derived from this report has to be fit on validation to be
    # usable; measuring it on test and applying it there would be test-set
    # tuning dressed up as a diagnostic.
    if partition == "validation":
        test_indices = val_indices
    elif partition == "test":
        test_indices = evaluated_indices
    else:
        raise ValueError("partition must be 'validation' or 'test'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model_from_payload(payload, torch).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    def trail_evidence(frames: torch.Tensor) -> torch.Tensor:
        """Per-frame orange-trail response, background-differenced."""
        steps = frames.shape[1]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        return ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

    records: list[dict] = []
    loader = DataLoader(Subset(data, test_indices), batch_size=batch_size)
    cursor = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        target = batch["target"].to(device)
        with torch.no_grad():
            prediction, start_scores, end_scores = model.forward_with_scores(frames)
        evidence = trail_evidence(frames)
        steps, height, width = evidence.shape[1:]
        xa = torch.linspace(0., 1., width, device=device)
        ya = torch.linspace(0., 1., height, device=device)
        grid = torch.stack((xa[None, :].expand(height, width),
                            ya[:, None].expand(height, width)), dim=2).reshape(-1, 2)
        # A trail pixel is one clearly above this clip's own background response.
        strong = evidence.flatten(2) > (evidence.flatten(2).amax(dim=2, keepdim=True) * .25).clamp_min(1e-6)
        for item in range(len(target)):
            index = test_indices[cursor + item]
            meta = data._meta(data.sample_paths[index])
            start_error = float(torch.linalg.vector_norm(prediction[item, :2] - target[item, :2]))
            end_error = float(torch.linalg.vector_norm(prediction[item, 2:4] - target[item, 2:4]))
            duration_error = float(torch.abs(prediction[item, 4] - target[item, 4]))
            recovered = start_error <= .03 and end_error <= .03 and duration_error <= .10
            any_strong = strong[item].any(dim=1)

            def nearest(point: torch.Tensor) -> dict:
                """Distance from a commanded point to the nearest trail pixel."""
                best, best_step = float("inf"), -1
                for step in range(steps):
                    if not bool(any_strong[step]):
                        continue
                    candidates = grid[strong[item, step]]
                    distance = float(torch.linalg.vector_norm(candidates - point[None, :], dim=1).min())
                    if distance < best:
                        best, best_step = distance, step
                return {"distance": best, "frame": best_step}

            commanded_start = nearest(target[item, :2])
            commanded_end = nearest(target[item, 2:4])
            records.append({
                "sample": str(data.sample_paths[index].relative_to(data.root)),
                "device": str(meta.get("device", "unknown")),
                "recovered": recovered,
                "start_error": start_error, "end_error": end_error,
                "duration_error": duration_error,
                "commanded": [float(v) for v in target[item, :5].cpu()],
                "predicted": [float(v) for v in prediction[item, :5].cpu()],
                # The decisive discriminator, per endpoint.
                "trail_gap_start": commanded_start["distance"],
                "trail_gap_end": commanded_end["distance"],
                "trail_frame_start": commanded_start["frame"],
                "trail_frame_end": commanded_end["frame"],
                "trail_frames_present": int(any_strong.sum()),
                # Where the model's end attention actually peaked, to separate a
                # misread from a collapse onto the other endpoint or the middle.
                "end_score_peak_frame": int(end_scores[item].flatten(1).amax(dim=1).argmax()),
                "start_score_peak_frame": int(start_scores[item].flatten(1).amax(dim=1).argmax()),
            })
        cursor += len(target)

    # Decompose every endpoint error along and perpendicular to the commanded
    # path.  A systematic along-path component is a bias (cheap to remove); a
    # perpendicular scatter is variance (needs better localisation).  The two
    # demand different fixes, and the aggregate error hides which one this is.
    for record in records:
        x0, y0, x1, y1, _duration = record["commanded"]
        direction = np.array([x1 - x0, y1 - y0], dtype=float)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
        for name, commanded, predicted in (
            ("start", np.array([x0, y0]), np.array(record["predicted"][:2])),
            ("end", np.array([x1, y1]), np.array(record["predicted"][2:4])),
        ):
            offset = predicted - commanded
            along = float(offset @ direction)
            record[f"{name}_along"] = along
            record[f"{name}_perp"] = float(np.linalg.norm(offset - along * direction))

    failures = [record for record in records if not record["recovered"]]
    gaps = [record["trail_gap_end"] for record in failures if record["end_error"] > .03]
    summary = {
        "checkpoint": checkpoint_name,
        "partition": partition,
        "test_samples": len(records),
        "failures": len(failures),
        "recovery": 1 - len(failures) / len(records),
        "median_trail_gap_all": float(np.median([r["trail_gap_end"] for r in records])),
        "along_perp_all": {
            f"{name}_{statistic}": float(function([r[f"{name}_{component}"] for r in records]))
            for name in ("start", "end")
            for component, statistic, function in (
                ("along", "along_mean", np.mean), ("along", "along_median", np.median),
                ("perp", "perp_median", np.median),
            )
        },
        "all_records": records,
        "failed_end_trail_gaps": sorted(gaps),
        "failing_records": failures,
    }
    (Path("/models") / f"basic_linear_{label}.json").write_text(_json.dumps(summary, indent=2))
    models.commit()
    return {key: value for key, value in summary.items() if key != "failing_records"}
