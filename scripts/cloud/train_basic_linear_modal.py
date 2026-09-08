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
# `gpu="any"` draws from {T4, L4, A10} and the draw is ~2.7x in epoch time (29 s vs 78 s
# measured on the same 2k config, 2026-08-21) AND changes the cuDNN algorithm choice, so
# identical-seed runs stop being bit-comparable.  For a SWEEP, pin this to one type so the
# settings are compared on one piece of hardware; leave it "any" for one-off runs, where
# a scarce named type can queue for hours without a container.
TRAIN_GPU = os.environ.get("MODAL_TRAIN_GPU", "any")
MODELS_VOLUME = "trueskate-models"

app = modal.App("trueskate-basic-linear")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0")
    # gesture_sampling imports the shared CMA-ES bounds, which transitively
    # imports the device gesture module.  The trainer never opens a WebDriver
    # session, but that module declares Selenium classes at import time.
    .pip_install("torch", "opencv-python-headless", "numpy", "scipy", "selenium")
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


def _training_inputs(data_subdir: str, experiment_manifest_name: str | None,
                     shard_manifest_name: str | None) -> tuple[Path, Path | None]:
    """Resolve directory or sequential-shard backed inputs for one run."""
    volume_root = Path("/corpus") / data_subdir
    if shard_manifest_name is None:
        return volume_root, (volume_root / experiment_manifest_name
                             if experiment_manifest_name else None)
    if experiment_manifest_name is not None:
        raise ValueError("shard_manifest_name already binds its experiment manifest")
    from trueskate_ai.data.sequential_shards import (
        materialize_sequential_shards, read_shard_manifest,
    )
    shard_path = volume_root / shard_manifest_name
    shard_payload = read_shard_manifest(shard_path)
    destination = Path("/tmp") / f"model1_shards_{shard_payload['fingerprint'][-16:]}"
    materialize_sequential_shards(shard_path, destination, verify_samples=False)
    return destination, shard_path.parent / shard_payload["experiment_manifest"]


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
# A full 13k-clip Model 1 corpus caches roughly 46 GiB of decoded RGB frames.
# Keep headroom for the model, batches, and Python runtime so the one-time cache
# avoids re-decoding every video on every epoch.  The observed full-corpus run
# takes just over eight hours, so leave a 50% timeout margin.  An atomic resume
# checkpoint is also committed after every completed epoch; a provider timeout
# can therefore cost at most the current epoch, never the entire run.
@app.function(image=image, gpu=TRAIN_GPU, timeout=24 * 3600, memory=65536,
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
                 image_width: int = 128, image_height: int = 288, knots: int = 2,
                 max_grad_norm: float | None = None,
                 experiment_manifest_name: str | None = None,
                 shard_manifest_name: str | None = None,
                 record_train_metrics: bool = False) -> dict:
    if shard_manifest_name is not None and cache_frames:
        raise ValueError("sequential shards require cache_frames=False; decode from staged local SSD "
                         "instead of retaining a >64 GiB large-rung corpus in RAM")
    trainer = _trainer()
    training_root, experiment_path = _training_inputs(
        data_subdir, experiment_manifest_name, shard_manifest_name,
    )
    checkpoint = Path("/models") / f"basic_linear_{run_label}.pth"
    resume_checkpoint = Path("/models") / f"basic_linear_{run_label}.resume.pth"
    payload = trainer.train(
        data=training_root,
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
        max_grad_norm=max_grad_norm,
        base_channels=base_channels,
        split_strategy=split_strategy,
        cache_frames=cache_frames,
        experiment_manifest=experiment_path,
        record_train_metrics=record_train_metrics,
        resume_path=resume_checkpoint,
        checkpoint_callback=models.commit,
    )
    result = {key: value for key, value in payload.items() if key != "state_dict"}
    result["checkpoint"] = checkpoint.name
    result["run_label"] = run_label
    (Path("/models") / f"basic_linear_{run_label}.json").write_text(json.dumps(result, indent=2))
    models.commit()
    return result


@app.function(image=image, cpu=8.0, timeout=24 * 3600, memory=16384,
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
                     image_width: int = 128, image_height: int = 288, knots: int = 2,
                     max_grad_norm: float | None = None,
                     experiment_manifest_name: str | None = None,
                     shard_manifest_name: str | None = None,
                     record_train_metrics: bool = False) -> dict:
    """Scheduler-independent execution fallback for the same compact protocol.

    This is intentionally a separate function rather than silently removing a
    GPU request.  The data split, model, optimiser and acceptance metric remain
    identical; only the hardware differs, and the result is separately labelled.
    """
    if shard_manifest_name is not None and cache_frames:
        raise ValueError("sequential shards require cache_frames=False")
    trainer = _trainer()
    training_root, experiment_path = _training_inputs(
        data_subdir, experiment_manifest_name, shard_manifest_name,
    )
    checkpoint = Path("/models") / f"basic_linear_{run_label}.pth"
    resume_checkpoint = Path("/models") / f"basic_linear_{run_label}.resume.pth"
    payload = trainer.train(
        data=training_root,
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
        max_grad_norm=max_grad_norm,
        base_channels=base_channels,
        split_strategy=split_strategy,
        cache_frames=cache_frames,
        experiment_manifest=experiment_path,
        record_train_metrics=record_train_metrics,
        resume_path=resume_checkpoint,
        checkpoint_callback=models.commit,
    )
    result = {key: value for key, value in payload.items() if key != "state_dict"}
    result["checkpoint"] = checkpoint.name
    result["run_label"] = run_label
    result["execution_hardware"] = "cpu"
    (Path("/models") / f"basic_linear_{run_label}.json").write_text(json.dumps(result, indent=2))
    models.commit()
    return result


@app.function(image=image, gpu=TRAIN_GPU, timeout=3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def evaluate_test_once(data_subdir: str, checkpoint_name: str, *, seed: int = 0,
                       batch_size: int = 8) -> dict:
    """Score ONE validation-selected checkpoint on the test split, once.

    Deliberately has no grid, no variants and no selection of any kind: a sweep
    is selected on validation and then exactly one candidate is brought here.
    Every other evaluator in this file either sweeps a knob on test
    (``evaluate_refinement``) or blends candidates (``evaluate_checkpoint_ensemble``),
    which makes them unusable as the final look at a held-out set.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_training import basic_linear_metrics

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    train_indices, val_indices, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    model = _model_from_payload(payload, torch).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    test = basic_linear_metrics(model, DataLoader(Subset(data, test_indices),
                                                  batch_size=batch_size), device)
    output = {"checkpoint": checkpoint_name,
              "split_sizes": [len(train_indices), len(val_indices), len(test_indices)],
              "split_seed": seed,
              "knots": payload.get("knots"),
              "line_fit": payload.get("line_fit"),
              "trajectory_weight": payload.get("trajectory_map_weight"),
              "validation_reported_at_train_time": payload.get("validation"),
              "test": test}
    # Persist AND print: `modal run module::function` does not surface a remote
    # return value, so a result that only lives in the return is simply lost.
    (Path("/models") / f"{Path(checkpoint_name).stem}_test_once.json").write_text(
        json.dumps(output, indent=2, sort_keys=True))
    models.commit()
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


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
    from trueskate_ai.vision.basic_linear_training import knot_component_labels

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
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
    # Every knot, not the first two: at K>2 ``values[:, :4]`` is knot 0 and the
    # INTERIOR knot, so "start_end" would silently compare the wrong pair.
    labels = knot_component_labels(values.shape[1])
    errors = np.linalg.norm(values[:, :len(labels) - 1].reshape(len(values), -1, 2), axis=2)
    first, last = errors[:, 0], errors[:, -1]
    output = {
        "checkpoint": checkpoint_name,
        "test_samples": int(len(values)),
        "knots": int(errors.shape[1]),
        "mean_signed_residual": dict(zip(labels, values.mean(axis=0).tolist())),
        "median_signed_residual": dict(zip(labels, np.median(values, axis=0).tolist())),
        # "start"/"end" keep their MVP-2 meaning: the first and last knot.
        "start_end_error_correlation": float(np.corrcoef(first, last)[0, 1]),
        "start_end_both_fail": float(np.mean((first > .03) & (last > .03))),
        "start_only_fail": float(np.mean((first > .03) & (last <= .03))),
        "end_only_fail": float(np.mean((first <= .03) & (last > .03))),
        "per_knot_fail": [float(np.mean(errors[:, index] > .03)) for index in range(errors.shape[1])],
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


@app.function(image=image, cpu=2.0, timeout=3600, memory=8192,
              volumes={"/corpus": corpus, "/models": models})
def audit_clip_headroom(data_subdir: str, *, sequence_length: int = 32) -> dict:
    """How much clip remains after each gesture's commanded liftoff?

    Metadata only — reads ``meta.json`` and never decodes a frame, so this is
    cheap enough to run over the whole corpus.

    EQ-003 found duration failures concentrate where the commanded liftoff sits
    at or past the end of the sampled window (headroom < 2 frames: 2/6 failed,
    versus 1/300 above it, Fisher p=9.6e-4).  ``frame_times`` are touch-start
    relative, so "headroom" is simply ``frame_times[-1] - duration`` expressed in
    sampled frames.  A model cannot read a liftoff that is not in its input, and
    no amount of capacity fixes that — which is why this measurement decides
    whether the fix is worth a retrain at all.
    """
    import json as _json
    import numpy as np

    root = Path("/corpus") / data_subdir
    lead_in, tail_frames, durations, spacings = [], [], [], []
    skipped = 0
    samples = sorted(path for path in root.rglob("meta.json"))
    for path in samples:
        try:
            meta = _json.loads(path.read_text())
            times = np.asarray(meta["frame_times"], dtype=float)
            duration = float(meta["duration"])
        except Exception:
            skipped += 1
            continue
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
            skipped += 1
            continue
        # The loader subsamples `sequence_length` frames evenly across the clip,
        # so the effective spacing is the clip span over the sampled intervals.
        spacing = (times[-1] - times[0]) / max(sequence_length - 1, 1)
        if spacing <= 0:
            skipped += 1
            continue
        lead_in.append(-times[0] / spacing)
        tail_frames.append((times[-1] - duration) / spacing)
        durations.append(duration)
        spacings.append(spacing)

    tail = np.asarray(tail_frames)
    lead = np.asarray(lead_in)
    def describe(values):
        return {"min": float(values.min()), "p01": float(np.quantile(values, .01)),
                "median": float(np.median(values)), "p99": float(np.quantile(values, .99)),
                "max": float(values.max())}
    output = {
        "data_subdir": data_subdir,
        "sequence_length": sequence_length,
        "clips": len(tail),
        "skipped": skipped,
        "frame_spacing_s": describe(np.asarray(spacings)),
        "commanded_duration_s": describe(np.asarray(durations)),
        # Frames of clip remaining AFTER commanded liftoff.  Negative means the
        # liftoff is not in the clip at all.
        "tail_frames_after_liftoff": describe(tail),
        "lead_in_frames_before_touch": describe(lead),
        "clips_tail_below_2_frames": int((tail < 2).sum()),
        "clips_tail_below_2_fraction": float((tail < 2).mean()),
        "clips_tail_negative": int((tail < 0).sum()),
        "clips_tail_negative_fraction": float((tail < 0).mean()),
    }
    (Path("/models") / f"basic_linear_headroom_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=2.0, timeout=3600, memory=8192,
              volumes={"/corpus": corpus, "/models": models})
def audit_split_session_overlap(data_subdir: str, checkpoint_name: str, *,
                                seed: int | None = None,
                                fresh_holdout_source: str | None = None,
                                fresh_stratify_by_device: bool | None = None) -> dict:
    """Do train / validation / test share recording sessions?

    The holdout protocol is exact-COMMAND disjoint, which it enforces.  It says
    nothing about sessions, so the same recording — same park, lighting, board
    pose and camera state — can appear on both sides of the split.  That does
    not violate the protocol, but it bounds how much generalisation a held-out
    number demonstrates, and EQ-007's certification should choose this
    deliberately rather than inherit it.

    Metadata only: the dataset reads meta.json to build its keys and never
    decodes a frame unless indexed.
    """
    import json as _json
    import torch
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command

    trainer = _trainer()
    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    seed = payload.get("split_seed") if seed is None else seed
    if fresh_holdout_source is None:
        fresh_holdout_source = payload.get("fresh_holdout_source")
    if fresh_stratify_by_device is None:
        fresh_stratify_by_device = bool(payload.get("fresh_stratify_by_device"))
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=False,
                                  **_payload_dataset_kwargs([payload]))
    if fresh_holdout_source is None:
        train_indices, val_indices, test_indices = split_by_command(data, seed=seed)
    else:
        train_indices, val_indices, test_indices = trainer.split_with_fresh_command_holdout(
            data, fresh_source=fresh_holdout_source, seed=seed,
            stratify_by_device=fresh_stratify_by_device,
        )

    import re as _re
    _SESSION = _re.compile(r"^iPhone_\w+?_\d{8}_\d{6}$")

    def session_of(index):
        """Session identity from the PATH, matched by PATTERN not by position.

        `_segment_key` falls back to `legacy:<dir>` when meta carries no
        `session`, collapsing a whole corpus into one bucket — but a fixed path
        index is no better, because the layouts differ: the 2k corpus is
        `<session>/<park>/sample` while the mixed corpus is
        `<source>/<session>/<park>/sample`, so "parts[1]" is the session in one
        and the PARK in the other.  Match the session directory's shape instead,
        and fall back to the full relative parent so a miss can never silently
        collapse distinct recordings into one.
        """
        parts = data.sample_paths[index].relative_to(data.root).parts
        for part in parts:
            if _SESSION.match(part):
                return part
        return "/".join(parts[:-1]) or parts[0]

    def sessions(indices):
        return {session_of(index) for index in indices}

    train, validation, test = sessions(train_indices), sessions(val_indices), sessions(test_indices)
    # How much of the evaluated data sits in a session the model also trained on?
    train_sessions = train
    test_in_train = sum(1 for index in test_indices if session_of(index) in train_sessions)
    val_in_train = sum(1 for index in val_indices if session_of(index) in train_sessions)
    output = {
        "data_subdir": data_subdir,
        "checkpoint": checkpoint_name,
        "seed": seed,
        "fresh_holdout_source": fresh_holdout_source,
        "clips": {"train": len(train_indices), "validation": len(val_indices), "test": len(test_indices)},
        "distinct_sessions": {"train": len(train), "validation": len(validation), "test": len(test)},
        "session_overlap": {
            "train_validation": len(train & validation),
            "train_test": len(train & test),
            "validation_test": len(validation & test),
        },
        "test_clips_in_a_training_session": test_in_train,
        "test_clips_in_a_training_session_fraction": test_in_train / max(len(test_indices), 1),
        "validation_clips_in_a_training_session": val_in_train,
        "validation_clips_in_a_training_session_fraction": val_in_train / max(len(val_indices), 1),
        "sessions_unique_to_test": sorted(test - train - validation),
    }
    (Path("/models") / f"basic_linear_session_overlap_{Path(checkpoint_name).stem}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=4.0, timeout=3600, memory=8192,
              volumes={"/corpus": corpus, "/models": models})
def audit_clip_frame_counts(data_subdir: str, *, limit: int = 0) -> dict:
    """Does each clip's video actually hold as many frames as its labels claim?

    `frame_times` is SYNTHESISED by the aligner from constants
    (`align_xctest_traces.py`: `[i/output_fps - pre_s for i in range(max_frames)]`),
    so it asserts a schedule rather than measuring one.  Nothing verifies the
    extracted mp4 against it, and `_decode_even_frames` stretches whatever frames
    exist across the requested sequence length.  A short video therefore yields a
    clip whose pixels are time-compressed relative to labels that still claim the
    nominal schedule — silently, and invisibly to any metadata-only audit.

    Decodes each clip to count frames: the header's CAP_PROP_FRAME_COUNT is
    frequently estimated from duration x fps and can itself be off by one, which
    is exactly the magnitude under test.
    """
    import json as _json
    import cv2
    import numpy as np

    from concurrent.futures import ThreadPoolExecutor

    root = Path("/corpus") / data_subdir
    rows, missing_video, unreadable = [], 0, 0

    def inspect(meta_path):
        """One clip: claimed frame count from metadata, actual from the container."""
        sample = meta_path.parent
        try:
            meta = _json.loads(meta_path.read_text())
            claimed = len(meta["frame_times"])
        except Exception:
            return ("unreadable", None)
        video = sample / "frames.mp4"
        if not video.exists():
            return ("missing", None)
        capture = cv2.VideoCapture(str(video))
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        # CAP_PROP_FRAME_COUNT is often estimated from duration x fps rather than
        # counted, so it can be off by one on its own.  Decode-and-count is the
        # only trustworthy number when the whole claim is an off-by-one.
        decoded = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded += 1
        capture.release()
        times = meta["frame_times"]
        return ("ok", {"sample": str(sample.relative_to(root)), "claimed": claimed,
                       "actual": decoded, "reported": reported, "fps": fps,
                       # Does the decoded video SPAN the window the labels assert?
                       "video_span_s": (decoded - 1) / fps if fps > 0 else None,
                       "label_span_s": float(times[-1]) - float(times[0])})

    # FUSE latency dominates and these are header reads, so fan out widely.
    paths = sorted(root.rglob("meta.json"))
    if limit:
        paths = paths[:limit]
    with ThreadPoolExecutor(max_workers=32) as pool:
        for status, row in pool.map(inspect, paths):
            if status == "ok":
                rows.append(row)
            elif status == "missing":
                missing_video += 1
            else:
                unreadable += 1

    claimed = np.array([r["claimed"] for r in rows])
    actual = np.array([r["actual"] for r in rows])
    short = actual < claimed
    output = {
        "data_subdir": data_subdir,
        "clips_with_video": len(rows),
        "missing_video": missing_video,
        "unreadable_meta": unreadable,
        "claimed_frames": {"min": int(claimed.min()), "max": int(claimed.max())} if len(rows) else {},
        "actual_frames": {"min": int(actual.min()), "max": int(actual.max())} if len(rows) else {},
        "header_disagrees_with_decode": int(sum(
            1 for r in rows if r["reported"] != r["actual"])),
        "clips_short": int(short.sum()),
        "clips_short_fraction": float(short.mean()) if len(rows) else 0.0,
        "shortfall_distribution": {
            str(int(deficit)): int((claimed - actual == deficit).sum())
            for deficit in sorted(set((claimed - actual).tolist()))
        },
        "timing": {
            "fps": sorted({round(r["fps"], 4) for r in rows}),
            "video_span_s": sorted({round(r["video_span_s"], 4) for r in rows}),
            "label_span_s": sorted({round(r["label_span_s"], 4) for r in rows}),
        },
        "worst_examples": sorted(
            ({"sample": r["sample"], "claimed": r["claimed"], "actual": r["actual"]}
             for r in rows if r["actual"] < r["claimed"]),
            key=lambda r: r["actual"] - r["claimed"])[:12],
    }
    (Path("/models") / f"basic_linear_frame_counts_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=2.0, timeout=3600, memory=8192,
              volumes={"/corpus": corpus, "/models": models})
def audit_corpus_coverage(data_subdir: str) -> dict:
    """What conditions does the corpus actually span — park, day, device, source?

    EQ-017 showed session identity is a weak nuisance (failures do not bunch by
    session) while the evaluated split is one park, one date and a four-hour
    window with 7 XR2 clips out of 153.  Coverage, not session-disjointness, is
    what bounds the held-out claim — so this measures what is available to
    evaluate against before EQ-007 predeclares its axes.

    Metadata only; no frames are decoded.
    """
    import json as _json
    import re
    from collections import Counter

    import datetime as _dt

    root = Path("/corpus") / data_subdir
    parks, devices, dates, sources, session_device = Counter(), Counter(), Counter(), Counter(), {}
    pairs = Counter()
    total = 0
    # Capture time from the manifest, not from the directory name: the date axis
    # is only meaningful if it reflects when the clip was recorded rather than
    # when it was staged.
    capture_times: dict[str, list[float]] = {}
    dirname_disagreements = 0
    for meta_path in sorted(root.rglob("meta.json")):
        relative = meta_path.parent.relative_to(root).parts
        if len(relative) < 3:
            continue
        source, session, park = relative[0], relative[1], relative[2]
        match = re.search(r"(iPhone_\w+?)_(\d{8})_(\d{6})$", session)
        device = match.group(1) if match else "unknown"
        date = match.group(2) if match else "unknown"
        total += 1
        parks[park] += 1
        devices[device] += 1
        dates[date] += 1
        sources[source] += 1
        pairs[f"{device}|{park}|{date}"] += 1
        session_device[f"{source}/{session}"] = device
        try:
            meta = _json.loads(meta_path.read_text())
            stamp = meta.get("gesture_start_monotonic")
            if stamp is not None:
                capture_times.setdefault(source, []).append(float(stamp))
                if match:
                    named = _dt.datetime.strptime(f"{match.group(2)}{match.group(3)}", "%Y%m%d%H%M%S")
                    observed = _dt.datetime.fromtimestamp(float(stamp))
                    if abs((observed - named).total_seconds()) > 24 * 3600:
                        dirname_disagreements += 1
        except Exception:
            pass

    output = {
        "data_subdir": data_subdir,
        "clips": total,
        "by_source": dict(sources.most_common()),
        "by_park": dict(parks.most_common()),
        "by_device": dict(devices.most_common()),
        "by_date": dict(dates.most_common()),
        "distinct_sessions": len(session_device),
        "sessions_by_device": dict(Counter(session_device.values()).most_common()),
        # The cell structure an evaluation could actually stratify over.
        "capture_time_span_by_source": {
            source: {
                "clips": len(values),
                "first": _dt.datetime.fromtimestamp(min(values)).isoformat(timespec="seconds"),
                "last": _dt.datetime.fromtimestamp(max(values)).isoformat(timespec="seconds"),
                "span_hours": round((max(values) - min(values)) / 3600, 2),
            }
            for source, values in sorted(capture_times.items())
        },
        "dirname_capture_time_disagreements": dirname_disagreements,
        "device_park_date_cells": len(pairs),
        "smallest_cells": dict(sorted(pairs.items(), key=lambda kv: kv[1])[:10]),
        "largest_cells": dict(pairs.most_common(10)),
    }
    (Path("/models") / f"basic_linear_coverage_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=8.0, timeout=2 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_trail_presence_threshold(data_subdir: str, checkpoint_name: str, *,
                                   clips: int = 200, batch_size: int = 8) -> dict:
    """Can trail PRESENCE recover the contact interval at any threshold?

    `trail_frames_present` is 32/32 on every one of 306 audited clips, so at the
    autopsy's threshold (0.25 x the clip's own max) the rendered trail is visible
    before touchdown and after liftoff.  If that is a threshold artefact, some
    stricter setting should make presence track the commanded contact window and
    duration becomes directly readable.  If no threshold separates them, the
    trail genuinely persists, duration is only recoverable from trail GEOMETRY
    rather than presence, and `trail_frames_present` should be renamed.

    Ground truth is the dataset's own `trajectory_mask` — `(t >= 0) & (t <=
    duration)` from the manifest — so this compares rendered evidence against the
    commanded contact interval without a model in the loop.
    """
    import json as _json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    indices = list(range(0, len(data), max(1, len(data) // max(clips, 1))))[:clips]

    def trail_evidence(frames: torch.Tensor) -> torch.Tensor:
        """Identical to the autopsy's per-frame orange-trail response."""
        steps = frames.shape[1]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        return ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

    fractions = [round(0.05 * step, 4) for step in range(1, 19)]  # 0.05 .. 0.90
    stats = {fraction: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "frames_flagged": 0}
             for fraction in fractions}
    total_frames = contact_frames = 0
    for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
        evidence = trail_evidence(batch["frames"])
        active = batch["trajectory_mask"].bool()
        peak = evidence.flatten(2).amax(dim=2)                       # [B, T]
        clip_max = peak.amax(dim=1, keepdim=True).clamp_min(1e-6)    # [B, 1]
        total_frames += active.numel()
        contact_frames += int(active.sum())
        for fraction in fractions:
            present = peak > (clip_max * fraction)
            entry_stats = stats[fraction]
            entry_stats["tp"] += int((present & active).sum())
            entry_stats["fp"] += int((present & ~active).sum())
            entry_stats["fn"] += int((~present & active).sum())
            entry_stats["tn"] += int((~present & ~active).sum())
            entry_stats["frames_flagged"] += int(present.sum())

    # THE BASELINE THAT DECIDES THIS. `frame_times` is a uniform synthesised grid
    # with a fixed 0.5s pre-roll, so the contact interval STARTS at frame 7 for
    # every clip and only its trailing edge varies.  A constant [7, E] mask uses
    # no pixels and no per-clip knowledge whatsoever.  If pixel evidence cannot
    # beat it, "trail presence carries contact information" is unsupported.
    constant_rows = []
    for end in range(8, 32):
        tp = fp = fn = tn = 0
        for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
            active = batch["trajectory_mask"].bool()
            window = torch.zeros_like(active)
            window[:, 7:end + 1] = True
            tp += int((window & active).sum())
            fp += int((window & ~active).sum())
            fn += int((~window & active).sum())
            tn += int((~window & ~active).sum())
        recall = tp / max(tp + fn, 1)
        constant_rows.append({
            "window": [7, end],
            "precision_contact": round(tp / max(tp + fp, 1), 4),
            "recall_contact": round(recall, 4),
            "balanced_accuracy": round(0.5 * (recall + tn / max(tn + fp, 1)), 4),
        })
    best_constant = max(constant_rows, key=lambda row: row["balanced_accuracy"])

    rows = []
    for fraction in fractions:
        entry_stats = stats[fraction]
        tp, fp, fn, tn = (entry_stats[key] for key in ("tp", "fp", "fn", "tn"))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({
            "threshold_fraction_of_clip_max": fraction,
            "frames_flagged_fraction": entry_stats["frames_flagged"] / max(total_frames, 1),
            "precision_contact": round(precision, 4),
            "recall_contact": round(recall, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
            "balanced_accuracy": round(
                0.5 * (recall + tn / max(tn + fp, 1)), 4),
        })
    best = max(rows, key=lambda row: row["balanced_accuracy"])
    output = {
        "data_subdir": data_subdir,
        "checkpoint": checkpoint_name,
        "clips_sampled": len(indices),
        "frames": total_frames,
        "contact_frame_fraction": round(contact_frames / max(total_frames, 1), 4),
        "autopsy_threshold_fraction": 0.25,
        "sweep": rows,
        "best_by_balanced_accuracy": best,
        "constant_window_baseline": constant_rows,
        "best_constant_window": best_constant,
        # The only comparison that matters: does pixel evidence beat no pixels?
        "evidence_beats_constant_window": best["balanced_accuracy"] > best_constant["balanced_accuracy"],
    }
    (Path("/models") / f"basic_linear_trail_threshold_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return {key: value for key, value in output.items() if key != "sweep"}


@app.function(image=image, cpu=8.0, timeout=2 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_liftoff_edge(data_subdir: str, checkpoint_name: str, *, clips: int = 300,
                       batch_size: int = 8) -> dict:
    """Can trail evidence locate LIFTOFF better than knowing nothing?

    EQ-016 showed frame-level presence loses to a constant window, because the
    contact interval starts at frame 7 in every clip and a third of the negatives
    are free.  Duration is decided entirely by the TRAILING edge, so that is what
    this measures: the estimated liftoff index against the commanded one, in
    frames, beside two baselines — predicting the corpus-mean liftoff index (no
    pixels, no per-clip knowledge) and the commanded index itself (oracle, zero
    by construction).

    Errors are reported in frames and in seconds, so they can be read directly
    against the 0.10 s duration gate and the 0.0731 s frame quantum.
    """
    import json as _json
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    indices = list(range(0, len(data), max(1, len(data) // max(clips, 1))))[:clips]

    def trail_evidence(frames: torch.Tensor) -> torch.Tensor:
        steps = frames.shape[1]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        return ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

    fractions = [round(0.05 * step, 4) for step in range(1, 19)]
    commanded_edges: list[float] = []
    estimated: dict[float, list[float]] = {fraction: [] for fraction in fractions}
    # Legacy and fresh are different corpora: legacy predates the anchor fix and
    # its trail is frequently already drawn during the lead-in, which poisons the
    # pre-touch reference.  The model's own test split is fresh-only, so an
    # aggregate over both is not comparable to it.
    sources = [data.sample_paths[index].relative_to(data.root).parts[0] for index in indices]
    for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
        active = batch["trajectory_mask"].bool()
        evidence = trail_evidence(batch["frames"])
        peak = evidence.flatten(2).amax(dim=2)
        normalised = peak / peak.amax(dim=1, keepdim=True).clamp_min(1e-6)
        steps = active.shape[1]
        grid = torch.arange(steps)
        for item in range(len(active)):
            # Commanded liftoff = the last frame the manifest calls contact.
            commanded_edges.append(float(grid[active[item]].max()) if bool(active[item].any()) else float("nan"))
            for fraction in fractions:
                above = normalised[item] > fraction
                # Search only at or after the lead-in: frames 0-6 are structurally
                # suppressed by the reference, so including them would hand the
                # estimator free credit (EQ-016).
                above[:7] = False
                estimated[fraction].append(float(grid[above].max()) if bool(above.any()) else float("nan"))

    commanded = np.array(commanded_edges, dtype=float)
    valid = ~np.isnan(commanded)
    mean_edge = float(np.nanmean(commanded))
    quantum_s = 2.2667 / 31

    source_array = np.array(sources)

    def summarise(estimate, mask=None):
        error = estimate - commanded
        keep = valid & ~np.isnan(error)
        if mask is not None:
            keep = keep & mask
        error = error[keep]
        absolute = np.abs(error)
        if not keep.any():
            return {"clips": 0}
        return {
            "clips": int(keep.sum()),
            "bias_frames": round(float(error.mean()), 3),
            "mae_frames": round(float(absolute.mean()), 3),
            "median_abs_frames": round(float(np.median(absolute)), 3),
            "p90_abs_frames": round(float(np.quantile(absolute, .9)), 3),
            "mae_seconds": round(float(absolute.mean() * quantum_s), 4),
            # A duration is recovered if the edge is within the 0.10s gate.
            "within_duration_gate": round(float((absolute * quantum_s <= 0.10).mean()), 4),
        }

    sweep = {str(fraction): summarise(np.array(estimated[fraction], dtype=float))
             for fraction in fractions}
    best_fraction = min(sweep, key=lambda key: sweep[key]["mae_frames"])
    baseline = summarise(np.full_like(commanded, mean_edge))
    output = {
        "data_subdir": data_subdir,
        "checkpoint": checkpoint_name,
        "clips_sampled": int(valid.sum()),
        "frame_quantum_s": round(quantum_s, 5),
        "commanded_edge": {"mean": round(mean_edge, 3),
                           "min": float(np.nanmin(commanded)), "max": float(np.nanmax(commanded)),
                           "sd": round(float(np.nanstd(commanded)), 3)},
        "constant_mean_edge_baseline": baseline,
        "best_evidence_threshold": best_fraction,
        "best_evidence": sweep[best_fraction],
        "evidence_beats_constant": sweep[best_fraction]["mae_frames"] < baseline["mae_frames"],
        # The comparison that decides whether the model is really "past" this:
        # only the fresh subset is the population the model's test split lives in.
        "by_source": {
            source: {
                "clips": int((source_array == source).sum()),
                "evidence": summarise(np.array(estimated[float(best_fraction)], dtype=float),
                                      source_array == source),
                "constant": summarise(np.full_like(commanded, mean_edge), source_array == source),
            }
            for source in sorted(set(sources))
        },
        "sweep": sweep,
    }
    (Path("/models") / f"basic_linear_liftoff_edge_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return {key: value for key, value in output.items() if key != "sweep"}


@app.function(image=image, cpu=8.0, timeout=2 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_liftoff_growth(data_subdir: str, checkpoint_name: str, *, clips: int = 300,
                         batch_size: int = 8, threshold: float = 0.35,
                         knee: float = 0.95) -> dict:
    """Liftoff from trail GROWTH rather than trail fade.

    EQ-025 estimated liftoff as the last frame whose peak evidence exceeded a
    threshold.  But `peak` is a spatial max, so it tracks the newest bright trail
    segment and then decays — that estimator is a fade timer, and on fresh clips
    it lost to a pixel-free constant on MAE.  The trail is cumulative, so the
    better-specified signal is EXTENT: it grows while the finger is down and
    plateaus at liftoff.  Liftoff is therefore the knee, not the fade.

    `threshold` and `knee` are PRE-COMMITTED (0.35 from EQ-016's balanced-accuracy
    optimum, chosen under a different objective; 0.95 as the plateau point) and
    the headline is a PAIRED sign test against the constant baseline — EQ-025's
    best-of-18 threshold selection and unpaired margin are not repeated.

    Fresh-source clips only: EQ-025 measured legacy and fresh to be different
    populations, and the model's test split is fresh-only.
    """
    import json as _json
    import math
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    fresh = [index for index in range(len(data))
             if data.sample_paths[index].relative_to(data.root).parts[0] == "fresh"]
    if not fresh:
        raise ValueError("no fresh-source clips found; this audit is fresh-only by design")
    indices = fresh[::max(1, len(fresh) // max(clips, 1))][:clips]

    def trail_evidence(frames: torch.Tensor) -> torch.Tensor:
        steps = frames.shape[1]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        return ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

    commanded, growth_edge, fade_edge = [], [], []
    argmax_edge, increase_edge = [], []
    for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
        active = batch["trajectory_mask"].bool()
        evidence = trail_evidence(batch["frames"]).flatten(2)          # [B, T, HW]
        clip_max = evidence.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        above = evidence > (clip_max * threshold)
        extent = above.sum(dim=2).float()                              # [B, T] trail area
        peak = (evidence / clip_max).amax(dim=2)                       # [B, T] fade signal
        steps = active.shape[1]
        grid = torch.arange(steps)
        for item in range(len(active)):
            commanded.append(float(grid[active[item]].max()) if bool(active[item].any())
                             else float("nan"))
            # GROWTH: first frame at which the trail has reached `knee` of its
            # final extent -- the plateau, i.e. the finger stopped adding to it.
            series = extent[item]
            ceiling = float(series.max())
            if ceiling <= 0:
                growth_edge.append(float("nan"))
                argmax_edge.append(float("nan"))
                increase_edge.append(float("nan"))
            else:
                reached = (series >= knee * ceiling) & (grid >= 7)
                growth_edge.append(float(grid[reached].min()) if bool(reached.any())
                                   else float("nan"))
                # argmax of extent: where the trail is largest, no floor, no knee.
                argmax_edge.append(float(int(series.argmax())))
                # The LITERAL spec: last frame at which extent still increases.
                rising = (series[1:] > series[:-1]) & (grid[1:] >= 7)
                increase_edge.append(float(grid[1:][rising].max()) if bool(rising.any())
                                     else float("nan"))
            # FADE: EQ-025's estimator, same clips, for a like-for-like contrast.
            lit = (peak[item] > threshold) & (grid >= 7)
            fade_edge.append(float(grid[lit].max()) if bool(lit.any()) else float("nan"))

    commanded = np.array(commanded, dtype=float)
    growth = np.array(growth_edge, dtype=float)
    fade = np.array(fade_edge, dtype=float)
    argmax_extent = np.array(argmax_edge, dtype=float)
    last_increase = np.array(increase_edge, dtype=float)
    # growth is early and fade is late by almost the same amount, so their
    # midpoint is the obvious untested member -- and a DIFFERENCE-based reader
    # cancels any clip-constant offset, which is the residual EQ-025 flagged.
    midpoint = (growth + fade) / 2
    # An INTEGER constant, so ties are possible and the sign test is like-for-like.
    constant = np.full_like(commanded, round(float(np.nanmean(commanded))))
    # Every arm scored on the SAME clips: a per-arm mask flatters whichever arm
    # drops its own degenerate cases.
    common = ~np.isnan(commanded)
    for series in (growth, fade, argmax_extent, last_increase, midpoint):
        common &= ~np.isnan(series)
    quantum_s = 2.1935 / 31   # the PIXEL quantum: the video spans 2.1935s (EQ-018)

    def summarise(estimate):
        error = (estimate - commanded)[common]
        absolute = np.abs(error)
        bias = float(error.mean())
        # A biased index reader loses to a mean-centred constant on MAE even when
        # it is MORE informative, so report the de-biased MAE too: that is the
        # resolution comparison, whereas raw MAE is a calibration comparison.
        debiased = np.abs(error - bias)
        return {"clips": int(common.sum()),
                "bias_frames": round(bias, 3),
                "mae_frames": round(float(absolute.mean()), 3),
                "debiased_mae_frames": round(float(debiased.mean()), 3),
                "median_abs_frames": round(float(np.median(absolute)), 3),
                "p90_abs_frames": round(float(np.quantile(absolute, .9)), 3),
                "within_duration_gate": round(float((absolute * quantum_s <= 0.10).mean()), 4)}

    def sign_test(a, b):
        """Two-sided paired sign test on |error|; ties dropped."""
        left, right = np.abs(a - commanded)[common], np.abs(b - commanded)[common]
        wins = int((left < right).sum())
        losses = int((left > right).sum())
        total = wins + losses
        if total == 0:
            return {"wins": 0, "losses": 0, "ties": int(common.sum()), "p": 1.0}
        extreme = max(wins, losses)
        tail = sum(math.comb(total, k) for k in range(extreme, total + 1)) / 2 ** total
        return {"wins": wins, "losses": losses, "ties": int(common.sum()) - total,
                "p": round(min(1.0, 2 * tail), 6)}

    output = {
        "data_subdir": data_subdir, "checkpoint": checkpoint_name,
        "precommitted": {"threshold": threshold, "knee": knee, "source": "fresh", "test": "paired sign"},
        "clips": len(indices), "frame_quantum_s": round(quantum_s, 5),
        "growth": summarise(growth), "fade": summarise(fade),
        "argmax_extent": summarise(argmax_extent), "last_increase": summarise(last_increase),
        "midpoint": summarise(midpoint), "constant": summarise(constant),
        # Is the growth estimator just pinned at its own floor?  memory
        # `sls-window-anchored-to-call-end` says the trace is already fully drawn
        # in frame_000 for ~half of samples; such a clip returns exactly 7.
        "growth_pinned_at_floor": {
            "count": int((growth[common] == 7).sum()),
            "fraction": round(float((growth[common] == 7).mean()), 4),
        },
        "growth_vs_constant": sign_test(growth, constant),
        "growth_vs_fade": sign_test(growth, fade),
        "midpoint_vs_constant": sign_test(midpoint, constant),
        "last_increase_vs_constant": sign_test(last_increase, constant),
    }
    (Path("/models") / f"basic_linear_liftoff_growth_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=8.0, timeout=2 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_duration_difference(data_subdir: str, checkpoint_name: str, *, clips: int = 400,
                              batch_size: int = 8, threshold: float = 0.35) -> dict:
    """Read DURATION as a difference of edges, not two absolute indices.

    Every estimator through EQ-026 compared a pixel-derived ABSOLUTE frame index
    against the label grid, so each paid the EQ-018 timebase skew and the `-ss`
    phase jitter in full.  A duration is a DIFFERENCE, and any clip-constant
    offset cancels in a difference — that is the one structural advantage the
    trained model has that none of these estimators were given.

    Both difference readers need a scale/offset (growth reads early, fade late),
    so the affine calibration is **cross-fitted**: fit on one half, score the
    other, and vice versa.  Fitting and scoring on the same clips would be the
    same in-sample error the constant baseline was criticised for in EQ-026.
    """
    import json as _json
    import math
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    fresh = [index for index in range(len(data))
             if data.sample_paths[index].relative_to(data.root).parts[0] == "fresh"]
    indices = fresh[::max(1, len(fresh) // max(clips, 1))][:clips]

    def trail_evidence(frames: torch.Tensor) -> torch.Tensor:
        steps = frames.shape[1]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        return ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

    durations, span_fade_growth, span_increase = [], [], []
    first_rising, last_rising = [], []
    for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
        evidence = trail_evidence(batch["frames"]).flatten(2)
        clip_max = evidence.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        extent = (evidence > (clip_max * threshold)).sum(dim=2).float()
        peak = (evidence / clip_max).amax(dim=2)
        steps = extent.shape[1]
        grid = torch.arange(steps)
        for item in range(len(extent)):
            durations.append(float(batch["target"][item, -1]))
            series = extent[item]
            ceiling = float(series.max())
            reached = (series >= 0.95 * ceiling) & (grid >= 7) if ceiling > 0 else None
            growth = float(grid[reached].min()) if reached is not None and bool(reached.any()) else float("nan")
            lit = (peak[item] > threshold) & (grid >= 7)
            fade = float(grid[lit].max()) if bool(lit.any()) else float("nan")
            span_fade_growth.append(fade - growth)
            rising = series[1:] > series[:-1]
            rising_idx = grid[1:][rising]
            if len(rising_idx) > 1:
                span_increase.append(float(rising_idx.max() - rising_idx.min()))
                # If sd(first) is ~0 this "span" is a single-edge reader wearing a
                # difference's clothes: the reference subtraction over frames 0-6
                # zeroes extent there, pinning the first rising frame.
                first_rising.append(float(rising_idx.min()))
                last_rising.append(float(rising_idx.max()))
            else:
                span_increase.append(float("nan"))
                first_rising.append(float("nan"))
                last_rising.append(float("nan"))

    durations = np.array(durations, dtype=float)
    readers = {"fade_minus_growth": np.array(span_fade_growth, dtype=float),
               "increase_span": np.array(span_increase, dtype=float)}
    quantum_s = 2.1935 / 31
    common = ~np.isnan(durations)
    for series in readers.values():
        common &= ~np.isnan(series)
    order = np.arange(len(durations))
    half_a = common & (order % 2 == 0)
    half_b = common & (order % 2 == 1)

    def cross_fitted(series):
        """Affine calibration fitted on the opposite half, applied out of sample."""
        predicted = np.full_like(durations, np.nan)
        for fit, score in ((half_a, half_b), (half_b, half_a)):
            if fit.sum() < 3:
                continue
            slope, intercept = np.polyfit(series[fit], durations[fit], 1)
            predicted[score] = slope * series[score] + intercept
        return predicted

    def constant_cross_fitted():
        predicted = np.full_like(durations, np.nan)
        for fit, score in ((half_a, half_b), (half_b, half_a)):
            predicted[score] = durations[fit].mean()
        return predicted

    def summarise(estimate):
        error = (estimate - durations)[common]
        absolute = np.abs(error)
        return {"clips": int(common.sum()),
                "bias_s": round(float(error.mean()), 4),
                "mae_s": round(float(absolute.mean()), 4),
                "median_abs_s": round(float(np.median(absolute)), 4),
                "p90_abs_s": round(float(np.quantile(absolute, .9)), 4),
                "mae_frames": round(float(absolute.mean() / quantum_s), 3),
                "within_duration_gate": round(float((absolute <= 0.10).mean()), 4)}

    def sign_test(a, b):
        left, right = np.abs(a - durations)[common], np.abs(b - durations)[common]
        wins, losses = int((left < right).sum()), int((left > right).sum())
        total = wins + losses
        if total == 0:
            return {"wins": 0, "losses": 0, "p": 1.0}
        extreme = max(wins, losses)
        tail = sum(math.comb(total, k) for k in range(extreme, total + 1)) / 2 ** total
        return {"wins": wins, "losses": losses, "ties": int(common.sum()) - total,
                "p": round(min(1.0, 2 * tail), 6)}

    constant = constant_cross_fitted()
    results, tests, correlations = {}, {}, {}
    for name, series in readers.items():
        estimate = cross_fitted(series)
        results[name] = summarise(estimate)
        tests[f"{name}_vs_constant"] = sign_test(estimate, constant)
        correlations[name] = round(float(np.corrcoef(series[common], durations[common])[0, 1]), 4)
    results["constant_duration"] = summarise(constant)
    output = {
        "data_subdir": data_subdir, "checkpoint": checkpoint_name,
        "clips": int(common.sum()), "frame_quantum_s": round(quantum_s, 5),
        "calibration": "affine, cross-fitted on alternating halves",
        "commanded_duration": {"mean": round(float(durations[common].mean()), 4),
                               "sd": round(float(durations[common].std()), 4)},
        "correlation_with_duration": correlations,
        # THE CHECK THAT DECIDES WHETHER THIS WAS A DIFFERENCE AT ALL.
        "rising_edges": {
            name: {
                "sd": round(float(np.nanstd(np.array(values, dtype=float)[common])), 3),
                "mean": round(float(np.nanmean(np.array(values, dtype=float)[common])), 3),
                "corr_with_duration": round(float(np.corrcoef(
                    np.array(values, dtype=float)[common], durations[common])[0, 1]), 4),
            }
            for name, values in (("first_rising", first_rising), ("last_rising", last_rising))
        },
        "results": results, "paired_sign_tests": tests,
        # The reference that decides whether any of this matters.
        # Read from the checkpoint, never hardcoded: a future run against another
        # checkpoint would otherwise silently report the wrong reference.
        "model_duration_mae_s": float((payload.get("test") or {}).get("duration_mae", float("nan"))),
    }
    (Path("/models") / f"basic_linear_duration_difference_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=8.0, timeout=2 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_trail_head(data_subdir: str, checkpoint_name: str, *, clips: int = 400,
                     batch_size: int = 8, threshold: float = 0.35,
                     advance: float = 0.02) -> dict:
    """Track the trail HEAD along the commanded chord.

    Every reader through EQ-027 collapsed each frame to a scalar — a thresholded
    pixel count or a spatial max brightness — and used none of the trail's
    POSITION.  The model localises endpoints to ~0.006 normalised units, so it
    plainly reads geometry spatially.  For a constant-velocity linear drag the
    head's advance along the chord stops at contact end, which is a kinematic
    read of liftoff and is immune to fade: the head stops moving whether or not
    the trail is still bright.

    **This is an ORACLE-ASSISTED upper bound.** It projects onto the *commanded*
    chord, i.e. it is handed the true endpoint direction that the model must
    infer.  A reader given that advantage which still cannot match the model is
    strong evidence; one that beats the model would prove nothing about
    inference-time feasibility.

    `threshold` (0.35) and `advance` (0.02 of the chord) are pre-committed.
    """
    import json as _json
    import math
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    fresh = [index for index in range(len(data))
             if data.sample_paths[index].relative_to(data.root).parts[0] == "fresh"]
    # Spread across the whole fresh set rather than taking a contiguous block:
    # EQ-027's stride collapsed to 1 and sampled the first N in path order.
    step = max(1, len(fresh) // max(clips, 1))
    indices = fresh[::step][:clips] if step > 1 else fresh[:clips]

    def trail_evidence(frames: torch.Tensor) -> torch.Tensor:
        steps = frames.shape[1]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        return ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

    durations, head_edge, head_reach = [], [], []
    for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
        frames = batch["frames"]
        target = batch["target"]
        evidence = trail_evidence(frames)
        height, width = evidence.shape[2], evidence.shape[3]
        flat = evidence.flatten(2)
        clip_max = flat.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        above = flat > (clip_max * threshold)
        ys = torch.linspace(0., 1., height)[:, None].expand(height, width).reshape(-1)
        xs = torch.linspace(0., 1., width)[None, :].expand(height, width).reshape(-1)
        steps = evidence.shape[1]
        grid = torch.arange(steps)
        for item in range(len(target)):
            durations.append(float(target[item, -1]))
            start = target[item, :2]
            end = target[item, 2:4]
            chord = end - start
            length = float(torch.linalg.vector_norm(chord))
            if length < 1e-6:
                head_edge.append(float("nan")); head_reach.append(float("nan")); continue
            unit = chord / length
            # Fraction along the commanded chord for every pixel.
            projection = ((xs - start[0]) * unit[0] + (ys - start[1]) * unit[1]) / length
            masked = torch.where(above[item], projection[None, :].expand(steps, -1),
                                 torch.full_like(projection[None, :].expand(steps, -1), -1e9))
            head = masked.amax(dim=1)                       # [T] furthest point drawn so far
            head = torch.where(head < -1e8, torch.full_like(head, float("nan")), head)
            head_reach.append(float(np.nanmax(head.numpy())))
            # Liftoff = the last frame at which the head still advanced materially.
            series = head.numpy()
            advancing = [t for t in range(1, steps)
                         if np.isfinite(series[t]) and np.isfinite(series[t - 1])
                         and series[t] - series[t - 1] > advance]
            head_edge.append(float(advancing[-1]) if advancing else float("nan"))

    durations = np.array(durations, dtype=float)
    head_edge = np.array(head_edge, dtype=float)
    head_reach = np.array(head_reach, dtype=float)
    quantum_s = 2.1935 / 31
    common = ~np.isnan(durations) & ~np.isnan(head_edge)
    order = np.arange(len(durations))
    half_a, half_b = common & (order % 2 == 0), common & (order % 2 == 1)

    def cross_fitted(series):
        predicted = np.full_like(durations, np.nan)
        for fit, score in ((half_a, half_b), (half_b, half_a)):
            if fit.sum() < 3:
                continue
            slope, intercept = np.polyfit(series[fit], durations[fit], 1)
            predicted[score] = slope * series[score] + intercept
        return predicted

    def summarise(estimate):
        error = (estimate - durations)[common]
        absolute = np.abs(error)
        return {"clips": int(common.sum()), "bias_s": round(float(error.mean()), 4),
                "mae_s": round(float(absolute.mean()), 4),
                "median_abs_s": round(float(np.median(absolute)), 4),
                "p90_abs_s": round(float(np.quantile(absolute, .9)), 4),
                "mae_frames": round(float(absolute.mean() / quantum_s), 3),
                "within_duration_gate": round(float((absolute <= 0.10).mean()), 4)}

    def sign_test(a, b):
        left, right = np.abs(a - durations)[common], np.abs(b - durations)[common]
        wins, losses = int((left < right).sum()), int((left > right).sum())
        total = wins + losses
        if total == 0:
            return {"wins": 0, "losses": 0, "p": 1.0}
        tail = sum(math.comb(total, k) for k in range(max(wins, losses), total + 1)) / 2 ** total
        return {"wins": wins, "losses": losses, "p": round(min(1.0, 2 * tail), 6)}

    constant = np.full_like(durations, np.nan)
    for fit, score in ((half_a, half_b), (half_b, half_a)):
        constant[score] = durations[fit].mean()
    estimate = cross_fitted(head_edge)
    correlation = float(np.corrcoef(head_edge[common], durations[common])[0, 1])
    output = {
        "data_subdir": data_subdir, "checkpoint": checkpoint_name,
        "oracle_assisted": "projects onto the COMMANDED chord; upper bound, not inference-feasible",
        "precommitted": {"threshold": threshold, "advance": advance, "source": "fresh"},
        "clips": int(common.sum()), "clips_dropped": int((~common).sum()),
        "correlation_with_duration": round(correlation, 4),
        "r_squared": round(correlation ** 2, 4),
        "head_reach": {"median": round(float(np.nanmedian(head_reach)), 4),
                       "p10": round(float(np.nanquantile(head_reach, .1)), 4),
                       "p90": round(float(np.nanquantile(head_reach, .9)), 4)},
        "head_tracking": summarise(estimate), "constant_duration": summarise(constant),
        "head_vs_constant": sign_test(estimate, constant),
        "best_scalar_reader_r": 0.4511,
        "model_duration_mae_s": float((payload.get("test") or {}).get("duration_mae", float("nan"))),
    }
    (Path("/models") / f"basic_linear_trail_head_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=8.0, timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_duration_head_attribution(data_subdir: str, checkpoint_name: str, *,
                                    batch_size: int = 8, epochs: int = 300,
                                    threshold: float = 0.35, seed: int = 0) -> dict:
    """Is the model's duration advantage the EVIDENCE MAP or the DECODER?

    EQ-028 established that `duration_head` consumes only a 2xT scalar series —
    the spatial max and mean of a learned evidence map — so the model's duration
    path is the same reader family as every hand-crafted estimator tried.  That
    leaves two explanations for the ~10x gap, which no experiment so far
    separates:

      (a) the learned evidence map beats a hand-crafted colour x motion filter;
      (b) the learned temporal decoder beats a single hand-picked event.

    This holds the decoder fixed and swaps the front end: the hand-crafted
    `trail_evidence` series is fed into a freshly initialised copy of the real
    `duration_head`, trained on the same split.  Landing near the model's
    0.0189 s implicates the DECODER; staying near the hand-picked reader's
    0.163 s implicates the EVIDENCE MAP.

    Protocol matches the checkpoint's: same split seed and fresh holdout source,
    epoch chosen on validation, test scored once.
    """
    import json as _json
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.data.gesture_sampling import BASIC_LINEAR_MAX_S, BASIC_LINEAR_MIN_S
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command

    trainer = _trainer()
    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    split_seed = payload.get("split_seed", seed)
    fresh_source = payload.get("fresh_holdout_source")
    if fresh_source is None:
        train_idx, val_idx, test_idx = split_by_command(data, seed=split_seed)
    else:
        train_idx, val_idx, test_idx = trainer.split_with_fresh_command_holdout(
            data, fresh_source=fresh_source, seed=split_seed,
            stratify_by_device=bool(payload.get("fresh_stratify_by_device")))

    def trail_evidence(frames: torch.Tensor) -> torch.Tensor:
        steps = frames.shape[1]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        return ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

    def extract(indices):
        """The same 2xT summary the real duration_head consumes."""
        series, targets = [], []
        for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
            evidence = trail_evidence(batch["frames"])
            flat = evidence.flatten(2)
            # Normalise per clip so the scale matches the learned map's dynamic
            # range rather than raw filter output.
            scale = flat.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            normalised = (flat / scale)
            series.append(torch.stack((normalised.amax(dim=2), normalised.mean(dim=2)), dim=1))
            targets.append(batch["target"][:, -1])
        return torch.cat(series), torch.cat(targets)

    train_x, train_y = extract(train_idx)
    val_x, val_y = extract(val_idx)
    test_x, test_y = extract(test_idx)

    channels = int(payload.get("base_channels") or 16)
    torch.manual_seed(seed)
    head = nn.Sequential(
        nn.Conv1d(2, channels, 3, padding=1), nn.SiLU(),
        nn.Conv1d(channels, channels, 3, padding=1), nn.SiLU(),
        nn.AdaptiveAvgPool1d(8), nn.Flatten(),
        nn.Linear(channels * 8, channels * 2), nn.SiLU(), nn.Linear(channels * 2, 1),
    )
    optimiser = torch.optim.Adam(head.parameters(), lr=2e-3)
    span = BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S

    def predict(inputs):
        return BASIC_LINEAR_MIN_S + torch.sigmoid(head(inputs))[:, 0] * span

    best = {"val_mae": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(epochs):
        head.train()
        permutation = torch.randperm(len(train_x))
        for start in range(0, len(train_x), 64):
            chunk = permutation[start:start + 64]
            optimiser.zero_grad()
            loss = nn.functional.smooth_l1_loss(predict(train_x[chunk]), train_y[chunk], beta=0.05)
            loss.backward()
            optimiser.step()
        head.eval()
        with torch.no_grad():
            val_mae = float((predict(val_x) - val_y).abs().mean())
        history.append(round(val_mae, 5))
        if val_mae < best["val_mae"]:
            best = {"val_mae": val_mae, "epoch": epoch,
                    "state": {k: v.clone() for k, v in head.state_dict().items()}}

    head.load_state_dict(best["state"])
    head.eval()
    with torch.no_grad():
        error = (predict(test_x) - test_y).abs()

    # THE CHECK THAT DECIDES WHAT THE RESIDUAL GAP IS.  Run the real checkpoint on
    # the SAME 153 test clips: if the two arms fail on the same clips, the residual
    # is a shared data defect (low rendered headroom / late onset per EQ-025) and is
    # not addressable by front-end work; if the failures are disjoint, it is a
    # genuine front-end advantage on exactly the hard mode.
    model = _model_from_payload(payload, torch)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model_errors = []
    with torch.no_grad():
        for batch in DataLoader(Subset(data, test_idx), batch_size=batch_size):
            prediction = model(batch["frames"])
            model_errors.append((prediction[:, -1] - batch["target"][:, -1]).abs())
    model_error = torch.cat(model_errors)
    ours, theirs = error.numpy(), model_error.numpy()
    gate = 0.10
    both = int(((ours > gate) & (theirs > gate)).sum())
    ours_only = int(((ours > gate) & (theirs <= gate)).sum())
    theirs_only = int(((ours <= gate) & (theirs > gate)).sum())
    neither = int(((ours <= gate) & (theirs <= gate)).sum())
    distinct_commands = len({data.command_keys[index] for index in test_idx})
    output = {
        "data_subdir": data_subdir, "checkpoint": checkpoint_name,
        "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx),
                  "seed": split_seed, "fresh_holdout_source": fresh_source},
        "selected_epoch": best["epoch"], "validation_mae_s": round(best["val_mae"], 5),
        "test_mae_s": round(float(error.mean()), 5),
        "test_median_abs_s": round(float(error.median()), 5),
        "test_p90_abs_s": round(float(np.quantile(error.numpy(), .9)), 5),
        "test_within_gate": round(float((error <= 0.10).float().mean()), 4),
        # The two anchors this number is interpreted against.
        "handcrafted_event_reader_mae_s": 0.163,
        "model_duration_mae_s": float((payload.get("test") or {}).get("duration_mae", float("nan"))),
        "validation_history_tail": history[-20:],
        "test_distinct_commands": distinct_commands,
        # Recomputed here rather than quoted, so both arms are the same 153 clips.
        "model_test_mae_s_recomputed": round(float(model_error.mean()), 5),
        "per_clip_error_agreement": {
            "pearson": round(float(np.corrcoef(ours, theirs)[0, 1]), 4),
            "spearman": round(float(np.corrcoef(
                np.argsort(np.argsort(ours)).astype(float),
                np.argsort(np.argsort(theirs)).astype(float))[0, 1]), 4),
            "both_out_of_gate": both, "handcrafted_only": ours_only,
            "model_only": theirs_only, "neither": neither,
        },
    }
    (Path("/models") / f"basic_linear_duration_attribution_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=8.0, timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_duration_decomposition(data_subdir: str, checkpoint_name: str, *,
                                 batch_size: int = 8, epochs: int = 300, seed: int = 0) -> dict:
    """Separate the three variables EQ-029 bundled into "the decoder is worth 2.6x".

    EQ-029 changed decoder architecture, fitting budget and evidence
    normalisation at once.  Two of the three separate cheaply on data already
    extracted:

    * **shape vs multivariate** — a ridge over a handful of hand-picked scalars
      of the same series, on the same train/test budget.  If it reaches the conv
      decoder's 0.0629 s, the decoder's win is reading SEVERAL scalars rather
      than reading temporal SHAPE; if it stalls near the single-event reader's
      0.163 s, shape is what matters.
    * **normalisation** — the same conv decoder on raw, per-clip-normalised and
      corpus-scaled series.  EQ-029 imposed a per-clip scale the hand-picked
      reader never had, and charged the difference to the decoder.

    The corpus scale is computed from TRAIN ONLY, so no variant leaks test
    statistics.  (Separating `temporal_mixer` from the learned filter needs a
    retrain and is deliberately out of scope.)
    """
    import json as _json
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.data.gesture_sampling import BASIC_LINEAR_MAX_S, BASIC_LINEAR_MIN_S
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command

    trainer = _trainer()
    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    split_seed = payload.get("split_seed", seed)
    fresh_source = payload.get("fresh_holdout_source")
    if fresh_source is None:
        train_idx, val_idx, test_idx = split_by_command(data, seed=split_seed)
    else:
        train_idx, val_idx, test_idx = trainer.split_with_fresh_command_holdout(
            data, fresh_source=fresh_source, seed=split_seed,
            stratify_by_device=bool(payload.get("fresh_stratify_by_device")))
    recorded = payload.get("split_sizes") or {}
    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        if name in recorded and recorded[name] != len(indices):
            raise ValueError(f"re-derived {name} split disagrees with the checkpoint")

    def trail_evidence(frames: torch.Tensor) -> torch.Tensor:
        steps = frames.shape[1]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        red, green, blue = frames.unbind(dim=2)
        motion = torch.abs(frames - reference).mean(dim=2)
        return ((red - green + .12).relu() * (green - blue + .12).relu()
                * (red - .20).relu() * motion)

    def extract_raw(indices):
        peaks, means, scales, targets = [], [], [], []
        for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
            flat = trail_evidence(batch["frames"]).flatten(2)
            peaks.append(flat.amax(dim=2))
            means.append(flat.mean(dim=2))
            scales.append(flat.amax(dim=(1, 2)))
            targets.append(batch["target"][:, -1])
        return (torch.cat(peaks), torch.cat(means), torch.cat(scales), torch.cat(targets))

    raw = {name: extract_raw(indices) for name, indices in
           (("train", train_idx), ("validation", val_idx), ("test", test_idx))}
    corpus_scale = float(raw["train"][2].max())   # TRAIN ONLY

    def build(name, mode):
        peaks, means, scales, targets = raw[name]
        if mode == "per_clip":
            denominator = scales[:, None].clamp_min(1e-6)
        elif mode == "corpus":
            denominator = torch.full_like(scales[:, None], corpus_scale)
        else:
            denominator = torch.ones_like(scales[:, None])
        return torch.stack((peaks / denominator, means / denominator), dim=1), targets

    channels = int(payload.get("base_channels") or 16)
    span = BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S

    def train_head(mode):
        torch.manual_seed(seed)
        head = nn.Sequential(
            nn.Conv1d(2, channels, 3, padding=1), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(channels * 8, channels * 2), nn.SiLU(), nn.Linear(channels * 2, 1))
        optimiser = torch.optim.Adam(head.parameters(), lr=2e-3)
        train_x, train_y = build("train", mode)
        val_x, val_y = build("validation", mode)
        test_x, test_y = build("test", mode)

        def predict(inputs):
            return BASIC_LINEAR_MIN_S + torch.sigmoid(head(inputs))[:, 0] * span

        best = {"val": float("inf"), "state": None}
        for _epoch in range(epochs):
            head.train()
            order = torch.randperm(len(train_x))
            for start in range(0, len(train_x), 64):
                chunk = order[start:start + 64]
                optimiser.zero_grad()
                nn.functional.smooth_l1_loss(predict(train_x[chunk]), train_y[chunk],
                                             beta=0.05).backward()
                optimiser.step()
            head.eval()
            with torch.no_grad():
                score = float((predict(val_x) - val_y).abs().mean())
            if score < best["val"]:
                best = {"val": score, "state": {k: v.clone() for k, v in head.state_dict().items()}}
        head.load_state_dict(best["state"])
        head.eval()
        with torch.no_grad():
            error = (predict(test_x) - test_y).abs()
        return {"test_mae_s": round(float(error.mean()), 5),
                "test_within_gate": round(float((error <= 0.10).float().mean()), 4)}

    def features(mode, name):
        """Hand-picked scalars of the same series -- no temporal shape modelling."""
        series, targets = build(name, mode)
        peaks, means = series[:, 0], series[:, 1]
        steps = peaks.shape[1]
        grid = torch.arange(steps).float()
        rows = []
        for item in range(len(peaks)):
            a = peaks[item]
            rising = (a[1:] > a[:-1]).nonzero().flatten()
            lit = (a > 0.35).nonzero().flatten()
            rows.append([
                float(a.argmax()),
                float(grid[1:][rising].max()) if len(rising) else 0.0,
                float(a.sum()),
                float(len(lit)),
                float(lit.max()) if len(lit) else 0.0,
                float(means[item].max()),
            ])
        return np.array(rows, dtype=float), targets.numpy()

    def ridge(mode, alpha=1.0):
        train_f, train_y = features(mode, "train")
        test_f, test_y = features(mode, "test")
        centre, scale = train_f.mean(0), train_f.std(0) + 1e-9
        design = np.hstack([(train_f - centre) / scale, np.ones((len(train_f), 1))])
        target_design = np.hstack([(test_f - centre) / scale, np.ones((len(test_f), 1))])
        penalty = alpha * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
        error = np.abs(target_design @ weights - test_y)
        return {"test_mae_s": round(float(error.mean()), 5),
                "test_within_gate": round(float((error <= 0.10).mean()), 4),
                "features": ["argmax_peak", "last_rising", "peak_sum", "lit_count",
                             "last_lit", "mean_max"]}

    def train_tabular(inputs_by_split, hidden):
        """Same loss and same [MIN,MAX] range constraint as the conv decoder.

        The ridge minimised SQUARED error and could emit impossible durations,
        while the conv minimised smooth_l1 inside a sigmoid range — so part of
        the gap was loss and range handling, not model class.  These arms match
        both, leaving only functional class and input width to vary.
        """
        torch.manual_seed(seed)
        width = inputs_by_split["train"][0].shape[1]
        layers = ([nn.Linear(width, 1)] if hidden == 0 else
                  [nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                   nn.Linear(hidden, 1)])
        net = nn.Sequential(*layers)
        optimiser = torch.optim.Adam(net.parameters(), lr=2e-3)

        def predict(inputs):
            return BASIC_LINEAR_MIN_S + torch.sigmoid(net(inputs))[:, 0] * span

        train_x, train_y = inputs_by_split["train"]
        val_x, val_y = inputs_by_split["validation"]
        test_x, test_y = inputs_by_split["test"]
        best = {"val": float("inf"), "state": None}
        for _epoch in range(epochs):
            net.train()
            order = torch.randperm(len(train_x))
            for start in range(0, len(train_x), 64):
                chunk = order[start:start + 64]
                optimiser.zero_grad()
                nn.functional.smooth_l1_loss(predict(train_x[chunk]), train_y[chunk],
                                             beta=0.05).backward()
                optimiser.step()
            net.eval()
            with torch.no_grad():
                score = float((predict(val_x) - val_y).abs().mean())
            if score < best["val"]:
                best = {"val": score, "state": {k: v.clone() for k, v in net.state_dict().items()}}
        net.load_state_dict(best["state"])
        net.eval()
        with torch.no_grad():
            error = (predict(test_x) - test_y).abs()
        return {"test_mae_s": round(float(error.mean()), 5),
                "test_within_gate": round(float((error <= 0.10).float().mean()), 4)}

    def tabular_inputs(kind):
        """`scalars` = the 6 hand-picked summaries; `raw` = the flattened 2x32 series."""
        prepared, statistics = {}, None
        for name in ("train", "validation", "test"):
            if kind == "scalars":
                values, targets = features("per_clip", name)
                values = torch.tensor(values, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)
            else:
                series, targets = build(name, "per_clip")
                values = series.flatten(1)
            if statistics is None:
                statistics = (values.mean(0), values.std(0) + 1e-6)
            prepared[name] = ((values - statistics[0]) / statistics[1], targets)
        return prepared

    scalars, raws = tabular_inputs("scalars"), tabular_inputs("raw")
    output = {
        "data_subdir": data_subdir, "checkpoint": checkpoint_name,
        "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "corpus_scale_from_train": round(corpus_scale, 6),
        "conv_decoder_by_normalisation": {mode: train_head(mode)
                                          for mode in ("per_clip", "corpus", "none")},
        "ridge_over_handpicked_scalars": ridge("per_clip"),
        # Matched loss + range, so only functional class and input width vary.
        "linear_6_scalars": train_tabular(scalars, hidden=0),
        "mlp_6_scalars": train_tabular(scalars, hidden=32),
        "linear_raw_64": train_tabular(raws, hidden=0),
        "mlp_raw_64": train_tabular(raws, hidden=32),
        "anchors": {"single_event_reader": 0.163,
                    "conv_decoder_eq029": 0.06289,
                    "model": float((payload.get("test") or {}).get("duration_mae", float("nan")))},
    }
    (Path("/models") / f"basic_linear_duration_decomposition_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
    models.commit()
    return output


@app.function(image=image, cpu=8.0, timeout=3 * 3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def audit_learned_series_capacity(data_subdir: str, checkpoint_name: str, *,
                                  batch_size: int = 8, epochs: int = 300, seed: int = 0) -> dict:
    """Split the front-end factor: is it the MAP, or is it joint training?

    EQ-029 put ~3.3x in the front end; EQ-031 showed the decoder half is
    capacity, not temporal structure.  This takes the model's OWN evidence
    series — `max(start_scores, end_scores)` reduced exactly as `duration_head`
    reduces it — and trains a fresh head on it.

    * lands near the model's 0.0189 s ⇒ the front end IS the learned map, and a
      frozen map plus a fresh head recovers the model's duration accuracy;
    * stays near the hand-crafted 0.0629 s ⇒ the map alone is not the advantage
      and the credit belongs to end-to-end, duration-supervised training.

    The map is frozen (no gradients reach it), so nothing here can leak the
    duration loss back into the encoder.
    """
    import json as _json
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.data.gesture_sampling import BASIC_LINEAR_MAX_S, BASIC_LINEAR_MIN_S
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command

    trainer = _trainer()
    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
    split_seed = payload.get("split_seed", seed)
    fresh_source = payload.get("fresh_holdout_source")
    if fresh_source is None:
        train_idx, val_idx, test_idx = split_by_command(data, seed=split_seed)
    else:
        train_idx, val_idx, test_idx = trainer.split_with_fresh_command_holdout(
            data, fresh_source=fresh_source, seed=split_seed,
            stratify_by_device=bool(payload.get("fresh_stratify_by_device")))
    recorded = payload.get("split_sizes") or {}
    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        if name in recorded and recorded[name] != len(indices):
            raise ValueError(f"re-derived {name} split disagrees with the checkpoint")

    model = _model_from_payload(payload, torch)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    def extract(indices):
        """The model's own 2xT reduction — the exact input `duration_head` sees."""
        series, targets = [], []
        with torch.no_grad():
            for batch in DataLoader(Subset(data, indices), batch_size=batch_size):
                _prediction, start_scores, end_scores = model.forward_with_scores(batch["frames"])
                evidence = torch.maximum(start_scores, end_scores)
                series.append(torch.stack((evidence.amax(dim=(2, 3)),
                                           evidence.mean(dim=(2, 3))), dim=1))
                targets.append(batch["target"][:, -1])
        return torch.cat(series), torch.cat(targets)

    prepared = {name: extract(indices) for name, indices in
                (("train", train_idx), ("validation", val_idx), ("test", test_idx))}
    channels = int(payload.get("base_channels") or 16)
    span = BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S

    def run(kind):
        torch.manual_seed(seed)
        if kind == "conv":
            net = nn.Sequential(
                nn.Conv1d(2, channels, 3, padding=1), nn.SiLU(),
                nn.Conv1d(channels, channels, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool1d(8), nn.Flatten(),
                nn.Linear(channels * 8, channels * 2), nn.SiLU(), nn.Linear(channels * 2, 1))
            shape = lambda tensor: tensor
        else:
            width = prepared["train"][0].shape[1] * prepared["train"][0].shape[2]
            net = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(),
                                nn.Linear(32, 1))
            shape = lambda tensor: tensor.flatten(1)
        optimiser = torch.optim.Adam(net.parameters(), lr=2e-3)

        def predict(inputs):
            return BASIC_LINEAR_MIN_S + torch.sigmoid(net(shape(inputs)))[:, 0] * span

        train_x, train_y = prepared["train"]
        val_x, val_y = prepared["validation"]
        test_x, test_y = prepared["test"]
        best = {"val": float("inf"), "state": None}
        for _epoch in range(epochs):
            net.train()
            order = torch.randperm(len(train_x))
            for start in range(0, len(train_x), 64):
                chunk = order[start:start + 64]
                optimiser.zero_grad()
                nn.functional.smooth_l1_loss(predict(train_x[chunk]), train_y[chunk],
                                             beta=0.05).backward()
                optimiser.step()
            net.eval()
            with torch.no_grad():
                score = float((predict(val_x) - val_y).abs().mean())
            if score < best["val"]:
                best = {"val": score, "state": {k: v.clone() for k, v in net.state_dict().items()}}
        net.load_state_dict(best["state"])
        net.eval()
        with torch.no_grad():
            error = (predict(test_x) - test_y).abs()
        return {"test_mae_s": round(float(error.mean()), 5),
                "test_within_gate": round(float((error <= 0.10).float().mean()), 4),
                "test_median_abs_s": round(float(error.median()), 5)}

    output = {
        "data_subdir": data_subdir, "checkpoint": checkpoint_name,
        "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "frozen_map_plus_fresh_head": {"conv": run("conv"), "mlp": run("mlp")},
        "anchors": {
            "handcrafted_series_conv_head": 0.06289,
            "handcrafted_series_mlp6": 0.07295,
            "model_end_to_end": float((payload.get("test") or {}).get("duration_mae", float("nan"))),
        },
    }
    (Path("/models") / f"basic_linear_learned_series_{data_subdir.replace('/', '_')}.json").write_text(
        _json.dumps(output, indent=2))
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
         evaluate_test: bool = True, fresh_stratify_by_device: bool = False,
         line_fit: bool = False, irls_iterations: int = 3, huber_delta: float = .02,
         image_width: int = 128, image_height: int = 288, knots: int = 2,
         max_grad_norm: float | None = None,
         experiment_manifest_name: str | None = None,
         shard_manifest_name: str | None = None,
         record_train_metrics: bool = False,
         provider_timeout_retries: int = 6) -> None:
    if provider_timeout_retries < 0:
        raise ValueError("provider_timeout_retries must be non-negative")
    kwargs = dict(
        epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
        base_channels=base_channels, split_strategy=split_strategy,
        cache_frames=cache_frames, split_seed=split_seed, map_weight=map_weight,
        start_onset=start_onset, start_sigma=start_sigma, end_onset=end_onset,
        temporal_mixer=temporal_mixer, trajectory_weight=trajectory_weight,
        trajectory_track=trajectory_track, fresh_holdout_source=fresh_holdout_source,
        evaluate_test=evaluate_test, fresh_stratify_by_device=fresh_stratify_by_device,
        line_fit=line_fit, irls_iterations=irls_iterations, huber_delta=huber_delta,
        image_width=image_width, image_height=image_height, knots=knots,
        max_grad_norm=max_grad_norm, experiment_manifest_name=experiment_manifest_name,
        shard_manifest_name=shard_manifest_name, record_train_metrics=record_train_metrics,
    )
    for attempt in range(provider_timeout_retries + 1):
        try:
            result = train_remote.remote(data_subdir, run_label, **kwargs)
            break
        except (modal.exception.FunctionTimeoutError, modal.exception.InternalFailure):
            if attempt == provider_timeout_retries:
                raise
            print(f"provider interruption; retrying durable run ({attempt + 1}/"
                  f"{provider_timeout_retries})")
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
    from trueskate_ai.vision.basic_linear_training import (
        decompose_endpoint_error, knot_columns, knot_errors, nearest_trail_gaps, target_knots,
    )

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True,
                                  **_payload_dataset_kwargs([payload]))
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

    # A line-fit checkpoint builds its knots entirely from the trajectory map;
    # start/end score maps only feed duration/onset there.  Reporting them as
    # "where the endpoint attention peaked" would describe a head that produces
    # no coordinate -- and knots>2 REQUIRES line_fit, so every k>2 checkpoint is
    # on that path.  Report the map that actually produced the prediction.
    line_fit = bool(payload.get("line_fit"))
    records: list[dict] = []
    loader = DataLoader(Subset(data, test_indices), batch_size=batch_size)
    cursor = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        target = batch["target"].to(device)
        with torch.no_grad():
            if line_fit:
                prediction, _start_scores, _end_scores, decode_scores = (
                    model.forward_with_track_scores(frames))
                start_scores = end_scores = None
            else:
                prediction, start_scores, end_scores = model.forward_with_scores(frames)
                decode_scores = None
        all_errors = knot_errors(prediction, target)
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
            errors = all_errors[item]
            start_error, end_error = float(errors[0]), float(errors[-1])
            duration_error = float(torch.abs(prediction[item, -1] - target[item, -1]))
            # Gate EVERY knot, as basic_linear_metrics does -- at K>2 a clip with
            # both endpoints right and a bad interior knot is not recovered.
            recovered = bool(errors.max() <= .03) and duration_error <= .10
            any_strong = strong[item].any(dim=1)

            first_x, first_y = knot_columns(target.shape[1], 0)
            last_x, last_y = knot_columns(target.shape[1], -1)
            # Every knot gets an evidence column, not just the endpoints:
            # `recovered` gates every knot (EQ-012), so a clip can fail on a knot
            # the report would otherwise say nothing about.  start/end keys below
            # are retained unchanged for k=2 artefact compatibility.
            # Loop bound from the TARGET width, not from len(errors): knot_errors
            # derives K from the prediction and truncates silently if the two ever
            # disagree, which would make per_knot_trail[-1] an interior knot and
            # change what trail_gap_end means.
            knot_points = torch.stack([
                target[item, knot_columns(target.shape[1], knot)[0]:
                             knot_columns(target.shape[1], knot)[1] + 1]
                for knot in range(target_knots(target.shape[1]))])
            per_knot_trail = nearest_trail_gaps(grid, strong[item], knot_points)
            commanded_start, commanded_end = per_knot_trail[0], per_knot_trail[-1]
            records.append({
                "sample": str(data.sample_paths[index].relative_to(data.root)),
                "device": str(meta.get("device", "unknown")),
                "recovered": recovered,
                "start_error": start_error, "end_error": end_error,
                "duration_error": duration_error,
                "knot_errors": [float(value) for value in errors.cpu()],
                "commanded": [float(v) for v in target[item].cpu()],
                "predicted": [float(v) for v in prediction[item].cpu()],
                # The decisive discriminator, per endpoint.
                "trail_gap_start": commanded_start["distance"],
                "trail_gap_end": commanded_end["distance"],
                "trail_frame_start": commanded_start["frame"],
                "trail_frame_end": commanded_end["frame"],
                "trail_frames_present": int(any_strong.sum()),
                **{f"trail_gap_knot{knot}": entry["distance"]
                   for knot, entry in enumerate(per_knot_trail)},
                **{f"trail_frame_knot{knot}": entry["frame"]
                   for knot, entry in enumerate(per_knot_trail)},
                # Where the decoding map peaked, to separate a misread from a
                # collapse onto the other endpoint or the middle.  Named for the
                # map that actually produced the prediction.
                **({"trajectory_score_peak_frame":
                    int(decode_scores[item].flatten(1).amax(dim=1).argmax())} if line_fit else
                   {"end_score_peak_frame": int(end_scores[item].flatten(1).amax(dim=1).argmax()),
                    "start_score_peak_frame": int(start_scores[item].flatten(1).amax(dim=1).argmax())}),
            })
        cursor += len(target)

    # First- and last-knot error split along and perpendicular to the chord.
    # Interior knots are deliberately not decomposed: the path bends through
    # them, so there is no single meaningful "along" direction to report.
    for record in records:
        record.update(decompose_endpoint_error(record["commanded"], record["predicted"]))

    failures = [record for record in records if not record["recovered"]]
    gaps = [record["trail_gap_end"] for record in failures if record["end_error"] > .03]
    # Per-knot, so a clip that failed only on an interior knot contributes to the
    # evidence-vs-misread split instead of vanishing from it.  `failed_end_trail_gaps`
    # is retained unchanged for comparability with the k=2 reports already quoted.
    knot_count = len(records[0]["commanded"]) // 2 if records else 0
    failed_knot_gaps = {
        f"knot{knot}": sorted(record[f"trail_gap_knot{knot}"] for record in failures
                              if record["knot_errors"][knot] > .03)
        for knot in range(knot_count)
    }
    summary = {
        "checkpoint": checkpoint_name,
        "partition": partition,
        # recovery gates EVERY knot, so it is NOT comparable across K: a k=3
        # report is scored against a strictly harder criterion than a k=2 one.
        "knots": int(target_knots(data[test_indices[0]]["target"].shape[0])),
        "line_fit": line_fit,
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
        "failed_knot_trail_gaps": failed_knot_gaps,
        "median_trail_gap_by_knot": [
            float(np.median([r[f"trail_gap_knot{knot}"] for r in records]))
            for knot in range(knot_count)
        ],
        "failing_records": failures,
    }
    (Path("/models") / f"basic_linear_{label}.json").write_text(_json.dumps(summary, indent=2))
    models.commit()
    return {key: value for key, value in summary.items() if key != "failing_records"}
