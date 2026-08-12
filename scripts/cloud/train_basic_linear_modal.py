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


@app.function(image=image, gpu="A10G", timeout=3 * 3600, memory=32768,
              volumes={"/corpus": corpus, "/models": models})
def train_remote(data_subdir: str, run_label: str, *, epochs: int = 40,
                 batch_size: int = 8, lr: float = 1e-3, seed: int = 0,
                 base_channels: int = 16, split_strategy: str = "command",
                 cache_frames: bool = True, split_seed: int | None = None,
                 map_weight: float = 0.0) -> dict:
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


@app.function(image=image, gpu="A10G", timeout=3 * 3600, memory=32768,
              volumes={"/corpus": corpus, "/models": models})
def evaluate_refinement(data_subdir: str, checkpoint_name: str, *, seed: int = 0,
                        batch_size: int = 8) -> dict:
    """Grid-evaluate post-inference orange refinement on the held-out commands."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_regressor import BasicLinearRegressor
    from trueskate_ai.vision.basic_linear_refinement import refine_linear_endpoints
    from trueskate_ai.vision.basic_linear_training import basic_linear_metrics

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, _val, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    model = BasicLinearRegressor(base_channels=int(payload["base_channels"])).to(device)
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


@app.function(image=image, gpu="A10G", timeout=3 * 3600, memory=32768,
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
    from trueskate_ai.vision.basic_linear_regressor import BasicLinearRegressor
    from trueskate_ai.vision.basic_linear_training import basic_linear_metrics

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, _val, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    model = BasicLinearRegressor(base_channels=int(payload["base_channels"])).to(device)
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


@app.function(image=image, gpu="A10G", timeout=3 * 3600, memory=32768,
              volumes={"/corpus": corpus, "/models": models})
def audit_endpoint_residuals(data_subdir: str, checkpoint_name: str, *, seed: int = 0,
                             batch_size: int = 8) -> dict:
    """Describe signed endpoint residuals for an already-evaluated checkpoint."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_regressor import BasicLinearRegressor

    payload = torch.load(Path("/models") / checkpoint_name, map_location="cpu", weights_only=False)
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, _val, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    model = BasicLinearRegressor(base_channels=int(payload["base_channels"])).to(device)
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


@app.function(image=image, gpu="A10G", timeout=3 * 3600, memory=32768,
              volumes={"/corpus": corpus, "/models": models})
def evaluate_checkpoint_ensemble(data_subdir: str, checkpoint_names: str, *, seed: int = 0,
                                 batch_size: int = 8) -> dict:
    """Validation-select a convex checkpoint ensemble, then test it once.

    Every candidate has been trained on the same corpus/split.  We enumerate
    fixed convex weights using validation commands only, so the test commands
    remain untouched until the one selected evaluation.
    """
    import itertools
    import torch
    from torch.utils.data import DataLoader, Subset
    from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset, split_by_command
    from trueskate_ai.vision.basic_linear_regressor import BasicLinearRegressor
    from trueskate_ai.vision.basic_linear_training import basic_linear_metrics

    checkpoint_names = [name.strip() for name in checkpoint_names.split(",") if name.strip()]
    if len(checkpoint_names) < 2:
        raise ValueError("need at least two checkpoints")
    payloads = [torch.load(Path("/models") / name, map_location="cpu", weights_only=False)
                for name in checkpoint_names]
    data = BasicLinearClipDataset(Path("/corpus") / data_subdir, cache_frames=True)
    _train, val_indices, test_indices = split_by_command(data, seed=seed)
    device = torch.device("cuda")
    models_local = []
    for payload in payloads:
        model = BasicLinearRegressor(base_channels=int(payload["base_channels"])).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models_local.append(model)

    def batches(indices):
        return [(batch["frames"].to(device), batch["target"].to(device))
                for batch in DataLoader(Subset(data, indices), batch_size=batch_size)]

    val_batches, test_batches = batches(val_indices), batches(test_indices)
    candidate_weights = []
    # 0.1 granularity is deliberately pre-declared and compact.  It covers
    # individual checkpoints and two/three-way averages without test tuning.
    for units in itertools.product(range(11), repeat=len(models_local)):
        if sum(units) == 10:
            candidate_weights.append(tuple(value / 10 for value in units))

    def metrics_for(weights, data_batches):
        class Ensemble(torch.nn.Module):
            def forward(self, frames):
                return sum(weight * model(frames) for weight, model in zip(weights, models_local))
        return basic_linear_metrics(
            Ensemble(), [{"frames": frames, "target": target} for frames, target in data_batches], device,
        )

    ranked = []
    for weights in candidate_weights:
        metric = metrics_for(weights, val_batches)
        rank = (-metric["gesture_recovery_accuracy"],
                metric["start_coordinate_median"] + metric["end_coordinate_median"] + metric["duration_mae"])
        ranked.append((rank, weights, metric))
    ranked.sort(key=lambda item: item[0])
    _rank, selected_weights, validation = ranked[0]
    test = metrics_for(selected_weights, test_batches)
    output = {
        "checkpoints": checkpoint_names,
        "selected_weights": dict(zip(checkpoint_names, selected_weights)),
        "validation": validation,
        "test": test,
        "validation_top5": [
            {"weights": dict(zip(checkpoint_names, weights)), "metrics": metric}
            for _rank, weights, metric in ranked[:5]
        ],
    }
    (Path("/models") / "basic_linear_checkpoint_ensemble_20260812.json").write_text(
        json.dumps(output, indent=2),
    )
    models.commit()
    return output


@app.local_entrypoint()
def main(data_subdir: str, run_label: str = "baseline", epochs: int = 40,
         batch_size: int = 8, lr: float = 1e-3, seed: int = 0,
         base_channels: int = 16, split_strategy: str = "command",
         cache_frames: bool = True, split_seed: int | None = None,
         map_weight: float = 0.0) -> None:
    result = train_remote.remote(
        data_subdir, run_label, epochs=epochs, batch_size=batch_size, lr=lr,
        seed=seed, base_channels=base_channels, split_strategy=split_strategy,
        cache_frames=cache_frames, split_seed=split_seed, map_weight=map_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
