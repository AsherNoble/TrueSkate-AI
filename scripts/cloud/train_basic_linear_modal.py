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
                 cache_frames: bool = True) -> dict:
    trainer = _trainer()
    checkpoint = Path("/models") / f"basic_linear_{run_label}.pth"
    payload = trainer.train(
        data=Path("/corpus") / data_subdir,
        out=checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
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
    for onset in (.20, .22, .24, .26, .28):
        for sigma in (.04, .06, .08, .10, .13, .17):
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


@app.local_entrypoint()
def main(data_subdir: str, run_label: str = "baseline", epochs: int = 40,
         batch_size: int = 8, lr: float = 1e-3, seed: int = 0,
         base_channels: int = 16, split_strategy: str = "command",
         cache_frames: bool = True) -> None:
    result = train_remote.remote(
        data_subdir, run_label, epochs=epochs, batch_size=batch_size, lr=lr,
        seed=seed, base_channels=base_channels, split_strategy=split_strategy,
        cache_frames=cache_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
