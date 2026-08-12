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
    .pip_install("torch", "opencv-python-headless", "numpy")
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
