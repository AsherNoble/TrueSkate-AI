"""Train Model 1 (the trace extractor) on Modal, over the trueskate-corpus volume.

Wraps the EXISTING trainer (scripts/train/train_trace_extractor.py — dataset,
labels, GaussianBumpPredictor, checkpoint format all unchanged) in a GPU
container: trueskate-corpus mounted read-side at /corpus, checkpoints committed
to the trueskate-models volume. The local box is MPS-only; this is the burst
path for real runs.

Known cost: SelfLabeledTraceDataset's warm-trace gate cv2.imreads every active
frame ONCE during __init__, over the volume FUSE mount — the first pass on a
full corpus is slow (mitigate later with a gate cache or --no-require-trace).

Run (repo root, ~/.modal.toml auth):
    .venv/bin/modal run scripts/cloud/train_trace_extractor_modal.py --smoke
        # cents: CPU container, synthetic one-step train + volume/import checks
    .venv/bin/modal run scripts/cloud/train_trace_extractor_modal.py --data-subdir <session>
        # PAID A10G mini-run on one session
    .venv/bin/modal run scripts/cloud/train_trace_extractor_modal.py --latency-s 0.45
        # PAID A10G full-corpus run — kick off on explicit GO only

Fetch a checkpoint:
    .venv/bin/modal volume get trueskate-models <name>.pth notebooks/models/
"""
from __future__ import annotations

from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

GPU = "A10G"          # ~$1.10/h burst tier from the cloud plan
CORPUS_VOLUME = "trueskate-corpus"
MODELS_VOLUME = "trueskate-models"

app = modal.App("trueskate-train-m1")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0")  # opencv-python-headless runtime dep
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "selenium",  # pulled by self_label -> sim.touch_actions (pure-python)
    )
    .add_local_dir(str(_REPO_ROOT / "src" / "trueskate_ai"), remote_path="/root/trueskate_ai")
    .add_local_file(str(_REPO_ROOT / "scripts" / "train" / "train_trace_extractor.py"),
                    remote_path="/root/train_trace_extractor.py")
)
corpus = modal.Volume.from_name(CORPUS_VOLUME)
models = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)


def _load_trainer():
    """Import the existing trainer script as a module inside the container."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_trace_extractor", "/root/train_trace_extractor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@app.function(image=image, gpu=GPU, timeout=4 * 3600,
              volumes={"/corpus": corpus, "/models": models})
def train_remote(epochs: int = 20, latency_s: float = 0.45, base_channels: int = 32,
                 batch_size: int = 8, lr: float = 1e-3, img_h: int = 288, img_w: int = 128,
                 data_subdir: str = "", no_require_trace: bool = False) -> str:
    """One training run on the mounted corpus; checkpoint → trueskate-models."""
    import time

    m = _load_trainer()
    m._H, m._W = img_h, img_w  # the trainer sizes its dataset/model off these globals
    root = Path("/corpus") / data_subdir if data_subdir else Path("/corpus")
    ds = m.SelfLabeledTraceDataset(root, latency_s=latency_s,
                                   require_trace=not no_require_trace)
    tag = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"/models/trace_extractor_v2_{tag}_lat{latency_s:g}.pth")
    m.train(ds, epochs=epochs, batch_size=batch_size, lr=lr, out_path=out,
            base_channels=base_channels)
    models.commit()
    return str(out)


@app.function(image=image, volumes={"/corpus": corpus}, timeout=900)
def smoke_remote() -> str:
    """Cents-cheap CPU validation: imports, one optimizer step, volume visible,
    spin-hold labelling works in-container."""
    m = _load_trainer()
    ds = m._SyntheticTraceDataset(8)
    m.train(ds, epochs=1, batch_size=4, lr=1e-3, out_path=Path("/tmp/smoke.pth"), smoke=True)
    from trueskate_ai.vision.self_label import label_frames
    labs = label_frames([(0.2, 0.8), (0.5, 0.5), (0.8, 0.2)], 1.0, 1.0,
                        [0.5, 1.4], spin_hold=(0.1, 1.5))
    assert labs[0].active and labs[0].spin_on and labs[1].spin_on and not labs[1].active
    top = [p.name for _, p in zip(range(3), Path("/corpus").iterdir())]
    return f"smoke OK (dataset→model→loss→step + spin labels); corpus mounted, e.g. {top}"


@app.local_entrypoint()
def main(smoke: bool = False, epochs: int = 20, latency_s: float = 0.45,
         base_channels: int = 32, batch_size: int = 8, lr: float = 1e-3,
         img_h: int = 288, img_w: int = 128, data_subdir: str = "",
         no_require_trace: bool = False) -> None:
    if smoke:
        print(smoke_remote.remote())
        return
    scope = f"subdir {data_subdir}" if data_subdir else "FULL corpus"
    print(f"PAID {GPU} run on {scope} (epochs={epochs}, latency_s={latency_s})...")
    out = train_remote.remote(epochs=epochs, latency_s=latency_s,
                              base_channels=base_channels, batch_size=batch_size, lr=lr,
                              img_h=img_h, img_w=img_w, data_subdir=data_subdir,
                              no_require_trace=no_require_trace)
    print(f"checkpoint on {MODELS_VOLUME}: {out}")
    print(f"fetch: .venv/bin/modal volume get {MODELS_VOLUME} {Path(out).name} notebooks/models/")
