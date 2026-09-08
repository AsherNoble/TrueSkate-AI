"""Train temporal Model 1 on Modal over the SLS Super Crown corpus.

The production path is the causal recurrent tracker in
``train_temporal_trace_extractor.py``: full gesture sequences, RGB/heatmap
history, scheduled sampling, multi-touch targets, and a strict 90% positive +
90% negative acceptance gate.  Legacy single-frame code remains mounted only
for the already-established XCTest latency audit.

Dataset startup reads every retained frame once over the volume FUSE mount.
Modal uses bounded sample-level cache workers so those independent high-latency
reads overlap while deterministic candidate ordering and statistics are kept.

Run (repo root, ~/.modal.toml auth):
    .venv/bin/modal run scripts/cloud/train_trace_extractor_modal.py --smoke
        # cents: CPU container, synthetic one-step train + volume/import checks
    .venv/bin/modal run scripts/cloud/train_trace_extractor_modal.py --data-subdir <session>
        # PAID A10G mini-run on one session
    .venv/bin/modal run scripts/cloud/train_trace_extractor_modal.py --latency-s 0.2
        # PAID A10G temporal run, SLS Super Crown only

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
        "scipy",
        "matplotlib",  # legacy latency sweep imports gaussian_bump_predictor
        "selenium",  # pulled by self_label -> sim.touch_actions (pure-python)
    )
    .env({"PYTHONPATH": "/root/src"})
    # Preserve the repository layout expected by both trainer entry points:
    # scripts/train/<file>.py resolves the import root as /root/src.
    .add_local_dir(
        str(_REPO_ROOT / "src" / "trueskate_ai"),
        remote_path="/root/src/trueskate_ai",
    )
    .add_local_file(
        str(_REPO_ROOT / "scripts" / "train" / "train_temporal_trace_extractor.py"),
        remote_path="/root/scripts/train/train_temporal_trace_extractor.py",
    )
    .add_local_file(str(_REPO_ROOT / "scripts" / "train" / "train_trace_extractor.py"),
                    remote_path="/root/scripts/train/train_trace_extractor_legacy.py")
    .add_local_file(str(_REPO_ROOT / "scripts" / "data" / "build_bc_clips.py"),
                    remote_path="/root/scripts/data/build_bc_clips.py")
)
corpus = modal.Volume.from_name(CORPUS_VOLUME)
models = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)


def _load_trainer():
    """Import the temporal trainer script inside the container."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_temporal_trace_extractor",
        "/root/scripts/train/train_temporal_trace_extractor.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_legacy_trainer():
    """Import the old single-frame trainer solely for latency sweeps."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_trace_extractor_legacy",
        "/root/scripts/train/train_trace_extractor_legacy.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_build_bc_clips():
    """Import the Model-1 -> peaks -> tracks -> strokes bridge inside the container."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "build_bc_clips", "/root/scripts/data/build_bc_clips.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves cls.__module__ via sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@app.function(image=image, gpu=GPU, timeout=8 * 3600, memory=32768,
              volumes={"/corpus": corpus, "/models": models})
def train_remote(epochs: int = 40, latency_s: float = 0.2, base_channels: int = 16,
                 hidden_channels: int = 32, downsample_stages: int = 2,
                 batch_size: int = 2, lr: float = 1e-3, img_h: int = 288, img_w: int = 128,
                 sequence_length: int = 24, max_touches: int = 4,
                 data_subdir: str = "", no_require_trace: bool = False,
                 data_match: str = "supercrown", val_fraction: float = 0.15,
                 split_seed: int = 0, resume_name: str = "",
                 resume_weights_only: bool = False, max_samples: int = 3000,
                 cache_workers: int = 16, teacher_start: float = 0.90,
                 teacher_end: float = 0.05, teacher_warmup_epochs: int = 1,
                 teacher_decay_epochs: int = 0, feedback_dropout: float = 0.10,
                 feedback_noise_std: float = 0.03,
                 heatmap_loss_weight: float = 1.0,
                 activity_loss_weight: float = 1.0,
                 heatmap_positive_fraction: float = 0.5,
                 activity_positive_fraction: float = 0.5,
                 overlap_sampling_fraction: float = 0.25,
                 hard_negative_weight: float = 0.0,
                 hard_negative_top_k: int = 64,
                 hard_negative_target_exclusion_threshold: float = 0.05,
                 model_seed: int = 0, lr_plateau_patience: int = 3,
                 cosine_min_lr: float = -1.0,
                 peak_threshold_grid: str = "0.30,0.50,0.65,0.75,0.85,0.90,0.95",
                 activity_threshold_grid: str = "0.50,0.70,0.80,0.90,0.95,0.98",
                 peak_nms_radius_px: int = 6, heatmap_sigma: float = 6.0) -> str:
    """One causal temporal run; best v3 checkpoint -> trueskate-models."""
    import time

    m = _load_trainer()
    peak_thresholds = m._parse_probability_grid(
        peak_threshold_grid, name="peak_threshold_grid"
    )
    activity_thresholds = m._parse_probability_grid(
        activity_threshold_grid, name="activity_threshold_grid"
    )
    root = Path("/corpus") / data_subdir if data_subdir else Path("/corpus")
    ds = m.TemporalTraceSequenceDataset(
        root,
        sequence_length=sequence_length,
        image_height=img_h,
        image_width=img_w,
        max_touches=max_touches,
        latency_s=latency_s,
        heatmap_sigma=heatmap_sigma,
        require_trace=not no_require_trace,
        include_path_term=data_match or None,
        max_samples=max_samples or None,
        cache_frames=True,
        cache_workers=cache_workers,
        detect_menu_frames=True,
    )
    print(f"temporal dataset stats: {ds.stats}")
    train_ds, val_ds = m.split_by_sample(
        ds, val_fraction=val_fraction, seed=split_seed)
    print(f"sample-level split: {len(train_ds)} train gestures / {len(val_ds)} val gestures")
    tag = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"/models/trace_extractor_temporal_v1_{tag}_lat{latency_s:g}.pth")
    resume = Path("/models") / resume_name if resume_name else None
    m.train_temporal(
        train_ds,
        val_dataset=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        out_path=out,
        image_height=img_h,
        image_width=img_w,
        sequence_length=sequence_length,
        latency_s=latency_s,
        heatmap_sigma=heatmap_sigma,
        base_channels=base_channels,
        hidden_channels=hidden_channels,
        downsample_stages=downsample_stages,
        sampling_seed=split_seed,
        model_seed=model_seed,
        resume_path=resume,
        resume_weights_only=resume_weights_only,
        target_accuracy=0.9,
        peak_thresholds=peak_thresholds,
        activity_thresholds=activity_thresholds,
        peak_nms_radius_px=peak_nms_radius_px,
        teacher_start=teacher_start,
        teacher_end=teacher_end,
        teacher_warmup_epochs=teacher_warmup_epochs,
        teacher_decay_epochs=teacher_decay_epochs or None,
        feedback_dropout=feedback_dropout,
        feedback_noise_std=feedback_noise_std,
        heatmap_loss_weight=heatmap_loss_weight,
        activity_loss_weight=activity_loss_weight,
        heatmap_positive_fraction=heatmap_positive_fraction,
        activity_positive_fraction=activity_positive_fraction,
        overlap_sampling_fraction=overlap_sampling_fraction,
        hard_negative_weight=hard_negative_weight,
        hard_negative_top_k=hard_negative_top_k,
        hard_negative_target_exclusion_threshold=(
            hard_negative_target_exclusion_threshold
        ),
        lr_plateau_patience=lr_plateau_patience or None,
        cosine_min_lr=(cosine_min_lr if cosine_min_lr >= 0.0 else None),
        checkpoint_callback=models.commit,
    )
    models.commit()
    return str(out)


@app.function(image=image, volumes={"/corpus": corpus}, timeout=900)
def smoke_remote() -> str:
    """CPU validation: causal rollout, balanced loss, v3 save, volume mount."""
    import tempfile

    m = _load_trainer()
    ds = m._SyntheticTemporalDataset(6)
    train_ds, val_ds = m.split_by_sample(ds, val_fraction=0.33, seed=0)
    out = Path(tempfile.gettempdir()) / "temporal_trace_smoke.pth"
    m.train_temporal(
        train_ds, val_dataset=val_ds, epochs=1, batch_size=2, learning_rate=1e-3,
        out_path=out, image_height=ds.height, image_width=ds.width,
        sequence_length=ds.steps, latency_s=0.2, base_channels=4,
        hidden_channels=8, peak_thresholds=(0.3, 0.7),
        activity_thresholds=(0.5, 0.9), hard_negative_weight=0.10,
        hard_negative_top_k=4, heatmap_positive_fraction=0.70,
        activity_positive_fraction=0.50, overlap_sampling_fraction=0.25,
        lr_plateau_patience=None, smoke=True,
    )
    import torch
    checkpoint = torch.load(out, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_version"] == 3
    assert checkpoint["model_type"] == "temporal_trace_predictor_v1"
    assert checkpoint["training_config"]["heatmap_positive_fraction"] == 0.70
    assert checkpoint["training_config"]["overlap_sampling_fraction"] == 0.25
    top = [p.name for _, p in zip(range(3), Path("/corpus").iterdir())]
    return f"temporal smoke OK (sequence->rollout->loss->v3 checkpoint); corpus mounted, e.g. {top}"


@app.function(image=image, volumes={"/corpus": corpus}, timeout=3600)
def sweep_latency_one(latency: float, data_match: str, max_samples: int) -> dict:
    """Measure one candidate on the deterministic XCTest calibration sample."""
    m = _load_legacy_trainer()
    ds = m.SelfLabeledTraceDataset(
        Path("/corpus"), latency_s=latency, require_trace=True,
        include_path_term=data_match or None, max_samples=max_samples,
        negative_keep_frac=0.0, cache_frames=False, allow_empty=True)
    return {"latency_s": latency, **ds.stats, "retained_frames": len(ds)}


@app.function(image=image, gpu=GPU, timeout=3600, memory=16384,
              volumes={"/corpus": corpus, "/models": models})
def stroke_recovery_remote(checkpoint_name: str, data_match: str = "supercrown",
                           max_samples: int = 600, peak_threshold: float = 0.45,
                           activity_threshold: float = 0.5, tol: float = 0.12) -> dict:
    """STROKE-level recovery of Model 1 over the corpus (the Model-2-relevant metric).

    Model 2 consumes assembled strokes, so we measure whether Model 1 -> peaks ->
    causal tracks -> assemble_strokes recovers the command-manifest ground-truth
    strokes.  Matching is on start+end endpoint geometry (latency-independent).
    """
    import json
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import torch

    from trueskate_ai.vision.temporal_trace_dataset import (
        _schedule_from_meta, _UnsupportedSample, discover_sample_paths)
    bb = _load_build_bc_clips()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = bb.load_trace_extractor(Path(f"/models/{checkpoint_name}"), device)
    inf, _ = bb.resolve_inference_config(ext)
    inf["peak_threshold"] = peak_threshold
    inf["activity_threshold"] = activity_threshold

    def gt_eps(touches):
        out = []
        for t in touches:
            if t.constant_xy is not None or len(t.waypoints) < 2:
                continue
            out.append((np.asarray(t.waypoints[0], float),
                        np.asarray(t.waypoints[-1], float)))
        return out

    def rec_eps(strokes):
        return [(np.array([s.params[0], s.params[1]]),
                 np.array([s.params[4], s.params[5]])) for s in strokes]

    def match(gt, rec, t):
        pairs = sorted(
            (0.5 * (np.linalg.norm(gs - rs) + np.linalg.norm(ge - re)), gi, ri)
            for gi, (gs, ge) in enumerate(gt) for ri, (rs, re) in enumerate(rec))
        ug, ur, m = set(), set(), []
        for d, gi, ri in pairs:
            if d <= t and gi not in ug and ri not in ur:
                ug.add(gi); ur.add(ri); m.append(d)
        return m

    paths = discover_sample_paths(Path("/corpus"), include_path_term=data_match or None,
                                  max_samples=max_samples or None)
    tols = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
    n_gt = n_rec = n_samples = skipped = 0
    matched = {t: 0 for t in tols}
    per_kind: dict = {}
    ep_errs: list = []
    for sd in paths:
        if any((sd / f).exists() for f in (".menu", ".editor", ".modal")):
            skipped += 1; continue
        try:
            meta = json.loads((sd / "meta.json").read_text())
        except (OSError, json.JSONDecodeError):
            skipped += 1; continue
        frames = sorted(sd.glob("frame_*.png"))
        if len(frames) < 2:
            skipped += 1; continue
        try:
            touches, _total, kind, _req = _schedule_from_meta(
                meta, sd, max_touches=4, finger_stagger_s=None)
        except (_UnsupportedSample, ValueError, KeyError):
            skipped += 1; continue
        gt = gt_eps(touches)
        if not gt:
            skipped += 1; continue
        ft = meta.get("frame_times")
        times = (np.asarray(ft, float) if ft and len(ft) == len(frames)
                 else np.arange(len(frames), dtype=float) / 30.0)
        try:
            tracks = bb.frames_to_touch_tracks(
                ext.model, frames, times, ext.h, ext.w, device,
                model_type=bb._TEMPORAL_MODEL_TYPE,
                active_thresh=float(inf["peak_threshold"]),
                activity_thresh=float(inf["activity_threshold"]),
                max_touches=int(inf["max_touches"]),
                peak_nms_radius_px=int(inf["peak_nms_radius_px"]))
        except Exception:  # noqa: BLE001
            skipped += 1; continue
        rec = rec_eps(bb.touch_tracks_to_strokes(tracks))
        n_samples += 1; n_gt += len(gt); n_rec += len(rec)
        pk = per_kind.setdefault(kind, [0, 0, 0])
        pk[0] += len(gt); pk[1] += len(rec)
        for t in tols:
            m = match(gt, rec, t)
            matched[t] += len(m)
            if abs(t - tol) < 1e-9:
                pk[2] += len(m); ep_errs.extend(m)
    result = {
        "checkpoint": checkpoint_name, "peak_threshold": peak_threshold,
        "n_samples": n_samples, "skipped": skipped, "n_gt": n_gt, "n_rec": n_rec,
        "recall_by_tol": {f"{t:.2f}": (matched[t] / n_gt if n_gt else 0.0) for t in tols},
        "precision_by_tol": {f"{t:.2f}": (matched[t] / n_rec if n_rec else 0.0) for t in tols},
        "matched_by_tol": {f"{t:.2f}": matched[t] for t in tols},
        "endpoint_err_mean_at_tol": (float(np.mean(ep_errs)) if ep_errs else None),
        "endpoint_err_median_at_tol": (float(np.median(ep_errs)) if ep_errs else None),
        "tol": tol,
        "per_kind": {k: {"gt": v[0], "rec": v[1], "matched": v[2],
                         "recall": (v[2] / v[0] if v[0] else 0.0)}
                     for k, v in sorted(per_kind.items())},
    }
    # Persist to the models volume so a flaky client connection cannot lose the
    # result: fetch with `modal volume get trueskate-models stroke_recovery_*.json`.
    out = Path(f"/models/stroke_recovery_{checkpoint_name.replace('.pth','')}_pt{peak_threshold:g}.json")
    out.write_text(json.dumps(result, indent=2))
    models.commit()
    print(json.dumps(result, indent=2))
    return result


@app.function(image=image, volumes={"/corpus": corpus}, timeout=3600)
def audit_selection_remote(data_match: str = "supercrown", max_samples: int = 100) -> dict:
    """Inspect selected metadata without paying to read any frame pixels."""
    import json
    from collections import Counter

    from trueskate_ai.vision.temporal_trace_dataset import discover_sample_paths

    paths = discover_sample_paths(
        Path("/corpus"), include_path_term=data_match or None,
        max_samples=max_samples or None,
    )
    kinds = Counter()
    spin_kinds = Counter()
    sessions = Counter()
    flags = Counter()
    for path in paths:
        if (path / ".menu").exists():
            flags["menu"] += 1
        if (path / ".editor").exists():
            flags["editor"] += 1
        try:
            meta = json.loads((path / "meta.json").read_text())
        except (OSError, json.JSONDecodeError):
            flags["bad_meta"] += 1
            continue
        kind = str(meta.get("gesture_distribution", "?"))
        kinds[kind] += 1
        if bool(meta.get("spin_active", kind in ("spin", "spin_flick"))):
            spin_kinds[kind] += 1
        relative = path.relative_to("/corpus")
        sessions[relative.parts[0] if relative.parts else "?"] += 1
    return {
        "selected": len(paths),
        "kinds": dict(kinds),
        "spin_active": sum(spin_kinds.values()),
        "spin_by_kind": dict(spin_kinds),
        "flags": dict(flags),
        "sessions": dict(sessions),
    }


@app.local_entrypoint()
def main(smoke: bool = False, sweep_latency: bool = False,
         audit_selection: bool = False, stroke_recovery: bool = False,
         recovery_checkpoint: str = "", recovery_peak_threshold: float = 0.45,
         recovery_tol: float = 0.12, epochs: int = 40, latency_s: float = 0.2,
         base_channels: int = 16, hidden_channels: int = 32, downsample_stages: int = 2,
         batch_size: int = 2, lr: float = 1e-3,
         img_h: int = 288, img_w: int = 128, sequence_length: int = 24,
         max_touches: int = 4, data_subdir: str = "",
         no_require_trace: bool = False, data_match: str = "supercrown",
         val_fraction: float = 0.15, split_seed: int = 0,
         resume_name: str = "", resume_weights_only: bool = False,
         max_samples: int = 3000,
         cache_workers: int = 16, teacher_start: float = 0.90,
         teacher_end: float = 0.05, teacher_warmup_epochs: int = 1,
         teacher_decay_epochs: int = 0, feedback_dropout: float = 0.10,
         feedback_noise_std: float = 0.03,
         heatmap_loss_weight: float = 1.0,
         activity_loss_weight: float = 1.0,
         heatmap_positive_fraction: float = 0.5,
         activity_positive_fraction: float = 0.5,
         overlap_sampling_fraction: float = 0.25,
         hard_negative_weight: float = 0.0,
         hard_negative_top_k: int = 64,
         hard_negative_target_exclusion_threshold: float = 0.05,
         model_seed: int = 0, lr_plateau_patience: int = 3,
         cosine_min_lr: float = -1.0,
         peak_threshold_grid: str = "0.30,0.50,0.65,0.75,0.85,0.90,0.95",
         activity_threshold_grid: str = "0.50,0.70,0.80,0.90,0.95,0.98",
         peak_nms_radius_px: int = 6, heatmap_sigma: float = 6.0) -> None:
    if smoke:
        print(smoke_remote.remote())
        return
    if sweep_latency:
        import json
        latencies = (0.1, 0.15, 0.2, 0.25, 0.3, 0.45)
        results = list(sweep_latency_one.map(
            latencies, [data_match] * len(latencies), [max_samples] * len(latencies)))
        print(json.dumps(sorted(results, key=lambda x: x["latency_s"]), indent=2))
        return
    if audit_selection:
        import json
        print(json.dumps(audit_selection_remote.remote(data_match, max_samples), indent=2))
        return
    if stroke_recovery:
        import json
        if not recovery_checkpoint:
            raise SystemExit("--recovery-checkpoint <name.pth> is required for --stroke-recovery")
        result = stroke_recovery_remote.remote(
            recovery_checkpoint, data_match=data_match, max_samples=max_samples,
            peak_threshold=recovery_peak_threshold, tol=recovery_tol)
        print(json.dumps(result, indent=2))
        return
    scope = f"subdir {data_subdir}" if data_subdir else "corpus"
    scope += f" filtered by {data_match!r}" if data_match else ""
    print(f"PAID {GPU} run on {scope} (epochs={epochs}, latency_s={latency_s})...")
    out = train_remote.remote(epochs=epochs, latency_s=latency_s,
                              base_channels=base_channels, hidden_channels=hidden_channels,
                              downsample_stages=downsample_stages,
                              batch_size=batch_size, lr=lr,
                              img_h=img_h, img_w=img_w,
                              sequence_length=sequence_length, max_touches=max_touches,
                              data_subdir=data_subdir,
                              no_require_trace=no_require_trace, data_match=data_match,
                              val_fraction=val_fraction, split_seed=split_seed,
                              resume_name=resume_name,
                              resume_weights_only=resume_weights_only,
                              max_samples=max_samples,
                              cache_workers=cache_workers,
                              teacher_start=teacher_start, teacher_end=teacher_end,
                              teacher_warmup_epochs=teacher_warmup_epochs,
                              teacher_decay_epochs=teacher_decay_epochs,
                              feedback_dropout=feedback_dropout,
                              feedback_noise_std=feedback_noise_std,
                              heatmap_loss_weight=heatmap_loss_weight,
                              activity_loss_weight=activity_loss_weight,
                              heatmap_positive_fraction=heatmap_positive_fraction,
                              activity_positive_fraction=activity_positive_fraction,
                              overlap_sampling_fraction=overlap_sampling_fraction,
                              hard_negative_weight=hard_negative_weight,
                              hard_negative_top_k=hard_negative_top_k,
                              hard_negative_target_exclusion_threshold=(
                                  hard_negative_target_exclusion_threshold
                              ), model_seed=model_seed,
                              lr_plateau_patience=lr_plateau_patience,
                              cosine_min_lr=cosine_min_lr,
                              peak_threshold_grid=peak_threshold_grid,
                              activity_threshold_grid=activity_threshold_grid,
                              peak_nms_radius_px=peak_nms_radius_px,
                              heatmap_sigma=heatmap_sigma)
    print(f"checkpoint on {MODELS_VOLUME}: {out}")
    print(f"fetch: .venv/bin/modal volume get {MODELS_VOLUME} {Path(out).name} notebooks/models/")
