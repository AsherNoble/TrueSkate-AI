"""Trick Model 1 MVP v0 training: curved v1's fit() with the model, loss and scoring swapped.

Kept: seeds and deterministic algorithms, a checkpoint and validation file every epoch,
the checkpoint chosen on validation only, exact resume, and a test split scored once.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from trueskate_ai.model1.curved.capture import file_sha256
from trueskate_ai.model1.curved.contract import digest, save_json
from trueskate_ai.model1.curved.training import atomic_checkpoint
from .contract import SCHEMA
from .dataset import TrickDataset, load_bundle
from .decoder import LEVELS, decode, score_clip
from .model import TrickModel, loss_function


@dataclass(frozen=True)
class Recipe:
    epochs: int = 60
    batch_size: int = 8
    base_channels: int = 16
    lr: float = .001
    weight_decay: float = .01
    seed: int = 0


def implementation_hash():
    model1 = Path(__file__).parents[1]
    return digest({name: file_sha256(model1/name) for name in (
        "trick/training.py", "trick/model.py", "trick/dataset.py", "trick/decoder.py", "trick/contract.py",
        "curved/model.py")})


def duration_bin(ms) -> str:
    return "50–99" if ms < 100 else "100–199" if ms < 200 else "200–399" if ms < 400 else "400–600"


def spread(values) -> dict | None:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return dict(count=len(values), p10=float(np.percentile(values, 10)), p50=float(np.percentile(values, 50)),
                p90=float(np.percentile(values, 90)), mean_abs=float(np.mean(np.abs(values))))


def rate(items, key) -> float:
    return float(np.mean([bool(item[key]) for item in items])) if items else 0.


def summarize(rows) -> dict:
    if not rows:
        raise ValueError("empty evaluation")
    strokes = [dict(s, device=row["device"]) for row in rows for s in row["strokes"]]
    matched = [s for s in strokes if s["matched"]]
    passes = [f"pass_{level:g}" for level in LEVELS]
    result = dict(clips=len(rows), strokes=len(strokes), predicted=sum(r["predicted_strokes"] for r in rows),
                  count_accuracy=rate(rows, "count_ok"), matched_rate=rate(strokes, "matched"),
                  extra_per_clip=float(np.mean([r["extra_strokes"] for r in rows])),
                  timing_pass=rate(strokes, "timing_ok"),
                  mean_path_error=float(np.mean([s["path_mean"] for s in matched])) if matched else 1.,
                  **{f"stroke_{key}": rate(strokes, key) for key in passes},
                  **{f"clip_{key}": rate(rows, f"clip_{key}") for key in passes},
                  errors={key: spread([s[key] for s in matched]) for key in (
                      "onset_error_ms", "release_error_ms", "duration_error_ms",
                      "path_mean", "path_p95", "path_max", "start_error", "end_error")},
                  strata={})
    for field in ("family", "pieces", "duration_bin", "device"):
        groups = {}
        for s in strokes:
            groups.setdefault(str(s[field]), []).append(s)
        result["strata"][field] = {
            name: dict(count=len(items), matched=rate(items, "matched"), timing_pass=rate(items, "timing_ok"),
                       path_max_p50=spread([s["path_max"] for s in items if s["matched"]]),
                       **{key: rate(items, key) for key in passes})
            for name, items in sorted(groups.items())}
    return result


def selection_key(summary) -> tuple:
    return (summary["stroke_pass_0.05"], summary["timing_pass"], -summary["mean_path_error"])


@torch.no_grad()
def predict(model, video, times, *, device) -> list[list[dict]]:
    """Decoded strokes for a batch of 120-frame clips (uint8 video, clip times in seconds)."""
    model.eval()
    pred = model(video.to(device), times.to(device))
    values = {k: v.float().cpu().numpy() for k, v in pred.items() if k != "heatmap"}
    return [decode(values["xy"][i], values["contact"][i], values["down"][i], values["up"][i], times[i].numpy())
            for i in range(len(video))]


def evaluate(model, dataset, *, device, batch_size=8) -> dict:
    rows = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        for index, strokes in zip(batch["index"].tolist(), predict(model, batch["video"], batch["times"], device=device)):
            row, score = dataset.rows[index], score_clip(strokes, dataset.meta[index])
            for s in score["strokes"]:
                s["duration_bin"] = duration_bin(s["duration_ms"])
            rows.append(dict(id=row["id"], device=row["device"], recording=row["recording"], run=row["run"],
                             prediction=strokes, **score))
    return dict(summary=summarize(rows), samples=rows)


def fit(train, validation, out: Path, *, recipe=Recipe(), identity, device="cpu",
        resume=False, stop_after_epochs=None, on_commit=lambda: None):
    """Core engine; also takes tiny synthetic datasets for deterministic resume tests."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if recipe.epochs < 1 or recipe.batch_size < 1:
        raise ValueError("positive training limits required")
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = TrickModel(recipe.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    generator = torch.Generator().manual_seed(recipe.seed)
    config = dict(recipe=asdict(recipe), identity=identity, implementation=implementation_hash(),
                  train_count=len(train), validation_count=len(validation), device=str(device),
                  torch_version=torch.__version__, numpy_version=np.__version__)
    start, history, best_key, best_epoch = 0, [], None, None
    latest = out/"latest.json"
    if resume:
        pointer = json.loads(latest.read_text())
        path = out/pointer["checkpoint"]
        if file_sha256(path) != pointer["sha256"]:
            raise ValueError("checkpoint changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["config"] != config:
            raise ValueError("resume inputs, implementation or recipe changed")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start, history = payload["epoch"], payload["history"]
        best_key, best_epoch = tuple(payload["best_key"]), payload["best_epoch"]
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        generator.set_state(payload["loader_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    elif latest.exists() or (out/"config.json").exists():
        raise ValueError("existing training run requires --resume")
    else:
        save_json(out/"config.json", config)
    loader = DataLoader(train, batch_size=recipe.batch_size, shuffle=True, generator=generator, num_workers=0)
    end = recipe.epochs if stop_after_epochs is None else min(recipe.epochs, start + stop_after_epochs)
    for epoch in range(start + 1, end + 1):
        began = time.monotonic()
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch["video"], batch["times"]), batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation, device=device, batch_size=recipe.batch_size)
        key = selection_key(metrics["summary"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation=metrics["summary"],
                            elapsed_s=time.monotonic() - began))
        name = f"epoch-{epoch:03}.pt"
        payload = dict(schema=SCHEMA, config=config, epoch=epoch, model=model.state_dict(),
                       optimizer=optimizer.state_dict(), history=history, best_key=best_key, best_epoch=best_epoch,
                       python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                       loader_rng=generator.get_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None)
        atomic_checkpoint(out/name, payload)
        save_json(out/f"validation-{epoch:03}.json", metrics, exclusive=False)
        save_json(latest, dict(checkpoint=name, sha256=file_sha256(out/name),
                               best_checkpoint=f"epoch-{best_epoch:03}.pt", epoch=epoch,
                               complete=epoch == recipe.epochs), exclusive=False)
        on_commit()  # Modal volume commit, after a complete atomic epoch.
        summary = metrics["summary"]
        print(json.dumps(dict(epoch=epoch, loss=round(history[-1]["loss"], 4), elapsed_s=round(history[-1]["elapsed_s"], 1),
                              count=round(summary["count_accuracy"], 3), timing=round(summary["timing_pass"], 3),
                              path=round(summary["mean_path_error"], 4),
                              **{k: round(summary[f"stroke_pass_{k}"], 3) for k in ("0.02", "0.05")})), flush=True)
    return json.loads(latest.read_text())


def train(bundle: Path, out: Path, *, recipe=Recipe(), device="cpu", resume=False, stop_after_epochs=None,
          on_commit=lambda: None):
    index, manifest = load_bundle(bundle)
    return fit(TrickDataset(bundle, "train"), TrickDataset(bundle, "validation"), out, recipe=recipe,
               identity=dict(bundle=index["sha256"], manifest=manifest["sha256"]), device=device, resume=resume,
               stop_after_epochs=stop_after_epochs, on_commit=on_commit)


def load_model(checkpoint: Path, *, device="cpu") -> TrickModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TrickModel(payload["config"]["recipe"]["base_channels"]).to(device)
    model.load_state_dict(payload["model"])
    return model


def test_once(bundle: Path, out: Path, *, device="cpu", on_commit=lambda: None):
    """Score the validation-chosen checkpoint on the test split; a second call refuses."""
    out = Path(out)
    pointer = json.loads((out/"latest.json").read_text())
    if not pointer["complete"]:
        raise ValueError("finish training before exposing test")
    chosen = out/pointer["best_checkpoint"]
    index, _ = load_bundle(bundle)
    if torch.load(chosen, map_location="cpu", weights_only=False)["config"]["identity"]["bundle"] != index["sha256"]:
        raise ValueError("checkpoint was trained on another bundle")
    # Consume the exposure before loading test labels: an interrupted test still counts.
    save_json(out/"test-exposure.json", dict(bundle=index["sha256"], checkpoint=chosen.name,
                                             checkpoint_sha256=file_sha256(chosen)))
    on_commit()
    result = evaluate(load_model(chosen, device=device), TrickDataset(bundle, "test"), device=device)
    save_json(out/"test-results.json", result)
    on_commit()
    return result["summary"]
