"""Reproducible curved training, epoch checkpoints, and one-time final test."""
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

from .audit import verify_manifest
from .capture import file_sha256
from .contract import SCHEMA, digest, save_json
from .dataset import CurvedDataset
from .decoder import decode, evaluate_path
from .model import CurvedModel, loss_function


@dataclass(frozen=True)
class Recipe:
    epochs: int = 40
    batch_size: int = 8
    base_channels: int = 16
    lr: float = .001
    weight_decay: float = .01
    seed: int = 0


def implementation_hash():
    here = Path(__file__).parent
    return digest({p.name: file_sha256(p) for p in [here/n for n in
                   ("training.py", "model.py", "dataset.py", "decoder.py", "contract.py")]})


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def summarize(rows):
    if not rows:
        raise ValueError("empty evaluation")
    result = dict(count=len(rows), pass_rate=float(np.mean([r["passed"] for r in rows])),
                  mean_path_error=float(np.mean([r["mean_path_error"] for r in rows])),
                  mean_logical_error_pt=float(np.mean([r["mean_logical_error_pt"] for r in rows])),
                  strata={})
    for field in ("park", "sampler", "duration_bin", "segment_count", "onset_bin"):
        groups = {}
        for row in rows:
            groups.setdefault(str(row[field]), []).append(row)
        result["strata"][field] = {name: dict(count=len(items), pass_rate=float(np.mean([r["passed"] for r in items])))
                                   for name, items in sorted(groups.items())}
    result["target_met"] = result["pass_rate"] >= .90 and all(
        group["pass_rate"] >= .80 for group in result["strata"]["park"].values())
    return result


@torch.no_grad()
def evaluate(model, dataset, *, device, batch_size=8):
    model.eval()
    rows = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        pred = model(batch["video"].to(device), batch["times"].to(device))
        values = {k: v.cpu().numpy() for k, v in pred.items() if k != "heatmap"}
        for i, index in enumerate(batch["index"].tolist()):
            meta = dataset.meta[index]
            path = decode(values["xy"][i], values["onset"][i], values["release"][i], batch["times"][i].numpy())
            metrics = evaluate_path(path, meta)
            duration = meta["gesture"]["times_ms"][-1]
            rows.append(dict(id=meta["candidate_id"], park=meta["park"], sampler=meta["gesture"]["sampler"],
                             duration_bin="300–599" if duration < 600 else "600–899" if duration < 900 else "900–1200",
                             segment_count=len(meta["gesture"]["times_ms"])-1,
                             onset_bin=int(np.searchsorted(meta["frame_times_s"], meta["onset_s"]))/120//.25,
                             prediction=path, **metrics))
    return dict(summary=summarize(rows), samples=rows)


def fit(train, validation, out: Path, *, recipe=Recipe(), identity, device="cpu",
        resume=False, stop_after_epochs=None, on_commit=lambda: None):
    """Core engine also accepts tiny synthetic datasets for deterministic resume tests."""
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
    model = CurvedModel(recipe.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    generator = torch.Generator().manual_seed(recipe.seed)
    config = dict(recipe=asdict(recipe), identity=identity, implementation=implementation_hash(),
                  train_count=len(train), validation_count=len(validation), device=str(device),
                  torch_version=torch.__version__, numpy_version=np.__version__)
    start, history, best_key, best_epoch = 0, [], (-1., -float("inf")), None
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
    end = recipe.epochs if stop_after_epochs is None else min(recipe.epochs, start+stop_after_epochs)
    for epoch in range(start+1, end+1):
        began = time.monotonic()
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["video"], batch["times"])
            loss = loss_function(pred, batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, validation, device=device, batch_size=recipe.batch_size)
        key = (metrics["summary"]["pass_rate"], -metrics["summary"]["mean_path_error"])
        if key > best_key:
            best_key, best_epoch = key, epoch
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation=metrics["summary"],
                            elapsed_s=time.monotonic()-began))
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
        print(json.dumps(history[-1]), flush=True)
    return json.loads(latest.read_text())


def training_inputs(manifest, human):
    corpus = json.loads(Path(manifest).read_text())
    frozen_human = json.loads(Path(human).read_text())
    verify_manifest(corpus)
    verify_manifest(frozen_human)
    parks = corpus["definition"]["parks"]
    if (len(corpus["samples"]) != 12500 or corpus.get("pool_count") != 14000
            or frozen_human.get("kind") != "frozen-human-qualitative"
            or len(frozen_human["samples"]) != 10 or len(parks) != 5):
        raise ValueError("complete production corpus and frozen human set required before training")
    for park in parks:
        if sum(r["park"] == park for r in frozen_human["samples"]) != 2:
            raise ValueError("human park balance changed")
        for split, count in (("train", 2000), ("validation", 250), ("test", 250)):
            if sum(r["park"] == park and r["split"] == split for r in corpus["samples"]) != count:
                raise ValueError("frozen park/partition quotas changed")
    return dict(dataset=corpus["sha256"], human=frozen_human["sha256"])


def train(manifest, human, out, *, device="cpu", cache=False, resume=False, on_commit=lambda: None):
    identity = training_inputs(manifest, human)
    training = CurvedDataset(manifest, "train", cache=cache)
    validation = CurvedDataset(manifest, "validation", cache=cache)
    return fit(training, validation, out, identity=identity, device=device, resume=resume, on_commit=on_commit)


def test_once(manifest, out, *, device="cpu", on_commit=lambda: None):
    out = Path(out)
    pointer = json.loads((out/"latest.json").read_text())
    if not pointer["complete"]:
        raise ValueError("finish training before exposing test")
    chosen = out/pointer["best_checkpoint"]
    payload = torch.load(chosen, map_location="cpu", weights_only=False)
    corpus = json.loads(Path(manifest).read_text())
    verify_manifest(corpus)
    if payload["config"]["identity"]["dataset"] != corpus["sha256"]:
        raise ValueError("test corpus changed")
    # Consume exposure before loading test labels: an interrupted test is recorded.
    save_json(out/"test-exposure.json", dict(dataset=corpus["sha256"], checkpoint=chosen.name,
                                            checkpoint_sha256=file_sha256(chosen)))
    on_commit()
    dataset = CurvedDataset(manifest, "test")
    model = CurvedModel(payload["config"]["recipe"]["base_channels"]).to(device)
    model.load_state_dict(payload["model"])
    result = evaluate(model, dataset, device=device)
    save_json(out/"test-results.json", result)
    on_commit()
    return result["summary"]
