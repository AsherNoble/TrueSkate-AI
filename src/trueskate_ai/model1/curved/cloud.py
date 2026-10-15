"""Offline cost estimates and bounded sequential bundles for the curved corpus."""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from .audit import verify_manifest
from .capture import file_sha256
from .contract import digest, save_json

RATES = dict(as_of="2026-10-11", source="https://modal.com/pricing", gpu_l4_s=.000222,
             cpu_core_s=.0000131, memory_gib_s=.00000222, storage_gib_month=.09)
HOURLY_USD = 3600*(RATES["gpu_l4_s"]+4*RATES["cpu_core_s"]+64*RATES["memory_gib_s"])


def quote(benchmark=None, *, storage_gib=0):
    result = dict(rates=RATES, gpu="L4", cpu=4, memory_gib=64, compute_hour_usd=HOURLY_USD,
                  benchmark_execution_limit_s=900, benchmark_compute_usd=HOURLY_USD*.25,
                  storage_upper_usd_month=storage_gib*RATES["storage_gib_month"],
                  note="Estimate excludes image build, cold start, transfer and account credits; price is not a spending authorization.")
    if benchmark is not None:
        seconds = (11250*benchmark["load_seconds_per_clip"]
                   +40*(1250*benchmark["train_step_s"]+1250*benchmark["evaluation_s_per_clip"])
                   +1250*(benchmark["load_seconds_per_clip"]+benchmark["evaluation_s_per_clip"]))
        result.update(estimated_train_and_test_s=seconds, compute_estimate_usd=seconds/3600*HOURLY_USD,
                      suggested_compute_budget_usd=math.ceil(seconds/3600*HOURLY_USD*1.5+1),
                      benchmark=benchmark)
    return result


def bundle(manifest: Path, human: Path, out: Path, *, max_bytes=2*2**30, max_samples=512):
    manifest, human, out = Path(manifest).resolve(), Path(human).resolve(), Path(out).resolve()
    data = json.loads(manifest.read_text())
    verify_manifest(data)
    verify_manifest(json.loads(human.read_text()))
    if max_bytes <= 0 or max_samples <= 0:
        raise ValueError("positive shard bounds required")
    out.mkdir(parents=True, exist_ok=False)
    groups, current, size = [], [], 0
    for row in data["samples"]:
        folder = (manifest.parent/row["sample_dir"]).resolve()
        if not folder.is_relative_to(manifest.parent):
            raise ValueError("sample escapes corpus")
        if any((folder/m).exists() for m in (".menu", ".trace_mismatch", ".contaminated")):
            raise ValueError("contamination marker excludes sample from upload")
        files = [folder/"meta.json", folder/"clip.mp4"]
        if file_sha256(files[0]) != row["meta_sha256"] or file_sha256(files[1]) != row["media_sha256"]:
            raise ValueError("corpus changed before upload")
        length = sum(f.stat().st_size for f in files)
        if length > max_bytes:
            raise ValueError("individual sample exceeds shard bound")
        if current and (size+length > max_bytes or len(current) == max_samples):
            groups.append(current)
            current, size = [], 0
        current.append(files)
        size += length
    if current:
        groups.append(current)
    shards = []
    for n, group in enumerate(groups):
        path = out/f"part-{n:04}.tar"
        with tarfile.open(path, "w") as archive:
            for files in group:
                for source in files:
                    archive.add(source, arcname=str(source.relative_to(manifest.parent)), recursive=False)
        shards.append(dict(file=path.name, sha256=file_sha256(path), bytes=path.stat().st_size))
    shutil.copy2(manifest, out/"dataset.json")
    shutil.copy2(human, out/"human.json")
    result = dict(kind="curved-native-bundle", dataset_sha256=data["sha256"],
                  human_file_sha256=file_sha256(human), shards=shards)
    result["sha256"] = digest(result)
    save_json(out/"bundle.json", result)
    return result


def stage(bundle_root: Path, destination: Path):
    bundle_root, destination = Path(bundle_root), Path(destination)
    index = json.loads((bundle_root/"bundle.json").read_text())
    if index.get("sha256") != digest({k: v for k, v in index.items() if k != "sha256"}):
        raise ValueError("bundle fingerprint mismatch")
    data = json.loads((bundle_root/"dataset.json").read_text())
    verify_manifest(data)
    if data["sha256"] != index["dataset_sha256"] or file_sha256(bundle_root/"human.json") != index["human_file_sha256"]:
        raise ValueError("bundle manifest changed")
    allowed = {str(PurePosixPath(r["sample_dir"])/name) for r in data["samples"] for name in ("meta.json", "clip.mp4")}
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    for shard in index["shards"]:
        source = bundle_root/shard["file"]
        if source.parent.resolve() != bundle_root.resolve() or file_sha256(source) != shard["sha256"]:
            raise ValueError("shard changed or escapes bundle")
        with tarfile.open(source) as archive:
            for member in archive:
                path = (destination/member.name).resolve()
                if (not member.isfile() or member.name not in allowed or member.name in seen
                        or not path.is_relative_to(destination.resolve())):
                    raise ValueError("unsafe/duplicate/unexpected archive member")
                seen.add(member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
    if seen != allowed:
        raise ValueError("incomplete bundle")
    for name in ("dataset.json", "human.json"):
        shutil.copy2(bundle_root/name, destination/name)
    return destination/"dataset.json", destination/"human.json"


def benchmark(manifest, *, device="cuda"):
    import itertools
    import time
    import torch
    from torch.utils.data import DataLoader
    from .dataset import CurvedDataset
    from .model import CurvedModel, loss_function
    from .training import evaluate, implementation_hash
    started = time.monotonic()
    train = CurvedDataset(manifest, "train", cache=True, limit=64)
    validation = CurvedDataset(manifest, "validation", cache=True, limit=16)
    loading = time.monotonic()-started
    model = CurvedModel().to(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    steps = []
    for i, batch in enumerate(itertools.islice(itertools.cycle(DataLoader(train, batch_size=8)), 24)):
        torch.cuda.synchronize()
        started = time.monotonic()
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(batch["video"], batch["times"]), batch)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if i >= 4:
            steps.append(time.monotonic()-started)
    started = time.monotonic()
    evaluate(model, validation, device=device)
    elapsed = time.monotonic()-started
    return dict(dataset_sha256=train.manifest["sha256"], implementation=implementation_hash(),
                gpu=torch.cuda.get_device_name(), train_step_s=sum(steps)/len(steps),
                evaluation_s_per_clip=elapsed/len(validation), load_seconds_per_clip=loading/(len(train)+len(validation)),
                measured_train_steps=len(steps), measured_validation_clips=len(validation))
