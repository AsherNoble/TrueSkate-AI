"""Offline bundle/quote, then explicitly budgeted Modal upload or compute.

Use Python (not `modal run`): budget validation occurs before creating the app.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from trueskate_ai.model1.curved.cloud import HOURLY_USD, quote, bundle

CORPUS_VOLUME = "trueskate-curved-model1-v1"
MODELS_VOLUME = "trueskate-models"


def approved_seconds(mode, amount, seconds):
    if not math.isfinite(amount) or amount <= 0 or not 1 <= seconds <= 86400:
        raise ValueError("explicit positive approved budget and bounded runtime required")
    if mode == "benchmark" and seconds > 900:
        raise ValueError("benchmark is capped at 15 minutes")
    # Keep one dollar for cold-start/build/teardown variability. This is an estimate,
    # not a provider billing cap; report the timeout and use account budgets too.
    if amount < seconds/3600*HOURLY_USD+1:
        raise ValueError("approved compute budget is smaller than the timeout estimate plus $1 overhead")
    return seconds


def execute(args):
    seconds = approved_seconds(args.command, args.approved_budget_usd, args.max_seconds)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run):
        raise ValueError("run name must be one simple directory name")
    if not re.fullmatch(r"[a-f0-9]{64}", args.bundle_hash):
        raise ValueError("exact bundle hash required")
    import modal
    app = modal.App("trueskate-curved-model1")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libglib2.0-0")
             .pip_install("torch==2.12.0", "opencv-python-headless==4.13.0.92", "numpy==2.4.6",
                          "scipy==1.17.1", "selenium==4.43.0")
             .env({"PYTHONPATH": "/root/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
             .add_local_dir(str(ROOT/"src/trueskate_ai"), remote_path="/root/src/trueskate_ai"))
    corpus = modal.Volume.from_name(CORPUS_VOLUME)
    models = modal.Volume.from_name(MODELS_VOLUME)

    @app.function(image=image, gpu="L4", cpu=(4, 4), memory=(65536, 65536), timeout=seconds, serialized=True,
                  retries=0, max_containers=1, scaledown_window=2,
                  volumes={"/corpus": corpus, "/models": models})
    def worker(mode, bundle_hash, run, resume):
        import tempfile
        import torch
        from trueskate_ai.model1.curved.cloud import stage, benchmark
        from trueskate_ai.model1.curved.contract import save_json
        from trueskate_ai.model1.curved.training import train, test_once
        if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name():
            raise RuntimeError("expected pinned L4")
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as scratch:
            manifest, human = stage(Path("/corpus")/bundle_hash, Path(scratch)/"corpus")
            out = Path("/models/curved-v1")/run
            if mode == "benchmark":
                result = benchmark(manifest)
                save_json(out/"benchmark.json", result)
                models.commit()
                return result
            if mode == "train":
                return train(manifest, human, out, device="cuda", cache=True, resume=resume,
                             on_commit=models.commit)
            return test_once(manifest, out, device="cuda", on_commit=models.commit)

    with app.run():
        return worker.remote(args.command, args.bundle_hash, args.run, args.resume)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("quote")
    q.add_argument("--benchmark", type=Path)
    q.add_argument("--storage-gib", type=float, default=0)
    b = sub.add_parser("bundle")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--human", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("upload")
    u.add_argument("--bundle", type=Path, required=True)
    u.add_argument("--approved-storage-usd-month", type=float, required=True)
    for mode in ("benchmark", "train", "test"):
        r = sub.add_parser(mode)
        r.add_argument("--bundle-hash", required=True)
        r.add_argument("--run", required=True)
        r.add_argument("--approved-budget-usd", type=float, required=True)
        r.add_argument("--max-seconds", type=int, required=True)
        r.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.command == "quote":
        result = quote(json.loads(a.benchmark.read_text()) if a.benchmark else None, storage_gib=a.storage_gib)
    elif a.command == "bundle":
        result = bundle(a.manifest, a.human, a.out)
    elif a.command == "upload":
        from trueskate_ai.model1.curved.capture import file_sha256
        from trueskate_ai.model1.curved.contract import digest
        index = json.loads((a.bundle/"bundle.json").read_text())
        if index["sha256"] != digest({k: v for k, v in index.items() if k != "sha256"}):
            raise ValueError("bundle index changed")
        for row in index["shards"]:
            if file_sha256(a.bundle/row["file"]) != row["sha256"]:
                raise ValueError("shard changed")
        size = sum(path.stat().st_size for path in a.bundle.iterdir() if path.is_file())/2**30
        if not math.isfinite(a.approved_storage_usd_month) or a.approved_storage_usd_month < size*.09:
            raise ValueError("storage estimate exceeds approved monthly amount")
        import modal
        volume = modal.Volume.from_name(CORPUS_VOLUME, create_if_missing=True)
        with volume.batch_upload(force=False) as upload:
            upload.put_directory(str(a.bundle), "/"+index["sha256"])
        result = dict(volume=CORPUS_VOLUME, bundle_hash=index["sha256"], gib=size)
    else:
        result = execute(a)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
