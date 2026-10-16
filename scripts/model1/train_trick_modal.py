"""Trick Model 1 MVP v0: freeze and pack the clips on the laptop, then train on one Modal L4.

Use Python (not `modal run`): the budget check runs before the app is created. Every
Modal call has a runtime cap, and `--approved-budget-usd` must cover that cap at the
L4 + 4 CPU + 16 GiB rate plus $0.50. Checkpoints go to the models volume, committed every
epoch; fetch them with `modal volume get trueskate-models trick-model1-v0/<run> <dir>`.

  freeze   --data-root DIR --roots trial trial-xr2 --out manifest.json
  pack     --manifest manifest.json --data-root DIR --out BUNDLE [--width 64 --height 144]
  upload   --bundle BUNDLE
  train    --bundle-hash H --run NAME --approved-budget-usd X --max-seconds S [--epochs N] [--batch-size N]
           [--resume]
  test     --bundle-hash H --run NAME --approved-budget-usd X --max-seconds S   (once per run)
  summary  --run-dir DIR   (a fetched run: the chosen epoch's validation, and test if scored)
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

DATA_VOLUME = "trueskate-trick-model1"
MODELS_VOLUME = "trueskate-models"
RUNS = "trick-model1-v0"
OVERHEAD_USD = .5


def hourly_usd() -> float:
    from trueskate_ai.model1.curved.cloud import RATES
    return 3600*(RATES["gpu_l4_s"] + 4*RATES["cpu_core_s"] + 16*RATES["memory_gib_s"])


def approved_seconds(amount: float, seconds: int) -> int:
    if not math.isfinite(amount) or amount <= 0 or not 60 <= seconds <= 3600:
        raise ValueError("explicit positive budget and a runtime cap of 60–3600 s required")
    if amount < seconds/3600*hourly_usd() + OVERHEAD_USD:
        raise ValueError(f"the cap costs up to ${seconds/3600*hourly_usd():.2f}; "
                         f"the approved budget must cover it plus ${OVERHEAD_USD:.2f}")
    return seconds


def execute(args) -> dict:
    seconds = approved_seconds(args.approved_budget_usd, args.max_seconds)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run) or not re.fullmatch(r"[a-f0-9]{64}", args.bundle_hash):
        raise ValueError("a simple run name and an exact bundle hash are required")
    recipe = {k: getattr(args, k, None) for k in ("epochs", "base_channels", "batch_size")}
    recipe = {k: v for k, v in recipe.items() if v is not None}
    import modal
    app = modal.App("trueskate-trick-model1")
    image = (modal.Image.debian_slim(python_version="3.11")
             .apt_install("ffmpeg", "libglib2.0-0")
             .pip_install("torch==2.12.0", "opencv-python-headless==4.13.0.92", "numpy==2.4.6", "scipy==1.17.1",
                          "pillow==12.2.0")
             .env({"PYTHONPATH": "/root/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
             .add_local_dir(str(ROOT/"src/trueskate_ai"), remote_path="/root/src/trueskate_ai"))
    data = modal.Volume.from_name(DATA_VOLUME)
    models = modal.Volume.from_name(MODELS_VOLUME)

    @app.function(image=image, gpu="L4", cpu=(4, 4), memory=(16384, 16384), timeout=seconds, serialized=True,
                  retries=0, max_containers=1, scaledown_window=2, volumes={"/data": data, "/models": models})
    def worker(mode, bundle_hash, run, resume, recipe):
        import torch
        from trueskate_ai.model1.trick.training import Recipe, test_once, train
        if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name():
            raise RuntimeError("expected pinned L4")
        torch.set_num_threads(4)
        bundle, out = Path("/data")/bundle_hash, Path("/models")/RUNS/run
        if mode == "train":
            return train(bundle, out, recipe=Recipe(**recipe), device="cuda", resume=resume,
                         on_commit=models.commit)
        return test_once(bundle, out, device="cuda", on_commit=models.commit)

    with app.run():
        return worker.remote(args.command, args.bundle_hash, args.run, getattr(args, "resume", False), recipe)


def upload(bundle: Path) -> dict:
    from trueskate_ai.model1.curved.capture import file_sha256
    from trueskate_ai.model1.trick.dataset import load_bundle
    index, _ = load_bundle(bundle)
    for entry in index["files"].values():
        if file_sha256(bundle/entry["frames"]) != entry["frames_sha256"]:
            raise ValueError("bundle frames changed")
    import modal
    volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    with volume.batch_upload(force=False) as batch:
        batch.put_directory(str(bundle), "/" + index["sha256"])
    gib = sum(p.stat().st_size for p in Path(bundle).rglob("*") if p.is_file())/2**30
    return dict(volume=DATA_VOLUME, bundle_hash=index["sha256"], gib=round(gib, 3))


def summary(run_dir: Path) -> dict:
    pointer = json.loads((run_dir/"latest.json").read_text())
    chosen = int(pointer["best_checkpoint"][6:9])
    validation = json.loads((run_dir/f"validation-{chosen:03}.json").read_text())["summary"]
    result = dict(epochs=pointer["epoch"], complete=pointer["complete"], chosen_epoch=chosen, validation=validation)
    if (run_dir/"test-results.json").exists():
        result["test"] = json.loads((run_dir/"test-results.json").read_text())["summary"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--data-root", type=Path, required=True)
    f.add_argument("--roots", nargs="+", required=True)
    f.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pack")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=144)
    p.add_argument("--jobs", type=int, default=4)
    u = sub.add_parser("upload")
    u.add_argument("--bundle", type=Path, required=True)
    for mode in ("train", "test"):
        r = sub.add_parser(mode)
        r.add_argument("--bundle-hash", required=True)
        r.add_argument("--run", required=True)
        r.add_argument("--approved-budget-usd", type=float, required=True)
        r.add_argument("--max-seconds", type=int, required=True)
        if mode == "train":
            r.add_argument("--epochs", type=int)
            r.add_argument("--base-channels", type=int)
            r.add_argument("--batch-size", type=int)
            r.add_argument("--resume", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        from trueskate_ai.model1.trick.dataset import freeze
        manifest = freeze(args.data_root, args.roots, args.out)
        result = dict(sha256=manifest["sha256"], counts=manifest["counts"], excluded=len(manifest["excluded"]))
    elif args.command == "pack":
        from trueskate_ai.model1.trick.dataset import pack
        index = pack(args.manifest, args.data_root, args.out, size=(args.width, args.height), jobs=args.jobs)
        result = dict(bundle_hash=index["sha256"], clips=len(index["files"]))
    elif args.command == "upload":
        result = upload(args.bundle)
    elif args.command == "summary":
        result = summary(args.run_dir)
    else:
        result = execute(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
