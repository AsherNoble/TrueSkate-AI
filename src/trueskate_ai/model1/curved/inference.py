"""Inference artifacts only; predicted gestures are never automatically executed."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .audit import verify_manifest
from .capture import file_sha256
from .contract import save_json
from .dataset import read_frames
from .decoder import decode
from .model import CurvedModel


def execution_check(path):
    from trueskate_ai.sim.timed_waypoints import TimedWaypoints
    try:
        payload = TimedWaypoints(tuple(map(tuple, path["points"])), tuple(path["times_ms"])).payload()
        return dict(executable=True, payload=payload)
    except ValueError as exc:
        return dict(executable=False, reason=str(exc))


@torch.no_grad()
def human_review(human_manifest, run_dir, *, device="cpu"):
    run_dir = Path(run_dir)
    human = json.loads(Path(human_manifest).read_text())
    verify_manifest(human)
    pointer = json.loads((run_dir/"latest.json").read_text())
    if not pointer["complete"]:
        raise ValueError("complete training before human evaluation")
    checkpoint = run_dir/pointer["best_checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["config"]["identity"]["human"] != human["sha256"]:
        raise ValueError("human evaluation set differs from the pre-training freeze")
    model = CurvedModel(payload["config"]["recipe"]["base_channels"]).to(device).eval()
    model.load_state_dict(payload["model"])
    out = run_dir/"human-review"
    out.mkdir(exist_ok=False)
    samples = []
    for i, row in enumerate(human["samples"]):
        video = Path(row["video"])
        if file_sha256(video) != row["sha256"]:
            raise ValueError("frozen human video changed")
        frames = torch.from_numpy(read_frames(video).copy()).permute(0, 3, 1, 2)[None].to(device)
        times = np.asarray(row["frame_times_s"], dtype=np.float32)
        result = model(frames, torch.from_numpy(times)[None].to(device))
        path = decode(result["xy"][0].cpu(), result["onset"][0].cpu(), result["release"][0].cpu(), times)
        name = f"human-{i:02}{video.suffix}"
        shutil.copy2(video, out/name)
        samples.append(dict(park=row["park"], video=name, frame_times_s=row["frame_times_s"],
                            prediction=path, execution=execution_check(path)))
    result = dict(human_sha256=human["sha256"], checkpoint_sha256=file_sha256(checkpoint), samples=samples)
    save_json(out/"predictions.json", result)
    template = Path(__file__).with_name("human.html").read_text()
    (out/"index.html").write_text(template.replace("__DATA__", json.dumps(result).replace("<", "\\u003c")))
    return dict(review=str(out/"index.html"), predictions=str(out/"predictions.json"))
