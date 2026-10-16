"""Trick clips for Model 1 MVP v0: a split by minute, per-frame targets and a frame bundle.

A clip holds one run of 1–4 gestures (contract.py), labelled from the visible trail at
admission (admission.py). Copies of a seeded run stay in one split: recordings that share
a run number form one group (both trial phones sent the same minutes), and the group takes
the split of its earliest recording's minute number, earliest in the order the roots are
given. New roots (the 1,000-clip batch) add in through ``freeze``'s root list without moving
earlier clips. Frames are decoded once on the laptop into a bundle, so training reads small
arrays instead of 828×1792 video.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

from trueskate_ai.model1.curved.capture import file_sha256, probe_pts
from trueskate_ai.model1.curved.contract import digest, save_json
from .contract import FRAME_COUNT, MAX_GESTURES, SCHEMA, Gesture

KIND = "trick-model1-dataset"
BUNDLE_KIND = "trick-model1-bundle"
IMAGE_WIDTH, IMAGE_HEIGHT = 64, 144  # curved v1's input size
MARKERS = (".menu", ".trace_mismatch", ".contaminated")
SPLITS = ("train", "validation", "test")
SPLIT_RULE = ("minute number mod 7: 1-5 train, 6 validation, 0 test; recordings that share "
              "a run number take the split of the earliest of them, in root order")
# Touch-down and lift targets are Gaussians about a frame wide: trail labels sit within
# about a frame (TRICK-SPEED-20261015), and the peak's shape carries the sub-frame time.
EVENT_SIGMA_S = .75/60


def nearest(times, t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(times) - t)))


def split_for(recording: str) -> str:
    match = re.fullmatch(r"m(\d+)", recording)
    if not match:
        raise ValueError(f"recording {recording!r} has no minute number")
    rest = int(match.group(1)) % 7
    return "validation" if rest == 6 else "test" if rest == 0 else "train"


def check_clip(meta: dict) -> list[Gesture]:
    """The data contract for one admitted clip. Returns its gestures; raises ValueError."""
    admission = meta.get("admission", {})
    if (meta.get("schema") != SCHEMA or admission.get("accepted") is not True
            or admission.get("frame_count") != FRAME_COUNT):
        raise ValueError("not an admitted trick clip")
    times = np.asarray(meta["frame_times_s"], dtype=float)
    if times.shape != (FRAME_COUNT,) or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("invalid native clip timeline")
    if np.max(np.diff(times)) > .0500001 or not 58 <= (FRAME_COUNT - 1)/times[-1] <= 62:
        raise ValueError("invalid 60 Hz clip cadence")
    numbers = meta["source_frame_numbers"]
    if len(numbers) != FRAME_COUNT or any(b - a != 1 for a, b in zip(numbers, numbers[1:])):
        raise ValueError("source frames are not consecutive")
    source = np.asarray(meta["source_frame_pts_s"], dtype=float)
    if source.shape != (FRAME_COUNT,) or not np.allclose(source - source[0], times, atol=1e-8):
        raise ValueError("source/clip PTS disagree")
    rate = meta["clock_rate"]
    if not .98 <= rate <= 1.02:
        raise ValueError("implausible video clock rate")
    rows = meta["strokes"]
    if not 1 <= len(rows) <= MAX_GESTURES:
        raise ValueError(f"a clip holds 1–{MAX_GESTURES} gestures")
    gestures, previous = [], -1
    for row in rows:
        gesture = Gesture.from_dict(row)
        down, up = row["onset_s"], row["release_s"]
        if not np.isfinite([down, up]).all() or abs(up - down - rate*gesture.times_ms[-1]/1000) > 1e-6:
            raise ValueError("gesture times and calibrated duration disagree")
        if not times[4] < down < up < times[-5]:
            raise ValueError("missing five-frame buffers before touch-down and after lift")
        if nearest(times, down) <= previous:
            raise ValueError("gestures overlap or touch")
        previous = nearest(times, up)
        gestures.append(gesture)
    return gestures


def targets(meta: dict) -> dict:
    """Per-frame targets: touching (the "contact" signal), touch-down, lift, and the gesture's position."""
    gestures = check_clip(meta)
    times = np.asarray(meta["frame_times_s"], dtype=float)
    contact, down, up = (np.zeros(FRAME_COUNT) for _ in range(3))
    xy = np.zeros((FRAME_COUNT, 2))
    for gesture, row in zip(gestures, meta["strokes"]):
        a, b = nearest(times, row["onset_s"]), nearest(times, row["release_s"])
        contact[a:b+1] = 1
        xy[a:b+1] = gesture.positions(times[a:b+1] - row["onset_s"], rate=meta["clock_rate"])
        down = np.maximum(down, np.exp(-.5*((times - row["onset_s"])/EVENT_SIGMA_S)**2))
        up = np.maximum(up, np.exp(-.5*((times - row["release_s"])/EVENT_SIGMA_S)**2))
    return dict(times=times.astype(np.float32), contact=contact.astype(np.float32), down=down.astype(np.float32),
                up=up.astype(np.float32), xy=xy.astype(np.float32), mask=contact.astype(bool),
                count=np.int64(len(gestures)))


def read_frames(video: Path, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> np.ndarray:
    """Exactly 120 native frames, area-scaled (curved v1's read_frames at any size)."""
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0",
                                   "-vf", f"scale={width}:{height}:flags=area", "-fps_mode", "passthrough",
                                   "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    if len(raw) != FRAME_COUNT*height*width*3:
        raise ValueError("expected exactly 120 decoded native frames; no padding/resampling")
    return np.frombuffer(raw, np.uint8).reshape(FRAME_COUNT, height, width, 3)


def clip_rows(data_root: Path, roots) -> tuple[list[dict], list[str]]:
    """Every admitted clip under data_root/<root>/recordings, checked; marked clips are excluded."""
    data_root, rows, excluded = Path(data_root).resolve(), [], []
    for root in roots:
        folders = sorted(p.parent for p in (data_root/root).glob("recordings/*/run-*/meta.json"))
        if not folders:
            raise ValueError(f"no admitted clips under {root}")
        for folder in folders:
            if any((folder/m).exists() for m in MARKERS):
                excluded.append(str(folder.relative_to(data_root)))
                continue
            meta = json.loads((folder/"meta.json").read_text())
            gestures = check_clip(meta)
            if meta["recording"] != folder.parent.name or folder.name != f"run-{meta['run']:05d}":
                raise ValueError(f"{folder}: folder and metadata disagree")
            if file_sha256(folder/"clip.mp4") != meta["clip_sha256"]:
                raise ValueError(f"{folder}: clip changed since admission")
            rows.append(dict(id=f"{root}/{meta['recording']}/{meta['run']:05d}",
                             sample_dir=str(folder.relative_to(data_root)), root=root,
                             recording=meta["recording"], run=meta["run"], device=meta["device"],
                             park=meta["park"], strokes=len(gestures), meta_sha256=file_sha256(folder/"meta.json"),
                             media_sha256=meta["clip_sha256"]))
    return rows, excluded


def assign_splits(rows: list[dict], roots=None) -> None:
    """Group recordings linked by a shared run number; each group takes one split."""
    order = {root: i for i, root in enumerate(roots or sorted({r["root"] for r in rows}))}
    parent = {}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first = {}
    for row in rows:
        key = (row["root"], row["recording"])
        parent.setdefault(key, key)
        if row["run"] in first:
            parent[find(key)] = find(first[row["run"]])
        else:
            first[row["run"]] = key
    groups = {}
    for key in parent:
        groups.setdefault(find(key), []).append(key)
    chosen = {}
    for members in groups.values():
        earliest = min(members, key=lambda m: (order[m[0]], m[1]))
        for key in members:
            chosen[key] = (split_for(earliest[1]), "+".join("/".join(m) for m in sorted(members)))
    for row in rows:
        row["split"], row["group"] = chosen[(row["root"], row["recording"])]


def verify_manifest(manifest: dict) -> dict:
    if manifest.get("kind") != KIND or manifest.get("sha256") != digest(
            {k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("a frozen trick Model 1 manifest is required")
    ids, splits = set(), {}
    for row in manifest["samples"]:
        if row["id"] in ids or row["split"] not in SPLITS:
            raise ValueError("duplicate sample or unknown split")
        ids.add(row["id"])
        for key in (("run", row["run"]), ("recording", row["root"], row["recording"])):
            if splits.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("a seeded run or recording leaks across splits")
    return manifest


def freeze(data_root: Path, roots, out: Path) -> dict:
    rows, excluded = clip_rows(data_root, roots)
    assign_splits(rows, roots)
    counts = {split: dict(clips=sum(r["split"] == split for r in rows),
                          strokes=sum(r["strokes"] for r in rows if r["split"] == split)) for split in SPLITS}
    manifest = dict(schema=SCHEMA, kind=KIND, data_root=Path(data_root).resolve().name, roots=list(roots),
                    split_rule=SPLIT_RULE, counts=counts, excluded=excluded, samples=rows)
    manifest["sha256"] = digest(manifest)
    verify_manifest(manifest)
    save_json(Path(out), manifest)
    return manifest


def _pack_one(data_root: Path, out: Path, row: dict, size) -> dict:
    folder = data_root/row["sample_dir"]
    if any((folder/m).exists() for m in MARKERS):
        raise ValueError(f"{row['id']}: contamination marker excludes the clip")
    if file_sha256(folder/"meta.json") != row["meta_sha256"] or file_sha256(folder/"clip.mp4") != row["media_sha256"]:
        raise ValueError(f"{row['id']}: frozen clip changed")
    meta = json.loads((folder/"meta.json").read_text())
    label = targets(meta)
    encoded = probe_pts(folder/"clip.mp4", recording=False)
    if encoded.shape != (FRAME_COUNT,) or not np.allclose(encoded - encoded[0], label["times"], atol=.001):
        raise ValueError(f"{row['id']}: encoded native timeline differs from source PTS")
    name = row["id"].replace("/", "_")
    np.savez_compressed(out/"frames"/f"{name}.npz", frames=read_frames(folder/"clip.mp4", *size))
    (out/"meta"/f"{name}.json").write_bytes((folder/"meta.json").read_bytes())
    return dict(frames=f"frames/{name}.npz", frames_sha256=file_sha256(out/"frames"/f"{name}.npz"),
                meta=f"meta/{name}.json")


def pack(manifest_path: Path, data_root: Path, out: Path, *, size=(IMAGE_WIDTH, IMAGE_HEIGHT), jobs=4) -> dict:
    """Decode every clip once into compressed uint8 arrays, beside a copy of its metadata."""
    manifest = verify_manifest(json.loads(Path(manifest_path).read_text()))
    data_root, out = Path(data_root).resolve(), Path(out)
    if data_root.name != manifest["data_root"]:
        raise ValueError("manifest was frozen under another data root")
    (out/"frames").mkdir(parents=True, exist_ok=False)
    (out/"meta").mkdir()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        entries = list(pool.map(lambda row: _pack_one(data_root, out, row, size), manifest["samples"]))
    save_json(out/"manifest.json", manifest)
    index = dict(kind=BUNDLE_KIND, manifest_sha256=manifest["sha256"], image_width=size[0], image_height=size[1],
                 files={row["id"]: entry for row, entry in zip(manifest["samples"], entries)})
    index["sha256"] = digest(index)
    save_json(out/"bundle.json", index)
    return index


def load_bundle(bundle: Path) -> tuple[dict, dict]:
    bundle = Path(bundle)
    index = json.loads((bundle/"bundle.json").read_text())
    if index.get("kind") != BUNDLE_KIND or index.get("sha256") != digest({k: v for k, v in index.items() if k != "sha256"}):
        raise ValueError("bundle fingerprint mismatch")
    manifest = verify_manifest(json.loads((bundle/"manifest.json").read_text()))
    if manifest["sha256"] != index["manifest_sha256"]:
        raise ValueError("bundle manifest changed")
    return index, manifest


class TrickDataset(Dataset):
    """One split of a bundle, held in memory as uint8 frames with precomputed targets."""

    def __init__(self, bundle: Path, split: str, *, limit=None):
        bundle = Path(bundle)
        index, self.manifest = load_bundle(bundle)
        self.rows = [r for r in self.manifest["samples"] if r["split"] == split]
        if limit is not None:
            if limit < 1:
                raise ValueError("positive limit required")
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError("empty partition")
        self.split, self.meta, self.frames, self.labels = split, [], [], []
        shape = (FRAME_COUNT, index["image_height"], index["image_width"], 3)
        for row in self.rows:
            entry = index["files"][row["id"]]
            if (file_sha256(bundle/entry["meta"]) != row["meta_sha256"]
                    or file_sha256(bundle/entry["frames"]) != entry["frames_sha256"]):
                raise ValueError(f"{row['id']}: bundle sample changed")
            meta = json.loads((bundle/entry["meta"]).read_text())
            with np.load(bundle/entry["frames"]) as data:
                frames = data["frames"]
            if frames.shape != shape or frames.dtype != np.uint8:
                raise ValueError(f"{row['id']}: frames have the wrong shape")
            self.meta.append(meta)
            self.frames.append(frames)
            self.labels.append(targets(meta))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # uint8 frames; float conversion happens per batch on the training device.
        return dict(video=torch.from_numpy(self.frames[index].copy()).permute(0, 3, 1, 2),
                    **{k: torch.as_tensor(v) for k, v in self.labels[index].items()}, index=index)
