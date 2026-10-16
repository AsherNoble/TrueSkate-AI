"""Strict, additive 120-native-frame corpus loader; no legacy temporal sampling."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audit import verify_manifest
from .capture import file_sha256, probe_pts
from .contract import FRAME_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, SCHEMA, Gesture


def read_frames(video: Path):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0",
                                   "-vf", f"scale={IMAGE_WIDTH}:{IMAGE_HEIGHT}:flags=area",
                                   "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "pipe:1"])
    if len(raw) != FRAME_COUNT*IMAGE_HEIGHT*IMAGE_WIDTH*3:
        raise ValueError("expected exactly 120 decoded native frames; no padding/resampling")
    return np.frombuffer(raw, np.uint8).reshape(FRAME_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, 3)


def targets(meta):
    times = np.asarray(meta["frame_times_s"], dtype=np.float64)
    if times.shape != (120,) or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("invalid native clip timeline")
    gaps = np.diff(times)
    if np.max(gaps) > .0500001 or not 58 <= 119/(times[-1]-times[0]) <= 62:
        raise ValueError("invalid 60 Hz clip cadence")
    down, up = meta["onset_s"], meta["release_s"]
    if not np.isfinite([down, up]).all() or not times[4] < down < up < times[-5]:
        raise ValueError("missing five-frame contact buffers")
    gesture = Gesture.from_dict(meta["gesture"])
    rate = meta["clock_rate"]
    if not .98 <= rate <= 1.02 or abs(up-down-rate*gesture.times_ms[-1]/1000) > 1e-6:
        raise ValueError("gesture and calibrated duration disagree")
    onset = int(np.argmin(abs(times-down)))
    release = int(np.argmin(abs(times-up)))
    mask = (np.arange(120) >= onset) & (np.arange(120) <= release)
    xy = gesture.positions(times-down, rate=rate)
    xy[onset], xy[release] = gesture.command_points()[[0, -1]]
    return dict(times=times.astype(np.float32), xy=xy.astype(np.float32), mask=mask,
                onset=np.int64(onset), release=np.int64(release))


class CurvedDataset(Dataset):
    def __init__(self, manifest: Path, split: str, *, cache=False, limit=None):
        self.root = Path(manifest).resolve().parent
        self.manifest = json.loads(Path(manifest).read_text())
        verify_manifest(self.manifest)
        if self.manifest.get("kind") != "curved-training":
            raise ValueError("a frozen curved-training manifest is required")
        all_rows = self.manifest["samples"]
        ids, dirs, groups = set(), set(), {}
        for row in all_rows:
            if row["id"] in ids or row["sample_dir"] in dirs:
                raise ValueError("duplicate corpus sample")
            ids.add(row["id"])
            dirs.add(row["sample_dir"])
            for group in (row["session"], row["recording"]):
                if group in groups and groups[group] != row["split"]:
                    raise ValueError("session/recording leaks across partitions")
                groups[group] = row["split"]
        self.rows = [r for r in all_rows if r["split"] == split]
        if limit is not None:
            if limit < 1:
                raise ValueError("positive benchmark limit required")
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError("empty corpus partition")
        self.meta = []
        self.cache = [] if cache else None
        self.split = split
        for row in self.rows:
            path = (self.root/row["sample_dir"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("sample path escapes corpus")
            if any((path/m).exists() for m in (".menu", ".trace_mismatch", ".contaminated")):
                raise ValueError("contamination marker excludes sample")
            if (file_sha256(path/"meta.json") != row["meta_sha256"] or
                    file_sha256(path/"clip.mp4") != row["media_sha256"]):
                raise ValueError("frozen sample changed")
            meta = json.loads((path/"meta.json").read_text())
            admission, calibration = meta.get("admission", {}), meta.get("calibration", {})
            if (meta.get("schema") != SCHEMA or meta.get("phase") != "production"
                    or meta.get("candidate_id") != row["id"] or meta.get("park") != row["park"]
                    or meta.get("split") != split or meta.get("session_id") != row["session"]
                    or meta.get("source_recording_id") != row["recording"]
                    or meta.get("fingerprint") != self.manifest["qualified"]["fingerprint"]
                    or meta.get("clip_sha256") != row["media_sha256"]
                    or admission.get("accepted") is not True or admission.get("frame_count") != 120
                    or admission.get("menu_checked_frames") != 120
                    or admission.get("native_frame_selection") is not True
                    or calibration.get("accepted") is not True):
                raise ValueError("incomplete curved admission/provenance")
            numbers = meta["source_frame_numbers"]
            if len(numbers) != 120 or any(b-a != 1 for a, b in zip(numbers, numbers[1:])):
                raise ValueError("source frames are not consecutive")
            source_pts = np.asarray(meta["source_frame_pts_s"])
            if source_pts.shape != (120,) or not np.allclose(source_pts-source_pts[0], meta["frame_times_s"], atol=1e-8):
                raise ValueError("source/clip PTS disagree")
            label = targets(meta)
            encoded = probe_pts(path/"clip.mp4", recording=False)
            if encoded.shape != (120,) or not np.allclose(encoded-encoded[0], label["times"], atol=.001):
                raise ValueError("encoded native timeline differs from source PTS")
            self.meta.append(meta)
            if cache:
                self.cache.append(read_frames(path/"clip.mp4"))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        frames = self.cache[index] if self.cache is not None else read_frames(
            self.root/self.rows[index]["sample_dir"]/"clip.mp4")
        # Keep cache uint8. Float conversion is per batch on the training device.
        return dict(video=torch.from_numpy(frames.copy()).permute(0, 3, 1, 2),
                    **{k: torch.as_tensor(v) for k, v in targets(self.meta[index]).items()},
                    index=index)
