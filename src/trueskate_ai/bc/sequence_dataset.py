"""Causal Model-2 examples built at executable action-group boundaries.

Overlapping strokes form one action group.  A decision is made at clip start
(cold start) or when the preceding complete group finishes, matching deployment:
observe -> predict a group (including its wait) -> execute it.
"""
from __future__ import annotations

import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from trueskate_ai.bc.frame_prep import prep_frame_rgb
from trueskate_ai.bc.gesture_tokens import STROKE_DIM, encode
from trueskate_ai.bc.model2 import SequencePolicyConfig

_MAX_CACHED_FRAMES = 20_000


class _FrameCache:
    def __init__(self, maxsize: int = _MAX_CACHED_FRAMES):
        self._maxsize = maxsize
        self._store: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def get(self, clip_idx: int, frame_idx: int, load) -> np.ndarray:
        key = (clip_idx, frame_idx)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        self._store[key] = load()
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)
        return self._store[key]


def _load_frame(path: Path, h: int, w: int) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return prep_frame_rgb(img, h, w).transpose(2, 0, 1)


def group_overlapping_strokes(strokes: list[dict]) -> list[list[dict]]:
    """Return interval-connected groups, ordered by start time."""
    ordered = sorted(strokes, key=lambda s: (float(s["t_start"]), float(s["t_end"])))
    groups: list[list[dict]] = []
    group_end = float("-inf")
    for stroke in ordered:
        start, end = float(stroke["t_start"]), float(stroke["t_end"])
        if not groups or start > group_end:
            groups.append([stroke])
            group_end = end
        else:
            groups[-1].append(stroke)
            group_end = max(group_end, end)
    return groups


def _group_params(group: list[dict], decision_time: float) -> np.ndarray:
    """Native tokens with delays relative to decision/previous stroke completion."""
    out = np.array([s["params"] for s in group], dtype=np.float64)
    out[0, -1] = float(group[0]["t_start"]) - decision_time
    for i in range(1, len(group)):
        out[i, -1] = float(group[i]["t_start"]) - float(group[i - 1]["t_end"])
    return out


class _Clip:
    def __init__(self, clip_idx: int, clip_dir: Path, h: int, w: int, cache: _FrameCache):
        meta = json.loads((clip_dir / "clip.json").read_text())
        self.path = clip_dir
        self.fps = float(meta["fps"])
        self.groups = group_overlapping_strokes(meta["strokes"])
        self.frame_paths = sorted(clip_dir.glob("frame_*.png"))
        self.frame_times = np.arange(len(self.frame_paths), dtype=np.float64) / self.fps
        self._h, self._w, self._clip_idx, self._cache = h, w, clip_idx, cache

    def frame(self, i: int) -> np.ndarray:
        return self._cache.get(self._clip_idx, i, lambda: _load_frame(self.frame_paths[i], self._h, self._w))

    def frame_indices_at_or_before(self, t: float, n: int) -> list[int] | None:
        idx = np.nonzero(self.frame_times <= t)[0]
        if idx.size == 0:
            return None
        sel = list(idx[-n:])
        while len(sel) < n:
            sel.insert(0, sel[0])
        return sel


class SequenceDataset(Dataset):
    def __init__(self, root: str | Path, *, cfg: SequencePolicyConfig):
        self.cfg = cfg
        self.clips: list[_Clip] = []
        self.index: list[tuple[int, int, float, list[int]]] = []
        counts: Counter[str] = Counter()
        root = Path(root)
        clip_dirs = sorted(d for d in root.glob("**/") if (d / "clip.json").exists())
        cache = _FrameCache()
        for ci, directory in enumerate(clip_dirs):
            clip = _Clip(ci, directory, cfg.img_h, cfg.img_w, cache)
            self.clips.append(clip)
            for gi, group in enumerate(clip.groups):
                if len(group) > cfg.m_out:
                    raise ValueError(f"{directory}: overlap group {gi} requires m_out={len(group)} (configured {cfg.m_out})")
                start = float(group[0]["t_start"])
                if start < 0:
                    counts["group_before_clip"] += 1
                    continue
                if gi == 0:
                    decision = 0.0
                else:
                    prev = clip.groups[gi - 1]
                    # A clip that starts mid-action establishes no cold-start state.
                    # Its first complete group is history for the following decision.
                    if float(prev[0]["t_start"]) < 0:
                        counts["cold_start_mid_action"] += 1
                        continue
                    decision = max(float(s["t_end"]) for s in prev)
                frame_idx = clip.frame_indices_at_or_before(decision, cfg.n_frames)
                if frame_idx is None:
                    counts["no_causal_frame"] += 1
                    continue
                self.index.append((ci, gi, decision, frame_idx))
                counts["retained"] += 1
        if not self.index:
            raise RuntimeError(f"No causal action-group decisions found under {root}; exclusions={dict(counts)}")
        self.exclusion_counts = dict(counts)
        print(f"SequenceDataset: {len(self.clips)} clips, {len(self.index)} retained decisions; "
              f"exclusions={{{', '.join(f'{k}: {v}' for k, v in counts.items() if k != 'retained')}}}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        ci, gi, decision, frame_idx = self.index[k]
        clip, cfg = self.clips[ci], self.cfg
        frames = np.stack([clip.frame(i) for i in frame_idx]).astype(np.float32)

        history = []
        for hist_i, group in enumerate(clip.groups[:gi]):
            if float(group[0]["t_start"]) >= 0:
                hist_decision = (0.0 if hist_i == 0 else
                                 max(float(s["t_end"]) for s in clip.groups[hist_i - 1]))
                history.extend(_group_params(group, hist_decision))
        history = history[-cfg.m_past:]
        past = np.zeros((cfg.m_past, STROKE_DIM), dtype=np.float32)
        past_mask = np.zeros(cfg.m_past, dtype=bool)
        if history:
            enc = encode(np.asarray(history)).astype(np.float32)
            past[-len(enc):], past_mask[-len(enc):] = enc, True

        native = _group_params(clip.groups[gi], decision)
        target = np.zeros((cfg.m_out, STROKE_DIM), dtype=np.float32)
        target_mask = np.zeros(cfg.m_out, dtype=bool)
        target[:len(native)] = encode(native).astype(np.float32)
        target_mask[:len(native)] = True
        return {"frames": torch.from_numpy(frames), "past_strokes": torch.from_numpy(past),
                "past_mask": torch.from_numpy(past_mask), "target": torch.from_numpy(target),
                "target_mask": torch.from_numpy(target_mask)}
