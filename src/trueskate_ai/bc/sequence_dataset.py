"""SequenceDataset — assembled expert clips -> Model 2 training tuples.

Consumes CLIPS (a continuous expert recording's frames + the strokes assembled
from it by Model 1 + `assemble.py`) and yields, per stroke decision point:

    (frames[t-n_frames : t],  past_strokes[t-m_past : t])  ->  next strokes[t : t+m_out]

Frames are the most-recent `n_frames` whose time precedes the stroke's start
(CAUSAL — only pre-decision frames). Strokes are encoded to normalised [0,1]
tokens (`gesture_tokens.encode`). Past strokes are front-padded (+ a mask);
decision points without a full `m_out` target are skipped.

Clip on-disk format (one dir per clip under `root`):
    clip_dir/frame_000000.png ...           # frames, ascending
    clip_dir/clip.json  = {"fps": float, "strokes": [{"params":[9], "t_start":s, "t_end":s}, ...]}

`clip.json` is produced downstream by running Model 1 over an expert recording and
`assemble.assemble_strokes`; this module is the training-time reader (pure torch).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from trueskate_ai.bc.gesture_tokens import STROKE_DIM, encode
from trueskate_ai.bc.model2 import SequencePolicyConfig


def _load_frame(path: Path, h: int, w: int) -> np.ndarray:
    import cv2  # local: keep numpy/torch the only hard deps for token/model code
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (C,H,W)


class _Clip:
    def __init__(self, clip_dir: Path, h: int, w: int):
        meta = json.loads((clip_dir / "clip.json").read_text())
        self.fps = float(meta["fps"])
        self.strokes = meta["strokes"]
        self.frame_paths = sorted(clip_dir.glob("frame_*.png"))
        self.frame_times = np.array([i / self.fps for i in range(len(self.frame_paths))], dtype=np.float64)
        self._h, self._w = h, w
        self._cache: dict[int, np.ndarray] = {}

    def frame(self, i: int) -> np.ndarray:
        if i not in self._cache:
            self._cache[i] = _load_frame(self.frame_paths[i], self._h, self._w)
        return self._cache[i]

    def frames_before(self, t: float, n: int) -> np.ndarray:
        """The n most-recent frames with time <= t, front-padded by repeat if short."""
        idx = np.nonzero(self.frame_times <= t)[0]
        if idx.size == 0:
            idx = np.array([0])
        sel = list(idx[-n:])
        while len(sel) < n:
            sel.insert(0, sel[0])
        return np.stack([self.frame(i) for i in sel], axis=0)  # (n,C,H,W)


class SequenceDataset(Dataset):
    def __init__(self, root: str | Path, *, cfg: SequencePolicyConfig):
        self.cfg = cfg
        self.clips: list[_Clip] = []
        self.index: list[tuple[int, int]] = []  # (clip_idx, stroke_idx = decision point)
        root = Path(root)
        clip_dirs = sorted(d for d in root.glob("**/") if (d / "clip.json").exists())
        for ci, d in enumerate(clip_dirs):
            clip = _Clip(d, cfg.img_h, cfg.img_w)
            self.clips.append(clip)
            n = len(clip.strokes)
            for si in range(n - cfg.m_out + 1):   # need a full m_out target
                self.index.append((ci, si))
        if not self.index:
            raise RuntimeError(f"No decision points found under {root} (need clip.json with strokes)")
        print(f"SequenceDataset: {len(self.clips)} clips, {len(self.index)} decision points")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        ci, si = self.index[k]
        clip = self.clips[ci]
        cfg = self.cfg
        params = np.array([s["params"] for s in clip.strokes], dtype=np.float64)  # (N,9) native
        t_start = float(clip.strokes[si]["t_start"])

        frames = clip.frames_before(t_start, cfg.n_frames).astype(np.float32)

        lo = max(0, si - cfg.m_past)
        past_native = params[lo:si]                                   # (<=m_past, 9)
        past = np.zeros((cfg.m_past, STROKE_DIM), dtype=np.float32)
        mask = np.zeros((cfg.m_past,), dtype=bool)
        if len(past_native):
            enc = encode(past_native).astype(np.float32)
            past[cfg.m_past - len(enc):] = enc
            mask[cfg.m_past - len(enc):] = True

        target = encode(params[si:si + cfg.m_out]).astype(np.float32)  # (m_out, 9)
        return {
            "frames": torch.from_numpy(frames),
            "past_strokes": torch.from_numpy(past),
            "past_mask": torch.from_numpy(mask),
            "target": torch.from_numpy(target),
        }
