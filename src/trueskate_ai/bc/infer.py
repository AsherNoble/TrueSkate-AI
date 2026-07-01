"""Model 2 inference — load a trained SequencePolicy and drive it receding-horizon.

Pure torch/numpy (no Appium/device deps), so it is unit-testable offline and is
reused verbatim by the on-device deploy loop
(`scripts/inspect/run_sequence_policy.py`).

Per decision:

    runner.observe(frame_bgr)                 # push a frame into the rolling n_frames buffer
    strokes = runner.act()                    # -> (m_out, 9) NATIVE strokes (decoded from [0,1])
    vec, n, pre_delay_s = runner.to_param_vector(strokes[:k])
    #   execute vec via action_param.execute_gesture_params after waiting pre_delay_s
    runner.commit(strokes[:k])                # record executed strokes so the next act() conditions on them

Frame convention matches training exactly (`sequence_dataset._load_frame`):
`observe` takes a **BGR uint8 H×W×3** array (as cv2 returns), resizes to
(img_w, img_h), converts to RGB, scales to [0, 1], and stores CHW.

delay-timing note: `strokes_to_param_vector` packs only the N-1 delays *between*
strokes (each stroke's `delay_before`, skipping the first), so the first stroke's
own `delay_before` — which the policy does predict — would otherwise be dropped.
`to_param_vector` returns it as `pre_delay_s` so the caller can honour it as a
wait before firing, keeping predicted inter-stroke timing intact end to end.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from trueskate_ai.bc.gesture_tokens import STROKE_DIM, decode, encode, strokes_to_param_vector
from trueskate_ai.bc.model2 import SequencePolicy, SequencePolicyConfig


def load_policy(model_path, device):
    """Load a Model 2 checkpoint -> (model.eval(), SequencePolicyConfig). Pure torch."""
    import torch

    ckpt = torch.load(model_path, map_location=device)
    cfg = SequencePolicyConfig(**ckpt["config"])
    model = SequencePolicy(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


class SequencePolicyRunner:
    """Stateful receding-horizon driver around a SequencePolicy. Device-free."""

    def __init__(self, model: SequencePolicy, cfg: SequencePolicyConfig, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self._frames: deque[np.ndarray] = deque(maxlen=cfg.n_frames)   # each (C,H,W) float32 RGB
        self._past: deque[np.ndarray] = deque(maxlen=cfg.m_past)       # each (9,) NATIVE stroke

    def reset(self) -> None:
        self._frames.clear()
        self._past.clear()

    def observe(self, frame_bgr: np.ndarray) -> None:
        """Push one BGR uint8 H×W×3 frame (cv2 convention), transformed as in training."""
        import cv2

        img = cv2.resize(np.asarray(frame_bgr), (self.cfg.img_w, self.cfg.img_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self._frames.append(img.transpose(2, 0, 1))                    # (C,H,W)

    def _frames_tensor(self):
        import torch

        if not self._frames:
            raise RuntimeError("observe() at least one frame before act()")
        sel = list(self._frames)
        while len(sel) < self.cfg.n_frames:                            # front-pad by repeat (as frames_before)
            sel.insert(0, sel[0])
        arr = np.stack(sel, axis=0)[None]                             # (1,n,C,H,W)
        return torch.from_numpy(arr).to(self.device)

    def _past_tensors(self):
        import torch

        m = self.cfg.m_past
        past = np.zeros((m, STROKE_DIM), dtype=np.float32)
        mask = np.zeros((m,), dtype=bool)
        hist = list(self._past)
        if hist:
            enc = encode(np.stack(hist)).astype(np.float32)           # native -> [0,1]
            past[m - len(enc):] = enc                                 # front-pad, matching the dataset
            mask[m - len(enc):] = True
        return (torch.from_numpy(past[None]).to(self.device),
                torch.from_numpy(mask[None]).to(self.device))

    def act(self) -> np.ndarray:
        """Predict the next chunk. Returns (m_out, STROKE_DIM) NATIVE strokes."""
        import torch

        frames = self._frames_tensor()
        past, mask = self._past_tensors()
        with torch.no_grad():
            pred = self.model(frames, past, past_mask=mask)           # (1,m_out,9) in [0,1]
        return decode(pred[0].cpu().numpy())                          # -> native

    def to_param_vector(self, strokes: np.ndarray) -> tuple[list[float], int, float]:
        """NATIVE strokes -> (CMA-ES 9N-1 vector, N, pre_delay_s).

        `pre_delay_s` is the first stroke's `delay_before` (not carried by the
        vector); honour it as a wait before executing so predicted timing holds.
        """
        strokes = np.asarray(strokes, dtype=np.float64).reshape(-1, STROKE_DIM)
        vec, n = strokes_to_param_vector(strokes)
        pre_delay_s = float(strokes[0, STROKE_DIM - 1]) if n else 0.0
        return vec, n, pre_delay_s

    def commit(self, strokes: np.ndarray) -> None:
        """Record executed NATIVE strokes into history for the next decision."""
        for s in np.asarray(strokes, dtype=np.float64).reshape(-1, STROKE_DIM):
            self._past.append(s)
