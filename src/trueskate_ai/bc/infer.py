"""Model 2 inference — load a trained SequencePolicy and drive it receding-horizon.

Pure torch/numpy (no Appium/device deps), so it is unit-testable offline and is
reused verbatim by the on-device deploy loop
(`scripts/inspect/run_sequence_policy.py`).

Per decision:

    runner.replace_window(recent_frames)      # deployment: genuine recent video window
    strokes = runner.act()                    # -> active NATIVE stroke prefix
    vec, n, pre_delay_s = runner.to_param_vector(strokes)
    #   execute vec via action_param.execute_gesture_params after waiting pre_delay_s
    runner.commit(strokes)                    # next decision follows group completion

Frame convention matches training exactly — both go through `bc.frame_prep.
prep_frame_rgb`: `observe` takes a **BGR uint8 H×W×3** array (as cv2 returns),
resizes to (img_w, img_h), converts to RGB, scales to [0, 1], and stores CHW.

delay-timing note: `strokes_to_param_vector` packs only the N-1 delays *between*
strokes (each stroke's `delay_before`, skipping the first), so the first stroke's
own `delay_before` — which the policy does predict — would otherwise be dropped.
`to_param_vector` returns it as `pre_delay_s` so the caller can honour it as a
wait before firing, keeping predicted inter-stroke timing intact end to end.
"""
from __future__ import annotations

from collections import deque
import threading
import time

import numpy as np

from trueskate_ai.bc.frame_prep import prep_frame_rgb
from trueskate_ai.bc.gesture_tokens import STROKE_DIM, decode, encode, strokes_to_param_vector
from trueskate_ai.bc.model2 import SequencePolicy, SequencePolicyConfig


def load_policy(model_path, device):
    """Load a Model 2 checkpoint -> (model.eval(), SequencePolicyConfig). Pure torch."""
    import torch

    ckpt = torch.load(model_path, map_location=device)
    if ckpt.get("checkpoint_version") != 2:
        raise RuntimeError("Incompatible Model 2 checkpoint: retrain for causal action groups and slot activity (v2)")
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
        img = prep_frame_rgb(frame_bgr, self.cfg.img_h, self.cfg.img_w)
        self._frames.append(img.transpose(2, 0, 1))                    # (C,H,W)

    def replace_window(self, frames_bgr: list[np.ndarray]) -> None:
        """Replace, rather than append to, the visual window used by the next act."""
        if not frames_bgr:
            raise RuntimeError("cannot decide without a genuinely observed frame")
        self._frames.clear()
        for frame in frames_bgr:
            self.observe(frame)

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
        """Predict the next active action-group prefix (at least one stroke)."""
        import torch

        frames = self._frames_tensor()
        past, mask = self._past_tensors()
        with torch.no_grad():
            pred, activity = self.model(frames, past, past_mask=mask)
        native = decode(pred[0].cpu().numpy())
        active = activity[0].cpu().numpy() >= 0
        # Activity is defined as a prefix.  A hole terminates it; one stroke is mandatory.
        n = 1
        while active[0] and n < self.cfg.m_out and active[n]:
            n += 1
        return native[:n]

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


class TimestampedMjpegBuffer:
    """Bounded WDA-MJPEG reader and recent-window temporal resampler."""

    def __init__(self, max_seconds: float = 1.0):
        self.max_seconds = max_seconds
        self._frames: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self, url: str) -> None:
        if self._thread is not None:
            raise RuntimeError("MJPEG buffer already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(url,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def add(self, timestamp: float, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._frames.append((timestamp, frame_bgr))
            cutoff = timestamp - self.max_seconds
            while self._frames and self._frames[0][0] < cutoff:
                self._frames.popleft()

    def recent_window(self, n_frames: int, window_s: float = 0.2,
                      now: float | None = None) -> list[np.ndarray]:
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        if window_s < 0:
            raise ValueError(f"window_s must be >= 0, got {window_s}")
        now = time.monotonic() if now is None else now
        with self._lock:
            available = [(t, f) for t, f in self._frames if t <= now]
        if not available:
            return []
        targets = np.linspace(now - window_s, now, n_frames)
        # Anchor the left edge with the latest observation at or before the
        # requested window start.  This preserves the configured temporal span
        # when capture is sparse.  Interior targets use the nearest frame that
        # has already arrived by decision time (the target grid is synthetic,
        # so a frame a few milliseconds after an interior target is not future
        # information).  Clamp indices monotonically to preserve chronology.
        # If capture began after the window start, front-pad with that earliest
        # genuinely available observation; never invent a frame.
        times = np.asarray([x[0] for x in available])
        start_idx = max(0, int(np.searchsorted(times, targets[0], side="right") - 1))
        indices = [start_idx]
        for target in targets[1:]:
            right = int(np.searchsorted(times, target, side="left"))
            if right <= 0:
                idx = 0
            elif right >= len(times):
                idx = len(times) - 1
            else:
                before = right - 1
                idx = (before if target - times[before] <= times[right] - target else right)
            indices.append(max(indices[-1], idx))
        selected = [available[idx][1] for idx in indices]
        return selected

    def _loop(self, url: str) -> None:
        import cv2
        import requests
        try:
            response = requests.get(url, stream=True, timeout=5)
            response.raise_for_status()
            buf = bytearray()
            for chunk in response.iter_content(8192):
                if self._stop.is_set():
                    break
                buf.extend(chunk)
                while True:
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start < 0 or end < 0:
                        break
                    jpg = bytes(buf[start:end + 2])
                    del buf[:end + 2]
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.add(time.monotonic(), frame)
        except Exception as exc:
            self.error = exc
