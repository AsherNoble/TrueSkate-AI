"""Warm-glow excess along the commanded path's shape (timing-blind matched filter)."""
import cv2
import numpy as np
from trail_lag import W, H

STEP, ACROSS, BG = 4., 12, 44     # px: sampling step, perpendicular search, background offset


def profile(frame, path, board, overlap=True):
    """For each arc length s (every STEP px): max over perpendicular offsets |o|<=ACROSS of
    warm excess = (R+G)/2 at the line minus the mean of both background sides (|o|=BG),
    requiring the excess to be warm (R-B rises). Board-masked points are NaN (hidden)."""
    img = frame.astype(np.float32)
    s = np.arange(0, path.S[-1]+1e-9, STEP)
    pts = np.array([path.at_s(v) for v in s])
    i = np.clip(np.searchsorted(path.S, s, side="right")-1, 0, len(path.len)-1)
    d = path.p[i+1]-path.p[i]; n = np.linalg.norm(d, axis=1, keepdims=True); n[n == 0] = 1
    normal = np.stack([-d[:, 1], d[:, 0]], 1)/n
    def sample(offset):
        q = pts+normal*offset
        x = np.clip(q[:, 0], 0, W-1).astype(np.float32); y = np.clip(q[:, 1], 0, H-1).astype(np.float32)
        return cv2.remap(img, x[None], y[None], cv2.INTER_LINEAR)[0]          # (n, 3) BGR
    bg = (sample(-BG)+sample(BG))/2
    offsets = list(range(-ACROSS-9, ACROSS+10, 3))
    bright, warm = [], []
    for o in offsets:
        dif = sample(o)-bg
        bright.append((dif[:, 2]+dif[:, 1])/2); warm.append(dif[:, 2]-dif[:, 0])
    bright, warm = np.array(bright), np.array(warm)
    centre = [i for i, o in enumerate(offsets) if abs(o) <= ACROSS]
    cand = np.where(warm[centre] >= -2, bright[centre], -np.inf)
    j = np.argmax(cand, 0); best = cand[j, np.arange(len(s))]
    peak = np.array(centre)[j]
    # The trail is a broad soft ridge (>half peak at +-12 px); scenery lines are thin spikes.
    sides = np.minimum(bright[peak-3, np.arange(len(s))], bright[peak+3, np.arange(len(s))])
    best = np.minimum(best, np.maximum(sides, 0)/.45)
    hit = np.array(offsets)[peak]
    hidden = board[np.clip(pts[:, 1], 0, H-1).astype(int), np.clip(pts[:, 0], 0, W-1).astype(int)]
    # Geometry only: where the path returns within reach of an earlier stretch (old glow nearby,
    # including under the background samples), the head cannot be attributed; treat as hidden.
    if overlap:
        gap = np.linalg.norm(pts[:, None]-pts[None], axis=2)
        earlier = (s[None, :] < s[:, None]-80)
    if overlap:
        hidden = hidden | ((gap < BG+ACROSS) & earlier).any(1)
    best[hidden] = np.nan
    return s, best, hit
