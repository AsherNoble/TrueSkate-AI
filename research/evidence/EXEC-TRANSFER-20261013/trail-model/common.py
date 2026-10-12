"""Shared helpers for trail characterisation (copied from scratchpad/lag, not edited there)."""
import json, subprocess
from pathlib import Path
import cv2
import numpy as np

SP = Path("/private/tmp/claude-503/-Users-ashernoble-Projects-Robotics---hardware-TrueSkate-AI/e0f1c48e-973a-4ff3-a26a-f5f2350d9a43/scratchpad")
M = SP/"audit6-mirror"
OUT = SP/"research/2-trail-model"
W, H = 828, 1792
FRAME_MS = 1000/60


def manifest():
    return {s["id"]: s for s in json.loads((M/"audit-6/manifest.json").read_text())["samples"]}


_cache = {}
def decode(sample):
    p = M/sample["sample_dir"]/"clip.mp4"
    if p in _cache:
        return _cache[p]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(p), "-fps_mode", "passthrough",
                                   "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    f = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    _cache.clear(); _cache[p] = f
    return f


class Path_:
    def __init__(self, sample):
        g = sample["gesture"]
        self.p = np.asarray(g["command_points"], float)*[W, H]
        self.t = np.asarray(g["times_ms"], float)
        seg = np.diff(self.p, axis=0)
        self.len = np.linalg.norm(seg, axis=1)
        self.S = np.concatenate([[0], np.cumsum(self.len)])

    def at_time(self, ms):
        ms = np.clip(ms, 0, self.t[-1])
        return np.stack([np.interp(ms, self.t, self.p[:, 0]), np.interp(ms, self.t, self.p[:, 1])], -1)

    def s_at_time(self, ms):
        return np.interp(np.clip(ms, 0, self.t[-1]), self.t, self.S)

    def at_s(self, s):
        i = int(np.clip(np.searchsorted(self.S, s, side="right")-1, 0, len(self.len)-1))
        u = 0 if self.len[i] == 0 else (s-self.S[i])/self.len[i]
        return self.p[i]+np.clip(u, 0, 1)*(self.p[i+1]-self.p[i])


def elapsed_ms(sample):
    t = np.asarray(sample["frame_times_s"])
    return (t-sample["onset_s"])/sample["clock_rate"]*1000


def board_mask(frame):
    """Copied from scratchpad/lag/trail_lag.py: speckled deck texture energy."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    energy = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (21, 21))
    ellipse = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    region = cv2.morphologyEx((energy > 35).astype(np.uint8), cv2.MORPH_OPEN, ellipse(15))
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, ellipse(31))
    return cv2.dilate(region, ellipse(21)) > 0
