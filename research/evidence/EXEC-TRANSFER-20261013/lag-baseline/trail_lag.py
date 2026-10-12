"""Prototype: signed trace-vs-command lag on sealed round-6 clips (offline, read-only).

Detection uses only the commanded path's *shape* (a corridor) and pixel colour,
never the commanded timing, so it cannot pull the measured head toward the ring.
The head is the furthest path position (arc length s) reached by trail pixels,
searched forward from the previous head. Lag(t) = commanded time at t minus the
commanded time at which the command reaches the observed head:
positive = trace behind the ring (trace lags), negative = trace ahead (ring lags).
"""
import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np

W, H = 828, 1792                # native clip pixels; logical points = pixels / 2
CORRIDOR = 80                   # px either side of the path (40 logical points)
R_CAP = 24                      # px the glow's round cap extends past the touch (stationary holds)


def decode(path):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
                                   "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


class Path_:
    def __init__(self, sample):
        g = sample["gesture"]
        self.p = np.asarray(g["command_points"], float)*[W, H]
        self.t = np.asarray(g["times_ms"], float)
        seg = np.diff(self.p, axis=0)
        self.len = np.linalg.norm(seg, axis=1)
        self.S = np.concatenate([[0], np.cumsum(self.len)])

    def at_s(self, s):
        i = int(np.clip(np.searchsorted(self.S, s, side="right")-1, 0, len(self.len)-1))
        u = 0 if self.len[i] == 0 else (s-self.S[i])/self.len[i]
        return self.p[i]+np.clip(u, 0, 1)*(self.p[i+1]-self.p[i])

    def s_at_time(self, ms):
        ms = np.clip(ms, 0, self.t[-1])
        i = int(np.clip(np.searchsorted(self.t, ms, side="right")-1, 0, len(self.len)-1))
        u = (ms-self.t[i])/(self.t[i+1]-self.t[i])
        return self.S[i]+u*self.len[i]

    def time_at_s(self, s):
        """Commanded time at arc length s; None inside a zero-length (pause) segment boundary."""
        s = float(np.clip(s, 0, self.S[-1]))
        for i in range(len(self.len)):
            if self.len[i] > 0 and self.S[i] <= s <= self.S[i+1]:
                return self.t[i]+(s-self.S[i])/self.len[i]*(self.t[i+1]-self.t[i])
        return None

    def project(self, xy, lo, hi):
        """Nearest arc length within [lo, hi] for each point; returns s and distance."""
        best_s = np.full(len(xy), np.nan); best_d = np.full(len(xy), np.inf)
        for i in range(len(self.len)):
            a, b = self.S[i], self.S[i+1]
            if b < lo or a > hi or self.len[i] == 0:
                continue
            d = self.p[i+1]-self.p[i]
            u = np.clip(((xy-self.p[i])@d)/(d@d), (max(lo, a)-a)/self.len[i], (min(hi, b)-a)/self.len[i])
            q = self.p[i]+u[:, None]*d
            dist = np.linalg.norm(xy-q, axis=1)
            better = dist < best_d
            best_d[better] = dist[better]; best_s[better] = a+u[better]*self.len[i]
        return best_s, best_d

    def corridor(self):
        m = np.zeros((H, W), np.uint8)
        for a, b in zip(self.p[:-1], self.p[1:]):
            cv2.line(m, tuple(int(v) for v in a), tuple(int(v) for v in b), 255, 2*CORRIDOR)
        for a in self.p:
            cv2.circle(m, tuple(int(v) for v in a), CORRIDOR, 255, -1)
        return m > 0


def trail_mask(frame, hue=(3, 22), sat=80, val=190):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= hue[0]) & (h <= hue[1]) & (s >= sat) & (v >= val)


def board_mask(frame):
    """Regions that hide the trail or mimic it: the speckled deck (True Skate draws the trail
    *under* the board) and busy scenery. Fine-scale texture energy is ~100 on the deck,
    1-4 on park ground and ~5 on the smooth trail glow; thin edges and grid lines are opened away."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    energy = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (21, 21))
    ellipse = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    region = cv2.morphologyEx((energy > 35).astype(np.uint8), cv2.MORPH_OPEN, ellipse(15))
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, ellipse(31))
    return cv2.dilate(region, ellipse(21)) > 0


def measure(sample, frames, *, threshold=25, back_ms=40, ahead_ms_per_frame=50, link_ms=25, max_dip=2):
    """Track the head along the timing-blind glow profile (see profile.py).

    Per frame: runs of excess >= threshold, bridging hidden (board) spans and dips of
    <= max_dip samples. The chosen run must start within link of the previous head, or
    be reachable from it across hidden samples only. The head is the half-plateau
    crossing at the run's leading edge (the endpoint of a glowing line). Windows come
    from the previous observed head, never from the ring's nominal time.
    """
    from profile import profile, STEP
    path = Path_(sample); times = np.asarray(sample["frame_times_s"])
    onset, rate, total = sample["onset_s"], sample["clock_rate"], path.t[-1]
    head_s, head_ms, missed, streak, rows = 0.0, 0.0, 0, 0, []
    for k, (frame, t) in enumerate(zip(frames, times)):
        nominal = (t-onset)/rate*1000
        if nominal < -20 or nominal > total+150:
            continue
        board = board_mask(frame)
        sv, e, hit = profile(frame, path, board)
        row = dict(frame=k, t=float(t), nominal_ms=float(nominal))
        lo = int(path.s_at_time(head_ms-back_ms)//STEP)
        hi = int(np.ceil(path.s_at_time(head_ms+ahead_ms_per_frame*(missed+1))/STEP))
        link = int(np.ceil(path.s_at_time(head_ms+link_ms+ahead_ms_per_frame*missed)/STEP))
        on = np.where(np.isnan(e), False, e >= threshold); hidden = np.isnan(e)
        chosen = None; j = lo
        while j <= min(hi, len(e)-1):
            if not on[j]:
                j += 1; continue
            # Is this run's start connected to the previous head (within link, or across hidden only)?
            between = range(link, j)
            if j > link and any(not hidden[x] and not on[x] for x in between):
                break
            a = j; dip = 0; last = j
            while j < len(e)-1:
                j += 1
                if on[j]:
                    last, dip = j, 0
                elif hidden[j]:
                    continue
                else:
                    dip += 1
                    if dip > max_dip:
                        break
            chosen = (a, last); j = last+1
            # Prefer the furthest connected run; keep scanning for runs reachable across hidden spans.
            link = last
        if chosen is None:
            missed += 1; streak = 0
        else:
            a, last = chosen
            run = [x for x in range(a, last+1) if not hidden[x]]
            plateau = float(np.median(e[run[-10:]]))
            edge = last
            while edge+1 < len(e) and not hidden[edge+1] and e[edge+1] >= plateau/2:
                edge += 1
            nxt = e[edge+1] if edge+1 < len(e) else np.nan
            frac = 0. if np.isnan(nxt) or e[edge] == nxt else float(np.clip((e[edge]-plateau/2)/(e[edge]-nxt), 0, 1))
            tip = max(0., (edge+frac)*STEP-R_CAP)
            # The glow's end must be seen: no hidden samples within 2 caps + 16 px beyond it.
            occluded = bool(any(hidden[edge+1:edge+1+int(np.ceil((2*R_CAP+16)/STEP))]))
            capped = bool(last >= hi or tip > hi*STEP)
            # Only in-window detections move the head; anything else counts as a miss.
            accepted = not capped and not occluded and tip >= path.s_at_time(head_ms-back_ms)
            streak = streak+1 if accepted else 0
            missed = 0 if accepted else missed+1
            if accepted and tip >= head_s:
                head_s = tip; head_ms = path.time_at_s(head_s) or head_ms
            observed = path.time_at_s(tip)
            i = int(np.clip(np.searchsorted(path.S, tip, side="right")-1, 0, len(path.len)-1))
            speed = path.len[i]/(path.t[i+1]-path.t[i])
            ring = path.at_s(path.s_at_time(nominal)) if 0 <= nominal <= total else None
            xy = path.at_s(tip)
            row.update(head_s=tip, head_xy=[float(v) for v in xy], plateau=plateau, segment=i,
                       speed_px_ms=float(speed), occluded=occluded, capped=capped,
                       streak=streak,
                       reliable=bool(speed >= .4 and not occluded and accepted and streak >= 2 and tip-path.S[i] >= R_CAP),
                       observed_ms=observed,
                       lag_ms=None if observed is None or not 0 <= nominal <= total else float(nominal-observed),
                       ring_gap_pt=None if ring is None else float(np.linalg.norm(xy-ring)/2))
        rows.append(row)
    return rows


if __name__ == "__main__":
    M = Path(sys.argv[1]); OUT = Path(sys.argv[2]); ids = sys.argv[3]
    manifest = json.loads((M/"audit-6/manifest.json").read_text())
    chosen = [s for s in manifest["samples"] if ids == "all" or s["id"] in {int(x) for x in ids.split(",")}]
    results = {}
    for s in chosen:
        results[s["id"]] = measure(s, decode(M/s["sample_dir"]/"clip.mp4"))
        lags = [r["lag_ms"] for r in results[s["id"]] if r.get("lag_ms") is not None and r["reliable"]]
        gaps = [r["ring_gap_pt"] for r in results[s["id"]] if r.get("ring_gap_pt") is not None]
        print(s["id"], s["gesture"]["sampler"], s["gesture"].get("profile"), "frames", len(lags),
              "lag ms median/min/max", np.round([np.median(lags), min(lags), max(lags)], 1) if lags else "-",
              "max ring gap pt", round(max(gaps), 1) if gaps else "-", flush=True)
    (OUT/f"lag-{ids if ids != 'all' else 'all'}.json").write_text(json.dumps(results))
