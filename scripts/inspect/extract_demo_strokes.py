"""Timed finger paths from an expert screen recording, read from True Skate's orange trail.

Each video frame adds a new piece of trail: the ribbon from where the finger was on
the previous frame to where it is now, plus the glow around the finger. Its
centroid sits mid-segment, so centroid paths (the 2026-10-03 extraction) start about
half a frame of motion late and stop about half a frame short. For a 60 pt/frame
flick that is ~20 pt at the press and ~40 pt at the lift.

This version reads the finger, not the segment:
- on frames where the finger moved at least 1.5 glow radii, the position is the
  leading edge of the new trail moved back by the glow radius. On slower frames the
  new trail is a sliver of the glow, so its centroid is kept (the trail edge only
  adds jitter there);
- the press is the trailing edge of the first piece, timed by extrapolating the first
  frame's speed back to it;
- frames after the segment's last one are added while the leading edge still advances.

Stroke segmentation (which frames belong to which gesture) comes from an existing
extraction file and is kept as is.

Usage: extract_demo_strokes.py --clip CLIP.MP4 --segments extracted-gestures.json --out strokes.json
Output has the same shape as the input ({name: [{frame, t, x, y}, ...]}, x and y normalised),
with "kind" press | frame | extended on each sample.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np

PT_W, PT_H = 414, 896
HSV_LO, HSV_HI = (4, 89, 140), (19, 255, 255)        # hue 8-38 deg, s >= 0.35, v >= 0.55


def orange(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_LO, HSV_HI) > 0


def load_masks(clip: Path, wanted: set[int]):
    cap = cv2.VideoCapture(str(clip))
    masks, times, k = {}, {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k in wanted:
            masks[k] = orange(f)
            times[k] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        k += 1
    h, w = next(iter(masks.values())).shape
    return masks, times, (PT_W / w, PT_H / h)


def new_pixels(masks, k, scale, centre, radius):
    """New orange pixels on frame k (points) within radius of centre."""
    ys, xs = np.nonzero(masks[k] & ~masks[k - 1])
    p = np.stack([xs * scale[0], ys * scale[1]], 1)
    return p[np.hypot(*(p - centre).T) <= radius]


def edge(p, u, front=True, frac=0.03):
    """Mean of the pixels in the leading (or trailing) frac along direction u."""
    proj = p @ u
    cut = np.quantile(proj, 1 - frac) if front else np.quantile(proj, frac)
    sel = proj >= cut if front else proj <= cut
    return p[sel].mean(0)


def unit(v):
    n = np.hypot(*v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def extract(rows, masks, times, scale):
    c = np.array([[r['x'] * PT_W, r['y'] * PT_H] for r in rows])
    frames = [r['frame'] for r in rows]
    n = len(rows)
    dirs = [unit(c[min(i + 1, n - 1)] - c[max(i - 1, 0)]) for i in range(n)]
    steps = [np.hypot(*(c[i] - c[i - 1])) if i else (np.hypot(*(c[1] - c[0])) if n > 1 else 0.0) for i in range(n)]
    blobs = [new_pixels(masks, k, scale, c[i], max(40.0, 0.75 * steps[i] + 30)) for i, k in enumerate(frames)]
    # glow radius: half the trail's width across the motion, median over frames
    widths = []
    for p, u in zip(blobs, dirs):
        if len(p) >= 30:
            perp = p @ np.array([-u[1], u[0]])
            widths.append((np.quantile(perp, 0.97) - np.quantile(perp, 0.03)) / 2)
    r_raw = float(np.median(widths)) if widths else 6.0
    r = float(np.clip(r_raw, 3.0, 16.0))
    # How far the finger moved during each frame. The first piece holds the whole ribbon
    # from the press with a glow cap at both ends, so its length less two radii is the
    # move; after that the step between consecutive centroids is the steadier measure.
    first = blobs[0] @ dirs[0] if len(blobs[0]) >= 15 else np.zeros(1)
    moved = [max(0.0, float(np.quantile(first, 0.97) - np.quantile(first, 0.03)) - 2 * r)] + steps[1:]
    pos = [edge(p, u) - r * u if m >= 1.5 * r else ci for p, u, m, ci in zip(blobs, dirs, moved, c)]
    # press: trailing edge of the first piece, when the finger was already moving on it
    u0 = dirs[0]
    if moved[0] >= 0.5 * r:
        pos[0] = edge(blobs[0], u0) - r * u0
        press = edge(blobs[0], u0, front=False) + r * u0
    else:
        press = pos[0]
    if np.dot(pos[0] - press, u0) < 0:              # degenerate first piece: no separate press point
        press = pos[0]
    out = []
    speed = np.hypot(*(pos[1] - pos[0])) / max(times[frames[1]] - times[frames[0]], 1e-3) if n > 1 else 0.0
    lead = np.hypot(*(pos[0] - press)) / speed if speed > 1e-6 else 0.0
    lead = min(lead, times[frames[0]] - times[frames[0] - 1])   # the press cannot be before the previous frame
    out.append(dict(frame=frames[0], t=round(times[frames[0]] - lead, 6), kind='press'))
    out[-1].update(x=float(press[0] / PT_W), y=float(press[1] / PT_H))
    for k, q in zip(frames, pos):
        out.append(dict(frame=k, t=round(times[k], 6), kind='frame', x=float(q[0] / PT_W), y=float(q[1] / PT_H)))
    # extend while the leading edge keeps advancing after the segment's last frame (fast ends only)
    last, u = pos[-1], dirs[-1]
    for k in ((frames[-1] + 1, frames[-1] + 2) if steps[-1] >= 1.5 * r else ()):
        if k not in masks:
            break
        p = new_pixels(masks, k, scale, last, 60.0)
        p = p[(p - last) @ u > r + 2]
        if len(p) < 15:
            break
        q = edge(p, u) - r * u
        if np.dot(q - last, u) < 2:
            break
        out.append(dict(frame=k, t=round(times[k], 6), kind='extended', x=float(q[0] / PT_W), y=float(q[1] / PT_H)))
        last = q
    return out, (r, r_raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clip', type=Path, required=True)
    ap.add_argument('--segments', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    seg = json.loads(a.segments.read_text())
    wanted = {k for rows in seg.values() for r in rows for k in (r['frame'] - 1, r['frame'], r['frame'] + 1, r['frame'] + 2)}
    masks, times, scale = load_masks(a.clip, wanted)
    result = {}
    for name, rows in seg.items():
        samples, r = extract(rows, masks, times, scale)
        result[name] = samples
        old = np.array([[rows[0]['x'] * PT_W, rows[0]['y'] * PT_H], [rows[-1]['x'] * PT_W, rows[-1]['y'] * PT_H]])
        new = np.array([[samples[0]['x'] * PT_W, samples[0]['y'] * PT_H], [samples[-1]['x'] * PT_W, samples[-1]['y'] * PT_H]])
        length = lambda s: sum(np.hypot((b['x'] - a_['x']) * PT_W, (b['y'] - a_['y']) * PT_H) for a_, b in zip(s, s[1:]))
        print(f"{name:8s} glow r {r[0]:4.1f} pt (measured {r[1]:4.1f}) | press moved {np.hypot(*(new[0] - old[0])):5.1f} pt, "
              f"{1000 * (rows[0]['t'] - samples[0]['t']):4.1f} ms earlier | end moved {np.hypot(*(new[1] - old[1])):5.1f} pt "
              f"| path {length(rows):6.1f} -> {length(samples):6.1f} pt | samples {len(rows)} -> {len(samples)}")
    a.out.write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
