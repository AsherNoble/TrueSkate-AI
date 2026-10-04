"""Per-drag cursor displacement from a gain_probe.py recording.

Usage: gain_measure.py <gain_TAG.mov> <gain_TAG.json>  -> table + <mov>.hover.json (points)

The AssistiveTouch cursor is a ~17 pt grey disc, translucent while hovering and darker
while pressed. True Skate draws its orange touch trail only while a touch moves, so
the first orange frame is the start of the moves (press + hold) and the last is the
lift. Start and end are the median disc centre (Hough circle) in hover windows just
before the press and just after the lift. Pick parking spots on clear floor: the disc
is hard to find over the dark back wall or on the yellow floor line.
"""
import json, sys
import cv2, numpy as np

ORANGE = dict(h=(4, 19), s=89, v=140)


def orange_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= ORANGE['h'][0]) & (hsv[..., 0] <= ORANGE['h'][1]) & (hsv[..., 1] >= ORANGE['s'])
         & (hsv[..., 2] >= ORANGE['v']))
    m[760:1280, 300:540] = False      # the board's own orange
    m[1600:, :] = False               # bottom bar
    m[:160, :] = False                # top HUD
    m[:, :90] = False                 # left HUD column
    return m


def find_cursor_anywhere(frame, y_range_pt=(150, 800)):
    """Best cursor-disc candidate over the play area (hovering or pressed)."""
    y0, y1 = y_range_pt[0] * 2, y_range_pt[1] * 2
    gray = cv2.medianBlur(cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY), 3)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=10,
                               minRadius=14, maxRadius=20)
    if circles is None:
        return None
    best = None
    for cx, cy, r in circles[0]:
        X, Y = cx / 2, (cy + y0) / 2
        if 340 <= X and 665 <= Y <= 740:          # AssistiveTouch menu button
            continue
        if 150 <= X <= 270 and 380 <= Y <= 640:   # the board
            continue
        xa, ya = int(cx) - 30, int(cy) - 30
        if xa < 0 or ya < 0 or xa + 60 > gray.shape[1] or ya + 60 > gray.shape[0]:
            continue
        patch = gray[ya:ya + 60, xa:xa + 60].astype(float)
        yy, xx = np.mgrid[0:60, 0:60]
        rr = np.hypot(xx - (cx - xa), yy - (cy - ya))
        inside, ring = rr < r - 4, (rr > r + 3) & (rr < r + 9)
        score = (patch[ring].mean() - patch[inside].mean()) - 2 * patch[inside].std()
        if best is None or score > best[0]:
            best = (score, X, Y, r / 2)
    return best



mov, marks_path = sys.argv[1], sys.argv[2]
marks = json.load(open(marks_path))
HOLD = 0.2
cap = cv2.VideoCapture(mov)
times, counts = [], []
while True:
    ok, f = cap.read()
    if not ok:
        break
    times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    counts.append(int(orange_mask(f).sum()))
times, counts = np.array(times), np.array(counts)
on = np.nonzero(counts >= 40)[0]
events = np.split(on, np.nonzero(np.diff(on) > 2)[0] + 1) if len(on) else []
first = np.array([times[e[0]] for e in events]); last = np.array([times[e[-1]] for e in events])
drags = marks['marks']
for k in drags:
    if 'dx' not in k:
        k['dx'] = k['speed'] if k['axis'] == 'x' else 0
        k['dy'] = k['speed'] if k['axis'] == 'y' else 0
expected = np.array([k['go_epoch'] + (k['press_us'] / 1e6 + HOLD) - marks['started_at'] for k in drags])
cands = (first[:, None] - expected[None, :]).ravel()
offset = max(cands, key=lambda o: int((np.abs(first[:, None] - expected[None, :] - o) < 0.1).any(axis=0).sum()))
print(f'{len(events)} trail events, {len(drags)} drags, video offset {offset:.3f} s')
windows = []
for i, (k, e) in enumerate(zip(drags, expected)):
    j = int(np.argmin(np.abs(first - e - offset)))
    if abs(first[j] - e - offset) >= 0.1:
        windows.append(None); continue
    press = first[j] - HOLD
    windows.append(((press - 0.33, press - 0.04), (last[j] + 0.3, last[j] + 0.6)))
# second pass: detect the hovering cursor in the windows
dets = [([], []) for _ in drags]
cap = cv2.VideoCapture(mov)
while True:
    ok, f = cap.read()
    if not ok:
        break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    for i, w in enumerate(windows):
        if w is None:
            continue
        for side in (0, 1):
            if w[side][0] <= t <= w[side][1]:
                d = find_cursor_anywhere(f)
                if d and d[0] > 8:
                    dets[i][side].append(d)
out = []
for k, w, (pre, post) in zip(drags, windows, dets):
    tag = f"step ({k['dx']:4d},{k['dy']:4d}) x{k['reports']:3d} @{k['interval_us'] // 1000:3d} ms"
    if w is None or not pre or not post:
        print(tag, 'missing', 'no trail' if w is None else f'hover detections {len(pre)}/{len(post)}'); continue
    p0 = np.median([d[1:3] for d in pre], axis=0); p1 = np.median([d[1:3] for d in post], axis=0)
    s0 = np.ptp([d[1:3] for d in pre], axis=0).max(); s1 = np.ptp([d[1:3] for d in post], axis=0).max()
    cx, cy = k['dx'] * k['reports'], k['dy'] * k['reports']
    mv = p1 - p0
    gx = mv[0] / cx if cx else float('nan'); gy = mv[1] / cy if cy else float('nan')
    p0, p1, mv = p0.astype(float), p1.astype(float), mv.astype(float)
    out.append(dict(k, counts=[cx, cy], start=p0.round(2).tolist(), end=p1.round(2).tolist(), moved_pt=mv.round(2).tolist(),
                    gain=[round(float(gx), 4), round(float(gy), 4)], n_pre=len(pre), n_post=len(post), spread_pt=[round(float(s0), 2), round(float(s1), 2)]))
    print(f"{tag}: start ({p0[0]:6.1f},{p0[1]:6.1f}) moved ({mv[0]:6.1f},{mv[1]:6.1f}) pt  gain ({gx:.3f},{gy:.3f})  "
          f"n {len(pre)}/{len(post)} spread {s0:.1f}/{s1:.1f}")
json.dump(out, open(mov + '.hover.json', 'w'), indent=1)
