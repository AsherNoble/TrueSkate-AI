"""Per-pass pointer delivery from a hover_rate_probe.py recording.

Usage: PYTHONPATH=<repo>/src hover_rate_measure.py <hover_rate_TAG.mov> <hover_rate_TAG.json> [out.json]

Each pass is a run of equal horizontal hover reports. Per pass:
- delivered: how many reports reached the screen, as net cursor travel (median hover
  position before and after the pass) over the modelled distance of one report;
- delivery span against send span: how long the cursor kept moving, against how long the
  board took to send;
- reports per frame, from the frame the first report shows to the frame the last one
  does. Reports are whole, so the cumulative travel is rounded to whole reports; this
  absorbs the tracker's ~0.5 pt jitter. At one report per 15 ms a 60 fps frame should
  carry one report, or two in about one frame in nine. "Frozen" frames carry none;
  "catch-up" frames carry three or more, i.e. late reports arriving together.
  Both are counted on regular (16.7 ms) frames only. The recorder drops a few frames
  (33 ms gaps, shown in brackets), and those carry two frames' worth of reports.
- displaced reports, for passes that delivered every report: the frame each report first
  shows in, against the frame a perfectly regular link would put it in. The regular link
  sends report i at i x step plus one constant latency, fitted per pass.
Passes are found from motion and matched to the probe's passes in order.
"""
import json, sys
import cv2, numpy as np
from trueskate_ai.control.hid_pointer import gain_distance

MOVING = 0.3          # reports per frame; finds the passes (the tracker jitters ~0.25 report)


def find_cursor_anywhere(frame, y_range_pt=(150, 800)):
    """Best cursor-disc candidate over the play area (same finder as gain_measure.py)."""
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


def displacement_frames(per_frame, frame_ms, step_ms):
    """Frames each report is late (+) or early (-) against a regular link with the best constant latency."""
    times = np.concatenate([[0.0], np.cumsum(frame_ms)])          # time of frame k, from the frame before the first report
    counts = np.concatenate([[0], np.cumsum(per_frame)])
    shown = np.array([int(np.argmax(counts >= i)) for i in range(1, counts[-1] + 1)])   # frame report i first shows in
    best = None
    for latency in np.arange(-step_ms, frame_ms.sum() / 2, 0.1):
        ideal = np.searchsorted(times, np.arange(len(shown)) * step_ms + latency)   # first frame at or after arrival
        d = shown - ideal
        cost = (np.abs(d).sum(), (d ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, d)
    return best[1]


mov, probe_path = sys.argv[1], sys.argv[2]
probe = json.load(open(probe_path))
cap = cv2.VideoCapture(mov)
t, x = [], []
while True:
    ok, f = cap.read()
    if not ok:
        break
    t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    d = find_cursor_anywhere(f, (300, 450))   # the parking row, y ~368 pt
    x.append(d[1] if d and d[0] > 8 else np.nan)
t, x = np.array(t), np.array(x)
fps = (len(t) - 1) / (t[-1] - t[0])

# motion segments: runs of moving frames (gaps of up to 4 still frames allowed), longer than 10 frames
per_report = gain_distance(abs(probe['passes'][0]['dx']))
step = np.diff(x) / per_report
idx = np.nonzero(np.abs(step) > MOVING)[0]
segs = [s for s in np.split(idx, np.nonzero(np.diff(idx) > 4)[0] + 1) if len(s) > 10]
assert len(segs) == len(probe['passes']), f"{len(segs)} motion segments for {len(probe['passes'])} passes"

rows = []
for p, s in zip(probe['passes'], segs):
    a, b = s[0], s[-1] + 1                      # frames a..b: rest before, moving, rest after
    before, after = np.nanmedian(x[max(a - 10, 0):a + 1]), np.nanmedian(x[b:b + 11])
    unit = per_report * np.sign(p['dx'])
    delivered = (after - before) / unit
    total = max(int(np.floor(delivered + 0.5)), 1)
    n = np.round((x[a:b + 1] - before) / (unit * delivered / total))   # whole reports shown so far, per frame
    lost = int(np.isnan(n).sum())
    first = int(np.argmax(n > 0))               # first frame showing a report
    last = int(np.argmax(n >= total))           # first frame showing all of them
    per_frame = np.diff(n[first - 1:last + 1]).astype(int)
    frame_ms = 1000 * np.diff(t[a + first - 1:a + last + 1])
    regular = frame_ms < 25
    longest = run = 0
    for c, ok in zip(per_frame, regular):
        run = run + 1 if (c == 0 and ok) else 0
        longest = max(longest, run)
    rows.append(dict(dx=p['dx'], step_ms=p['step_us'] / 1000, sent=p['reports'], delivered=round(float(delivered), 1),
                     send_ms=round(p['reports'] * p['step_us'] / 1000),
                     deliver_ms=round(1000 * float(t[a + last] - t[a + first - 1])),
                     frames=int(regular.sum()), frozen=int(((per_frame == 0) & regular).sum()), longest_freeze=longest,
                     catch_up=int(((per_frame >= 3) & regular).sum()),
                     dropped_frames=[int(c) for c in per_frame[~regular]],
                     backwards=int((per_frame < 0).sum()), untracked=lost,
                     per_frame=per_frame.tolist(), frame_ms=[round(float(v), 1) for v in frame_ms]))
    if total == p['reports'] and not lost and (per_frame >= 0).all():
        rows[-1]['displaced'] = displacement_frames(per_frame, frame_ms, p['step_us'] / 1000).tolist()
print(f'{mov}: {len(t)} frames at {fps:.1f} fps ({int((np.diff(t) > 0.025).sum())} dropped), '
      f'cursor found in {int(np.isfinite(x).sum())}; one {abs(probe["passes"][0]["dx"])}-count report = {per_report:.3f} pt')
print(f'{"step ms":>7} {"dx":>3} {"sent":>5} {"delivered":>9} {"send ms":>8} {"deliver ms":>10} '
      f'{"frames":>6} {"frozen":>6} {"longest":>7} {"catch-up":>8}  reports per frame ([n] = after a dropped frame)')
for r in rows:
    digits = ''.join((str(c) if 0 <= c <= 9 else '?') if ms < 25 else f'[{c}]' for c, ms in zip(r['per_frame'], r['frame_ms']))
    print(f'{r["step_ms"]:7.0f} {r["dx"]:+3d} {r["sent"]:5d} {r["delivered"]:9.1f} {r["send_ms"]:8d} {r["deliver_ms"]:10d} '
          f'{r["frames"]:6d} {r["frozen"]:6d} {r["longest_freeze"]:7d} {r["catch_up"]:8d}  ' + digits
          + (f'  ({r["backwards"]} backwards, {r["untracked"]} untracked)' if r['backwards'] or r['untracked'] else ''))
by_step = {}
for r in rows:
    g = by_step.setdefault(r['step_ms'], dict(passes=0, sent=0, delivered=0.0, frames=0, frozen=0, longest_freeze=0, catch_up=0,
                                               per_frame_counts={}, dropped_frames=[], displaced={}))
    g['passes'] += 1; g['sent'] += r['sent']; g['delivered'] = round(g['delivered'] + r['delivered'], 1)
    g['frames'] += r['frames']; g['frozen'] += r['frozen']; g['catch_up'] += r['catch_up']
    g['longest_freeze'] = max(g['longest_freeze'], r['longest_freeze'])
    g['dropped_frames'] += r['dropped_frames']
    for d in r.get('displaced', []):
        g['displaced'][str(d)] = g['displaced'].get(str(d), 0) + 1
    for c, ms in zip(r['per_frame'], r['frame_ms']):
        if ms < 25:
            g['per_frame_counts'][str(c)] = g['per_frame_counts'].get(str(c), 0) + 1
for k, g in by_step.items():
    g['per_frame_counts'] = dict(sorted(g['per_frame_counts'].items(), key=lambda kv: int(kv[0])))
    g['frozen_pct'] = round(100 * g['frozen'] / max(g['frames'], 1), 1)
    g['displaced'] = dict(sorted(g['displaced'].items(), key=lambda kv: int(kv[0])))
    g['displaced_pct'] = round(100 * sum(v for k_, v in g['displaced'].items() if k_ != '0') / max(sum(g['displaced'].values()), 1), 1)
    print(f'every {k:g} ms: {g["delivered"]:.0f} of {g["sent"]} reports delivered; on {g["frames"]} regular frames: '
          f'frozen {g["frozen"]} ({g["frozen_pct"]}%), longest {g["longest_freeze"]}, catch-up {g["catch_up"]}; '
          f'frames by reports carried {g["per_frame_counts"]}; dropped frames carried {g["dropped_frames"]}'
          + (f'; reports by frames displaced {g["displaced"]} ({g["displaced_pct"]}% off their frame)' if g['displaced'] else ''))
if len(sys.argv) > 3:
    json.dump(dict(recording=mov, fps=round(fps, 2), per_report_pt=round(per_report, 4), board_max_late_us=probe['max_late_us'],
                   passes=rows, by_step={f'{k:g}ms': g for k, g in by_step.items()}), open(sys.argv[3], 'w'), indent=1)
