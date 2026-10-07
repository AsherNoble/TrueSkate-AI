"""Per-pass USB pointer delivery from a pico_hover_rate recording, in hover_rate_measure.py's terms.

Usage: PYTHONPATH=<repo>/src measure.py <movie.mov> <schedule.json> <out_dir>

- The park row is found from the cursor itself (sparse full-area pass), because the
  Bluetooth gain may not hold over USB; the cursor is then tracked on every frame within
  +/-40 pt of that row, as the Bluetooth probe did. The movie is streamed, never held.
- Passes are placed by schedule time, anchored on the first 3 ms pass. USB can deliver a
  60-report 1 ms pass in a few frames, too short for motion-segment counting.
- One report's travel is measured from the 15 ms passes. At 3 and 1 ms, travel short of
  that is loss or rate-dependent acceleration; video alone cannot separate them.
- Press tests: contact sheets around each press for visual review, plus an orange-trail
  pixel count as a hint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hid_pointer'))
from hover_rate_measure import MOVING, find_cursor_anywhere, pass_row, print_rows, summarise  # noqa: E402

ROW_HALF_PT = 40
WINDOW_PAD_S = (0.15, 0.45)


def track(mov, y_range_pt, every=1):
    """Stream the movie: frame times and the cursor (x, y pt) found within y_range_pt."""
    cap = cv2.VideoCapture(mov)
    t, x, y = [], [], []
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        d = None
        if i % every == 0:
            ok, f = cap.retrieve()
            d = find_cursor_anywhere(f, y_range_pt) if ok else None
        good = d and d[0] > 8
        x.append(d[1] if good else np.nan)
        y.append(d[2] if good else np.nan)
        i += 1
    return np.array(t), np.array(x), np.array(y)


def grab(mov, indices):
    """Frames at the given indices, from a second streaming pass."""
    want, out = set(int(i) for i in indices), {}
    cap = cv2.VideoCapture(mov)
    i = 0
    while want - out.keys():
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    return out


def park_row(y):
    """Most common cursor row (5 pt bins) over the whole recording."""
    v = y[np.isfinite(y)]
    if not len(v):
        raise RuntimeError('Cursor never found')
    hist, edges = np.histogram(v, bins=np.arange(150, 805, 5))
    return float(edges[np.argmax(hist)] + 2.5)


def moving_frames(x, per_report):
    return np.nonzero(np.abs(np.diff(x) / per_report) > MOVING)[0]


def anchor_time(t, x, per_report):
    """Video time the first 3 ms pass starts: the first run of more than 10 moving frames."""
    idx = moving_frames(x, per_report)
    for s in np.split(idx, np.nonzero(np.diff(idx) > 4)[0] + 1):
        if len(s) > 10:
            return float(t[s[0]])
    raise RuntimeError('No pass found to anchor the schedule')


def window_segment(t, x, per_report, start_s, end_s):
    """Moving frames within one pass's time window, as pass_row's segment."""
    lo, hi = np.searchsorted(t, start_s - WINDOW_PAD_S[0]), np.searchsorted(t, end_s + WINDOW_PAD_S[1])
    idx = moving_frames(x, per_report)
    return idx[(idx >= lo) & (idx < hi)]


def orange_fraction(img):
    b, g, r = (img[..., i].astype(int) for i in range(3))
    return float(((r > 180) & (g > 70) & (g < 175) & (b < 100)).mean())


def press_sheets(mov, t, video_s, presses, park_xy, out):
    """Crops around the park point from 0.1 s before each press to 0.6 s after its lift."""
    px, py = int(park_xy[0] * 2), int(park_xy[1] * 2)
    x0, y0 = max(px - 160, 0), max(py - 300, 0)
    plan = []
    for p in presses:
        a = int(np.searchsorted(t, video_s(p['start_us']) - 0.1))
        b = min(int(np.searchsorted(t, video_s(p['lift_us']) + 0.6)), len(t) - 1)
        plan.append((p, max(a - 30, 0), np.linspace(a, b, 8).astype(int)))
    frames = grab(mov, [i for _, base, picks in plan for i in [base, *picks]])
    results = []
    for p, base, picks in plan:
        crops = [frames[i][y0:y0 + 420, x0:x0 + 320] for i in picks if i in frames]
        if not crops or base not in frames:
            results.append(dict(kind=p['kind'], error='frames not found'))
            continue
        cv2.imwrite(str(out / f"press-{p['kind']}.png"), np.hstack(crops))
        results.append(dict(kind=p['kind'], frames=[round(float(t[i]), 3) for i in picks],
                            orange_before=round(orange_fraction(frames[base][y0:y0 + 420, x0:x0 + 320]), 4),
                            orange_peak=round(max(orange_fraction(c) for c in crops), 4)))
    return results


def main(argv):
    mov, sched_path, out = argv[0], argv[1], Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    schedule = json.load(open(sched_path))
    _, _, y_sparse = track(mov, (150, 800), every=6)            # where the cursor mostly sits
    row = park_row(y_sparse)
    t, x, _ = track(mov, (int(row - ROW_HALF_PT), int(row + ROW_HALF_PT)))   # every frame on that row, as the BLE probe did
    fps = (len(t) - 1) / (t[-1] - t[0])

    passes = schedule['passes']
    first = passes[0]
    rough = 1.9                                               # pt per 4-count report, Bluetooth fit; only finds the anchor
    t0 = anchor_time(t, x, rough)

    def video_s(us):
        return t0 + (us - first['start_us']) / 1e6

    # One report's travel from the 15 ms passes, where every report is expected to arrive.
    travel = []
    for p in passes:
        if p['step_us'] == 15000:
            s = window_segment(t, x, rough, video_s(p['start_us']), video_s(p['end_us']))
            if len(s):
                a, b = s[0], s[-1] + 1
                travel.append(abs(np.nanmedian(x[b:b + 11]) - np.nanmedian(x[max(a - 10, 0):a + 1])) / p['reports'])
    if not travel:
        raise RuntimeError('No 15 ms pass measured; cannot calibrate one report')
    per_report = float(np.median(travel))

    rows = []
    for p in passes:
        s = window_segment(t, x, per_report, video_s(p['start_us']), video_s(p['end_us']))
        if not len(s):
            rows.append(dict(mode=p['mode'], dx=p['dx'], step_ms=p['step_us'] / 1000, sent=p['reports'], missing=True))
            continue
        r = pass_row(t, x, s, p['dx'], p['step_us'], p['reports'], per_report)
        r['mode'] = p['mode']
        r['start_offset_s'] = round(float(t[s[0]] - video_s(p['start_us'])), 3)   # anchor sanity: should stay small
        rows.append(r)
    measured = [r for r in rows if not r.get('missing')]
    offsets = np.array([r['start_offset_s'] for r in measured])
    if len(offsets) and np.ptp(offsets) > 0.2:
        print(f'WARNING: pass start offsets span {np.ptp(offsets):.3f} s; the schedule anchor may be wrong')
    print(f'{mov}: {len(t)} frames at {fps:.1f} fps ({int((np.diff(t) > 0.025).sum())} dropped); park row {row:.0f} pt; '
          f'cursor found in {int(np.isfinite(x).sum())}; one {schedule["pass_dx"]}-count report = {per_report:.3f} pt '
          f'(Bluetooth fit 1.896); anchor {t0:.3f} s; {len(rows) - len(measured)} passes not found')
    print_rows(measured)
    by_step = {}
    for mode in ('rates', 'stall', 'usb1ms'):
        print(f'-- {mode}')
        by_step[mode] = {f'{k:g}ms': g for k, g in summarise([r for r in measured if r['mode'] == mode]).items()}
    presses = press_sheets(mov, t, video_s, schedule['presses'], (float(np.nanmedian(x)), row), out)
    for p in presses:
        print(f"press {p['kind']}: orange {p['orange_before']} -> {p['orange_peak']} (review press-{p['kind']}.png)")
    json.dump(dict(recording=mov, fps=round(fps, 2), park_row_pt=row, per_report_pt=round(per_report, 4),
                   anchor_s=round(t0, 3), start_offset_span_s=round(float(np.ptp(offsets)), 3) if len(offsets) else None,
                   passes=rows, by_step=by_step, presses=presses),
              open(out / 'measure.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
