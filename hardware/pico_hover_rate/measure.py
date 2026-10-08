"""Per-pass pointer delivery from a hover-rate recording, in hover_rate_measure.py's terms.

Usage: PYTHONPATH=<repo>/src measure.py <movie> <schedule.json | Bluetooth probe .json> <out_dir> [stats.json]

Run the USB and the Bluetooth recordings through this same tool, so the comparison uses one
tracker and one method:
- The cursor is found by the Bluetooth finder's circle search, refined to a sub-pixel
  centroid. The iOS pointer adapts to what is under it, so its polarity (darker or lighter
  than the floor) is read on the park row first and only that polarity is scored after.
  Isolated one-frame jumps out and straight back are tracker misses; they are repaired and counted.
  The park row comes from the cursor itself; tracking uses the Bluetooth band around it,
  with the board exclusion kept below the row.
- Passes are placed by schedule time, anchored on the first pass, each window running to
  the next pass. With read_stats.py's stats.json the windows use the board's actual send
  times instead, so a slow host cannot push a pass out of its window.
- One report's travel comes from the 15 ms passes. At 3 and 1 ms, travel short of that is
  loss or rate-dependent acceleration; video alone cannot separate them.
- Regularity: 'displaced' (each report's frame against a perfectly regular link with one
  fitted latency) is comparable at every spacing. 'frozen' and 'catch-up' frames only mean
  something at 15 ms, where a frame should carry one report.
- Noise: cursor jitter while parked, and per pass the in-motion distance from whole report
  counts (residual p99; flagged above 0.35 of a report). Passes reaching a screen edge, or
  sent late by the board (stats), are flagged; late ones are not counted as link timing.
- With board stats, a pass sent fully on time also gets "displaced_all_delivered": frame
  landing using that pass's own gain, valid when no report was lost.
- Decoding uses FFmpeg when available and records the decoder; measure both links on one
  machine.
- Press tests (USB schedule only): contact sheets around each press for visual review, and
  an orange-trail pixel count as a hint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hid_pointer'))
from hover_rate_measure import MOVING, pass_row, print_rows, summarise  # noqa: E402

BAND_PT = (-68, 82)            # the Bluetooth probe's (300, 450) band around its 368 pt row
BOARD = (150, 270, 380, 640)   # x0, x1, y0, y1 pt: the board, never the cursor
WINDOW_LEAD_S, WINDOW_TAIL_S = 0.15, 0.05
MIN_SCORE = 8


def find_cursor(frame, y_range_pt, board_top_pt=BOARD[2], polarity=0):
    """Best cursor-disc candidate within y_range_pt: (score, x_pt, y_pt, r_pt, sign) or None.

    polarity +1 scores a disc darker than its ring (the Bluetooth finder), -1 a lighter one,
    0 either; sign is +1 when the chosen disc is the darker.
    """
    y0, y1 = max(int(y_range_pt[0]) * 2, 0), int(y_range_pt[1]) * 2
    raw = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(raw, 3)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=10,
                               minRadius=14, maxRadius=20)
    if circles is None:
        return None
    yy, xx = np.mgrid[0:60, 0:60]
    best = None
    for cx, cy, r in circles[0]:
        X, Y = cx / 2, (cy + y0) / 2
        if 340 <= X and 665 <= Y <= 740:                          # AssistiveTouch menu button
            continue
        if BOARD[0] <= X <= BOARD[1] and board_top_pt <= Y <= BOARD[3]:
            continue
        xa, ya = int(cx) - 30, int(cy) - 30
        if xa < 0 or ya < 0 or xa + 60 > gray.shape[1] or ya + 60 > gray.shape[0]:
            continue
        patch = gray[ya:ya + 60, xa:xa + 60].astype(float)
        rr = np.hypot(xx - (cx - xa), yy - (cy - ya))
        inside, ring = rr < r - 4, (rr > r + 3) & (rr < r + 9)
        contrast = patch[ring].mean() - patch[inside].mean()
        score = (abs(contrast) if polarity == 0 else polarity * contrast) - 2 * patch[inside].std()
        if best is None or score > best[0]:
            best = (score, cx, cy, r, xa, ya, 1 if contrast >= 0 else -1)
    if best is None:
        return None
    score, cx, cy, r, xa, ya, sign = best
    # Sub-pixel centre: contrast-weighted centroid of the disc on the unblurred image.
    patch = raw[ya:ya + 60, xa:xa + 60].astype(float)
    rr = np.hypot(xx - (cx - xa), yy - (cy - ya))
    ring_mean = patch[(rr > r + 3) & (rr < r + 9)].mean()
    w = np.abs(patch - ring_mean) * (rr < r + 2)
    if w.sum() > 0:
        cx, cy = xa + (w * xx).sum() / w.sum(), ya + (w * yy).sum() / w.sum()
    return score, cx / 2, (cy + y0) / 2, r / 2, sign


def open_movie(mov):
    """FFmpeg when this OpenCV has it, so results do not depend on the machine's decoder."""
    cap = cv2.VideoCapture(mov, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(mov)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {mov}')
    return cap


def track(mov, y_range_pt, every=1, board_top_pt=BOARD[2], polarity=0):
    """Stream the movie: frame times, cursor x and y (pt), and the chosen disc's sign per frame."""
    cap = open_movie(mov)
    t, x, y, sign = [], [], [], []
    i = 0
    while cap.grab():
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        d = None
        if i % every == 0:
            ok, f = cap.retrieve()
            d = find_cursor(f, y_range_pt, board_top_pt, polarity) if ok else None
        good = d is not None and d[0] > MIN_SCORE
        x.append(d[1] if good else np.nan)
        y.append(d[2] if good else np.nan)
        sign.append(d[4] if good else 0)
        i += 1
    return np.array(t), np.array(x), np.array(y), np.array(sign)


def repair_spikes(x, limit_pt=15.0):
    """Replace isolated one-frame jumps out and straight back (tracker misses) by the neighbours' mean.

    Real motion never jumps more than limit_pt out and back within one frame.
    Returns the repaired track and the number of frames repaired.
    """
    x = x.copy()
    d_in, d_out = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    spike = ((np.abs(d_in) > limit_pt) & (np.abs(d_out) > limit_pt) & (np.sign(d_in) != np.sign(d_out))
             & (np.abs(x[2:] - x[:-2]) < limit_pt))
    idx = np.nonzero(spike)[0] + 1
    x[idx] = (x[idx - 1] + x[idx + 1]) / 2
    return x, int(len(idx))


def grab(mov, indices):
    """Frames at the given indices, from a second streaming pass."""
    want, out = set(int(i) for i in indices), {}
    cap = open_movie(mov)
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
    """Most common cursor row (5 pt bins) over the recording."""
    v = y[np.isfinite(y)]
    if not len(v):
        raise RuntimeError('Cursor never found')
    hist, edges = np.histogram(v, bins=np.arange(150, 805, 5))
    return float(edges[np.argmax(hist)] + 2.5)


def moving_frames(x, per_report):
    return np.nonzero(np.abs(np.diff(x) / per_report) > MOVING)[0]


def anchor_time(t, x, per_report):
    """Video time the first pass starts: the first run of more than 10 moving frames."""
    idx = moving_frames(x, per_report)
    for s in np.split(idx, np.nonzero(np.diff(idx) > 4)[0] + 1):
        if len(s) > 10:
            return float(t[s[0]])
    raise RuntimeError('No pass found to anchor the schedule')


def window_segment(t, x, per_report, start_s, end_s, dx=0):
    """The pass's motion within [start_s - lead, end_s), as pass_row's segment.

    Moving frames are split into runs (gaps of up to 4 still frames allowed, as in
    motion_segments). The pass is the run with the most net travel along dx, so a tracking
    glitch out and back elsewhere in the window (no net travel) cannot win or stretch it.
    """
    lo, hi = np.searchsorted(t, start_s - WINDOW_LEAD_S), np.searchsorted(t, end_s)
    idx = moving_frames(x, per_report)
    idx = idx[(idx >= lo) & (idx < hi)]
    if not len(idx):
        return idx
    runs = np.split(idx, np.nonzero(np.diff(idx) > 4)[0] + 1)
    sign = np.sign(dx) if dx else 1

    def net(r):
        a, b = x[r[0]], x[min(r[-1] + 1, len(x) - 1)]
        travel = sign * (b - a) if dx else abs(b - a)
        return travel if np.isfinite(travel) else -np.inf
    return max(runs, key=net)


def pass_times(schedule, stats):
    """Per pass (start_us, end_us) on the board's clock: actual send times when stats are given."""
    return [(a, b) for a, b, _ in pass_board(schedule, stats)]


def pass_board(schedule, stats):
    """Per pass (start_us, end_us, max_late_us); max_late_us is None without stats."""
    passes = schedule['passes']
    if stats is None:
        return [(p['start_us'], p['end_us'], None) for p in passes]
    from schedule import build, schedule_hash
    events, *_ = build()
    if schedule_hash(events) != schedule['schedule_hash'] or len(stats['lateness_us']) != len(events):
        raise ValueError('stats/schedule do not match the schedule.py build')
    t = np.array([e[0] for e in events])
    late = np.array(stats['lateness_us'])
    out = []
    for p in passes:
        idx = np.nonzero((t >= p['start_us']) & (t < p['end_us']))[0]
        if (late[idx] < 0).any():
            raise ValueError('a pass was not fully sent; measure without stats')
        sent = t[idx] + late[idx]
        out.append((int(sent[0]), int(sent[-1] + p['step_us']), int(late[idx].max())))
    return out


TRACK_PT = (15.0, 399.0)         # the finder cannot see the cursor beyond these x (a 60 px patch must fit)
RESIDUAL_FLAG = 0.35             # in-motion distance from a whole report count, in reports
BOARD_LATE_US = 1000             # a pass the board sent later than this is not a regular schedule


def in_motion_residual(x, s, unit):
    """Per-frame distance of the cumulative travel from a whole count of unit, within segment s.

    unit is the pass's own travel per counted report (pass_row's), so a rate-dependent gain
    does not read as noise.
    """
    a, b = s[0], s[-1] + 1
    before = np.nanmedian(x[max(a - 10, 0):a + 1])
    k = (x[a:b + 1] - before) / unit
    k = k[np.isfinite(k)]
    return np.abs(k - np.round(k)) if len(k) else np.array([np.nan])


def touches_edge(x, s, margin_pt=5.0):
    """Whether the pass may have run past where the cursor can be tracked or the screen ends.

    Fast passes move 30-100 pt per frame, so the last tracked in-pass point says little.
    The rest positions do: a pass clamped at the screen edge (0 or 414 pt) comes to rest
    outside the trackable range, so its rest frames are lost or sit at the tracking limit.
    """
    a, b = s[0], s[-1] + 2
    for rest in (x[max(a - 10, 0):a + 1], x[b - 1:b + 10]):
        seen = rest[np.isfinite(rest)]
        if len(rest) and len(seen) < 0.5 * len(rest):
            return True
        if len(seen) and (np.median(seen) <= TRACK_PT[0] + margin_pt or np.median(seen) >= TRACK_PT[1] - margin_pt):
            return True
    return False


def noise_floor(t, x, video_s, times, modes):
    """RMS cursor jitter (pt) while parked between consecutive passes of one section."""
    dev = []
    for (_, end), (nxt, _), m1, m2 in zip(times, times[1:], modes, modes[1:]):
        if m1 != m2:                                    # a re-home and re-park lies between sections
            continue
        lo, hi = np.searchsorted(t, video_s(end) + 0.3), np.searchsorted(t, video_s(nxt) - 0.05)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 5:
            dev.extend(seg - np.median(seg))
    return float(np.sqrt(np.mean(np.square(dev)))) if dev else None


def orange_fraction(img):
    b, g, r = (img[..., i].astype(int) for i in range(3))
    return float(((r > 180) & (g > 70) & (g < 175) & (b < 100)).mean())


def press_sheets(mov, t, video_s, presses, park_xy, out):
    """Crops around the park point from 0.1 s before each press to 0.6 s after its lift."""
    px, py = int(park_xy[0] * 2), int(park_xy[1] * 2)
    x0, y0 = max(px - 160, 0), max(py - 300, 0)
    plan = []
    results = []
    for p in presses:
        a = int(np.searchsorted(t, video_s(p['start_us']) - 0.1))
        b = min(int(np.searchsorted(t, video_s(p['lift_us']) + 0.6)), len(t) - 1)
        if a >= len(t) - 1:
            results.append(dict(kind=p['kind'], error='press is after the end of the recording'))
            continue
        plan.append((p, max(a - 30, 0), np.linspace(a, b, 8).astype(int)))
    frames = grab(mov, [i for _, base, picks in plan for i in [base, *picks]])
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
    stats = json.load(open(argv[3])) if len(argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    schedule = json.load(open(sched_path))
    passes = [dict(p, mode=p.get('mode', 'bluetooth')) for p in schedule['passes']]
    times = pass_times(schedule, stats)

    backend = open_movie(mov).getBackendName()
    board = pass_board(schedule, stats)
    _, _, y_sparse, sign_sparse = track(mov, (150, 800), every=6)   # where the cursor mostly sits
    row = park_row(y_sparse)
    on_row = np.abs(y_sparse - row) <= 10
    polarity = 1 if (sign_sparse[on_row] >= 0).mean() >= 0.5 else -1   # dark (as the Bluetooth finder) or light
    board_top = max(BOARD[2], row + 15)                          # never exclude the park row itself
    t, x, _, _ = track(mov, (row + BAND_PT[0], row + BAND_PT[1]), board_top_pt=board_top, polarity=polarity)
    x, repaired = repair_spikes(x)
    fps = (len(t) - 1) / (t[-1] - t[0])

    rough = 1.9                                                  # pt per 4-count report (Bluetooth fit); finds the anchor only
    t0 = anchor_time(t, x, rough)

    def video_s(us):
        return t0 + (us - times[0][0]) / 1e6

    def window(i):
        start = times[i][0]
        end = times[i + 1][0] if i + 1 < len(times) else times[i][1] + 1_000_000
        return video_s(start), video_s(end) - WINDOW_TAIL_S

    # One report's travel from the 15 ms passes, where every report is expected to arrive.
    travel = []
    for i, p in enumerate(passes):
        if p['step_us'] == 15000:
            s = window_segment(t, x, rough, *window(i), dx=p['dx'])
            if len(s) and not touches_edge(x, s):
                a, b = s[0], s[-1] + 1
                travel.append(abs(np.nanmedian(x[b:b + 11]) - np.nanmedian(x[max(a - 10, 0):a + 1])) / p['reports'])
    if not travel:
        raise RuntimeError('No 15 ms pass measured; cannot calibrate one report')
    per_report = float(np.median(travel))
    noise = noise_floor(t, x, video_s, times, [p['mode'] for p in passes])

    rows = []
    for i, p in enumerate(passes):
        s = window_segment(t, x, per_report, *window(i), dx=p['dx'])
        if not len(s):
            rows.append(dict(mode=p['mode'], dx=p['dx'], step_ms=p['step_us'] / 1000, sent=p['reports'], missing=True))
            continue
        r = pass_row(t, x, s, p['dx'], p['step_us'], p['reports'], per_report)
        r['mode'] = p['mode']
        r['start_offset_s'] = round(float(t[s[0]] - window(i)[0]), 3)   # anchor sanity: should stay small
        unit = per_report * r['delivered'] / max(round(r['delivered']), 1)     # pass_row's per-report unit
        r['residual_p99'] = round(float(np.nanpercentile(in_motion_residual(x, s, unit), 99)), 3)
        r['edge_clipped'] = touches_edge(x, s)
        if r['edge_clipped']:
            r.pop('displaced', None)                   # travel was cut by the screen edge: not a link measurement
        late = board[i][2]
        r['board_max_late_us'] = late
        if late is not None and late > BOARD_LATE_US:
            r.pop('displaced', None)                   # the board itself was irregular: not a link measurement
            r['board_late'] = True
        elif late is not None and not r['edge_clipped'] and not r['untracked'] and r['backwards'] == 0:
            # Every report left the board on time. If none was lost, the pass's own travel per
            # report gives its gain, and frame landing can be compared even when gain depends on rate.
            a, b = s[0], s[-1] + 1
            travel = abs(np.nanmedian(x[b:b + 11]) - np.nanmedian(x[max(a - 10, 0):a + 1]))
            scaled = pass_row(t, x, s, p['dx'], p['step_us'], p['reports'], travel / p['reports'])
            if 'displaced' in scaled:
                r['displaced_all_delivered'] = scaled['displaced']
                r['pass_gain_pt'] = round(travel / p['reports'], 4)
        rows.append(r)
    measured = [r for r in rows if not r.get('missing')]
    offsets = np.array([r['start_offset_s'] for r in measured])
    span = round(float(np.ptp(offsets)), 3) if len(offsets) else None
    print(f'{mov}: {len(t)} frames at {fps:.1f} fps ({int((np.diff(t) > 0.025).sum())} dropped); park row {row:.0f} pt; '
          f'cursor found in {int(np.isfinite(x).sum())}; one {schedule.get("pass_dx", 4)}-count report = {per_report:.3f} pt '
          f'(Bluetooth fit 1.896); {"dark" if polarity > 0 else "light"} pointer; {repaired} one-frame spikes repaired; '
          f'parked jitter {noise if noise is None else round(noise, 3)} pt rms; anchor {t0:.3f} s; '
          f'{len(rows) - len(measured)} passes not found; board times {"from stats" if stats else "scheduled"}; decoder {backend}')
    if span is not None and span > 0.2:
        print(f'WARNING: pass start offsets span {span:.3f} s; the schedule anchor may be wrong')
    flagged = [r for r in measured if r['residual_p99'] > RESIDUAL_FLAG or r['edge_clipped'] or r.get('board_late')]
    for r in flagged:
        why = ', '.join(k for k, v in (('residual', r['residual_p99'] > RESIDUAL_FLAG), ('edge', r['edge_clipped']),
                                       ('board late', r.get('board_late'))) if v)
        print(f"FLAG {r['mode']} {r['step_ms']:g} ms {r['dx']:+d}: {why} (residual p99 {r['residual_p99']})")
    print_rows(measured)
    by_step = {}
    for mode in dict.fromkeys(p['mode'] for p in passes):
        print(f'-- {mode}')
        by_step[mode] = {f'{k:g}ms': g for k, g in summarise([r for r in measured if r['mode'] == mode]).items()}
    all_delivered, gain_ratio = {}, {}
    for r in measured:
        if 'displaced_all_delivered' not in r:
            continue
        k = f"{r['step_ms']:g}ms"
        g = all_delivered.setdefault(k, {})
        for d in r['displaced_all_delivered']:
            g[str(d)] = g.get(str(d), 0) + 1
        gain_ratio.setdefault(k, []).append(r['pass_gain_pt'] / per_report)
    for k, g in all_delivered.items():
        n = sum(g.values())
        ratio = float(np.median(gain_ratio[k]))
        print(f'every {k}, assuming every on-time report arrived: {n} reports, '
              f'{100 * sum(v for d, v in g.items() if d != "0") / n:.1f}% off their frame '
              f'{dict(sorted(g.items(), key=lambda kv: int(kv[0])))}; travel per report {ratio:.2f}x the 15 ms gain. '
              + ('BELOW 1: reports may have been lost; do not cite as regularity.' if ratio < 0.97 else
                 'Loss and rate-dependent gain are inseparable on video; cite with that caveat.'))
    print('note: frozen/catch-up frames are meaningful at 15 ms only. "displaced" needs every report seen; '
          'at 1 and 3 ms rate-dependent gain can prevent that, so see the all-delivered line (board stats required).')
    presses = press_sheets(mov, t, video_s, schedule.get('presses', []), (float(np.nanmedian(x)), row), out)
    for p in presses:
        print(f"press {p['kind']}: orange {p.get('orange_before')} -> {p.get('orange_peak')} (review press-{p['kind']}.png)")
    json.dump(dict(recording=mov, decoder=backend, fps=round(fps, 2), park_row_pt=row, per_report_pt=round(per_report, 4),
                   displaced_all_delivered=all_delivered,
                   all_delivered_gain_ratio={k: round(float(np.median(v)), 3) for k, v in gain_ratio.items()},
                   parked_jitter_rms_pt=noise, pointer='dark' if polarity > 0 else 'light', spikes_repaired=repaired,
                   anchor_s=round(t0, 3), start_offset_span_s=span,
                   board_times='stats' if stats else 'scheduled', passes=rows, by_step=by_step, presses=presses),
              open(out / 'measure.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
