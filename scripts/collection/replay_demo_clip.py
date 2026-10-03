"""Replay timed gestures extracted from an expert demo clip on one XR; raw recordings only.

One touch contact per request (GESTURES.md): each gesture is its own
POST /wda/perform_trick_gestures call, fired from Python at its scheduled start.
Bundling several contacts into one request makes True Skate join them into a
single chain (May 2026, and again in the 2026-10-03 bundled replay).
--mode separate: one HTTP call per gesture, fired from Python. Calls block until
the gesture finishes plus ~0.3 s, so later gestures start late.
--mode scheduled: one POST /wda/perform_gesture_schedule; WDA submits each
gesture as its own record at its start time on the device.
Intended vs. actual times are logged in both modes.
Variants resample each gesture so points are at least N ms apart.
No training admission; output goes to a new directory.
"""
import argparse, json, math, random, socket, time
from pathlib import Path
from trueskate_ai.data.control_hitboxes import segment_is_safe
from trueskate_ai.research.curved_audit import reset_payload, save_new

W, H = 414, 896
SPACINGS = {'native': None, 'min33ms': 33, 'min50ms': 50}


def resample(samples, spacing_ms):
    """samples: [(t_ms, x, y)] from the clip. Returns [(t_ms, x, y)] with >= spacing between points."""
    if spacing_ms is None:
        return samples
    t0, t1 = samples[0][0], samples[-1][0]
    n = max(1, math.floor((t1 - t0) / spacing_ms))
    out = []
    for k in range(n + 1):
        t = t0 + (t1 - t0) * k / n
        j = max(i for i in range(len(samples)) if samples[i][0] <= t + 1e-9)
        if j == len(samples) - 1:
            out.append((t, samples[j][1], samples[j][2]))
            continue
        (ta, xa, ya), (tb, xb, yb) = samples[j], samples[j + 1]
        u = (t - ta) / (tb - ta)
        out.append((t, xa + u * (xb - xa), ya + u * (yb - ya)))
    return out


def gesture_requests(gestures, spacing_ms):
    """gestures: {name: [(t_s, x, y)]} in clip seconds, normalised. Returns [(name, start_s, payload)]."""
    origin = min(s[0][0] for s in gestures.values())
    out = []
    for name, rows in gestures.items():
        samples = [(round((t - origin) * 1000), x, y) for t, x, y in rows]
        points = resample(samples, spacing_ms)
        times = [round(t) for t, _, _ in points]
        q = [(int(x * W), int(y * H)) for _, x, y in points]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f'{name}: non-increasing times {times}')
        if any(not segment_is_safe((a[0] / W, a[1] / H), (b[0] / W, b[1] / H)) for a, b in zip(q, q[1:])):
            raise ValueError(f'{name}: path crosses a protected control')
        waypoints = [dict(x=q[0][0], y=q[0][1], duration_ms=0)]
        waypoints += [dict(x=p[0], y=p[1], duration_ms=b - a) for p, a, b in zip(q[1:], times, times[1:])]
        out.append((name, times[0] / 1000, {'gestures': [{'waypoints': waypoints}]}))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gestures', type=Path, required=True)
    p.add_argument('--device', default='iPhone_XR2')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261003)
    p.add_argument('--mode', choices=('separate', 'scheduled'), default='scheduled')
    p.add_argument('--schedule-mode', choices=('records', 'paths'), default='paths',
                   help="scheduled only: one record per gesture, or one record with indexed paths")
    a = p.parse_args()
    raw = json.loads(a.gestures.read_text())
    gestures = {name: [(r['t'], r['x'], r['y']) for r in rows] for name, rows in raw.items()}
    plan = [dict(variant=v, repeat=r) for r in range(a.repeats) for v in SPACINGS]
    rng = random.Random(a.seed)
    for start in range(0, len(plan), len(SPACINGS)):  # shuffle within each repeat block
        block = plan[start:start + len(SPACINGS)]; rng.shuffle(block); plan[start:start + len(SPACINGS)] = block
    requests = {v: gesture_requests(gestures, s) for v, s in SPACINGS.items()}
    if a.out.exists():
        raise SystemExit('output directory must be new')
    if 'training-server' not in socket.gethostname():
        p.error('run on the rig')
    a.out.mkdir(parents=True)
    save_new(a.out / 'plan.json', dict(mode=a.mode, schedule_mode=a.schedule_mode, gestures=raw, plan=plan, requests=requests, training_admission=False))
    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    from trueskate_ai.collection.wda_action_timing import _http_json
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    cfg = next(d for d in DEVICES if d['name'] == a.device)
    base = f"http://127.0.0.1:{cfg['wda_port']}"
    if _http_json(base + '/wda/activeAppInfo')['value']['bundleId'] != BUNDLE_ID:
        raise RuntimeError('True Skate must already be frontmost')
    if _http_json(base + '/wda/video').get('value') is not None:
        raise RuntimeError('recorder not idle')
    worker = DeviceSession(cfg); worker.connect(); driver = worker.driver
    log = []
    try:
        for k, item in enumerate(plan, 1):
            if driver.query_app_state(BUNDLE_ID) != 4:
                raise RuntimeError('foreground lost')
            driver.execute('actions', reset_payload()); time.sleep(3)
            recorder = XCTestScreenRecorder(driver, fps=30); recorder.start(); time.sleep(1)
            calls = []; t0 = time.monotonic()
            if a.mode == 'separate':
                for name, start_s, payload in requests[item['variant']]:
                    time.sleep(max(0., t0 + start_s - time.monotonic()))
                    begin = time.monotonic() - t0
                    _http_json(base + '/wda/perform_trick_gestures', payload)
                    calls.append(dict(gesture=name, intended_start_s=start_s, start_s=begin, end_s=time.monotonic() - t0))
            else:
                schedule = {'mode': a.schedule_mode,
                            'gestures': [dict(start_ms=round(start_s * 1000), waypoints=payload['gestures'][0]['waypoints'])
                                         for _, start_s, payload in requests[item['variant']]]}
                report = _http_json(base + '/wda/perform_gesture_schedule', schedule)['value']
                if not report.get('complete') or report.get('error') or any(g.get('error') for g in report.get('gestures', [])):
                    raise RuntimeError(f'schedule incomplete: {report}')
                if a.schedule_mode == 'records':
                    for (name, start_s, _), g in zip(requests[item['variant']], report['gestures']):
                        calls.append(dict(gesture=name, intended_start_s=start_s, start_s=g['submitted_s'], end_s=g['completed_s']))
                else:  # one record: timing is the record's, starts are as scheduled on device
                    calls = [dict(gesture=name, intended_start_s=start_s, start_s=start_s, record_completed_s=report['completed_s'])
                             for name, start_s, _ in requests[item['variant']]]
            time.sleep(3)
            name = f"{k:02d}_{item['variant']}_r{item['repeat'] + 1}.mov"
            recorder.stop_and_save(a.out / name)
            log.append(dict(**item, file=name, calls=calls))
            late = max(c['start_s'] - c['intended_start_s'] for c in calls)
            print(f'{k}/{len(plan)} {name} worst start delay {late * 1000:.0f} ms', flush=True)  # 0 by construction in paths mode
    finally:
        worker.disconnect()
        save_new(a.out / 'runs.json', dict(runs=log))


if __name__ == '__main__':
    main()
