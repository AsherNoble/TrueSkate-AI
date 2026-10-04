"""Measure WDA request preparation vs pointer-move count on one XR; no recording.

Sends single-touch timed paths of 2/5/8/11/15 waypoints and 16/32/56 segments in a
seeded interleaved order, with WDA action timing enabled, and writes the raw
records plus a per-count summary. Collection must be off and True Skate frontmost.
"""
import argparse, json, math, random, socket, statistics, time
from pathlib import Path
from trueskate_ai.data.control_hitboxes import segment_is_safe
from trueskate_ai.research.curved_audit import save_new
from trueskate_ai.sim.timed_waypoints import TimedWaypoints

COUNTS = (2, 5, 8, 11, 15, 17, 33, 57)  # waypoints; 17/33/57 = 16/32/56 segments
DURATIONS_MS = (300, 600, 900)
GAP_S = 1.5


def path_points(n, phase):
    """Gentle S-path inside the central safe rectangle."""
    pts = []
    for i in range(n):
        u = i / (n - 1)
        pts.append((0.35 + 0.45 * u, 0.50 + 0.15 * math.sin(2 * math.pi * u + phase)))
    return pts


def payload(n, duration_ms, phase):
    times = [round(duration_ms * i / (n - 1)) for i in range(n)]
    points = path_points(n, phase)
    if n <= 15:
        return TimedWaypoints(tuple(points), tuple(times)).payload()
    # TimedWaypoints caps at 15 points; same payload shape, built directly.
    q = [(int(x * 414), int(y * 896)) for x, y in points]
    if any(not segment_is_safe((a[0] / 414, a[1] / 896), (b[0] / 414, b[1] / 896)) for a, b in zip(q, q[1:])):
        raise ValueError('probe path intersects protected controls')
    moves = [dict(type='pointerMove', duration=b - a, x=p[0], y=p[1], origin='viewport')
             for p, a, b in zip(q[1:], times, times[1:])]
    return {'actions': [dict(type='pointer', id='timed_path', parameters={'pointerType': 'touch'}, actions=[
        dict(type='pointerMove', duration=0, x=q[0][0], y=q[0][1], origin='viewport'),
        dict(type='pointerDown', button=0), *moves, dict(type='pointerUp', button=0)])]}


def plan(reps, seed):
    rng = random.Random(seed)
    items = [dict(waypoints=n, duration_ms=DURATIONS_MS[r % 3], phase=round(rng.uniform(0, 2 * math.pi), 6))
             for n in COUNTS for r in range(reps)]
    rng.shuffle(items)
    for i, item in enumerate(items):
        item.update(index=i, payload=payload(item['waypoints'], item['duration_ms'], item['phase']))
    return items


def summarize(items, records):
    by = {}
    for item, r in zip(items, records):
        b = {k: v['monotonic_s'] for k, v in r.items() if isinstance(v, dict) and 'monotonic_s' in v}
        prep = b['preparation_finished'] - b['preparation_started']
        ios = b['ios_completion_callback'] - b['submitted_to_ios'] - item['duration_ms'] / 1000
        by.setdefault(item['waypoints'], []).append((prep, ios, r.get('outcome')))
    return {str(n): dict(n=len(v), prep_median_s=statistics.median(p for p, _, _ in v), prep_max_s=max(p for p, _, _ in v),
                         ios_minus_duration_median_s=statistics.median(i for _, i, _ in v),
                         outcomes=sorted({o for _, _, o in v}))
            for n, v in sorted(by.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True)
    p.add_argument('--wda-revision', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--seed', type=int, default=20261003)
    a = p.parse_args()
    if 'training-server' not in socket.gethostname():
        p.error('run on the rig')
    items = plan(a.reps, a.seed)
    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture, _http_json
    cfg = next(d for d in DEVICES if d['name'] == a.device)
    base = f"http://127.0.0.1:{cfg['wda_port']}"
    # Same strict preflight as the audit runner: never let connect activate the app.
    if _http_json(base + '/wda/activeAppInfo')['value']['bundleId'] != BUNDLE_ID:
        raise RuntimeError('True Skate must already be frontmost')
    worker = DeviceSession(cfg)
    worker.connect()
    timing = WDAActionTimingCapture(wda_port=cfg['wda_port'], expected_revision=a.wda_revision)
    calls, report, error = [], None, None
    try:
        timing.start()
        for item in items:
            start = time.monotonic()
            worker.driver.execute('actions', item['payload'])
            calls.append(dict(index=item['index'], client_s=time.monotonic() - start))
            time.sleep(GAP_S)
        report = timing.stop()
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
        if timing.active:
            report = timing.stop()
    finally:
        timing.cleanup()
        worker.disconnect()
    records = (report or {}).get('records', [])
    result = dict(device=a.device, wda_revision=a.wda_revision, seed=a.seed, reps=a.reps, error=error,
                  items=[{k: v for k, v in i.items() if k != 'payload'} for i in items], calls=calls,
                  report=report, summary=summarize(items, records) if len(records) == len(items) else None)
    save_new(a.out, result)
    print(json.dumps(dict(error=error, summary=result['summary']), indent=1))
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
