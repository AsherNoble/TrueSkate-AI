"""Replay timed gestures extracted from an expert demo clip on one XR; raw recordings only.

One touch contact per request (GESTURES.md): each gesture is its own
POST /wda/perform_trick_gestures call, fired from Python at its scheduled start.
Bundling several contacts into one request makes True Skate join them into a
single chain (May 2026, and again in the 2026-10-03 bundled replay).
--mode separate: one HTTP call per gesture, fired from Python. Calls block until
the gesture finishes plus ~0.3 s, so later gestures start late.
--mode scheduled: one POST /wda/perform_gesture_schedule; WDA submits each
gesture using the explicitly selected --schedule-mode. The default paths mode
uses a single indexed-path record; records mode uses separate records.
Host submission times and raw device responses are retained. Paths-mode offsets
are intended schedule times, not measured physical delivery.
Variants resample each gesture so points are at least N ms apart.
No training admission; output goes to a new directory.
"""
import argparse, json, math, random, signal, socket, subprocess, time
from dataclasses import asdict
from urllib.request import Request, urlopen
from pathlib import Path
from trueskate_ai.data.control_hitboxes import segment_is_safe
from trueskate_ai.research.curved_audit import reset_payload, save_new, digest

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
    if not gestures or any(len(rows) < 2 for rows in gestures.values()):
        raise ValueError('each gesture needs at least two samples')
    for name, rows in gestures.items():
        if any(len(r) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in r)
               or not 0 <= r[1] <= 1 or not 0 <= r[2] <= 1 for r in rows):
            raise ValueError(f'{name}: invalid normalized gesture sample')
        if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
            raise ValueError(f'{name}: non-increasing source timestamps')
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


def schedule_payload(requests, mode, anchor):
    if mode not in ('records', 'paths') or not requests:
        raise ValueError('explicit valid schedule mode and requests required')
    if anchor is not None:
        if mode != 'paths' or len(anchor) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in anchor):
            raise ValueError('valid normalized anchor requires paths mode')
        if not segment_is_safe(anchor, anchor):
            raise ValueError('anchor touches a protected control')
    lead = 100 if anchor is not None else 0
    gs = [dict(start_ms=round(start_s * 1000) + lead, waypoints=payload['gestures'][0]['waypoints'])
          for _, start_s, payload in requests]
    if anchor is not None:
        ax, ay = round(anchor[0] * W), round(anchor[1] * H)
        point = (ax / W, ay / H)
        if not segment_is_safe(point, point):raise ValueError('quantized anchor touches a protected control')
        end_ms = max(g['start_ms'] + sum(w['duration_ms'] for w in g['waypoints']) for g in gs) + 100
        gs.append(dict(start_ms=0, waypoints=[dict(x=ax, y=ay, duration_ms=0),
                                            dict(x=ax, y=ay, duration_ms=end_ms)]))
    for g in gs:
        if type(g['start_ms']) is not int or not 0 <= g['start_ms'] <= 60000:
            raise ValueError('replay exceeds one-minute recording budget')
        if len(g['waypoints']) < 2 or any(type(w[k]) is not int for w in g['waypoints'] for k in ('x','y','duration_ms')):
            raise ValueError('integer waypoints required')
        if g['waypoints'][0]['duration_ms'] != 0 or any(not 0 < w['duration_ms'] <= 60000 for w in g['waypoints'][1:]):
            raise ValueError('invalid waypoint duration')
    last_ms = max(g['start_ms'] + sum(w['duration_ms'] for w in g['waypoints']) for g in gs)
    if 1 + last_ms / 1000 + 3 > 59:
        raise ValueError('replay exceeds one-minute recording budget')
    return dict(mode=mode, gestures=gs)


def replay_trial(*, recorder, requests, mode, schedule_mode, anchor, guard, post, out,
                 metadata, clock=time.monotonic, sleep=time.sleep):
    """Persist a trial before returning, including exactly one post-start retrieval attempt."""
    if mode not in ('separate', 'scheduled') or anchor is not None and mode != 'scheduled':
        raise ValueError('invalid replay mode/anchor')
    schedule = schedule_payload(requests, schedule_mode, anchor)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    save_new(out / 'planned.json', dict(**metadata, mode=mode, schedule_mode=schedule_mode,
             requests=requests, schedule=schedule, training_admission=False))
    log = dict(**metadata, file='original.mov', calls=[], video=None, schedule=schedule,
               schedule_response=None, error=None, failures=[], training_admission=False)
    started = False
    def fail(phase, exc):
        detail = f'{type(exc).__name__}: {exc}'
        log['failures'].append(dict(phase=phase, error=detail))
        log['error'] = log['error'] or detail
    try:
        guard(); origin = clock(); recorder.start(); started = True
        deadline = origin + 59  # Include recorder start latency in the budget.
        def remaining():
            value = deadline - clock()
            if value <= 0: raise RuntimeError('one-minute replay deadline exceeded')
            return value
        def wait(seconds):
            target = clock() + seconds
            if target > deadline: raise RuntimeError('replay wait exceeds recording deadline')
            while clock() < target:
                sleep(min(.25, target - clock()))
            remaining()
        wait(1); t0 = clock()
        if mode == 'separate':
            for name, start_s, payload in requests:
                wait(max(0., t0 + start_s - clock()))
                guard(); timeout = min(10., remaining())
                call = dict(gesture=name, intended_start_s=start_s, start_s=clock()-t0,
                            payload=payload, payload_sha256=digest(payload), success=False)
                log['calls'].append(call)
                response = post('/wda/perform_trick_gestures', payload, timeout_s=timeout)
                call.update(response=response, end_s=clock()-t0)
                value = response.get('value')
                if response.get('error') or isinstance(value, dict) and value.get('error'):
                    raise RuntimeError('gesture response reports error')
                call['success'] = True; remaining()
        else:
            guard(); timeout = min(10., remaining())
            call = dict(payload=schedule, payload_sha256=digest(schedule), start_s=clock()-t0, success=False)
            log['calls'].append(call)
            response = post('/wda/perform_gesture_schedule', schedule, timeout_s=timeout)
            log['schedule_response'] = response
            call.update(response=response, end_s=clock()-t0)
            report = response.get('value')
            if (response.get('error') or not isinstance(report, dict) or report.get('complete') is not True
                    or report.get('error') or any(g.get('error') for g in report.get('gestures', []))):
                raise RuntimeError(f'schedule incomplete: {report}')
            if schedule_mode == 'records':
                if len(report.get('gestures', [])) != len(requests):
                    raise RuntimeError('schedule receipt count mismatch')
                previous = -1.
                for receipt in report['gestures']:
                    begin, end = receipt['submitted_s'], receipt['completed_s']
                    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (begin,end))
                            or not previous <= begin <= end):
                        raise RuntimeError('invalid or unordered schedule receipt times')
                    previous = begin
                log['gesture_timings'] = [dict(gesture=name, intended_start_s=start_s,
                    submitted_s=g['submitted_s'], completed_s=g['completed_s'])
                    for (name, start_s, _), g in zip(requests, report['gestures'])]
            else:
                completed = report.get('completed_s')
                if type(completed) not in (int,float) or not math.isfinite(completed) or completed < 0:
                    raise RuntimeError('invalid schedule completion receipt')
                log['gesture_timings'] = [dict(gesture=name, intended_start_s=start_s,
                    scheduled_start_s=start_s + (.1 if anchor is not None else 0),
                    record_completed_s=report['completed_s'], delivery_measured=False)
                    for name, start_s, _ in requests]
            call['success'] = True; remaining()
        wait(3); guard(); remaining()
    except BaseException as exc:
        fail('execution', exc)
    finally:
        if started:
            try:
                result = recorder.stop_and_save(out / 'original.mov')
                log['video'] = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(result).items()}
            except BaseException as exc:
                fail('recorder_stop', exc)
        # Never abort/retry after failed retrieval. This log exists before disconnect.
        save_new(out / 'execution.json', log)
    return log


def request_json(base, path, payload, *, timeout_s):
    request = Request(base + path, data=json.dumps(payload, allow_nan=False).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout_s) as response:
        value = json.load(response)
    if not isinstance(value, dict): raise ValueError('unexpected WDA response')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gestures', type=Path, required=True)
    p.add_argument('--device', default='iPhone_XR2')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261003)
    p.add_argument('--mode', choices=('separate', 'scheduled'), default='scheduled')
    p.add_argument('--variants', default=','.join(SPACINGS), help='comma-separated subset of ' + ','.join(SPACINGS))
    p.add_argument('--anchor', action='append', default=[], metavar='X,Y',
                   help='paths mode: also hold a still finger at normalised X,Y from 100 ms before the first gesture '
                        'to 100 ms after the last (repeatable: each value is its own condition)')
    p.add_argument('--schedule-mode', choices=('records', 'paths'), default='paths',
                   help="scheduled only: one record per gesture, or one record with indexed paths")
    a = p.parse_args()
    raw = json.loads(a.gestures.read_text())
    gestures = {name: [(r['t'], r['x'], r['y']) for r in rows] for name, rows in raw.items()}
    variants = a.variants.split(',')
    if a.repeats < 1 or not variants or len(set(variants)) != len(variants) or any(v not in SPACINGS for v in variants):
        p.error('positive repeats and unique known variants required')
    anchors = [None] if not a.anchor else [tuple(float(v) for v in x.split(',')) for x in a.anchor]
    if a.anchor and (a.mode != 'scheduled' or a.schedule_mode != 'paths'):
        p.error('--anchor needs --mode scheduled --schedule-mode paths')
    conditions = [(v, anc) for v in variants for anc in anchors]
    plan = [dict(variant=v, anchor=anc, repeat=r) for r in range(a.repeats) for v, anc in conditions]
    rng = random.Random(a.seed)
    for start in range(0, len(plan), len(conditions)):  # shuffle within each repeat block
        block = plan[start:start + len(conditions)]; rng.shuffle(block); plan[start:start + len(conditions)] = block
    requests = {v: gesture_requests(gestures, SPACINGS[v]) for v in variants}
    for v, anc in conditions:
        schedule_payload(requests[v], a.schedule_mode, anc)
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
    tunnel = subprocess.check_output(['launchctl', 'print', 'system/com.trueskate.remotexpc-tunnel'], text=True)
    if 'state = running' not in tunnel: raise RuntimeError('root recording tunnel unavailable')
    if _http_json(base + '/wda/activeAppInfo')['value']['bundleId'] != BUNDLE_ID:
        raise RuntimeError('True Skate must already be frontmost')
    if _http_json(base + '/wda/video').get('value') is not None:
        raise RuntimeError('recorder not idle')
    worker = DeviceSession(cfg); log = []
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        worker.connect(); driver = worker.driver
        def guard():
            if driver.query_app_state(BUNDLE_ID) != 4 or worker._active_bundle_id() != BUNDLE_ID:
                raise RuntimeError('foreground lost/unavailable')
        for k, item in enumerate(plan, 1):
            guard(); driver.execute('actions', reset_payload()); time.sleep(3); guard()
            result = replay_trial(recorder=XCTestScreenRecorder(driver, fps=30), requests=requests[item['variant']],
                 mode=a.mode, schedule_mode=a.schedule_mode, anchor=item['anchor'], guard=guard,
                 post=lambda path, payload, **kw: request_json(base, path, payload, **kw),
                 out=a.out / f'trial_{k:02d}', metadata=dict(**item, device=a.device, park_source='operator task'))
            log.append(result)
            if result['error']: raise RuntimeError(result['error'])
            print(f'{k}/{len(plan)} recording retained', flush=True)
    except BaseException as exc:
        save_new(a.out / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        try:
            save_new(a.out / 'runs.json', dict(runs=log, training_admission=False))
        finally:
            try:
                worker.disconnect()
            except BaseException as exc:
                save_new(a.out / 'disconnect-failure.json', dict(error=f'{type(exc).__name__}: {exc}'))
                raise


if __name__ == '__main__':
    main()
