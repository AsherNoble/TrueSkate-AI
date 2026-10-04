"""Play strokes through the ESP32 Bluetooth pointer on one XR and record the screen.

Strokes come from an extracted-gestures file (--gestures: normalised x, y in clip
seconds) or a strokes file (--strokes: {name: [[t_s, x_pt, y_pt], ...]}). The whole
schedule (home, park, hover moves, presses, moves, lifts) is uploaded to the board
and played from its clock on the 15 ms Bluetooth grid; Python only sends GO, so
every gap inside the schedule survives. Raw recordings only, no training
admission; output goes to a new directory.

Needs hid_bridge.py running on the rig with the board bonded to the phone and
AssistiveTouch on. True Skate must already be frontmost.
"""
import argparse, json, math, re, socket, sys, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'hardware' / 'hid_pointer'))
from hid_client import PointerClient  # noqa: E402
from trueskate_ai.control import hid_pointer as hp  # noqa: E402

W, H = 414, 896
RESET_POINT = (207, 49)                     # True Skate's reset-board button


def load_strokes(path: Path, normalised: bool) -> list:
    raw = json.loads(path.read_text())
    if normalised:
        return [hp.Stroke(name, [(r['t'], r['x'] * W, r['y'] * H) for r in rows]) for name, rows in raw.items()]
    return [hp.Stroke(name, [tuple(r[:3]) for r in rows]) for name, rows in raw.items()]


def http_json(url, body=None, timeout=20):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b'{}')


LEAD_S = 1.0
TAIL_S = 1.5
MAX_RECORDING_S = 60.0


def guard_foreground(driver, base, bundle):
    # Both independent guards must be available and agree; never reactivate here.
    state = driver.query_app_state(bundle)
    active = http_json(base + '/wda/activeAppInfo', timeout=2)
    if state != 4 or active.get('value', {}).get('bundleId') != bundle:
        raise RuntimeError(f'foreground unavailable or contradictory: appium={state}, wda={active!r}')


def receipt_budget(events):
    # Buffered SENT lines drain over 115200-baud serial after playback.
    return 1.0 + len(events) * 0.004


def validate_schedule(events, lead_s=LEAD_S, tail_s=TAIL_S):
    if not 1 <= len(events) <= 2048:
        raise ValueError('one to 2048 events required')
    previous = -1
    for event in events:
        if len(event) != 4 or any(type(v) is not int for v in event):
            raise ValueError('integer event values required')
        t, dx, dy, buttons = event
        if not 0 <= t <= 60_000_000 or t <= previous or not -127 <= dx <= 127 or not -127 <= dy <= 127 or buttons not in (0, 1):
            raise ValueError('invalid event values or timestamp order')
        previous = t
    if events[-1][3] != 0:
        raise ValueError('schedule must end with neutral buttons')
    if any(not math.isfinite(v) or v < 0 for v in (lead_s, tail_s)):
        raise ValueError('invalid recording lead or tail')
    budget = lead_s + events[-1][0] / 1e6 + receipt_budget(events) + tail_s
    if budget > MAX_RECORDING_S:
        raise ValueError('recording budget exceeds one minute including lead, receipts and tail')
    return budget


def board_ready(client) -> str:
    if client.ask('HELLO 2', 0.5) != ['HELLO 2']:
        raise RuntimeError('protocol v2 required; older firmware is unsafe')
    client.ask('PARAMS 12 12 0 400', 2.0)
    lines = client.ask('STATUS', 0.5)
    statuses = [line for line in lines if line.startswith('STATUS ')]
    if len(statuses) != 1:
        raise RuntimeError(f'pointer status unavailable: {lines!r}')
    fields = dict(part.split('=', 1) for part in statuses[0].split()[1:])
    if any(fields.get(k) != v for k, v in dict(protocol='2', connected='1', auth='1', subscribed='1', interval_us='15000').items()):
        raise RuntimeError(f'pointer not ready: {statuses[0]!r}')
    if client.ask('NEUTRAL', 0.1) != ['OK']:
        raise RuntimeError('neutral button state unavailable')
    return statuses[0]


def upload(client, events) -> None:
    validate_schedule(events)
    if client.ask('CLEAR', 0.1) != ['OK']:
        raise RuntimeError('schedule clear not acknowledged')
    client.send('\n'.join(f'E {t} {dx} {dy} {b}' for t, dx, dy, b in events))
    oks, end = 0, time.monotonic() + 10
    while oks < len(events) and time.monotonic() < end:
        lines = client.read_lines(min(0.05, end - time.monotonic()))
        PointerClient.check_errors(lines)
        if any(line != 'OK' for line in lines):
            raise RuntimeError(f'unexpected upload receipt: {lines}')
        oks += len(lines)
    if oks != len(events):
        raise RuntimeError(f'board acknowledged {oks}/{len(events)} events')


def validate_receipts(lines, events):
    PointerClient.check_errors(lines)
    sent, completions = [], []
    for line in lines:
        if line.startswith('SENT'):
            match = re.fullmatch(r'SENT (\d+) (\d+)', line)
            if not match or completions:
                raise RuntimeError('malformed or out-of-order event receipt')
            index, actual = map(int, match.groups())
            if index != len(sent) or index >= len(events) or not events[index][0] <= actual <= 60_000_000 or (sent and actual <= sent[-1]):
                raise RuntimeError('unordered, duplicate or invalid event receipt')
            sent.append(actual)
        elif line.startswith('DONE'):
            if line != f'DONE {len(events)}' or len(sent) != len(events) or completions:
                raise RuntimeError('incorrect completion count or incomplete receipts')
            completions.append(line)
        else:
            raise RuntimeError(f'unexpected playback receipt: {line}')
    if len(sent) != len(events) or len(completions) != 1:
        raise RuntimeError('incomplete playback receipts')
    return sent


def play(client, events, deadline=None, clock=time.monotonic) -> dict:
    validate_schedule(events)
    timeout = events[-1][0] / 1e6 + receipt_budget(events)
    if deadline is not None:
        timeout = min(timeout, deadline - clock())
    if timeout <= 0:
        raise RuntimeError('recording deadline exhausted')
    go = time.time()
    client.send('GO')
    lines = client.wait_for('DONE', timeout=timeout)
    sent = validate_receipts(lines, events)
    late = [s - e[0] for s, e in zip(sent, events)]
    return dict(go_epoch=go, max_late_us=max(late), min_late_us=min(late),
                receipt_semantics='notification attempts; not physical delivery', receipts=lines)


def run_once(*, client, events, recorder, guard, reset, out, clock=time.monotonic, sleep=time.sleep):
    """One bounded start and one stop attempt; retain partial video on every failure."""
    validate_schedule(events)
    guard()
    upload(client, events)
    guard()  # immediately before reset
    reset()
    sleep(3)
    guard()
    stop_attempted = False
    started = False
    result = timing = None
    failure = None
    deadline = clock() + MAX_RECORDING_S
    try:
        recorder.start()
        started = True
        sleep(LEAD_S)
        guard()  # recheck both sources after sleep and immediately before GO
        if clock() + events[-1][0] / 1e6 + receipt_budget(events) + TAIL_S > deadline:
            raise RuntimeError('insufficient recording budget after foreground guard')
        timing = play(client, events, deadline=deadline - TAIL_S, clock=clock)
        if clock() + TAIL_S > deadline:
            raise RuntimeError('recording deadline exhausted')
        sleep(TAIL_S)
    except BaseException as exc:
        failure = exc
        try:
            client.cancel()
        except Exception as cancel_error:
            save_new(out.with_suffix('.cancel-error.json'), dict(error=str(cancel_error)))
    finally:
        if started:
            stop_attempted = True  # mark before RPC; never retry a failed stop
            try:
                result = recorder.stop_and_save(out)
            except BaseException as stop_error:
                save_new(out.with_suffix('.retrieval-error.json'), dict(error=str(stop_error), prior_error=str(failure) if failure else None))
                if failure is None:
                    failure = stop_error
        if failure is not None:
            save_new(out.with_suffix('.failure.json'), dict(error=str(failure), started=started, stop_attempted=stop_attempted,
                                                          partial_video=str(out) if out.exists() else None, timing=timing))
    if failure is not None:
        raise failure
    return result, timing


def save_new(path: Path, value) -> None:
    with open(path, 'x') as f:
        json.dump(value, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--gestures', type=Path, help='extracted-gestures JSON (normalised x, y)')
    src.add_argument('--strokes', type=Path, help='strokes JSON in points: {name: [[t_s, x, y], ...]}')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['iPhone_XR2'], default='iPhone_XR2', help='only measured XR2 profile is validated')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--lift', choices=hp.LIFT_MODES, default='immediate',
                   help='lift report 1 ms after the last move, or on the next 15 ms slot')
    p.add_argument('--lift-for', action='append', default=[], metavar='STROKE=MODE',
                   help='per-stroke lift mode override (repeatable)')
    p.add_argument('--bridge-port', type=int, default=8765)
    p.add_argument('--fps', type=int, default=30, help='XCTest recording frame rate')
    a = p.parse_args()
    if a.repeats < 1 or a.fps not in (30, 60):
        p.error('positive repeats and 30/60 fps required')
    if 'training-server' not in socket.gethostname():
        p.error('run on the rig')
    if a.out.exists():
        raise SystemExit('output directory must be new')
    strokes = load_strokes(a.gestures or a.strokes, normalised=a.gestures is not None)
    for item in a.lift_for:
        name, mode = item.split('=')
        next(s for s in strokes if s.name == name).lift = mode
    events, info = hp.build_schedule(strokes, lift=a.lift)
    validate_schedule(events)

    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    cfg = next(d for d in DEVICES if d['name'] == a.device)
    base = f"http://127.0.0.1:{cfg['wda_port']}"
    if http_json(base + '/wda/activeAppInfo')['value']['bundleId'] != BUNDLE_ID:
        raise RuntimeError('True Skate must already be frontmost')
    if http_json(base + '/wda/video').get('value') is not None:
        raise RuntimeError('recorder not idle')
    client = PointerClient(port=a.bridge_port)
    worker = DeviceSession(cfg)
    runs = []
    try:
        status = board_ready(client)
        a.out.mkdir(parents=True)
        save_new(a.out / 'plan.json', dict(source=str(a.gestures or a.strokes), lift=a.lift, board_status=status,
                                          strokes={s.name: s.samples for s in strokes}, schedule_info=info,
                                          events=events, protocol=2, training_admission=False))
        worker.connect()
        driver = worker.driver
        guard = lambda: guard_foreground(driver, base, BUNDLE_ID)
        tap = {'gestures': [{'waypoints': [dict(x=RESET_POINT[0], y=RESET_POINT[1], duration_ms=0),
                                            dict(x=RESET_POINT[0], y=RESET_POINT[1], duration_ms=50)]}]}
        for k in range(1, a.repeats + 1):
            # Explicit neutral acknowledgement also gates each repeated run.
            if client.ask('NEUTRAL', 0.1) != ['OK']:
                raise RuntimeError('neutral button state unavailable')
            name = f'{k:02d}_pointer_{a.lift}_r{k}.mov'
            res, timing = run_once(client=client, events=events, recorder=XCTestScreenRecorder(driver, fps=a.fps),
                                   guard=guard, reset=lambda: http_json(base + '/wda/perform_trick_gestures', tap),
                                   out=a.out / name)
            runs.append(dict(repeat=k, file=name, recording_started_at=res.started_at_epoch_s, **timing))
            print(f"{k}/{a.repeats} {name} notification-attempt receipts complete", flush=True)
    finally:
        client.close()
        worker.disconnect()
        if a.out.exists():
            save_new(a.out / 'runs.json', dict(runs=runs))


if __name__ == '__main__':
    main()
