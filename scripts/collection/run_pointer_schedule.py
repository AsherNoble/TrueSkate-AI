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
import argparse, json, socket, sys, time, urllib.request
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


def board_ready(client) -> str:
    client.ask('PARAMS 12 12 0 400', 2.0)   # 15 ms interval, no peripheral latency
    status = next((l for l in client.ask('STATUS', 0.5) if l.startswith('STATUS')), '')
    if not all(k in status for k in ('connected=1', 'auth=1', 'subscribed=1', 'interval_us=15000')):
        raise RuntimeError(f'pointer not ready: {status!r}')
    return status


def upload(client, events) -> None:
    client.ask('CLEAR', 0.1)
    client.send('\n'.join(f'E {t} {dx} {dy} {b}' for t, dx, dy, b in events))
    oks, end = 0, time.time() + 10
    while oks < len(events) and time.time() < end:
        lines = client.read_lines(0.05)
        if any(l.startswith('ERR') for l in lines):
            raise RuntimeError(f'board rejected the schedule: {lines}')
        oks += sum(1 for l in lines if l == 'OK')
    if oks < len(events):
        raise RuntimeError(f'board acknowledged {oks}/{len(events)} events')


def play(client, events) -> dict:
    go = time.time()
    client.send('GO')
    lines = client.wait_for('DONE', timeout=events[-1][0] / 1e6 + 10)
    sent = [int(l.split()[2]) for l in lines if l.startswith('SENT')]
    if len(sent) != len(events):
        raise RuntimeError(f'board reported {len(sent)}/{len(events)} events sent')
    late = [s - e[0] for s, e in zip(sent, events)]
    return dict(go_epoch=go, max_late_us=max(late), min_late_us=min(late))


def save_new(path: Path, value) -> None:
    with open(path, 'x') as f:
        json.dump(value, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--gestures', type=Path, help='extracted-gestures JSON (normalised x, y)')
    src.add_argument('--strokes', type=Path, help='strokes JSON in points: {name: [[t_s, x, y], ...]}')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='iPhone_XR2')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--lift', choices=hp.LIFT_MODES, default='immediate',
                   help='lift report 1 ms after the last move, or on the next 15 ms slot')
    p.add_argument('--lift-for', action='append', default=[], metavar='STROKE=MODE',
                   help='per-stroke lift mode override (repeatable)')
    p.add_argument('--bridge-port', type=int, default=8765)
    p.add_argument('--fps', type=int, default=30, help='XCTest recording frame rate')
    a = p.parse_args()
    if 'training-server' not in socket.gethostname():
        p.error('run on the rig')
    if a.out.exists():
        raise SystemExit('output directory must be new')
    strokes = load_strokes(a.gestures or a.strokes, normalised=a.gestures is not None)
    for item in a.lift_for:
        name, mode = item.split('=')
        next(s for s in strokes if s.name == name).lift = mode
    events, info = hp.build_schedule(strokes, lift=a.lift)

    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    cfg = next(d for d in DEVICES if d['name'] == a.device)
    base = f"http://127.0.0.1:{cfg['wda_port']}"
    if http_json(base + '/wda/activeAppInfo')['value']['bundleId'] != BUNDLE_ID:
        raise RuntimeError('True Skate must already be frontmost')
    if http_json(base + '/wda/video').get('value') is not None:
        raise RuntimeError('recorder not idle')
    client = PointerClient(port=a.bridge_port)
    status = board_ready(client)
    a.out.mkdir(parents=True)
    save_new(a.out / 'plan.json', dict(source=str(a.gestures or a.strokes), lift=a.lift, board_status=status,
                                       strokes={s.name: s.samples for s in strokes}, schedule_info=info,
                                       events=events, training_admission=False))
    worker = DeviceSession(cfg); worker.connect(); driver = worker.driver
    runs = []
    try:
        for k in range(1, a.repeats + 1):
            if driver.query_app_state(BUNDLE_ID) != 4:
                raise RuntimeError('foreground lost')
            upload(client, events)
            tap = {'gestures': [{'waypoints': [dict(x=RESET_POINT[0], y=RESET_POINT[1], duration_ms=0),
                                               dict(x=RESET_POINT[0], y=RESET_POINT[1], duration_ms=50)]}]}
            http_json(base + '/wda/perform_trick_gestures', tap); time.sleep(3)
            recorder = XCTestScreenRecorder(driver, fps=a.fps); recorder.start(); time.sleep(1)
            try:
                timing = play(client, events)
                time.sleep(1.5)
            except BaseException:
                recorder.abort(); raise
            name = f'{k:02d}_pointer_{a.lift}_r{k}.mov'
            res = recorder.stop_and_save(a.out / name)
            runs.append(dict(repeat=k, file=name, recording_started_at=res.started_at_epoch_s, **timing))
            print(f"{k}/{a.repeats} {name} board lateness {timing['min_late_us']}..{timing['max_late_us']} us", flush=True)
    finally:
        client.close(); worker.disconnect()
        save_new(a.out / 'runs.json', dict(runs=runs))


if __name__ == '__main__':
    main()
