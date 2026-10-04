"""Measure the AssistiveTouch pointer gain on XR2 (run on the rig, True Skate frontmost).

Every drag is one board-timed schedule: home to the top-left corner, hover to a
parking spot, wait 400 ms, press, hold 200 ms, n equal reports, hold 200 ms, lift.
No host timing enters a drag. Recordings stay under a minute and an Appium call
between drags keeps the session alive. gain_measure.py reads the cursor from the
recording (hovering, before the press and after the lift).

Usage: PYTHONPATH=<repo>/src gain_probe.py <tag> <dx,dy:n:interval_ms:spot[:R]> ...
  spot C parks at ~(109, 368) pt on clear floor; D at ~(348, 315) pt (vertical room).
  R resets the board first (vertical moves push it).
Writes gain_<tag>.mov and gain_<tag>.json in the working directory.
"""
import json, sys, time, urllib.request
from hid_client import PointerClient
from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder

PARK = {'C': (3, 11), 'D': (10, 9)}   # 40 hover reports from the top-left corner
HOLD_US = 200_000
tag = sys.argv[1]
plan = []
for a in sys.argv[2:]:
    step, n, interval, spot, *flags = a.split(':')
    dx, dy = (int(v) for v in step.split(','))
    plan.append(dict(dx=dx, dy=dy, reports=int(n), interval_us=int(interval) * 1000, spot=spot, reset='R' in flags))
c = PointerClient()


def schedule(d):
    ev, t = [], 0
    for _ in range(25):
        ev.append((t, -127, -127, 0)); t += 15000
    t += 300_000
    px, py = PARK[d['spot']]
    for _ in range(40):
        ev.append((t, px, py, 0)); t += 15000
    t += 400_000
    press = t
    ev.append((t, 0, 0, 1)); t += HOLD_US
    for i in range(d['reports']):
        ev.append((t, d['dx'], d['dy'], 1)); t += d['interval_us']
    t += HOLD_US - d['interval_us']
    ev.append((t, 0, 0, 0))
    return press, ev


def reset_board():
    """Tap True Skate's reset button so earlier pushes cannot leave the board elsewhere."""
    body = {'gestures': [{'waypoints': [{'x': 207, 'y': 49, 'duration_ms': 0}, {'x': 207, 'y': 49, 'duration_ms': 50}]}]}
    req = urllib.request.Request('http://127.0.0.1:8103/wda/perform_trick_gestures', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    urllib.request.urlopen(req, timeout=20).read()
    time.sleep(1.5)


def play(ev):
    c.ask("CLEAR", 0.05)
    c.send('\n'.join(f"E {t} {x} {y} {b}" for t, x, y, b in ev))
    oks = 0
    end = time.time() + 5
    while oks < len(ev) and time.time() < end:
        oks += sum(1 for l in c.read_lines(0.05) if l == 'OK')
    assert oks >= len(ev), f'board acknowledged {oks}/{len(ev)} events'
    go = time.time()
    c.send("GO")
    sent = [l for l in c.wait_for("DONE", timeout=15) if l.startswith("SENT")]
    return go, [int(l.split()[2]) for l in sent]


cfg = next(d for d in DEVICES if d['name'] == 'iPhone_XR2')
w = DeviceSession(cfg); w.connect()
print(c.ask("STATUS", 0.4))
marks, rec = [], None
try:
    w.driver.activate_app(BUNDLE_ID); time.sleep(1.5)
    reset_board()
    rec = XCTestScreenRecorder(w.driver, fps=30); rec.start(); time.sleep(1.0)
    for d in plan:
        if d['reset']:
            reset_board()                                    # pushes move the board; put it back
        press_us, ev = schedule(d)
        go, sent = play(ev)
        late = max(s - e[0] for s, e in zip(sent, ev))
        marks.append(dict(d, kind='drag', press_us=press_us, go_epoch=go, n_events=len(ev), max_late_us=late))
        w.driver.query_app_state(BUNDLE_ID)                  # keep the Appium session alive
        time.sleep(0.6)
    res = rec.stop_and_save(f"gain_{tag}.mov")
    out = dict(tag=tag, started_at=res.started_at_epoch_s, fps=res.fps, marks=marks)
    json.dump(out, open(f'gain_{tag}.json', 'w'), indent=1)
    print('saved', res.mov_path, res.n_bytes, 'drags', len(marks), 'duration_s', round(time.time() - res.host_start_epoch_s, 1),
          'max_late_us', max(m['max_late_us'] for m in marks))
finally:
    if rec is not None and rec.is_recording:
        rec.abort()
    c.close(); w.disconnect()
