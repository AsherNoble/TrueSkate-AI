"""How finely does the phone take pointer updates? Hover only (no button, so no touches),
constant small moves at a fixed spacing, recorded at 60 fps. Per-frame cursor steps show
whether updates arrive one by one or in bursts on the 15 ms Bluetooth grid.

Usage: PYTHONPATH=<repo>/src hover_rate_probe.py <tag> [rates|stall]   (True Skate frontmost; nothing is touched)
  rates (default): 4-count reports every 3, 1 and 15 ms, there and back (100, 60, 30 reports)
  stall: 300 reports at the replay rate (one per 15 ms), in 10 passes
Writes hover_rate_<tag>.mov and hover_rate_<tag>.json; hover_rate_measure.py reads them.
"""
import json, sys, time
from hid_client import PointerClient
from trueskate_ai.sim.device import DeviceSession, DEVICES
from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder

tag = sys.argv[1]
ev, marks, t = [], [], 0


def add(dx, dy, step):
    global t
    ev.append((t, dx, dy, 0)); t += step


for _ in range(25):
    add(-127, -127, 15000)
t += 300_000
for _ in range(40):
    add(3, 11, 15000)                       # park at ~(109, 368) pt on clear floor
t += 600_000
MODES = {'rates': [(4, 3000, 100), (-4, 3000, 100), (4, 1000, 60), (-4, 1000, 60), (4, 15000, 30), (-4, 15000, 30)],
         'stall': [(4, 15000, 30), (-4, 15000, 30)] * 5}       # 300 reports at the replay rate (1 per 15 ms)
for dx, step, n in MODES[sys.argv[2] if len(sys.argv) > 2 else 'rates']:
    start = t
    for _ in range(n):
        add(dx, 0, step)
    marks.append(dict(dx=dx, step_us=step, reports=n, start_us=start, end_us=t))
    t += 600_000

c = PointerClient()
cfg = next(d for d in DEVICES if d['name'] == 'iPhone_XR2')
w = DeviceSession(cfg); w.connect()
rec = None
try:
    c.ask('PARAMS 12 12 0 400', 2.0)
    status = next(l for l in c.ask('STATUS', 0.5) if l.startswith('STATUS'))
    assert 'connected=1' in status and 'interval_us=15000' in status, status
    c.ask('CLEAR', 0.1)
    c.send('\n'.join(f'E {a} {b} {d} {e}' for a, b, d, e in ev))
    oks, end = 0, time.time() + 10
    while oks < len(ev) and time.time() < end:
        oks += sum(1 for l in c.read_lines(0.05) if l == 'OK')
    assert oks >= len(ev), f'acked {oks}/{len(ev)}'
    rec = XCTestScreenRecorder(w.driver, fps=60); rec.start(); time.sleep(1.0)
    go = time.time(); c.send('GO')
    lines = c.wait_for('DONE', timeout=t / 1e6 + 10)
    sent = [int(l.split()[2]) for l in lines if l.startswith('SENT')]
    late = [s - e[0] for s, e in zip(sent, ev)]
    time.sleep(1.0)
    res = rec.stop_and_save(f'hover_rate_{tag}.mov')
    json.dump(dict(status=status, go_epoch=go, started_at=res.started_at_epoch_s, passes=marks, n_events=len(ev),
                   sent_us=sent, max_late_us=max(late)), open(f'hover_rate_{tag}.json', 'w'), indent=1)
    print('saved', res.mov_path, 'events', len(ev), 'sent', len(sent), 'max lateness us', max(late))
finally:
    if rec is not None and rec.is_recording:
        rec.abort()
    c.close(); w.disconnect()
