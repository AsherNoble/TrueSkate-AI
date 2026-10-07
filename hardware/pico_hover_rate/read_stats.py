"""Board-side timing from a pico_hover_rate run.

Read the stats sector with the Pico in BOOTSEL (read-only):
  picotool save -r <EEPROM_START> <EEPROM_START+0x1000> stats.bin
then: PYTHONPATH=<repo>/src read_stats.py stats.bin schedule.json [out.json]

Lateness (stored in 16 us units, up to ~1 s) is the time from an event's schedule slot
to the moment TinyUSB accepted its report. A report is accepted only once the host has collected the previous one, so at
1 ms spacing the send-to-send interval shows how often the host actually polls.
"""
from __future__ import annotations

import json
import struct
import sys

import numpy as np

from schedule import build

HEADER = struct.Struct('<4sIIHHHHIII')
STATUS = {1: 'armed (did not finish)', 2: 'done', 3: 'no mount', 4: 'host lost', 5: 'send timeout'}
UNSENT = 0xFFFF
LATE_UNIT_US = 16


def parse(blob: bytes, schedule: dict) -> dict:
    magic, build_id, sched_hash, status, count, sent, clipped, mount_ms, waits, max_wait = HEADER.unpack_from(blob)
    if magic != b'PHR1':
        raise ValueError('No probe stats in this sector')
    if f'{sched_hash:08x}' != schedule['schedule_hash'][:8]:
        raise ValueError('Stats come from a different schedule')
    if count != schedule['n_events']:
        raise ValueError('Event count disagrees with the schedule')
    raw = np.frombuffer(blob, '<u2', count, HEADER.size).astype(int)
    late = np.where(raw == UNSENT, -1, raw * LATE_UNIT_US)        # -1: never sent
    return dict(build_id=f'{build_id:08x}', status=STATUS.get(status, f'unknown {status}'), events=count,
                events_sent=sent, clipped=clipped, mount_ms=mount_ms, not_ready_waits=waits, max_wait_us=max_wait,
                lateness_us=late.tolist())


def summarise(stats: dict) -> dict:
    events, passes, _, _ = build()
    late = np.array(stats['lateness_us'])
    sent = late >= 0
    t = np.array([e[0] for e in events])
    out = dict(status=stats['status'], events_sent=stats['events_sent'], events=stats['events'], clipped=stats['clipped'],
               not_ready_waits=stats['not_ready_waits'], max_wait_us=stats['max_wait_us'], by_step={})
    for p in passes:
        idx = np.nonzero((t >= p['start_us']) & (t < p['end_us']))[0]
        if not sent[idx].all():
            continue
        actual = t[idx] + late[idx]
        g = out['by_step'].setdefault(f"{p['step_us'] / 1000:g}ms", dict(reports=0, lateness=[], intervals=[]))
        g['reports'] += len(idx)
        g['lateness'] += late[idx].tolist()
        g['intervals'] += np.diff(actual).tolist()
    for g in out['by_step'].values():
        lat, gaps = np.array(g.pop('lateness')), np.array(g.pop('intervals'))
        g['lateness_us'] = {q: int(np.percentile(lat, p)) for q, p in (('p50', 50), ('p95', 95), ('p99', 99), ('max', 100))}
        g['send_interval_us'] = {q: int(np.percentile(gaps, p)) for q, p in (('p5', 5), ('p50', 50), ('p95', 95))}
    return out


def main(argv):
    blob = open(argv[0], 'rb').read()
    schedule = json.load(open(argv[1]))
    stats = parse(blob, schedule)
    summary = summarise(stats)
    print(json.dumps(summary, indent=1))
    if len(argv) > 2:
        json.dump(dict(stats, summary=summary), open(argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
