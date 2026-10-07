from pathlib import Path
import struct
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hardware/pico_hover_rate'))
import measure  # noqa: E402
import read_stats  # noqa: E402
import schedule as sched  # noqa: E402
from trueskate_ai.control import hid_pointer as hp  # noqa: E402


def test_schedule_is_ordered_bounded_and_fits_the_stats_sector():
    events, passes, presses, _ = sched.build()
    times = [e[0] for e in events]
    assert times == sorted(times)
    assert all(-127 <= dx <= 127 and -127 <= dy <= 127 and b in (0, 1) for _, dx, dy, b in events)
    assert events[-1][1:] == (0, 0, 0)                        # ends neutral
    assert events[-1][0] < 40e6                               # fits the recording after lead and mount
    assert 32 + 2 * len(events) <= 4096
    assert len(passes) == 6 + 10 + 10
    assert {(p['step_us'], p['reports']) for p in passes if p['mode'] == 'usb1ms'} == {(1000, 60)}


def test_hover_passes_never_press_and_return_to_the_park_column():
    events, passes, _, _ = sched.build()
    for p in passes:
        inside = [e for e in events if p['start_us'] <= e[0] < p['end_us']]
        assert len(inside) == p['reports'] and all(b == 0 and dy == 0 for _, _, dy, b in inside)
    for mode in ('rates', 'stall', 'usb1ms'):
        assert sum(p['dx'] * p['reports'] for p in passes if p['mode'] == mode) == 0
    first_press = min(e[0] for e in events if e[3])
    assert first_press > max(p['end_us'] for p in passes)


def test_presses_stay_on_clear_floor_and_lift_as_their_own_report():
    events, _, presses, sections = sched.build()
    repark_end = next(s['end_us'] for s in sections if s['kind'] == 'repark')
    tail = [e for e in events if e[0] >= repark_end]
    for _, x, y, _ in sched.modelled_path(tail):
        assert sched.SAFE_X[0] <= x <= sched.SAFE_X[1] and sched.SAFE_Y[0] <= y <= sched.SAFE_Y[1]
    for p in presses:
        lift = next(e for e in events if e[0] == p['lift_us'])
        assert lift[1:] == (0, 0, 0)
        held = [e for e in events if p['start_us'] <= e[0] < p['lift_us']]
        assert held and all(b == 1 for *_, b in held)
        if p['kind'] != 'tap':
            assert p['lift_us'] - held[-1][0] == sched.LIFT_DELAY_US
            assert sum(1 for e in held if e[1:3] != (0, 0)) == p['reports']


def test_park_matches_the_bluetooth_planner():
    events, _, _, sections = sched.build()
    park = [e for e in events if sections[0]['start_us'] <= e[0] < sections[0]['end_us']]
    assert park[:hp.HOME_REPORTS] == [(i * sched.STEP_US, -127, -127, 0) for i in range(hp.HOME_REPORTS)]
    assert [e[1:3] for e in park[hp.HOME_REPORTS:]] == [hp.PARK_STEP] * hp.PARK_REPORTS


def test_header_carries_every_event_and_the_hash():
    events, *_ = sched.build()
    digest = sched.schedule_hash(events)
    text = sched.header(events, digest)
    assert f'kEventCount = {len(events)};' in text and f'0x{digest[:8]}u' in text
    assert text.count('u, ') == len(events)


def blob(late_us, status=2, sent=None, digest=None):
    """Stats sector as the firmware writes it; None marks an unsent event."""
    events, *_ = sched.build()
    digest = digest or sched.schedule_hash(events)
    head = read_stats.HEADER.pack(b'PHR1', 0x1234, int(digest[:8], 16), status, len(events),
                                  len(events) if sent is None else sent, 0, 1500, 7, 900)
    slots = [read_stats.UNSENT if v is None else round(v / read_stats.LATE_UNIT_US) for v in late_us]
    return head + struct.pack(f'<{len(slots)}H', *slots), dict(schedule_hash=digest, n_events=len(events))


def test_stats_summarise_lateness_and_host_poll_interval():
    events, passes, _, _ = sched.build()
    late = [0] * len(events)
    t = np.array([e[0] for e in events])
    one_ms = [p for p in passes if p['step_us'] == 1000]
    for p in one_ms:                                          # a host polling every 3 ms
        idx = np.nonzero((t >= p['start_us']) & (t < p['end_us']))[0]
        for k, i in enumerate(idx):
            late[i] = k * 2000
    data, schedule = blob(late)
    stats = read_stats.parse(data, schedule)
    assert stats['status'] == 'done' and stats['not_ready_waits'] == 7
    summary = read_stats.summarise(stats)
    assert abs(summary['by_step']['1ms']['send_interval_us']['p50'] - 3000) <= read_stats.LATE_UNIT_US
    assert summary['by_step']['15ms']['lateness_us']['max'] == 0
    longest = max(p['reports'] for p in one_ms)
    assert abs(summary['by_step']['1ms']['lateness_us']['max'] - (longest - 1) * 2000) <= read_stats.LATE_UNIT_US
    assert (longest - 1) * 2000 > 65_535                       # beyond a microsecond uint16


def test_stats_reject_other_schedules_and_report_unsent_events():
    events, *_ = sched.build()
    data, schedule = blob([0] * len(events), digest='f' * 16)
    with pytest.raises(ValueError, match='different schedule'):
        read_stats.parse(data, dict(schedule, schedule_hash='0' * 16))
    late = [0] * 10 + [None] * (len(events) - 10)
    data, schedule = blob(late, status=4, sent=10)
    stats = read_stats.parse(data, schedule)
    assert stats['status'] == 'host lost' and stats['events_sent'] == 10
    assert read_stats.summarise(stats)['by_step'] == {}       # no pass was fully sent


def test_measure_finds_row_anchor_and_windows():
    t = np.arange(600) / 60
    x = np.full(600, 109.0)
    x[100:118] = 109 + np.arange(18) * 10.0                   # an 18-frame pass from t = 100/60 s
    x[118:] = x[117]
    y = np.where(np.arange(600) % 7 == 0, 500.0, 368.0)
    assert measure.park_row(y) == pytest.approx(367.5)
    t0 = measure.anchor_time(t, x, 1.9)
    assert t0 == pytest.approx(t[100])
    seg = measure.window_segment(t, x, 1.9, t0, t0 + 0.3)
    assert seg[0] == 100 and seg[-1] == 116
    assert len(measure.window_segment(t, x, 1.9, t0 + 3, t0 + 3.3)) == 0
