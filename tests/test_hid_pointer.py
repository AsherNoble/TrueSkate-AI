import math

import pytest

from trueskate_ai.control import hid_pointer as hp

# (counts per report, measured points per report) from the 2026-10-04 XR2 drags
MEASURED = [(1, 0.203), (2, 0.406), (3, 0.900), (4, 1.894), (5, 2.892), (8, 5.900), (16, 13.906),
            (22, 19.857), (27, 26.875), (32, 39.000), (45, 70.750), (64, 117.25), (72, 136.25),
            (90, 153.5), (127, 189.5)]


def test_curve_matches_measurements():
    for m, d in MEASURED:
        assert hp.gain_distance(m) == pytest.approx(d, abs=0.45)


def test_curve_is_increasing_and_inverts():
    ms = [3 + k * 0.25 for k in range(int((127 - 3) / 0.25) + 1)]
    ds = [hp.gain_distance(m) for m in ms]
    assert all(b > a for a, b in zip(ds, ds[1:]))
    for m in (3.0, 10.5, 25.77, 40.0, 71.72, 100.0, 127.0):
        assert hp.counts_length(hp.gain_distance(m)) == pytest.approx(m, abs=1e-6)


def test_displacement_follows_the_count_vector():
    px, py = hp.displacement(-3, 4)                      # length 5: measured (-1.75, 2.31) per report
    assert (px, py) == pytest.approx((-1.735, 2.314), abs=0.05)
    assert hp.displacement(0, 0) == (0.0, 0.0)


def test_best_report_lands_close():
    worst = 0.0
    for k in range(400):
        ang = k * 0.7
        dist = 0.5 + (k % 40) * 4.7                      # 0.5 .. 183.8 pt
        vx, vy = dist * math.cos(ang), dist * math.sin(ang)
        cx, cy = hp.best_report(vx, vy)
        assert max(abs(cx), abs(cy)) <= hp.MAX_COUNT
        px, py = hp.displacement(cx, cy)
        worst = max(worst, math.hypot(px - vx, py - vy))
    assert worst < 1.5                                   # integer counts; steepest part is 2.45 pt/count


def test_follow_does_not_accumulate_error():
    targets = [(200 + 80 * math.cos(2 * math.pi * k / 90), 500 + 80 * math.sin(2 * math.pi * k / 90))
               for k in range(1, 271)]                   # three laps
    reports, positions = hp.follow((280.0, 500.0), targets)
    errors = [math.hypot(px - tx, py - ty) for (px, py), (tx, ty) in zip(positions, targets)]
    assert max(errors) < 1.0
    assert len(reports) == len(targets)


def _strokes():
    flick = hp.Stroke('flick', [(2.30, 206.0, 622.0), (2.3167, 248.0, 667.0), (2.3333, 302.0, 708.0), (2.35, 330.0, 730.0)])
    push = hp.Stroke('push', [(0.485, 314.0, 231.0), (0.5, 304.0, 257.0), (0.5167, 294.0, 285.0), (0.60, 280.0, 400.0)])
    late = hp.Stroke('late', [(2.468, 279.0, 484.0), (2.485, 299.0, 444.0), (2.518, 336.0, 383.0)])
    return [flick, push, late]


@pytest.mark.parametrize('lift', hp.LIFT_MODES)
def test_schedule_shape(lift):
    events, info = hp.build_schedule(_strokes(), lift=lift)
    times = [e[0] for e in events]
    assert times == sorted(times) and len(set(times)) == len(times)
    lifts_at = {s['release_us'] for s in info['strokes']}
    assert all(t % hp.STEP_US == 0 for t in times if t not in lifts_at)
    assert all(max(abs(dx), abs(dy)) <= hp.MAX_COUNT for _, dx, dy, _ in events)
    buttons = [b for *_, b in events]
    presses = sum(1 for a, b in zip([0] + buttons, buttons) if a == 0 and b == 1)
    lifts = sum(1 for a, b in zip(buttons, buttons[1:]) if a == 1 and b == 0)
    assert presses == 3 and lifts == 3 and buttons[-1] == 0
    # strokes come out in time order; press and lift reports never move the cursor
    names = [s['name'] for s in info['strokes']]
    assert names == ['push', 'flick', 'late']
    by_time = {t: (dx, dy, b) for t, dx, dy, b in events}
    for s in info['strokes']:
        assert by_time[s['press_slot'] * hp.STEP_US] == (0, 0, 1)
        assert by_time[s['release_us']] == (0, 0, 0)
        assert by_time[s['last_move_slot'] * hp.STEP_US][2] == 1
        gap = s['release_us'] - s['last_move_slot'] * hp.STEP_US
        assert gap == (hp.LIFT_DELAY_US if lift == 'immediate' else hp.STEP_US)
        assert math.dist(s['modelled_end'], s['target_end']) < 1.5


def test_unknown_lift_mode_is_refused():
    with pytest.raises(ValueError, match='lift must be one of'):
        hp.build_schedule(_strokes(), lift='with-move')


def test_schedule_keeps_gaps_between_presses():
    events, info = hp.build_schedule(_strokes())
    slots = {s['name']: s['press_slot'] for s in info['strokes']}
    step = hp.STEP_US / 1e6
    for a, b, gap in (('push', 'flick', 2.30 - 0.485), ('flick', 'late', 2.468 - 2.30)):
        assert abs((slots[b] - slots[a]) * step - gap) <= step / 2 + 1e-9


def test_schedule_refuses_strokes_too_close_to_hover_between():
    a = hp.Stroke('a', [(0.0, 50.0, 300.0), (0.03, 60.0, 300.0)])
    b = hp.Stroke('b', [(0.06, 400.0, 800.0), (0.09, 390.0, 800.0)])  # 0 free slots, ~560 pt away
    with pytest.raises(ValueError, match='slots to reach the start'):
        hp.build_schedule([a, b])
