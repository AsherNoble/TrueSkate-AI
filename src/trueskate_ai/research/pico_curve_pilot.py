"""Preloaded XR2 USB curve diagnostic. No corpus admission or BLE planner changes.

All schedule times use the Pico play clock, after its eight-second mount lead.
Relative-report predictions are separate from intended normalized trajectories.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from bisect import bisect_right
from pathlib import Path

if Path(__file__).with_name('pico_curve_hid_reference.py').is_file():
    # The isolated rig stage carries exact reference bytes; its deployed checkout
    # need not have the newly developed HID planner or control geometry.
    from pico_curve_hid_reference import displacement as ble_displacement
    from pico_curve_controls_reference import CONTROL_START_MAP_VERSION, segment_is_safe
else:
    from trueskate_ai.control.hid_pointer import displacement as ble_displacement
    from trueskate_ai.data.control_hitboxes import CONTROL_START_MAP_VERSION, segment_is_safe

SIZE = (414, 896)
LEAD_MS = 8000
STEP_US = 3000
MAX_VECTOR = 16
MAX_EVENTS = (4096 - 32) // 2
MAX_DURATION_US = 34_000_000
HEADER = struct.Struct('<4sIIHHHHIII')
FAMILIES = {
    'arc': ([(-1, 0), (-.7, -.7), (0, -1), (.7, -.7), (1, 0)], [0, .25, .5, .75, 1]),
    's': ([(-1, 0), (-.5, -1), (0, 0), (.5, 1), (1, 0)], [0, .25, .5, .75, 1]),
    'reversal': ([(-1, 0), (1, -.5), (0, .5), (1, -.5), (-1, 0)], [0, .25, .5, .75, 1]),
    'pause': ([(-1, 0), (-.5, -1), (-.5, -1), (.5, 1), (1, 0)], [0, .2, .5, .8, 1]),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def schedule_hash(events):
    # Matches the proven PHR1 firmware/readback convention.
    return hashlib.sha256(json.dumps(events).encode()).hexdigest()[:16]


def seal(value):
    return dict(value, sha256=digest(value))


def checked(value):
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('content checksum mismatch')
    return value


def case(family, duration_ms, repeat=0):
    points, fractions = FAMILIES[family]
    # round() is deterministic here; frozen integer times remain authoritative.
    return dict(id=f'{family}-{duration_ms}-r{repeat}', family=family, duration_ms=duration_ms,
                points=[[.5 + .1*x, .48 + .06*y] for x, y in points],
                times_ms=[round(duration_ms*t) for t in fractions], repeat=repeat)


def position(path, ms):
    """Intended position, with no per-gesture phase adjustment."""
    ts, ps = path['times_ms'], path['points']
    if ms <= 0:
        return tuple(ps[0])
    if ms >= ts[-1]:
        return tuple(ps[-1])
    i = bisect_right(ts, ms)-1
    u = (ms-ts[i])/(ts[i+1]-ts[i])
    return tuple(a+u*(b-a) for a, b in zip(ps[i], ps[i+1]))


def speed_bound(path):
    return max(math.dist(a, b)/((tb-ta)/1000) for a, b, ta, tb in
               zip(path['points'], path['points'][1:], path['times_ms'], path['times_ms'][1:]))


def validate_profile(profile):
    checked(profile)
    if profile.get('schema') != 'pico-usb-gain-v1' or profile.get('accepted') is not True:
        raise ValueError('accepted measured USB profile required')
    if profile.get('device_size') != list(SIZE) or profile.get('training_admission') is not False:
        raise ValueError('unexpected device/profile purpose')
    if not profile.get('evidence_sha256') or not profile.get('calibration_schedule_sha256'):
        raise ValueError('gain evidence identity required')
    for step in (3000, 15000):
        nodes = profile['profiles'][str(step)]
        if nodes[0] != [0, 0] or nodes[-1][0] != 16:
            raise ValueError('gain profile must cover count lengths 0..16')
        if any(not math.isfinite(v) or v < 0 for p in nodes for v in p):
            raise ValueError('nonfinite gain profile')
        if any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(nodes, nodes[1:])):
            raise ValueError('nonmonotonic gain profile')
    if len(profile['park_point_pt']) != 2 or any(not math.isfinite(v) for v in profile['park_point_pt']):
        raise ValueError('invalid measured park')
    return profile


def distance(nodes, length):
    if not 0 <= length <= nodes[-1][0]+1e-9:
        raise ValueError('report outside measured gain range')
    for (m0, d0), (m1, d1) in zip(nodes, nodes[1:]):
        if length <= m1:
            return d0+(d1-d0)*(length-m0)/(m1-m0)
    return nodes[-1][1]


def movement(profile, dx, dy, step=STEP_US):
    length = math.hypot(dx, dy)
    if length == 0:
        return 0., 0.
    d = distance(profile['profiles'][str(step)], length)/length
    return dx*d, dy*d


def candidates(profile, step=STEP_US):
    # Exhaustive small integer vectors; deterministic ties and no inverse-fit assumptions.
    return [(dx, dy, *movement(profile, dx, dy, step)) for dx in range(-16, 17)
            for dy in range(-16, 17) if math.hypot(dx, dy) <= MAX_VECTOR]


def follow(profile, start, targets, step=STEP_US):
    options = candidates(profile, step)
    x, y = start
    reports, predicted = [], []
    for tx, ty in targets:
        dx, dy, px, py = min(options, key=lambda r: ((x+r[2]-tx)**2+(y+r[3]-ty)**2,
                                                    r[0]**2+r[1]**2, r[0], r[1]))
        x, y = x+px, y+py
        reports.append((dx, dy))
        predicted.append((x, y))
    return reports, predicted


class Program:
    def __init__(self, kind):
        self.kind, self.events, self.commands, self.sections = kind, [], [], []

    def add(self, t, dx=0, dy=0, buttons=0):
        self.events.append([int(t), int(dx), int(dy), int(buttons)])
        return len(self.events)-1

    def home(self, at):
        # Same proven home/park reports and rate; actual park is measured, not assumed.
        first = len(self.events)
        for i in range(25):
            self.add(at+i*15000, -127, -127)
        for i in range(40):
            self.add(at+675000+i*15000, 3, 11)
        self.sections.append(dict(kind='home-park', start_us=at, end_us=at+1_875_000,
                                  event_start=first, event_end=len(self.events)))
        return at+1_875_000

    def contact(self, at, path, profile, kind, role=None):
        start = tuple(p*s for p, s in zip(path['points'][0], SIZE))
        # Hover approach at measured 15 ms gain, entirely button-up. 300 ms permits
        # bounded count vectors even for the longest centre-screen approach.
        origin = tuple(profile['park_point_pt'])
        targets = [tuple(a+(b-a)*(i+1)/20 for a, b in zip(origin, start)) for i in range(20)]
        reports, predicted_hover = follow(profile, origin, targets, 15000)
        for i, (dx, dy) in enumerate(reports):
            self.add(at-450000+i*15000, dx, dy)
        pos = predicted_hover[-1]
        first = self.add(at, buttons=1)  # still press; no lost initial movement
        duration = path['duration_ms']*1000
        times = list(range(STEP_US, duration, STEP_US))+[duration]
        targets = [tuple(p*s for p, s in zip(position(path, t/1000), SIZE)) for t in times]
        reports, predicted = follow(profile, pos, targets)
        for t, (dx, dy) in zip(times, reports):
            self.add(at+t, dx, dy, 1)
        lift = self.add(at+duration+1000)  # still, separate lift after the endpoint
        modelled = [[0, *pos]]+[[t, *p] for t, p in zip(times, predicted)]
        command = dict(kind=kind, role=role, id=path['id'], path=path, press_us=at,
                       lift_us=at+duration+1000, event_start=first, event_end=lift+1,
                       predicted_pt=modelled)
        self.commands.append(command)
        for (_, x, y), target in zip(modelled, [path['points'][0]]+[position(path, t/1000) for t in times]):
            if math.dist((x/SIZE[0], y/SIZE[1]), target) > .005:
                raise ValueError(f'{path["id"]}: compiler position error exceeds .005')
        all_points = [(p[1]/414, p[2]/896) for p in modelled]
        for a, b in zip(all_points, all_points[1:]):
            if not segment_is_safe(a, b):
                raise ValueError('predicted contact crosses an expanded control')

    def finish(self, profile=None):
        self.add(max(e[0] for e in self.events)+15000)
        value = dict(schema='pico-curve-program-v1', kind=self.kind, device='iPhone_XR2',
                     device_size=list(SIZE), park='The Workshop', lead_ms=LEAD_MS,
                     events=self.events, n_events=len(self.events), commands=self.commands,
                     sections=self.sections, duration_us=self.events[-1][0],
                     schedule_hash=schedule_hash(self.events), control_map=CONTROL_START_MAP_VERSION,
                     training_admission=False, profile=profile)
        value = seal(value)
        validate_program(value, _regenerate=False)
        return value


def gain_program():
    p = Program('gain')
    vectors = [(n, 0) for n in (1, 2, 3, 4, 5, 8, 16)]+[(0, 4), (3, 3)]
    at = 0
    for step in (15000, 3000):
        at = p.home(at)
        for dx, dy in vectors:
            count = min(40, max(6, math.ceil(40/math.hypot(*ble_displacement(dx, dy)))))
            for sign in (1, -1):
                first, start = len(p.events), at
                for _ in range(count):
                    p.add(at, sign*dx, sign*dy)
                    at += step
                p.commands.append(dict(kind='gain', id=f'g{len(p.commands):02}', step_us=step,
                                       dx=sign*dx, dy=sign*dy, reports=count, start_us=start,
                                       end_us=at, event_start=first, event_end=len(p.events)))
                at += 400000
    return p.finish()


def curve_program(profile, batch):
    validate_profile(profile)
    if batch not in (1, 2):
        raise ValueError('batch must be 1 or 2')
    fast = [case(f, 150, batch-1) for f in FAMILIES]
    slow = [case(f, 600) for f in (('arc', 's') if batch == 1 else ('reversal', 'pause'))]
    p = Program(f'curves-{batch}')
    # Two anchors and one held-out check: same contact trajectory, no gameplay reset.
    marker = dict(id='marker', duration_ms=300, points=[[.5, .48], [.5, .42]], times_ms=[0, 300])
    schedule = [(3_000_000, 'control', 'start', dict(marker, id='control-start')),
                *[(t*1_000_000, 'curve', None, path) for t, path in zip((7, 11, 15), fast[:3])],
                (19_000_000, 'control', 'middle', dict(marker, id='control-middle')),
                (23_000_000, 'curve', None, fast[3]),
                (27_000_000, 'curve', None, slow[0]), (30_500_000, 'curve', None, slow[1]),
                (33_500_000, 'control', 'end', dict(marker, id='control-end'))]
    for at, kind, role, path in schedule:
        p.home(at-2_300_000)
        p.contact(at, path, profile, kind, role)
    return p.finish(profile)


def validate_program(p, *, _regenerate=True):
    checked(p)
    ev = p['events']
    if p.get('schema') != 'pico-curve-program-v1' or p.get('training_admission') is not False:
        raise ValueError('unexpected program schema/purpose')
    if p.get('device_size') != list(SIZE) or p.get('lead_ms') != LEAD_MS:
        raise ValueError('unexpected geometry/lead')
    if not ev or len(ev) != p['n_events'] or len(ev) > MAX_EVENTS:
        raise ValueError('event count or stats sector capacity exceeded')
    if schedule_hash(ev) != p['schedule_hash'] or p['duration_us'] != ev[-1][0]:
        raise ValueError('schedule identity mismatch')
    if ev[-1][0] > MAX_DURATION_US or ev[-1][1:] != [0, 0, 0]:
        raise ValueError('program duration/neutral ending invalid')
    if any(len(e) != 4 or any(type(v) is not int for v in e) or e[0] < 0 or
           max(abs(e[1]), abs(e[2])) > 127 or e[3] not in (0, 1) for e in ev):
        raise ValueError('invalid report')
    if any(b[0] <= a[0] for a, b in zip(ev, ev[1:])):
        raise ValueError('reports are not strictly chronological')
    home_indices = {i for s in p['sections'] for i in range(s['event_start'], s['event_start']+25)}
    if any(math.hypot(e[1], e[2]) > MAX_VECTOR and i not in home_indices for i, e in enumerate(ev)):
        raise ValueError('large count vector outside homing')
    if p['kind'] == 'gain':
        if any(e[3] for e in ev) or len(p['commands']) != 36:
            raise ValueError('gain program must contain 36 button-up passes')
        if _regenerate and p != gain_program():
            raise ValueError('gain program differs from frozen generation')
        return
    if p['kind'] not in ('curves-1', 'curves-2'):
        raise ValueError('unknown curve program')
    validate_profile(p['profile'])
    # Verify content against the deterministic compiler, without re-entering finish.
    curves = [c for c in p['commands'] if c['kind'] == 'curve']
    controls = [c for c in p['commands'] if c['kind'] == 'control']
    if len(curves) != 6 or [c['role'] for c in controls] != ['start', 'middle', 'end']:
        raise ValueError('six curves and three controls required')
    batch = int(p['kind'][-1])
    expected = [case(f, 150, batch-1) for f in FAMILIES]+[case(f, 600) for f in
               (('arc', 's') if batch == 1 else ('reversal', 'pause'))]
    if [c['path'] for c in curves] != expected:
        raise ValueError('curve cases differ from frozen geometry/order')
    covered = set()
    for c in p['commands']:
        lo, hi = c['event_start'], c['event_end']
        contact = ev[lo:hi]
        if contact[0] != [c['press_us'], 0, 0, 1] or contact[-1] != [c['lift_us'], 0, 0, 0]:
            raise ValueError('missing separate still press/lift')
        if any(e[3] != 1 for e in contact[:-1]):
            raise ValueError('contact interruption in compiled program')
        if contact[-1][0]-contact[-2][0] != 1000 or contact[-2][0]-contact[0][0] != c['path']['duration_ms']*1000:
            raise ValueError('endpoint/lift timing differs from target')
        for a, b in zip(c['path']['points'], c['path']['points'][1:]):
            if not segment_is_safe(a, b):
                raise ValueError('intended path crosses controls')
        covered.update(range(lo, hi-1))
    if {i for i, e in enumerate(ev) if e[3]} != covered:
        raise ValueError('unaccounted contact')
    if _regenerate:
        expected = curve_program(p['profile'], batch)
        actual = json.loads(json.dumps(p))
        # Retain exact reports, order, times, cases and gain evidence. Allow only
        # negligible libm variation in diagnostic floating position predictions.
        for a, b in zip(actual['commands'], expected['commands']):
            if len(a['predicted_pt']) != len(b['predicted_pt']) or any(
                len(x) != len(y) or any(abs(u-v) > 1e-10 for u, v in zip(x, y))
                for x, y in zip(a['predicted_pt'], b['predicted_pt'])):
                raise ValueError('predicted positions differ from USB compiler')
            a['predicted_pt'] = b['predicted_pt']
        actual.pop('sha256')
        if digest(actual) != digest({k: v for k, v in expected.items() if k != 'sha256'}):
            raise ValueError('program differs from deterministic USB compiler')


def firmware_header(program):
    validate_program(program)
    rows = ',\n'.join(f'  {{{t}u, {dx}, {dy}, {b}}}' for t, dx, dy, b in program['events'])
    return f'''#pragma once
// Generated USB curve diagnostic; program SHA256 {program['sha256']}.
#include <stdint.h>
struct Event {{ uint32_t t_us; int8_t dx; int8_t dy; uint8_t buttons; }};
static constexpr uint32_t kScheduleHash = 0x{program['schedule_hash'][:8]}u;
static constexpr uint32_t kLeadMs = {LEAD_MS}u;
static constexpr uint16_t kEventCount = {program['n_events']};
static const Event kEvents[kEventCount] = {{
{rows}
}};
'''


def receipts(blob, program):
    validate_program(program)
    if len(blob) != 4096:
        raise ValueError('exact 4096-byte stats sector required')
    magic, build, sched, status, count, sent, clipped, mount, waits, maxwait = HEADER.unpack_from(blob)
    if magic != b'PHR1' or sched != int(program['schedule_hash'][:8], 16) or count != program['n_events']:
        raise ValueError('board receipt does not match frozen program')
    raw = struct.unpack_from(f'<{count}H', blob, HEADER.size)
    lateness = [None if x == 65535 else x*16 for x in raw]
    if sent > count or any(v is None for v in lateness[:sent]) or any(v is not None for v in lateness[sent:]):
        raise ValueError('board sent count differs from slots')
    actual = [None if late is None else e[0]+late for e, late in zip(program['events'], lateness)]
    if any(b < a for a, b in zip(actual[:sent], actual[1:sent])):
        raise ValueError('unordered board acceptance times')
    return seal(dict(schema='pico-curve-receipts-v1', program_sha256=program['sha256'], build_id=f'{build:08x}',
                     stats_sha256=hashlib.sha256(blob).hexdigest(), status=status, events_sent=sent,
                     clipped=clipped, mount_ms=mount, waits=waits, max_wait_us=maxwait,
                     lateness_us=lateness, accepted_us=actual,
                     semantics='TinyUSB acceptance; not independently observed iOS delivery'))


def verify_receipts(receipt, program, *, complete=True):
    checked(receipt)
    if receipt.get('program_sha256') != program['sha256'] or len(receipt['accepted_us']) != program['n_events']:
        raise ValueError('receipt/program identity mismatch')
    if complete and (receipt['status'] != 2 or receipt['events_sent'] != program['n_events'] or receipt['clipped']):
        raise ValueError('incomplete or clipped board run; preserve without replacement')


def fit_gain(program, observations):
    """Fit radial profiles from validated stationary before/after positions.

    Observations bind the movie, receipt and displayed JPEGs externally. No measured
    curve is ever used to fit gain. Direction checks must agree within 0.5 pt/report.
    """
    validate_program(program)
    if program['kind'] != 'gain' or observations.get('program_sha256') != program['sha256']:
        raise ValueError('gain calibration identity mismatch')
    rows = observations['passes']
    if len(rows) != len(program['commands']):
        raise ValueError('all calibration passes required')
    measurements = {}
    for command, row in zip(program['commands'], rows):
        if row['id'] != command['id'] or row.get('confirmed') is not True:
            raise ValueError('validated stationary endpoints required')
        before, after = row['before_pt'], row['after_pt']
        if len(before) != 2 or len(after) != 2:
            raise ValueError('two-dimensional gain endpoints required')
        if any(not math.isfinite(v) or not 0 <= v <= SIZE[i] for point in (before, after) for i, v in enumerate(point)):
            raise ValueError('invalid gain endpoint')
        if row.get('uncertainty_pt', math.inf) > 2 or row.get('stable') is not True:
            raise ValueError('inadequate stationary cursor tracking')
        dx, dy = command['dx'], command['dy']
        length = math.hypot(dx, dy)
        vx, vy = ((b-a)/command['reports'] for a, b in zip(before, after))
        parallel = (vx*dx+vy*dy)/length
        orthogonal = abs(vx*dy-vy*dx)/length
        if parallel <= 0 or orthogonal > .5:
            raise ValueError('direction-dependent gain or incorrect tracking')
        group = measurements.setdefault(command['step_us'], {})
        group.setdefault(length, []).append(parallel)
    profiles = {}
    for step, group in measurements.items():
        nodes = [[0, 0]]
        for length, values in sorted(group.items()):
            if max(values)-min(values) > .5:
                raise ValueError('direction-dependent gain discrepancy exceeds .5 pt/report')
            nodes.append([length, sum(values)/len(values)])
        # Reject a bad/nonmonotonic calibration rather than silently smoothing it.
        if any(b[1] < a[1] for a, b in zip(nodes, nodes[1:])):
            raise ValueError('measured gain is not monotonic')
        profiles[str(step)] = nodes
    parks = observations['park_points_pt']
    if len(parks) != 2 or any(len(p) != 2 or any(not math.isfinite(v) for v in p) for p in parks) or math.dist(*parks) > 2:
        raise ValueError('home/park reproducibility not established')
    park = [sum(p[i] for p in parks)/len(parks) for i in range(2)]
    profile = seal(dict(schema='pico-usb-gain-v1', device_size=list(SIZE), accepted=True,
                        profiles=profiles, park_point_pt=park, evidence_sha256=digest(observations),
                        calibration_schedule_sha256=program['sha256'], training_admission=False))
    return validate_profile(profile)


def fit_timebase(program, receipt, controls, pts):
    """One affine map from first/end controls; middle is held out, never fitted.

    Each onset lies between its last clear pre-contact frame and first visible one.
    Fit interval midpoints and carry their uncertainty into every position score.
    """
    verify_receipts(receipt, program)
    commands = [c for c in program['commands'] if c['kind'] == 'control']
    if set(controls) != {'start', 'middle', 'end'}:
        raise ValueError('three independent Pico controls required')
    board, video, half = [], [], []
    for c in commands:
        onset = controls[c['role']]
        i = onset.get('first_contact_frame')
        if type(i) is not int or not 1 <= i < len(pts) or onset.get('confirmed') is not True:
            raise ValueError('visible control boundary required')
        a, b = pts[i-1], pts[i]
        if b-a > .025:
            raise ValueError('dropped frame at control boundary')
        board.append(receipt['accepted_us'][c['event_start']]/1e6)
        video.append((a+b)/2)
        half.append((b-a)/2)
    rate = (video[2]-video[0])/(board[2]-board[0])
    intercept = video[0]-rate*board[0]
    if not .98 <= rate <= 1.02:
        raise ValueError('implausible recording clock fit')
    residual = abs(video[1]-(intercept+rate*board[1]))
    frame = 2*half[1]
    return dict(intercept_s=intercept, rate=rate, middle_error_s=residual,
                accepted=residual <= frame+1e-9,
                uncertainty_s=max(half[0], half[2], residual+half[1]),
                control_onset_intervals_s=[[v-h, v+h] for v, h in zip(video, half)],
                source='Pico accepted reports and visible contact; no WDA execution timestamp')


def score_curve(command, receipt, fit, pts, annotations):
    """Conservative diagnostic with unknowns retained and no local time/shape fitting."""
    onset = fit['intercept_s']+fit['rate']*command['press_us']/1e6
    lift = fit['intercept_s']+fit['rate']*command['lift_us']/1e6
    indices = [i for i, t in enumerate(pts) if onset <= t < lift]
    rows = {row['source_frame']: row for row in annotations.get('frames', [])}
    if len(rows) != len(annotations.get('frames', [])):
        raise ValueError('duplicate frame annotation')
    errors, lower, unknown = [], [], []
    for i in indices:
        row = rows.get(i)
        if not row or row.get('contact_confirmed') is not True or row.get('point_pt') is None:
            unknown.append(i)
            continue
        point = row['point_pt']
        uncertainty = row.get('uncertainty_pt')
        if len(point) != 2 or any(not math.isfinite(x) for x in point) or not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError('invalid position annotation')
        ms = (pts[i]-onset)/fit['rate']*1000
        target = position(command['path'], ms)
        error = math.dist(tuple(p/s for p, s in zip(point, SIZE)), target)
        budget = uncertainty/min(SIZE)+speed_bound(command['path'])*fit['uncertainty_s']/fit['rate']
        errors.append(dict(source_frame=i, error=error, uncertainty=budget, upper=error+budget,
                           error_pt=math.dist(point, tuple(p*s for p, s in zip(target, SIZE)))))
        lower.append(max(0, error-budget))
    duration_error = duration_uncertainty = None
    bounds = []
    for key in ('first_contact_frame', 'first_lifted_frame'):
        i = annotations.get(key)
        if annotations.get('boundary_confirmed') is True and type(i) is int and 1 <= i < len(pts) and pts[i]-pts[i-1] <= .025:
            bounds.append((.5*(pts[i]+pts[i-1]), .5*(pts[i]-pts[i-1])))
    if len(bounds) == 2 and bounds[1][0] > bounds[0][0]:
        duration_error = abs(bounds[1][0]-bounds[0][0]-command['path']['duration_ms']/1000*fit['rate'])
        duration_uncertainty = bounds[0][1]+bounds[1][1]
    # Cadence gaps during fast motion cannot disappear from the denominator.
    # This remains an estimate of missing 60 Hz slots, never invented pixel data.
    expected_slots = max(len(indices), round((lift-onset)*60))
    coverage = len(errors)/expected_slots if expected_slots else 0
    interrupted = annotations.get('interrupted') is True
    if fit.get('accepted') is not True:
        verdict = 'inconclusive'
    elif interrupted or any(e > .03 for e in lower) or (duration_error is not None and duration_error-duration_uncertainty > .10):
        verdict = 'fail'
    elif (fit.get('accepted') is True and coverage >= .9 and errors and
          all(e['upper'] <= .03 for e in errors) and duration_error is not None and
          duration_error+duration_uncertainty <= .10 and annotations.get('interrupted') is False):
        verdict = 'pass'
    else:
        verdict = 'inconclusive'
    late = receipt['lateness_us'][command['event_start']:command['event_end']]
    return dict(id=command['id'], verdict=verdict, coverage=coverage, evaluated_frames=len(errors),
                expected_frames=expected_slots, native_contact_frames=len(indices),
                missing_nominal_slots=expected_slots-len(indices), unknown_frames=unknown, errors=errors,
                duration_error_s=duration_error, duration_uncertainty_s=duration_uncertainty,
                observed_onset_error_s=abs(bounds[0][0]-onset) if len(bounds) == 2 else None,
                observed_lift_error_s=abs(bounds[1][0]-lift) if len(bounds) == 2 else None,
                board_max_lateness_us=max(late), onset_s=onset, lift_s=lift,
                interpretation='A failed curve does not by itself identify timing as the cause')
