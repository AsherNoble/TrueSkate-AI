"""Synthetic USB evidence only: no research recordings or devices."""
import copy
import json
import math
from pathlib import Path
import struct
import importlib.util

import numpy as np
import pytest

from trueskate_ai.control.hid_pointer import displacement, STEP_US as BLE_STEP
from trueskate_ai.research import pico_curve_pilot as p
from trueskate_ai.research import pico_curve_review as r


def observations(program):
    rows = []
    for c in program['commands']:
        dx, dy = displacement(c['dx'], c['dy'])
        before = [150., 350.]
        after = [before[0]+c['reports']*dx, before[1]+c['reports']*dy]
        rows.append(dict(id=c['id'], before_pt=before, after_pt=after, uncertainty_pt=.5, stable=True, confirmed=True))
    return dict(program_sha256=program['sha256'], passes=rows, park_points_pt=[[109.1, 368.2]]*2)


@pytest.fixture(scope='module')
def profile():
    program = p.gain_program()
    return p.fit_gain(program, observations(program))


def receipt(program, *, late=None, sent=None, status=2):
    count = program['n_events']
    sent = count if sent is None else sent
    slots = [round((late or {}).get(i, 0)/16) if i < sent else 65535 for i in range(count)]
    header = p.HEADER.pack(b'PHR1', 1234, int(program['schedule_hash'][:8], 16), status, count, sent, 0, 550, 0, 0)
    blob = header+struct.pack(f'<{count}H', *slots)
    return p.receipts(blob+b'\xff'*(4096-len(blob)), program)


def reseal(value):
    return p.seal({k: v for k, v in value.items() if k != 'sha256'})


def test_frozen_programs_are_bounded_separate_contacts_and_deterministic(profile):
    gain = p.gain_program()
    assert gain == p.gain_program()
    assert gain['n_events'] == 971 and gain['duration_us'] == 25_322_000
    assert not any(e[3] for e in gain['events'])
    first, second = [p.curve_program(profile, i) for i in (1, 2)]
    assert first == p.curve_program(profile, 1)
    assert BLE_STEP == 15000
    for program in (first, second):
        p.validate_program(program)
        assert program['n_events'] == 1684 <= p.MAX_EVENTS
        assert program['duration_us'] == 33_816_000 < 34_000_000
        assert len([c for c in program['commands'] if c['kind'] == 'curve']) == 6
        for c in program['commands']:
            events = program['events'][c['event_start']:c['event_end']]
            assert events[0][1:] == [0, 0, 1] and events[-1][1:] == [0, 0, 0]
            assert all(e[3] == 1 for e in events[:-1])
            assert events[-1][0]-events[-2][0] == 1000
            assert max(math.hypot(e[1], e[2]) for e in events) <= 16
    cases = [c for prog in (first, second) for c in prog['commands'] if c['kind'] == 'curve']
    assert len(cases) == len({c['id'] for c in cases}) == 12
    assert {c['path']['duration_ms'] for c in cases} == {150, 600}
    assert p.position(p.case('pause', 600), 200) == p.position(p.case('pause', 600), 270)
    assert p.position(p.case('reversal', 600), 0) == p.position(p.case('reversal', 600), 600)


@pytest.mark.parametrize('fault', ['changed_path', 'missing_lift', 'interruption', 'overflow', 'unsorted', 'checksum'])
def test_bad_programs_rejected(profile, fault):
    program = copy.deepcopy(p.curve_program(profile, 1))
    c = program['commands'][1]
    if fault == 'changed_path':
        c['path']['points'][1][0] += .001
    elif fault == 'missing_lift':
        program['events'][c['event_end']-1][1] = 1
    elif fault == 'interruption':
        program['events'][c['event_start']+2][3] = 0
    elif fault == 'overflow':
        program['n_events'] = p.MAX_EVENTS+1
    elif fault == 'unsorted':
        program['events'][2][0] = program['events'][1][0]
    else:
        program['events'][2][1] = 0
    if fault != 'checksum':
        program['schedule_hash'] = p.schedule_hash(program['events'])
        program = reseal(program)
    with pytest.raises(ValueError):
        p.validate_program(program)


def test_measured_gain_requires_visible_stationary_and_direction_agreement():
    program = p.gain_program()
    good = observations(program)
    fit = p.fit_gain(program, good)
    assert set(fit['profiles']) == {'3000', '15000'}
    for bad in ('unconfirmed', 'uncertain', 'direction', 'park', 'negative_gain'):
        data = copy.deepcopy(good)
        if bad == 'unconfirmed':
            data['passes'][0]['confirmed'] = False
        elif bad == 'uncertain':
            data['passes'][0]['uncertainty_pt'] = 3
        elif bad == 'direction':
            row = data['passes'][7]  # -4 counts, 22 reports
            row['after_pt'][0] -= 25
        elif bad == 'park':
            data['park_points_pt'][1] = [115, 368.2]
        else:
            data['passes'][0]['after_pt'] = [149, 350]
        with pytest.raises(ValueError):
            p.fit_gain(program, data)


def test_board_readback_binds_schedule_and_preserves_partial_host_loss(profile):
    program = p.curve_program(profile, 1)
    normal = receipt(program)
    p.verify_receipts(normal, program)
    failed = receipt(program, sent=200, status=4)
    assert failed['accepted_us'][200:] == [None]*(program['n_events']-200)
    with pytest.raises(ValueError, match='incomplete'):
        p.verify_receipts(failed, program)
    p.verify_receipts(failed, program, complete=False)
    other = p.curve_program(profile, 2)
    with pytest.raises(ValueError, match='identity'):
        p.verify_receipts(normal, other)
    with pytest.raises(ValueError, match='4096'):
        p.receipts(b'short', program)
    first = program['commands'][1]['event_start']
    delayed = receipt(program, late={i: 32000 for i in range(first, program['n_events'])})
    assert delayed['accepted_us'][first]-normal['accepted_us'][first] == 32000


def controls_for(program, board, pts, intercept=10., rate=.998, middle_shift=0):
    controls = {}
    for c in program['commands']:
        if c['kind'] != 'control':
            continue
        t = intercept+rate*board['accepted_us'][c['event_start']]/1e6
        if c['role'] == 'middle':
            t += middle_shift
        controls[c['role']] = dict(first_contact_frame=int(np.searchsorted(pts, t)), confirmed=True)
    return controls


def test_global_drift_fit_uses_first_end_and_middle_is_held_out(profile):
    program = p.curve_program(profile, 1)
    board = receipt(program)
    pts = np.arange(0, 60, 1/60).tolist()
    good = p.fit_timebase(program, board, controls_for(program, board, pts), pts)
    assert good['accepted'] and good['rate'] == pytest.approx(.998, abs=.001)
    bad = p.fit_timebase(program, board, controls_for(program, board, pts, middle_shift=.12), pts)
    assert not bad['accepted'] and bad['middle_error_s'] > .10
    assert bad['intercept_s'] == good['intercept_s'] and bad['rate'] == good['rate']
    control = controls_for(program, board, pts)
    missing = pts.copy()
    i = control['start']['first_contact_frame']
    missing[i-1] = missing[i]-.04
    with pytest.raises(ValueError, match='dropped'):
        p.fit_timebase(program, board, control, missing)


def trace(c, fit, pts, *, shift_s=0., scale=1.):
    onset = fit['intercept_s']+fit['rate']*c['press_us']/1e6
    lift = fit['intercept_s']+fit['rate']*c['lift_us']/1e6
    rows = []
    for i, t in enumerate(pts):
        if onset <= t < lift:
            pos = p.position(c['path'], (t-onset-shift_s)/fit['rate']*1000)
            start = c['path']['points'][0]
            pos = tuple(a+scale*(b-a) for a, b in zip(start, pos))
            rows.append(dict(source_frame=i, point_pt=[pos[0]*414, pos[1]*896],
                             uncertainty_pt=.5, contact_confirmed=True))
    return dict(frames=rows, first_contact_frame=int(np.searchsorted(pts, onset+shift_s)),
                first_lifted_frame=int(np.searchsorted(pts, lift+shift_s)), interrupted=False, boundary_confirmed=True)


def test_scoring_distinguishes_delay_gain_error_unknown_and_uncertainty(profile):
    program = p.curve_program(profile, 1)
    board = receipt(program)
    pts = np.arange(0, 60, 1/60).tolist()
    c = next(c for c in program['commands'] if c['id'] == 's-600-r0')
    fit = dict(intercept_s=10., rate=1., accepted=True, uncertainty_s=1/120)
    good = trace(c, fit, pts)
    assert p.score_curve(c, board, fit, pts, good)['verdict'] == 'pass'
    shifted = trace(c, fit, pts, shift_s=.15)
    wrong_gain = trace(c, fit, pts, scale=.5)
    for bad in (shifted, wrong_gain, dict(good, interrupted=True)):
        assert p.score_curve(c, board, fit, pts, bad)['verdict'] == 'fail'
    invisible = dict(good, frames=good['frames'][:5])
    assert p.score_curve(c, board, fit, pts, invisible)['verdict'] == 'inconclusive'
    assert p.score_curve(c, board, dict(fit, accepted=False), pts, shifted)['verdict'] == 'inconclusive'
    assert p.score_curve(c, board, fit, pts, dict(good, boundary_confirmed=False))['verdict'] == 'inconclusive'
    assert p.score_curve(c, board, dict(fit, uncertainty_s=.08), pts, good)['verdict'] == 'inconclusive'
    # Board lateness remains diagnostic; it never shifts the intended ring per curve.
    delayed = copy.deepcopy(board)
    delayed['lateness_us'][c['event_start']] = 48000
    result = p.score_curve(c, delayed, fit, pts, good)
    assert result['board_max_lateness_us'] == 48000 and result['onset_s'] == 10+c['press_us']/1e6


def test_static_tracking_and_capture_admission_fail_closed(tmp_path):
    clip = dict(frames=[dict(source_frame=i) for i in range(5)])
    rows = [dict(source_frame=i, point_pt=[100+i*.1, 300.], uncertainty_pt=1.) for i in range(5)]
    assert r.static_position(clip, dict(static_confirmed=True, frames=rows))[0] == pytest.approx([100.2, 300])
    with pytest.raises(ValueError, match='uncertain'):
        r.static_position(clip, dict(static_confirmed=True, frames=[dict(x, point_pt=[100+i*4, 300]) for i, x in enumerate(rows)]))
    with pytest.raises(ValueError, match='inadequate'):
        r.static_position(clip, dict(static_confirmed=True, frames=rows[:2]))
    movie = tmp_path/'fixture.mov'
    movie.write_bytes(b'synthetic')
    program = p.gain_program()
    video = dict(frames=3600, duration_s=60., effective_fps=60., max_gap_s=1/60, sha256=r.file_sha256(movie))
    report = dict(pico_program_sha256=program['sha256'], park='The Workshop', park_provenance='operator readiness prompt',
                  cases=[dict(lifecycle='passed', fps_requested=60, video=video)], cleanup_errors=[])
    pts = np.arange(3600)/60
    r.capture_proof(report, program, movie, pts)
    for failed in (dict(report, cleanup_errors=['failure']), dict(report, pico_program_sha256='other'),
                   dict(report, park_provenance='assumed'), dict(report, error='aborted')):
        with pytest.raises(ValueError):
            r.capture_proof(failed, program, movie, pts)


def test_fast_reversal_clock_floor_and_local_capture_gaps_are_inconclusive(profile):
    program = p.curve_program(profile, 1)
    board = receipt(program)
    pts = np.arange(0, 60, 1/60).tolist()
    fit = dict(intercept_s=10., rate=1., accepted=True, uncertainty_s=1/120)
    reversal = next(c for c in program['commands'] if c['id'] == 'reversal-150-r0')
    ideal = p.score_curve(reversal, board, fit, pts, trace(reversal, fit, pts))
    assert ideal['verdict'] == 'inconclusive' and max(x['uncertainty'] for x in ideal['errors']) > .03
    arc = next(c for c in program['commands'] if c['id'] == 'arc-150-r0')
    assert p.score_curve(arc, board, fit, pts, trace(arc, fit, pts))['verdict'] == 'pass'
    onset = fit['intercept_s']+arc['press_us']/1e6
    dropped = [t for t in pts if not onset+.03 < t < onset+.08]
    lost = p.score_curve(arc, board, fit, dropped, trace(arc, fit, dropped))
    assert lost['verdict'] == 'inconclusive' and lost['missing_nominal_slots'] >= 2 and lost['coverage'] < .9


def test_review_binds_source_pts_pixels_receipts_and_annotations(tmp_path, monkeypatch, profile):
    program = p.curve_program(profile, 1)
    board = receipt(program)
    movie = tmp_path/'fixture.mov'
    movie.write_bytes(b'synthetic original')
    pts = np.arange(3600)/60
    report = dict(pico_program_sha256=program['sha256'], park='The Workshop', park_provenance='operator readiness prompt',
                  cases=[dict(lifecycle='passed', fps_requested=60, video=dict(frames=3600, duration_s=60.,
                    effective_fps=60., max_gap_s=1/60, sha256=r.file_sha256(movie)))], cleanup_errors=[])
    monkeypatch.setattr(r, 'native_pts', lambda _: pts.tolist())
    def decode(movie, pts, keep, inspect):
        for _ in pts:
            inspect(np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(r, '_decode_source_frames', decode)
    templates = Path(__file__).resolve().parents[1]/'scripts/inspect/templates'
    root = tmp_path/'controls'
    bp = r.build_review(program, board, movie, report, root, templates, origin_s=10.)
    private = json.loads(bp.read_text())
    controls = controls_for(program, board, pts, rate=1.)
    annotations = {c['id']:dict(first_contact_frame=controls[c['role']]['first_contact_frame'], boundary_confirmed=True)
                   for c in private['identity']['clips']}
    exported = dict(schema=private['schema'], version=2, bundle_sha256=private['bundle_sha256'], annotations=annotations)
    fit, proof = r.control_fit(private, exported, root)
    assert fit['accepted'] and proof['bundle_sha256'] == private['bundle_sha256']
    full = tmp_path/'curves'
    result_path = r.build_review(program, board, movie, report, full, templates, control_review=(private, exported, root))
    final = json.loads(result_path.read_text())
    assert len(final['identity']['clips']) == 6 and final['identity']['provenance']['receipts'] == board
    assert 'wda_timing' not in final['identity']['provenance']
    clip = final['identity']['clips'][0]
    for f in clip['frames']:
        assert f['pts_s'] == pts[f['source_frame']] and len(f['sha256']) == 64
    # Changing a displayed byte cannot retain the review identity.
    frame = private['identity']['clips'][0]['frames'][0]
    (root/frame['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        r.control_fit(private, exported, root)


def test_aggregate_requires_all_twelve_distinct_cases(profile):
    batches = []
    for batch in (1, 2):
        program = p.curve_program(profile, batch)
        batches.append(p.seal(dict(schema='pico-usb-curve-result-v1', kind=program['kind'],
            profile_sha256=profile['sha256'], results=[dict(id=c['id'], verdict='pass') for c in
            program['commands'] if c['kind'] == 'curve'])))
    assert r.aggregate(*batches)['verdict'] == 'pass'
    with pytest.raises(ValueError, match='distinct'):
        r.aggregate(batches[0], batches[0])
    missing = copy.deepcopy(batches[1])
    missing['results'].pop()
    with pytest.raises(ValueError, match='twelve'):
        r.aggregate(batches[0], reseal(missing))


def test_automatic_static_gain_retains_evidence_and_rejects_tracking_loss(tmp_path, monkeypatch):
    program = p.gain_program()
    board = receipt(program)
    movie = tmp_path/'fixture.mov'
    movie.write_bytes(b'synthetic original')
    pts = np.arange(3600)/60
    report = dict(pico_program_sha256=program['sha256'], park='The Workshop', park_provenance='operator readiness prompt',
                  cases=[dict(lifecycle='passed', fps_requested=60, video=dict(frames=3600, duration_s=60.,
                    effective_fps=60., max_gap_s=1/60, sha256=r.file_sha256(movie)))], cleanup_errors=[])
    monkeypatch.setattr(r, 'native_pts', lambda _: pts.tolist())
    image = np.full((1792, 828, 3), 120, np.uint8)
    state = {'i':0}
    def decode(movie, pts, keep, inspect):
        for i, _ in enumerate(pts):
            state['i'] = i
            inspect(image)
    monkeypatch.setattr(r, '_decode_source_frames', decode)
    modeled, at = [], []
    x = y = 0
    for t, dx, dy, b in program['events']:
        px, py = displacement(dx, dy)
        x, y = max(0, min(414, x+px)), max(0, min(896, y+py))
        at.append(t/1e6)
        modeled.append((x, y))
    def finder(*args, **kwargs):
        i = np.searchsorted(at, pts[state['i']]-10, side='right')-1
        x, y = modeled[max(0,i)]
        return 30., x, y, 8.5, 1
    out = tmp_path/'automatic'
    profile = r.measure_gain(program, board, movie, report, out, 10., finder=finder)
    assert profile['accepted'] and len(list(out.glob('*.jpg'))) == 74
    proof = json.loads((out/'evidence.json').read_text())
    assert len(proof['detections']) == 74 and proof['source_pts_s'] == pts.tolist()
    failed = tmp_path/'failed'
    with pytest.raises(ValueError, match='inadequate'):
        r.measure_gain(program, board, movie, report, failed, 10., finder=lambda *a, **kw:None)
    assert (failed/'evidence.json').exists() and (failed/'failure.json').exists() and not (failed/'profile.json').exists()


def test_uf2_package_guard_protects_timing_sector_and_board_family(tmp_path):
    path = Path(__file__).resolve().parents[1]/'hardware/pico_curve_pilot/pilot.py'
    spec = importlib.util.spec_from_file_location('usb_pilot_cli', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    def uf2(address=0x10000000, family=0xe48bff56):
        data = struct.pack('<8I', 0x0a324655, 0x9e5d5157, 0x2000, address, 256, 0, 1, family)
        return data+b'\0'*(508-len(data))+struct.pack('<I',0x0ab16f30)
    file = tmp_path/'firmware.uf2'
    file.write_bytes(uf2())
    cli.validate_uf2(file)
    for bad in (uf2(0x101ff000), uf2(family=0), b'short'):
        file.write_bytes(bad)
        with pytest.raises(ValueError):
            cli.validate_uf2(file)
