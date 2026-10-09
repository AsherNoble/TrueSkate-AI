"""Bounded, operator-reviewed kickflip replay diagnostic; never training data."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from trueskate_ai.collection.tap_timing_calibration import detect_tap_onset, fit_two_anchor_timeline
from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
from trueskate_ai.data.control_hitboxes import segment_is_safe
from trueskate_ai.research.curve_measurement import _decode_source_frames, read_native_frames
from trueskate_ai.research.curve_protocol import digest, save_new
from trueskate_ai.research.curved_audit import reset_payload
from trueskate_ai.research.linear_speed_probe import deadline_sleep

EXPERIMENT = 'KICKFLIP-REPEATABILITY-20261009'
# Operator (2026-10-09): keep iterating until the kickflip is reproduced.
SETUP_LIMIT = 200
REPEATS = 20
WDA_REVISION = 'ae50404aac12d9f8c41f6c3fa8776e97975eaef5'
SIZE = (414, 896)
FPS = 60
STOP_S = 59.
LATE_S = .1
# Operator's filmed kickflip (tmp/Example Kickflip.MP4, 60 fps), approximated with
# straight, constant-speed strokes; normalized points; t = 0 at pop touch-down.
# The pop runs from mid-board down off the tail; the flick from the board's upper
# half right and slightly down; the catch is a stationary hold on the upper half.
# The historical 20 ms push is in the unreliable flicker range (LINEAR-LENGTH-20261003)
# and failed to move the board in trial 19; use the operator's filmed 0.13 s push,
# which ended 0.37 s before the pop.
KICKFLIP = dict(push_start=[.813, .27], push_end=[.74, .522], push_s=.13, push_to_pop_s=.37,
                pop_start=[.505, .56], pop_end=[.555, .815], pop_s=.075, flick_gap_s=.13,
                flick_start=[.537, .51], flick_end=[.715, .565], flick_s=.1, catch_gap_s=.43,
                catch_point=[.5, .505], catch_s=.52)
# One XCTest record per stroke (GESTURES.md hard rule): multi-path records conjoin
# strokes and anchors draw lines (CURVE-AUDIT-20261003, attempts 08-12). Records
# cannot start within ~0.25 s of the previous gesture's end, so the flick is sent
# as soon as the pop returns; flick_gap_s and catch_gap_s are filmed targets.
# Strokes use WDA's direct endpoint (no W3C preparation or stability wait).
DIRECT_RETURN_S = .25  # Typical end-of-gesture to endpoint return; sets the catch wait.
VARIANTS = ({}, dict(pop_s=.05), dict(pop_s=.1), dict(flick_s=.06), dict(flick_s=.14),
            dict(catch_gap_s=.3), dict(catch_gap_s=.55),
            # Trial 14 (0.06 s flick) landed a HARD FLIP: try removing the flick's downward
            # slant (a likely shove-it input when the flick arrives ~0.3 s after the pop).
            dict(flick_s=.06, flick_end=[.715, .51]), dict(flick_s=.06, flick_end=[.715, .48]),
            dict(flick_s=.04, flick_end=[.715, .51]),
            # Operator: the pop should be quick and shorter. Ending just past the tail tip
            # (~0.70) removes post-pop stroke time that only delays the flick.
            dict(pop_end=[.52, .73], pop_s=.05, flick_s=.06, flick_end=[.715, .51]),
            dict(pop_end=[.52, .73], pop_s=.04, flick_s=.06, flick_end=[.715, .51]),
            dict(pop_end=[.52, .73], pop_s=.05, flick_s=.06),
            # Trials 17-18 (candidate_10) landed KICKFLIPs that the operator calls rocket
            # flips: the board flips nose-up. Angle the flick up the deck to level it.
            dict(flick_s=.04, flick_end=[.715, .46]), dict(flick_s=.04, flick_end=[.715, .42]),
            # Operator now allows 2-3 point flicks. Their filmed flick presses almost still on
            # the deck for ~50 ms (a likely levelling press), then snaps right in ~50 ms.
            dict(flick_points=[[.537, .51], [.545, .517], [.715, .51]], flick_durations_s=[.05, .04]),
            dict(flick_points=[[.537, .51], [.545, .517], [.715, .565]], flick_durations_s=[.05, .05]),
            dict(flick_points=[[.537, .51], [.545, .517], [.715, .51]], flick_durations_s=[.08, .04]),
            # Operator: the flick travels a curved, downward-ish arc before going straight
            # outward. Slower down-right first segment, then a fast outward segment.
            dict(flick_points=[[.537, .51], [.57, .54], [.715, .56]], flick_durations_s=[.06, .04]),
            dict(flick_points=[[.537, .51], [.57, .54], [.715, .56]], flick_durations_s=[.04, .03]),
            dict(flick_points=[[.537, .51], [.56, .545], [.715, .55]], flick_durations_s=[.05, .04]))
IMPLEMENTATION_PATHS = (
    'scripts/collection/probe_kickflip_repeatability.py',
    'src/trueskate_ai/research/kickflip_repeatability.py',
    'src/trueskate_ai/research/curve_measurement.py',
    'src/trueskate_ai/research/curved_audit.py',
    'src/trueskate_ai/research/linear_speed_probe.py',
    'src/trueskate_ai/sim/gestures.py', 'src/trueskate_ai/sim/touch_actions.py',
    'src/trueskate_ai/sim/device.py', 'src/trueskate_ai/data/control_hitboxes.py',
    'src/trueskate_ai/collection/wda_action_timing.py',
    'src/trueskate_ai/collection/xctest_capture.py',
    'src/trueskate_ai/collection/tap_timing_calibration.py',
    'src/trueskate_ai/collection/gameplay_filter.py',
    'src/trueskate_ai/collection/scene_settle.py',
)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seal(value):
    value = copy.deepcopy(value)
    value['sha256'] = digest(value)
    return value


def verify_seal(value):
    body = {k: v for k, v in value.items() if k != 'sha256'}
    if value.get('sha256') != digest(body):
        raise ValueError('frozen artifact hash mismatch')
    return value


def safe_scaled_path(points):
    if any(len(p) != 2 or not all(np.isfinite(v) and 0 <= v <= 1 for v in p) for p in points):
        raise ValueError('normalized finite points required')
    # Check both source and the actual Selenium-quantized path.
    scaled = [(x * SIZE[0], y * SIZE[1]) for x, y in points]
    quantized = [(int(x) / SIZE[0], int(y) / SIZE[1]) for x, y in scaled]
    for path in (points, quantized):
        if any(not segment_is_safe(a, b) for a, b in zip(path, path[1:])):
            raise ValueError('gesture crosses protected controls')
    return scaled


def direct_stroke(name, points, durations_s):
    """One contact for WDA's /wda/perform_trick_gestures: straight segments, one time each.

    Repeating a point holds it. The operator allows two or three points for the flick.
    """
    if len(points) != len(durations_s) + 1 or not 2 <= len(points) <= 3:
        raise ValueError('a stroke needs two or three points and one duration per segment')
    if any(not .005 <= d <= 1. for d in durations_s):
        raise ValueError('invalid stroke duration')
    scaled = safe_scaled_path(points)
    ms = [int(round(d * 1000)) for d in durations_s]
    waypoints = [dict(x=int(scaled[0][0]), y=int(scaled[0][1]))]
    waypoints += [dict(x=int(x), y=int(y), duration_ms=d) for (x, y), d in zip(scaled[1:], ms)]
    payload = {'gestures': [{'waypoints': waypoints}]}
    return dict(name=name, transport='wda_perform_trick_gestures', points=[list(p) for p in points],
                encoded_duration_s=sum(ms) / 1000, payload=payload, payload_sha256=digest(payload))


def prepare_manifest(repo, previous_setup=None, superseded_reason=None):
    repo = Path(repo)
    candidates, rejected = [], []
    for index, change in enumerate(VARIANTS, 1):
        candidate_id = f'candidate_{index:02d}'
        params = {**KICKFLIP, **change}
        try:
            flick = (direct_stroke('flick', params['flick_points'], params['flick_durations_s'])
                     if 'flick_points' in params else
                     direct_stroke('flick', [params['flick_start'], params['flick_end']], [params['flick_s']]))
            push = direct_stroke('push', [params['push_start'], params['push_end']], [params['push_s']])
            contacts = [push, direct_stroke('pop', [params['pop_start'], params['pop_end']], [params['pop_s']]),
                        flick, direct_stroke('catch', [params['catch_point']] * 2, [params['catch_s']])]
            candidates.append(seal(dict(id=candidate_id, seed='operator-filmed kickflip 2026-10-09',
                parameters=params, varied=sorted(change), contacts=contacts,
                catch_wait_s=round(max(0., params['catch_gap_s'] - DIRECT_RETURN_S), 3),
                pop_wait_s=round(max(0., params['push_to_pop_s'] - DIRECT_RETURN_S), 3),
                training_admission=False)))
        except ValueError as exc:
            rejected.append(dict(id=candidate_id, parameters=params, error=str(exc)))
    history, spent = None, 0
    if previous_setup is not None:
        previous_setup = Path(previous_setup)
        previous = load_manifest(previous_setup)
        if (previous_setup / 'approval.json').exists() or list((previous_setup / 'repeats').glob('trial_*')):
            raise ValueError('reviewed or repeatability runs cannot migrate')
        if previous['experiment'] != EXPERIMENT or previous['wda_revision'] != WDA_REVISION:
            raise ValueError('predecessor procedure changed')
        # A changed recipe set or execution code supersedes the old procedure: its gameplay
        # attempts still spend the cap. Only an identical sealed recipe can later count them,
        # and only through the explicit operator gate override.
        implementation = {p: file_sha(repo / p) for p in IMPLEMENTATION_PATHS}
        superseded = ([c['sha256'] for c in previous['candidates']] != [c['sha256'] for c in candidates]
                      or previous['implementation_hashes'] != implementation)
        if superseded and not (superseded_reason or '').strip():
            raise ValueError('predecessor procedure changed; an explicit superseded-procedure reason is required')
        classifications = {}
        for trial in setup_trials(previous_setup):
            execution = read_json(trial / 'execution.json')
            gameplay = any(e.get('kind') not in ('control', 'reset') for e in execution.get('events', []))
            if gameplay and not superseded:
                raise ValueError('only failures before gameplay can migrate')
            video = execution.get('video')
            movie = trial / 'original.mov'
            if (video is not None and (not movie.is_file() or video.get('n_bytes') != movie.stat().st_size)) or (video is None and movie.exists()):
                raise ValueError('only failures with consistent recording retrieval can migrate')
            if execution.get('error'):
                classifications[trial.name] = 'technical failure ' + ('during' if gameplay else 'before') + ' gameplay'
            elif not gameplay:
                raise ValueError('only failures before gameplay can migrate')
            else:
                admission = read_json(trial / 'admission.json') if (trial / 'admission.json').exists() else {}
                assessment = read_json(trial / 'assessment.json') if (trial / 'assessment.json').exists() else {}
                if not admission.get('accepted') or assessment.get('video_sha256') != file_sha(movie):
                    raise ValueError('completed gameplay must be admitted and assessed before it can migrate')
                classifications[trial.name] = f"superseded-procedure outcome: {assessment['trick']} / {assessment['status']}"
        spent = setup_attempt_count(previous_setup)
        history = dict(source_root=str(previous_setup), manifest_sha256=previous['sha256'],
            attempts=spent, trials=classifications, superseded_reason=superseded_reason if superseded else None,
            classification='retained evidence; never trick outcomes for these candidates',
            files={str(Path('history/predecessor') / p.relative_to(previous_setup)): file_sha(p)
                   for p in previous_setup.rglob('*') if p.is_file()})
    return seal(dict(experiment=EXPERIMENT, device='iPhone_XR', park='Workshop', size=list(SIZE),
        fps=FPS, wda_revision=WDA_REVISION, setup_limit=SETUP_LIMIT, repeats=REPEATS,
        candidates=candidates, rejected=rejected,
        calibration='centre controls at 1.5/30/57 seconds; start/end fit; held-out middle',
        implementation_hashes={p: file_sha(repo / p) for p in IMPLEMENTATION_PATHS},
        controls_reset_before=True, prior_setup_attempts=spent, prior_setup=history, training_admission=False))


def load_manifest(root):
    manifest = verify_seal(read_json(Path(root) / 'manifest.json'))
    for candidate in manifest['candidates']:
        verify_seal(candidate)
    history = manifest.get('prior_setup')
    if history:
        if history['attempts'] != manifest['prior_setup_attempts']:
            raise ValueError('predecessor budget mismatch')
        for relative, expected in history['files'].items():
            path = (Path(root) / relative).resolve()
            path.relative_to(Path(root).resolve() / 'history/predecessor')
            if file_sha(path) != expected:
                raise ValueError('predecessor evidence changed')
    return manifest


def verify_implementation(manifest, repo):
    expected = manifest['implementation_hashes']
    if expected != {p: file_sha(Path(repo) / p) for p in IMPLEMENTATION_PATHS}:
        raise ValueError('execution implementation changed; requires newly reviewed manifest')


def setup_trials(root):
    return sorted((Path(root) / 'setup').glob('trial_*'))


def setup_attempt_count(root):
    return load_manifest(root).get('prior_setup_attempts', 0) + len(setup_trials(root))


def reserve_setup(root, candidate_id):
    root = Path(root)
    manifest = load_manifest(root)
    if (root / 'approval.json').exists():
        raise ValueError('setup is frozen after review approval')
    trials = setup_trials(root)
    spent = manifest.get('prior_setup_attempts', 0) + len(trials)
    if spent >= SETUP_LIMIT:
        raise ValueError(f'{SETUP_LIMIT}-attempt setup limit reached; operator steering required')
    if any(not (p / 'admission.json').exists() or not read_json(p / 'admission.json')['accepted'] for p in trials):
        raise ValueError('a technical failure or unvalidated attempt requires resolution; no replacement')
    candidate = next((c for c in manifest['candidates'] if c['id'] == candidate_id), None)
    if candidate is None:
        raise ValueError('unknown candidate')
    trial = root / 'setup' / f'trial_{spent + 1:02d}'
    trial.mkdir(parents=True, exist_ok=False)  # Reservation counts even a failed start.
    save_new(trial / 'reservation.json', dict(candidate_id=candidate_id, candidate_sha256=candidate['sha256']))
    return trial, candidate


def marker():
    return {'actions': [dict(type='pointer', id='control', parameters={'pointerType': 'touch'}, actions=[
        dict(type='pointerMove', x=207, y=448, duration=0, origin='viewport'),
        dict(type='pointerDown', button=0), dict(type='pause', duration=50), dict(type='pointerUp', button=0)])]}


def run_trial(*, out, candidate, recorder, timing, perform, guard, settle, revision,
              context, clock=time.monotonic, sleep=deadline_sleep, epoch=time.time, foreground_guard=None,
              perform_direct=None):
    """One start/one retrieval; exact separate contacts, including on interruption."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    verify_seal(candidate)
    save_new(out / 'planned.json', dict(candidate=candidate, context=context, fps=FPS,
             stop_s=STOP_S, training_admission=False))
    events, failures = [], []
    started = False
    video = report = None
    def fail(phase, exc):
        failures.append(dict(phase=phase, error=f'{type(exc).__name__}: {exc}'))
    try:
        guard()
        timing.start()
        start_call = clock()
        recorder.start()
        started = True
        origin = clock()
        if origin - start_call > 1:
            raise RuntimeError('recorder start exceeded reserved lead')
        deadline = start_call + 60.
        def remaining():
            if clock() >= deadline:
                raise RuntimeError('one-minute recording deadline exceeded')
        def call(kind, payload, target, *, role=None, encoded_duration_s=.05, direct=False, guarded=True):
            remaining()
            if target >= deadline:
                raise RuntimeError('command target exceeds recording deadline')
            # Guards include a fresh screenshot; reserve time for them before the
            # submission deadline instead of adding their latency to every gap.
            fast = foreground_guard is not None and (kind == 'reset' or kind == 'gesture' and role == 'pop')
            if guarded:
                sleep(max(0., target - (.25 if fast else 1.5) - clock()))
                (foreground_guard if fast else guard)()
            sleep(max(0., target - clock()))
            remaining()
            lateness = clock() - target
            if lateness > LATE_S:
                raise RuntimeError(f'{kind} schedule overrun: {lateness:.6f}s')
            event = dict(kind=kind, role=role, intended_s=target-origin,
                         lateness_s=lateness, encoded_duration_s=encoded_duration_s,
                         payload=payload, payload_sha256=digest(payload), instrumented=not direct,
                         call_start_monotonic_s=clock(), call_start_epoch_s=epoch(), success=False)
            events.append(event)
            response = (perform_direct if direct else perform)(payload)
            event.update(response=response, call_end_monotonic_s=clock(), call_end_epoch_s=epoch())
            if isinstance(response, dict) and (response.get('error') or
                    isinstance(response.get('value'), dict) and response['value'].get('error')):
                raise RuntimeError('gesture response reports an error')
            event['success'] = True
            remaining()
            return event
        def reset(slot, following_slot):
            event = call('reset', reset_payload(), origin + slot)
            reserve = min(10., origin + following_slot - clock() - 1.6)
            if reserve <= 0:
                raise RuntimeError('reset exhausted its settling window')
            result = settle(reserve)
            event['settle'] = result.summary()
            if not result.settled or clock() >= origin + following_slot:
                raise RuntimeError('reset failed to settle before next command')
        call('control', marker(), origin + 1.5, role='start')
        reset(3., 14.)
        push, pop, flick, catch = candidate['contacts']
        call('gesture', push['payload'], origin + 14., role='push',
             encoded_duration_s=push['encoded_duration_s'], direct=True)
        # The full guard ran just before the push; a guard here would delay the pop.
        call('gesture', pop['payload'], clock() + candidate['pop_wait_s'], role='pop',
             encoded_duration_s=pop['encoded_duration_s'], direct=True, guarded=False)
        # No guards between strokes: each would add ~0.2 s to an already late flick.
        call('gesture', flick['payload'], clock(), role='flick',
             encoded_duration_s=flick['encoded_duration_s'], direct=True, guarded=False)
        call('gesture', catch['payload'], clock() + candidate['catch_wait_s'], role='catch',
             encoded_duration_s=catch['encoded_duration_s'], direct=True, guarded=False)
        guard()
        if clock() >= origin + 20.:
            raise RuntimeError('gameplay exceeded reserved observation window')
        reset(22., 30.)
        call('control', marker(), origin + 30., role='middle')
        reset(49., 57.)
        call('control', marker(), origin + 57., role='end')
        guard()
        sleep(max(0., start_call + STOP_S - clock()))
        if foreground_guard is not None:
            foreground_guard()
        remaining()
    except BaseException as exc:
        fail('execution', exc)
    finally:
        if started:
            try:
                result = recorder.stop_and_save(out / 'original.mov')
                video = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(result).items()}
            except BaseException as exc:
                fail('recorder_stop', exc)
        if timing.active:
            try:
                report = timing.stop()
                save_new(out / 'wda-timing.json', report)
            except BaseException as exc:
                fail('timing_stop', exc)
        if not failures:
            try:
                validate_action_timing_report(report, expected_revision=revision,
                                              expected_count=sum(e['instrumented'] for e in events))
                if len(events) != 10:
                    raise ValueError('incomplete fixed trial')
            except BaseException as exc:
                fail('timing_validation', exc)
        execution = dict(events=events, video=video, failures=failures,
                         error=failures[0]['error'] if failures else None,
                         candidate_sha256=candidate['sha256'], context=context, training_admission=False)
        save_new(out / 'execution.json', execution)
    if failures:
        raise RuntimeError(execution['error'])
    return execution


def admit_trial(out, revision):
    out = Path(out)
    try:
        execution = read_json(out / 'execution.json')
        if execution['error']:
            raise ValueError(execution['error'])
        instrumented = [e for e in execution['events'] if e.get('instrumented', True)]
        records = validate_action_timing_report(read_json(out / 'wda-timing.json'),
                    expected_revision=revision, expected_count=len(instrumented))
        video = out / 'original.mov'
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_streams', '-show_frames', '-show_entries',
            'stream=width,height:frame=best_effort_timestamp_time', '-of', 'json', str(video)]))
        pts = np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']])
        stream = probe['streams'][0]
        if (stream['width'], stream['height']) != (828, 1792):
            raise ValueError('unexpected XR native video dimensions')
        if len(pts) < 2 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
            raise ValueError('missing or unordered source PTS')
        native = float(np.median(np.diff(pts)))
        if abs(native - 1 / FPS) > .005 or pts[-1] - pts[0] > 60.:
            raise ValueError('unexpected native frame interval/recording duration')
        from trueskate_ai.collection.gameplay_filter import is_menu_frame, is_editor_frame
        def inspect(frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if is_editor_frame(rgb) or is_menu_frame(rgb, allow_idle_navigation=True):
                raise ValueError('gameplay contamination in original video')
        _decode_source_frames(video, pts, keep=False, inspect=inspect)
        onsets, stamps = {}, {}
        started = execution['video']['started_at_epoch_s']
        for event, record in zip(instrumented, records):
            if event['kind'] != 'control':
                continue
            approximate = record['submitted_to_ios']['epoch_s'] - started
            frames, times, indices = read_native_frames(video, pts, start=approximate-.7, end=approximate+.7)
            detection = detect_tap_onset(frames, times, point_xy=(.5, .5), command_s=approximate)
            if detection is None:
                raise ValueError(f"{event['role']} control onset not observable")
            onsets[event['role']] = detection.onset_s
            stamps[event['role']] = record['submitted_to_ios']['monotonic_s']
        fit = fit_two_anchor_timeline(stamps['start'], onsets['start'], stamps['end'], onsets['end'], min_anchor_span_s=55.)
        residual = onsets['middle'] - fit.video_time_s(stamps['middle'])
        if not onsets['start'] < onsets['middle'] < onsets['end'] or abs(residual) > 2 * native:
            raise ValueError('held-out middle calibration failed')
        # Direct strokes have no WDA timing record. Map their rig-clock call starts
        # with the median rig-to-WDA offset of the instrumented requests (approximate).
        offset = float(np.median([r['request_entered']['monotonic_s'] - e['call_start_monotonic_s']
                                  for e, r in zip(instrumented, records)]))
        by_event = {id(e): r for e, r in zip(instrumented, records)}
        gestures = []
        for event in execution['events']:
            if event['kind'] != 'gesture':
                continue
            record = by_event.get(id(event))
            if record is not None:
                stamp = record['submitted_to_ios']['monotonic_s']
                gestures.append(dict(role=event['role'], instrumented=True, submitted_monotonic_s=stamp,
                    video_start_s=fit.video_time_s(stamp), encoded_duration_s=event['encoded_duration_s'],
                    ios_callback_latency_s=record['ios_completion_callback']['monotonic_s']-stamp,
                    call_lateness_s=event['lateness_s']))
            else:
                stamp = event['call_start_monotonic_s'] + offset
                gestures.append(dict(role=event['role'], instrumented=False, submitted_monotonic_s=stamp,
                    video_start_s=fit.video_time_s(stamp), encoded_duration_s=event['encoded_duration_s'],
                    call_duration_s=event['call_end_monotonic_s'] - event['call_start_monotonic_s'],
                    call_lateness_s=event['lateness_s'], timing_source='rig clock + median WDA offset'))
        result = dict(accepted=True, fit=asdict(fit), control_onsets_s=onsets, rig_to_wda_offset_s=offset,
            heldout_middle_error_s=residual, native_frame_s=native, frame_count=len(pts),
            max_frame_gap_s=float(np.max(np.diff(pts))), original_pts_s=pts.tolist(),
            video_sha256=file_sha(video), gestures=gestures, training_admission=False)
    except Exception as exc:
        save_new(out / 'admission.json', dict(accepted=False, error=str(exc), training_admission=False))
        raise
    save_new(out / 'admission.json', result)
    return result


def trial_number(trial):
    return int(Path(trial).name.split('_')[1])


def qualified_trials(root, candidate_id, minimum=3):
    """First `minimum` attempts of the candidate; all of its attempts must be assessed.

    The default gate is three attempts with at least two landed KICKFLIPs. An operator
    override (minimum < 3) also counts preserved predecessor attempts of the *identical*
    sealed recipe, and requires every selected attempt to be a landed KICKFLIP.
    """
    root = Path(root)
    trials = []
    if minimum >= 3:
        pool = [(t, read_json(t / 'reservation.json')['candidate_id'] == candidate_id) for t in setup_trials(root)]
    else:
        candidate = next(c for c in load_manifest(root)['candidates'] if c['id'] == candidate_id)
        everywhere = sorted([*setup_trials(root), *root.glob('history/**/setup/trial_*')], key=trial_number)
        pool = [(t, read_json(t / 'reservation.json').get('candidate_sha256') == candidate['sha256'])
                for t in everywhere]
    for trial, matches in pool:
        if matches:
            if not (trial / 'assessment.json').exists():
                raise ValueError('all candidate attempts must be assessed; do not skip failures')
            admission = read_json(trial / 'admission.json')
            assessment = read_json(trial / 'assessment.json')
            if not admission['accepted'] or assessment['video_sha256'] != file_sha(trial / 'original.mov'):
                raise ValueError('assessment is not tied to admitted source video')
            trials.append((trial, assessment))
    if minimum < 1 or len(trials) < minimum:
        raise ValueError(f'{minimum} assessed candidate attempt(s) required before review')
    # Show the first attempts, not a cherry-picked group of later successful attempts.
    selected = trials[:minimum]
    landed = sum(a['trick'] == 'KICKFLIP' and a['status'] == 'landed' for _, a in selected)
    if landed < (2 if minimum >= 3 else minimum):
        raise ValueError('candidate lacks the required landed kickflips in its first assessed attempts')
    return [p for p, _ in selected]


def review_candidate(root, candidate_id, *, minimum=3, override_reason=None):
    root = Path(root)
    if minimum < 3 and not (override_reason or '').strip():
        raise ValueError('an operator gate override needs an explicit reason')
    manifest = load_manifest(root)
    candidate = next(c for c in manifest['candidates'] if c['id'] == candidate_id)
    trials = qualified_trials(root, candidate_id, minimum)
    destination = root / 'review' / candidate_id
    destination.mkdir(parents=True, exist_ok=False)
    clips, sources = [], []
    for index, trial in enumerate(trials, 1):
        admission = read_json(trial / 'admission.json')
        start = admission['gestures'][0]['video_start_s'] - 1.
        end = admission['gestures'][-1]['video_start_s'] + admission['gestures'][-1]['encoded_duration_s'] + 5.
        clip = destination / f'attempt_{index:02d}.mov'
        subprocess.run(['ffmpeg', '-v', 'error', '-n', '-ss', str(start), '-i', str(trial / 'original.mov'),
            '-t', str(end-start), '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', str(clip)], check=True)
        clips.append(clip)
        sources.append(dict(trial=str(trial.relative_to(root)), video_sha256=admission['video_sha256'],
                            start_s=start, end_s=end, assessment=read_json(trial / 'assessment.json')))
    listing = destination / 'clips.txt'
    listing.write_text(''.join("file '" + str(p.resolve()).replace("'", "'\\''") + "'\n" for p in clips))
    preview = destination / 'preview.mov'
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-f', 'concat', '-safe', '0', '-i', str(listing),
                    '-c', 'copy', str(preview)], check=True)
    review = seal(dict(candidate_id=candidate_id, candidate_sha256=candidate['sha256'],
        manifest_sha256=manifest['sha256'], context_sha256=digest(read_json(root / 'context.json')),
        sources=sources, preview=str(preview.relative_to(root)), preview_sha256=file_sha(preview),
        gate=dict(minimum=minimum, override_reason=override_reason if minimum < 3 else None),
        operator_approval_required=True))
    save_new(destination / 'review.json', review)
    return preview


def approve_review(root, candidate_id, statement):
    root = Path(root)
    if not statement.strip():
        raise ValueError('explicit operator approval statement required')
    review = verify_seal(read_json(root / 'review' / candidate_id / 'review.json'))
    if file_sha(root / review['preview']) != review['preview_sha256']:
        raise ValueError('review movie changed')
    gate = review.get('gate', dict(minimum=3))
    selected = qualified_trials(root, candidate_id, gate['minimum'])
    if [str(t.relative_to(root)) for t in selected] != [source['trial'] for source in review['sources']]:
        raise ValueError('reviewed attempts are no longer the first qualifying attempts')
    save_new(root / 'approval.json', seal(dict(review=review, operator_statement=statement,
        approved_at_epoch_s=time.time(), repeats=REPEATS, training_admission=False)))


def approved_candidate(root):
    root = Path(root)
    manifest = load_manifest(root)
    approval = verify_seal(read_json(root / 'approval.json'))
    review = verify_seal(approval['review'])
    candidate = next(c for c in manifest['candidates'] if c['id'] == review['candidate_id'])
    if (review['manifest_sha256'] != manifest['sha256'] or review['candidate_sha256'] != candidate['sha256']
            or review['context_sha256'] != digest(read_json(root / 'context.json'))
            or file_sha(root / review['preview']) != review['preview_sha256']
            or approval['repeats'] != REPEATS or not approval['operator_statement'].strip()):
        raise ValueError('approved recipe/settings/review changed')
    for source in review['sources']:
        if file_sha(root / source['trial'] / 'original.mov') != source['video_sha256']:
            raise ValueError('reviewed original video changed')
    return candidate


DIVERGENCE_WINDOW_S = (-.5, 4.)
# 104x224 grayscale; hide the HUD: top 15 %, bottom 10 % and the speedometer corner.
DIVERGENCE_MASK = np.ones((224, 104), bool)
DIVERGENCE_MASK[:34], DIVERGENCE_MASK[201:], DIVERGENCE_MASK[179:, 78:] = False, False, False


def divergence_frame(frame):
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (104, 224), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def divergence_curve(reference, frames):
    """Mean absolute masked difference between paired (already aligned) processed frames."""
    return np.array([float(np.abs(a - b)[DIVERGENCE_MASK].mean()) for a, b in zip(reference, frames)])


def divergence_onset(offsets, curve, *, factor=3., minimum=2., sustain=3):
    """First offset after the push where the difference stays above the pre-push noise floor."""
    offsets, curve = np.asarray(offsets), np.asarray(curve)
    before = curve[offsets < 0]
    floor = float(np.percentile(before, 95)) if len(before) else 0.
    threshold = max(factor * floor, floor + minimum)
    above = (curve > threshold) & (offsets >= 0)
    for i in range(len(curve) - sustain + 1):
        if above[i:i + sustain].all():
            return float(offsets[i]), floor, threshold
    return None, floor, threshold


def processed_window(trial, admission):
    """Processed native frames around the push, keyed by offset from the push's video start."""
    pts = np.array(admission['original_pts_s'])
    push = next(g for g in admission['gestures'] if g['role'] == 'push')['video_start_s']
    start, end = push + DIVERGENCE_WINDOW_S[0], push + DIVERGENCE_WINDOW_S[1]
    frames = []
    _decode_source_frames(Path(trial) / 'original.mov', pts, start=start, end=end, keep=False,
                          inspect=lambda frame: frames.append(divergence_frame(frame)))
    offsets = [float(t - push) for t in pts if start <= t <= end]
    if len(offsets) != len(frames):
        raise ValueError('divergence window decode count differs from source PTS')
    return np.array(offsets), frames


def divergence(root, runs, native_frame_s):
    """Each admitted repeat against the first, aligned once at the push; pairs within half a frame."""
    root = Path(root)
    window = lambda run: processed_window(root / run['trial'], read_json(root / run['trial'] / 'admission.json'))
    (ref_offsets, ref_frames), results = window(runs[0]), []
    for run in runs[1:]:
        offsets, frames = window(run)  # One repeat in memory at a time.
        nearest = np.abs(ref_offsets[None, :] - offsets[:, None]).argmin(axis=1)
        keep = np.abs(ref_offsets[nearest] - offsets) <= native_frame_s / 2
        paired = [i for i in range(len(offsets)) if keep[i]]
        curve = divergence_curve([ref_frames[nearest[i]] for i in paired], [frames[i] for i in paired])
        onset, floor, threshold = divergence_onset(offsets[paired], curve)
        results.append(dict(trial=run['trial'], offsets_s=offsets[paired].tolist(), mean_abs_difference=curve.tolist(),
                            onset_s=onset, noise_floor=floor, threshold=threshold))
    return dict(reference=runs[0]['trial'], alignment='push video start', window_s=list(DIVERGENCE_WINDOW_S),
                metric='mean absolute grayscale difference, 104x224, HUD masked', runs=results)


def report(root):
    root = Path(root)
    trials = sorted((root / 'repeats').glob('trial_*'))
    summary = dict(experiment=EXPERIMENT, expected_repeats=REPEATS, attempted=len(trials),
                   admitted=0, technical_failures=[], outcomes={}, runs=[], training_admission=False,
                   limitation='Whole-system repeatability; no proof of intrinsic game randomness. '
                   'Durations are encoded requests; callback latency and submission gaps are WDA measurements, '
                   'not physical contact boundaries. Visual divergence requires source-video review.')
    for trial in trials:
        if not (trial / 'admission.json').exists() or not read_json(trial / 'admission.json')['accepted']:
            summary['technical_failures'].append(str(trial.relative_to(root)))
            continue
        admission = read_json(trial / 'admission.json')
        if file_sha(trial / 'original.mov') != admission['video_sha256']:
            raise ValueError('admitted source video changed')
        summary['admitted'] += 1
        gestures = admission['gestures']
        assessment = read_json(trial / 'assessment.json') if (trial / 'assessment.json').exists() else None
        if assessment and assessment['video_sha256'] != admission['video_sha256']:
            raise ValueError('assessment belongs to another video')
        outcome = f"{assessment['trick']} / {assessment['status']}" if assessment else 'unassessed'
        summary['outcomes'][outcome] = summary['outcomes'].get(outcome, 0) + 1
        summary['runs'].append(dict(trial=str(trial.relative_to(root)), gestures=gestures,
            submission_gaps_s={f"{a['role']}_to_{b['role']}": b['submitted_monotonic_s']-a['submitted_monotonic_s']
                               for a, b in zip(gestures, gestures[1:])}, assessment=assessment,
            heldout_middle_error_s=admission['heldout_middle_error_s'], native_frame_s=admission['native_frame_s'],
            max_frame_gap_s=admission['max_frame_gap_s']))
    summary['complete'] = len(trials) == REPEATS and summary['admitted'] == REPEATS
    summary['timing_distributions'] = {}
    for key in ('push_to_pop', 'pop_to_flick', 'flick_to_catch'):
        values = [r['submission_gaps_s'][key] for r in summary['runs']]
        if values:
            summary['timing_distributions'][key] = dict(count=len(values), min_s=min(values),
                median_s=float(np.median(values)), p95_s=float(np.percentile(values, 95)), max_s=max(values))
    return summary


def render_report(root, out):
    """Inspection movies only: source PTS remain the measurement authority."""
    root, out = Path(root), Path(out)
    summary = report(root)
    out.mkdir(parents=True, exist_ok=False)
    save_new(out / 'summary.json', summary)
    snapshots, comparisons = [], []
    reference = None
    for run in summary['runs']:
        trial = root / run['trial']
        admission = read_json(trial / 'admission.json')
        pts = np.array(admission['original_pts_s'])
        gestures = {g['role']: g for g in admission['gestures']}
        flick_end = gestures['flick']['video_start_s'] + gestures['flick']['encoded_duration_s']
        targets = {f'before_{role}': g['video_start_s'] - admission['native_frame_s']
                   for role, g in gestures.items()}
        targets.update({f'after_flick_{offset:g}s': flick_end + offset for offset in (.1, .25, .5, 1., 2., 5.)})
        for label, target in targets.items():
            index = int(np.argmin(abs(pts-target)))
            images, times, indices = read_native_frames(trial / 'original.mov', pts, start=pts[index], end=pts[index])
            image = out / f'{trial.name}_{label}.png'
            if len(images) != 1 or not cv2.imwrite(str(image), images[0]):
                raise ValueError('source snapshot extraction failed')
            snapshots.append(dict(trial=run['trial'], phase=label, target_s=target, source_frame=index,
                source_pts_s=float(pts[index]), timing_difference_s=float(pts[index]-target),
                within_half_frame=abs(float(pts[index]-target)) <= admission['native_frame_s']/2 + 1e-9,
                image=image.name))
        start = gestures['push']['video_start_s'] - 1.
        if reference is None:
            reference = (trial, start)
            continue
        reference_trial, reference_start = reference
        movie = out / f'{reference_trial.name}_vs_{trial.name}.mov'
        # Only the first push aligns videos. Pop/flick/catch/landing retain their real timing.
        filters = (f'[0:v]trim=start={reference_start}:duration=11,setpts=PTS-STARTPTS,scale=414:896[left];'
                   f'[1:v]trim=start={start}:duration=11,setpts=PTS-STARTPTS,scale=414:896[right];'
                   '[left][right]hstack=inputs=2:shortest=1[v]')
        subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(reference_trial / 'original.mov'),
            '-i', str(trial / 'original.mov'), '-filter_complex', filters, '-map', '[v]', '-an',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', str(movie)], check=True)
        comparisons.append(dict(movie=movie.name, left=str(reference_trial.relative_to(root)),
            right=run['trial'], alignment='first push only', duration_s=11.,
            rendered_for_inspection=True, use_original_pts_for_measurement=True))
    save_new(out / 'visual-provenance.json', dict(snapshots=snapshots, comparisons=comparisons))
    onsets = {}
    if len(summary['runs']) > 1:
        result = divergence(root, summary['runs'], float(np.median([r['native_frame_s'] for r in summary['runs']])))
        save_new(out / 'divergence.json', result)
        onsets = {r['trial']: r['onset_s'] for r in result['runs']}
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(figsize=(10, 5))
        for r in result['runs']:
            axes.plot(r['offsets_s'], r['mean_abs_difference'], lw=1, alpha=.7)
        axes.axvline(0, color='k', lw=.8)
        axes.set(xlabel='seconds from push start (aligned once)', ylabel='mean |difference| vs reference',
                 title=f"{len(result['runs'])} repeats vs {result['reference']}")
        figure.tight_layout()
        figure.savefig(out / 'divergence.png', dpi=120)
        plt.close(figure)
    lines = [f'# {EXPERIMENT}', '',
             f"Attempts: {summary['attempted']}/{REPEATS}; admitted: {summary['admitted']}; complete: {summary['complete']}.",
             '', '## Trick and landing outcomes', '']
    lines += [f'- {key}: {count}' for key, count in summary['outcomes'].items()]
    lines += ['', '## Submission timing', '']
    for key, values in summary['timing_distributions'].items():
        lines.append(f"- {key}: min {values['min_s']*1000:.2f}, median {values['median_s']*1000:.2f}, "
                     f"p95 {values['p95_s']*1000:.2f}, max {values['max_s']*1000:.2f} ms.")
    lines += ['', '## Per-repeat outcome, divergence onset and measured gaps', '',
              '| Repeat | Outcome | Divergence onset (s) | push→pop (s) | pop→flick (s) | flick→catch (s) |',
              '|---|---|---|---|---|---|']
    for run in summary['runs']:
        a, g, onset = run['assessment'], run['submission_gaps_s'], onsets.get(run['trial'])
        lines.append(f"| {run['trial']} | {a['trick'] + ' / ' + a['status'] if a else 'unassessed'} | "
                     f"{'reference' if run is summary['runs'][0] else ('none' if onset is None else f'{onset:.3f}')} | "
                     + ' | '.join(f"{g.get(k, float('nan')):.3f}" for k in ('push_to_pop', 'pop_to_flick', 'flick_to_catch'))
                     + ' |')
    lines += ['', '## Interpretation', '', summary['limitation'], '',
              'Compare the snapshots and first-push-aligned movies through the full sequence. '
              'The first admitted repeat is the comparison reference, not the most similar or successful run. '
              'Record visible divergence in an accompanying analysis; these renders do not automatically grade board pose.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
