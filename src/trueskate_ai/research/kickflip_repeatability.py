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
from trueskate_ai.sim.gestures import PUSH_START, PUSH_END, PUSH_DURATION
from trueskate_ai.sim.touch_actions import build_curved_drag, easing_to_segment_durations, make_touch_pointer

EXPERIMENT = 'KICKFLIP-REPEATABILITY-20261009'
SETUP_LIMIT = 24
REPEATS = 20
WDA_REVISION = 'ae50404aac12d9f8c41f6c3fa8776e97975eaef5'
SIZE = (414, 896)
FPS = 60
STOP_S = 59.
LATE_S = .1
# Operator-described kickflip (2026-10-09). The mined library pop missed the board
# in this camera. Pop, flick and catch need ~0.1 s spacing, so they share one
# request with a finger each, as the trick executor bundles close strokes; t = 0
# is pop touch-down. Each variant changes one uncertain number from the centre.
TAIL_TIP = (.5, .67)
BOARD_CENTRE = (.5, .5)
FLICK_END = (.8, .5)
KICKFLIP = dict(pop_hold_s=.75, pop_length_pt=150., pop_move_s=.1, pop_easing_power=.5,
                flick_gap_s=.05, flick_s=.06, catch_delay_s=.25, catch_hold_s=.3)
# 190 points is the longest pop clear of the bottom bar's protected margin.
VARIANTS = ({}, dict(pop_length_pt=100.), dict(pop_length_pt=190.), dict(pop_move_s=.06),
            dict(pop_move_s=.16), dict(flick_gap_s=0.), dict(flick_gap_s=.12), dict(flick_s=.04),
            dict(flick_s=.1), dict(catch_delay_s=.15), dict(catch_delay_s=.4), dict(pop_hold_s=.4))
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


def contact_payload(name, points, duration_s, easing_power=1.):
    if (len(points) < 2 or not np.isfinite(duration_s) or not .01 <= duration_s <= 1.
            or not np.isfinite(easing_power) or not .3 <= easing_power <= 3.):
        raise ValueError('invalid gesture duration/easing/points')
    scaled = safe_scaled_path(points)
    finger = make_touch_pointer(name)
    finger.name = name
    build_curved_drag(finger, scaled, total_duration=duration_s,
                      easing=(lambda t: t ** easing_power) if easing_power != 1. else None)
    payload = {'actions': [finger.encode()]}
    movements = payload['actions'][0]['actions']
    return dict(name=name, points=points, requested_duration_s=duration_s,
                encoded_duration_s=sum(a.get('duration', 0) for a in movements) / 1000,
                easing_power=easing_power, payload=payload, payload_sha256=digest(payload))


def trick_payload(params):
    """Pop, flick and catch fingers in one W3C payload with independent timelines."""
    pop_end = (TAIL_TIP[0], TAIL_TIP[1] + params['pop_length_pt'] / SIZE[1])
    pop_points = [(TAIL_TIP[0], TAIL_TIP[1] + (pop_end[1] - TAIL_TIP[1]) * i / 4) for i in range(5)]
    pop_up = params['pop_hold_s'] + params['pop_move_s']
    flick_down = pop_up + params['flick_gap_s']
    catch_down = flick_down + params['flick_s'] + params['catch_delay_s']
    strokes = (('pop', pop_points, 0., params['pop_hold_s'], params['pop_move_s'], params['pop_easing_power']),
               ('flick', [BOARD_CENTRE, FLICK_END], flick_down, 0., params['flick_s'], 1.),
               ('catch', [BOARD_CENTRE, BOARD_CENTRE], catch_down, params['catch_hold_s'], 0., 1.))
    sources, fingers = [], []
    for role, points, down_s, hold_s, move_s, easing_power in strokes:
        if not (0 <= down_s and 0 <= hold_s and 0 <= move_s <= 1. and .3 <= easing_power <= 3. and hold_s + move_s > 0):
            raise ValueError('invalid trick stroke timing')
        scaled = safe_scaled_path(points)
        finger = make_touch_pointer(role)
        finger.name = role  # Stable ids keep identical recipes hash-identical.
        finger.create_pointer_move(x=scaled[0][0], y=scaled[0][1], duration=0)
        if down_s > 0:
            finger.create_pause(down_s)
            # WDA drops a zero-duration move followed by a pause; re-issue it before down.
            finger.create_pointer_move(x=scaled[0][0], y=scaled[0][1], duration=0)
        finger.create_pointer_down()
        if hold_s > 0:
            finger.create_pause(hold_s)
        if move_s > 0:
            # Progress -> time easing; a power below one starts slowly and accelerates.
            durations = easing_to_segment_durations(len(scaled) - 1, int(round(move_s * 1000)),
                                                    lambda t, p=easing_power: t ** p)
            for (x, y), duration in zip(scaled[1:], durations):
                finger.create_pointer_move(x=x, y=y, duration=duration)
        finger.create_pointer_up(0)
        source = finger.encode()
        elapsed, down, up = 0, None, None
        for action in source['actions']:
            down = elapsed if action['type'] == 'pointerDown' else down
            up = elapsed if action['type'] == 'pointerUp' else up
            elapsed += action.get('duration', 0)
        sources.append(source)
        fingers.append(dict(role=role, points=points, down_s=down / 1000, up_s=up / 1000))
    longest = max(len(source['actions']) for source in sources)
    for source in sources:  # Same padding as the trick executor's combined payload.
        source['actions'] += [dict(type='pause', duration=0)] * (longest - len(source['actions']))
    payload = {'actions': sources}
    return dict(name='trick', fingers=fingers, payload=payload, payload_sha256=digest(payload),
                encoded_duration_s=max(f['up_s'] for f in fingers))


def prepare_manifest(repo, previous_setup=None, superseded_reason=None):
    repo = Path(repo)
    push = contact_payload('push', [PUSH_START, PUSH_END], PUSH_DURATION, 2.)
    candidates, rejected = [], []
    for index, change in enumerate(VARIANTS, 1):
        candidate_id = f'candidate_{index:02d}'
        params = {**KICKFLIP, **change}
        try:
            candidates.append(seal(dict(id=candidate_id, seed='operator-described kickflip 2026-10-09',
                parameters=params, varied=sorted(change), contacts=[push, trick_payload(params)],
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
        # A changed recipe set supersedes the old procedure: its gameplay attempts
        # still spend the cap but can never count as evidence for the new candidates.
        superseded = [c['sha256'] for c in previous['candidates']] != [c['sha256'] for c in candidates]
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
        push_post_response_wait_s=.48, candidates=candidates, rejected=rejected,
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
        raise ValueError('24-attempt setup limit reached; operator steering required')
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
              context, clock=time.monotonic, sleep=deadline_sleep, epoch=time.time, foreground_guard=None):
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
        def call(kind, payload, target, *, role=None, encoded_duration_s=.05):
            remaining()
            if target >= deadline:
                raise RuntimeError('command target exceeds recording deadline')
            # Guards include a fresh screenshot; reserve time for them before the
            # submission deadline instead of adding their latency to every gap.
            fast = foreground_guard is not None and (kind == 'reset' or kind == 'gesture' and role == 'trick')
            sleep(max(0., target - (.25 if fast else 1.5) - clock()))
            (foreground_guard if fast else guard)()
            sleep(max(0., target - clock()))
            remaining()
            lateness = clock() - target
            if lateness > LATE_S:
                raise RuntimeError(f'{kind} schedule overrun: {lateness:.6f}s')
            event = dict(kind=kind, role=role, intended_s=target-origin,
                         lateness_s=lateness, encoded_duration_s=encoded_duration_s,
                         payload=payload, payload_sha256=digest(payload),
                         call_start_monotonic_s=clock(), call_start_epoch_s=epoch(), success=False)
            events.append(event)
            response = perform(payload)
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
        push, trick = candidate['contacts']
        call('gesture', push['payload'], origin + 14., role='push', encoded_duration_s=push['encoded_duration_s'])
        # The historic push sleeps AFTER the blocking call, not after physical lift.
        call('gesture', trick['payload'], clock() + .48, role='trick',
             encoded_duration_s=trick['encoded_duration_s'])
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
                validate_action_timing_report(report, expected_revision=revision, expected_count=len(events))
                if len(events) != 8:
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
        records = validate_action_timing_report(read_json(out / 'wda-timing.json'),
                    expected_revision=revision, expected_count=len(execution['events']))
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
        for event, record in zip(execution['events'], records):
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
        gestures = []
        for event, record in zip(execution['events'], records):
            if event['kind'] == 'gesture':
                stamp = record['submitted_to_ios']['monotonic_s']
                gestures.append(dict(role=event['role'], submitted_monotonic_s=stamp,
                    video_start_s=fit.video_time_s(stamp), encoded_duration_s=event['encoded_duration_s'],
                    ios_callback_latency_s=record['ios_completion_callback']['monotonic_s']-stamp,
                    call_lateness_s=event['lateness_s']))
        result = dict(accepted=True, fit=asdict(fit), control_onsets_s=onsets,
            heldout_middle_error_s=residual, native_frame_s=native, frame_count=len(pts),
            max_frame_gap_s=float(np.max(np.diff(pts))), original_pts_s=pts.tolist(),
            video_sha256=file_sha(video), gestures=gestures, training_admission=False)
    except Exception as exc:
        save_new(out / 'admission.json', dict(accepted=False, error=str(exc), training_admission=False))
        raise
    save_new(out / 'admission.json', result)
    return result


def qualified_trials(root, candidate_id):
    trials = []
    for trial in setup_trials(root):
        reservation = read_json(trial / 'reservation.json')
        if reservation['candidate_id'] == candidate_id:
            if not (trial / 'assessment.json').exists():
                raise ValueError('all candidate attempts must be assessed; do not skip failures')
            admission = read_json(trial / 'admission.json')
            assessment = read_json(trial / 'assessment.json')
            if not admission['accepted'] or assessment['video_sha256'] != file_sha(trial / 'original.mov'):
                raise ValueError('assessment is not tied to admitted source video')
            trials.append((trial, assessment))
    if len(trials) < 3:
        raise ValueError('three assessed candidate attempts required before review')
    # Show the first three, not a cherry-picked group of later successful attempts.
    selected = trials[:3]
    if sum(a['trick'] == 'KICKFLIP' and a['status'] == 'landed' for _, a in selected) < 2:
        raise ValueError('candidate requires at least two landed kickflips in its first three assessed attempts')
    return [p for p, _ in selected]


def review_candidate(root, candidate_id):
    root = Path(root)
    manifest = load_manifest(root)
    candidate = next(c for c in manifest['candidates'] if c['id'] == candidate_id)
    trials = qualified_trials(root, candidate_id)
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
    qualified_trials(root, candidate_id)
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
    for key in ('push_to_trick',):
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
        fingers = {f['role']: f for f in read_json(trial / 'planned.json')['candidate']['contacts'][1]['fingers']}
        trick_start = gestures['trick']['video_start_s']
        flick_end = trick_start + fingers['flick']['up_s']
        targets = {f'before_{role}': trick_start + f['down_s'] - admission['native_frame_s']
                   for role, f in fingers.items()}
        targets['before_push'] = gestures['push']['video_start_s'] - admission['native_frame_s']
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
        # Only the first push aligns videos. The trick and landing retain their real timing.
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
    lines = [f'# {EXPERIMENT}', '',
             f"Attempts: {summary['attempted']}/{REPEATS}; admitted: {summary['admitted']}; complete: {summary['complete']}.",
             '', '## Trick and landing outcomes', '']
    lines += [f'- {key}: {count}' for key, count in summary['outcomes'].items()]
    lines += ['', '## Submission timing', '']
    for key, values in summary['timing_distributions'].items():
        lines.append(f"- {key}: min {values['min_s']*1000:.2f}, median {values['median_s']*1000:.2f}, "
                     f"p95 {values['p95_s']*1000:.2f}, max {values['max_s']*1000:.2f} ms.")
    lines += ['', '## Interpretation', '', summary['limitation'], '',
              'Compare the snapshots and first-push-aligned movies through the full sequence. '
              'The first admitted repeat is the comparison reference, not the most similar or successful run. '
              'Record visible divergence in an accompanying analysis; these renders do not automatically grade board pose.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
