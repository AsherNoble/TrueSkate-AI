"""Prepare, run and review a bounded XR1 kickflip repeatability experiment."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from trueskate_ai.research import kickflip_repeatability as experiment
from trueskate_ai.research.curve_protocol import save_new


def live_batch(root, *, candidate_id, repeatability, env_file, ready_note, settings_note):
    if not ready_note.strip() or not settings_note.strip():
        raise ValueError('operator readiness and waypoint/settings notes are required')
    if 'training-server' not in socket.gethostname():
        raise ValueError('live runs belong on training-server, not laptop USB')
    root = Path(root)
    manifest = experiment.load_manifest(root)
    experiment.verify_implementation(manifest, REPO)
    # Review and limits are checked before contacting the phone.
    if repeatability:
        candidate = experiment.approved_candidate(root)
        existing = sorted((root / 'repeats').glob('trial_*'))
        if existing:
            raise ValueError('repeatability batch already started; no automatic retry/replacement')
    else:
        if (root / 'approval.json').exists() or len(experiment.setup_trials(root)) >= experiment.SETUP_LIMIT:
            raise ValueError('setup is frozen or has reached its attempt limit')
        candidate = next((c for c in manifest['candidates'] if c['id'] == candidate_id), None)
        if candidate is None:
            raise ValueError('unknown candidate')
        for trial in experiment.setup_trials(root):
            if not (trial / 'admission.json').exists() or not experiment.read_json(trial / 'admission.json')['accepted']:
                raise ValueError('previous technical failure requires resolution; no replacement')
    context = dict(device='iPhone_XR', park='Workshop', settings_note=settings_note,
                   park_source='operator-confirmed readiness statement', operator_ready_note=ready_note,
                   allow_idle_navigation=True, training_admission=False)
    if (root / 'context.json').exists():
        recorded = experiment.read_json(root / 'context.json')
        # Readiness may be restated; fixed scene/settings must not change.
        if recorded['settings_note'] != settings_note:
            raise ValueError('waypoint/settings changed; requires a new reviewed experiment')
        context = recorded
    else:
        if repeatability:
            raise ValueError('reviewed experiment context is missing')
        save_new(root / 'context.json', context)
    from dotenv import load_dotenv
    if not Path(env_file).is_absolute() or not Path(env_file).is_file():
        raise ValueError('existing absolute rig .env path required')
    load_dotenv(env_file, override=True)
    if not os.environ.get('IPHONE_XR_UDID'):
        raise ValueError('XR1 UDID must come from rig .env')
    from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture, _http_json
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    from trueskate_ai.collection.scene_settle import wait_for_centre_settle
    from trueskate_ai.collection.gameplay_filter import is_menu_frame, is_editor_frame
    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    tunnel = subprocess.check_output(['launchctl', 'print', 'system/com.trueskate.remotexpc-tunnel'], text=True)
    if 'state = running' not in tunnel:
        raise RuntimeError('root recording tunnel must be running; no start attempted')
    cfg = next(d for d in DEVICES if d['name'] == 'iPhone_XR')
    base = f"http://127.0.0.1:{cfg['wda_port']}"
    _http_json(base + '/status')
    if _http_json(base + '/wda/activeAppInfo')['value']['bundleId'] != BUNDLE_ID:
        raise RuntimeError('True Skate must already be frontmost')
    if _http_json(base + '/wda/video').get('value') is not None:
        raise RuntimeError('recorder is not idle; no start attempted')
    if _http_json('http://127.0.0.1:4723/sessions').get('value'):
        raise RuntimeError('XR1 Appium session is already owned; do not disturb it')
    worker = DeviceSession(cfg, calibrate_touch_on_connect=False)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    trial = None
    try:
        worker.connect()
        driver = worker.driver
        actual_udid = driver.capabilities.get('udid') or driver.capabilities.get('appium:udid')
        if actual_udid != os.environ['IPHONE_XR_UDID']:
            raise RuntimeError('Appium session does not confirm configured XR1 UDID')
        def screenshot():
            return base64.b64decode(_http_json(base + '/screenshot')['value'])
        def guard():
            if driver.query_app_state(BUNDLE_ID) != 4 or worker._active_bundle_id() != BUNDLE_ID:
                raise RuntimeError('True Skate foreground lost')
            png = screenshot()
            if is_editor_frame(png) or is_menu_frame(png, allow_idle_navigation=True):
                raise RuntimeError('gameplay contamination')
        def settle(max_wait_s):
            if max_wait_s <= 0:
                raise RuntimeError('no time left to settle')
            return wait_for_centre_settle(screenshot, threshold=2., max_wait_s=max_wait_s)
        def perform(payload):
            return driver.execute('actions', payload)
        guard()
        for index in range(1, experiment.REPEATS + 1) if repeatability else (1,):
            if repeatability:
                if experiment.approved_candidate(root)['sha256'] != candidate['sha256']:
                    raise ValueError('approval changed during batch')
                trial = root / 'repeats' / f'trial_{index:02d}'
                trial.mkdir(parents=True, exist_ok=False)
                save_new(trial / 'reservation.json', dict(candidate_id=candidate['id'],
                    candidate_sha256=candidate['sha256'], repeat=index))
            else:
                trial, candidate = experiment.reserve_setup(root, candidate_id)
            guard()
            if _http_json(base + '/wda/video').get('value') is not None:
                raise RuntimeError('recorder not idle')
            perform(experiment.reset_payload())
            initial = settle(10.)
            if not initial.settled:
                raise RuntimeError('initial reset failed to settle')
            guard()
            (trial / 'initial-scene.png').write_bytes(screenshot())
            save_new(trial / 'initial-settle.json', initial.summary())
            experiment.run_trial(out=trial, candidate=candidate,
                recorder=XCTestScreenRecorder(driver, fps=experiment.FPS),
                timing=WDAActionTimingCapture(wda_port=8100, expected_revision=manifest['wda_revision']),
                perform=perform, guard=guard, settle=settle, revision=manifest['wda_revision'], context=context)
            # Disconnect before lengthy native decoding; never leave a stale session.
            worker.disconnect()
            experiment.admit_trial(trial, manifest['wda_revision'])
            print(f"{'repeat' if repeatability else 'setup'} {index}: recording complete and validated: {trial}", flush=True)
            if repeatability and index < experiment.REPEATS:
                worker.connect()
                driver = worker.driver
        print('run complete; all requested recordings retrieved', flush=True)
    except BaseException as exc:
        if trial is not None and not (trial / 'live-failure.json').exists():
            save_new(trial / 'live-failure.json', dict(error=f'{type(exc).__name__}: {exc}', no_replacements=True))
        raise
    finally:
        worker.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='absolute isolated experiment output directory')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    for name in ('run-setup', 'run-repeatability'):
        command = sub.add_parser(name)
        if name == 'run-setup':
            command.add_argument('--candidate', required=True)
        command.add_argument('--env-file', type=Path, required=True)
        command.add_argument('--operator-ready-note', required=True)
        command.add_argument('--settings-note', required=True, help='fixed waypoint, stance, camera and physics settings')
    assessment = sub.add_parser('assess')
    assessment.add_argument('--trial', required=True, help='relative setup/trial_NN or repeats/trial_NN')
    assessment.add_argument('--trick', required=True)
    assessment.add_argument('--status', choices=['landed', 'failed', 'unknown'], required=True)
    assessment.add_argument('--evidence-note', required=True)
    preview = sub.add_parser('preview')
    preview.add_argument('--candidate', required=True)
    preview.add_argument('--no-open', action='store_true', help='render only on rig; open returned movie on laptop')
    approval = sub.add_parser('approve')
    approval.add_argument('--candidate', required=True)
    approval.add_argument('--operator-statement', required=True, help='exact explicit acceptance of the QuickTime review')
    analysis = sub.add_parser('report')
    analysis.add_argument('--out', type=Path, required=True, help='new absolute report directory')
    sub.add_parser('status')
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error('--root must be absolute')
    root = args.root
    if args.command == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
        save_new(root / 'manifest.json', experiment.prepare_manifest(REPO))
        print(root / 'manifest.json')
    elif args.command.startswith('run-'):
        live_batch(root, candidate_id=getattr(args, 'candidate', None),
                   repeatability=args.command == 'run-repeatability', env_file=args.env_file,
                   ready_note=args.operator_ready_note, settings_note=args.settings_note)
    elif args.command == 'assess':
        trial = (root / args.trial).resolve()
        relative = trial.relative_to(root.resolve())
        if len(relative.parts) != 2 or relative.parts[0] not in ('setup', 'repeats') or not relative.parts[1].startswith('trial_'):
            parser.error('--trial must identify an experiment trial')
        admission = experiment.read_json(trial / 'admission.json')
        if not admission['accepted'] or not args.evidence_note.strip():
            parser.error('admitted recording and source-video evidence note required')
        save_new(trial / 'assessment.json', dict(trick=args.trick.strip().upper(), status=args.status,
                 evidence_note=args.evidence_note, video_sha256=experiment.file_sha(trial / 'original.mov')))
    elif args.command == 'preview':
        movie = experiment.review_candidate(root, args.candidate)
        print(movie)
        if not args.no_open:
            if sys.platform != 'darwin' or 'training-server' in socket.gethostname():
                parser.error('open the copied preview on the laptop; use --no-open on rig')
            subprocess.run(['open', '-a', 'QuickTime Player', str(movie)], check=True)
    elif args.command == 'approve':
        experiment.approve_review(root, args.candidate, args.operator_statement)
        print('review approval recorded; recipe frozen for 20 repeats')
    elif args.command == 'report':
        if not args.out.is_absolute():
            parser.error('--out must be absolute')
        experiment.render_report(root, args.out)
        print(args.out)
    else:
        manifest = experiment.load_manifest(root)
        print(json.dumps(dict(setup_attempts=len(experiment.setup_trials(root)), setup_limit=experiment.SETUP_LIMIT,
            review_approved=(root / 'approval.json').exists(), candidates=[dict(id=c['id'], seed=c['seed'],
            durations_s=[g['encoded_duration_s'] for g in c['contacts']], gap_s=c['pop_to_flick_gap_s'])
            for c in manifest['candidates']], repeats=experiment.report(root)), indent=2))


if __name__ == '__main__':
    main()
