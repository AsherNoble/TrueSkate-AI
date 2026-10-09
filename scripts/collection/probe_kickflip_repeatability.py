"""Prepare, run and review a bounded XR1 kickflip repeatability experiment."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from trueskate_ai.research import kickflip_repeatability as experiment
from trueskate_ai.research.curve_protocol import save_new

TUNNEL_REGISTRY = 'http://127.0.0.1:42314/remotexpc/tunnels/'


def recording_cleanup_ready(udid):
    """Require XR1's RemoteXPC tunnel and zero leftover XCTest attachments.

    A running tunnel daemon is not enough: with no tunnel for the phone, every
    retrieved recording stays on the device (KICKFLIP-REPEATABILITY-20261009).
    """
    try:
        with urllib.request.urlopen(TUNNEL_REGISTRY + udid, timeout=5) as response:
            tunnel = response.read().decode()
    except OSError as exc:
        raise RuntimeError(f'XR1 RemoteXPC tunnel unavailable; no start attempted ({exc})') from exc
    if udid not in tunnel:
        raise RuntimeError('tunnel registry did not return XR1; no start attempted')
    listing = subprocess.run(['appium', 'driver', 'run', 'xcuitest', 'cleanup-videos', '--', '--udid', udid, '--dry-run'],
                             capture_output=True, text=True, timeout=180)
    found = re.search(r'Found (\d+) UUID-shaped attachment', listing.stdout + listing.stderr)
    if listing.returncode or not found:
        raise RuntimeError('could not list XR1 recording attachments; no start attempted')
    if int(found.group(1)):
        raise RuntimeError(f'{found.group(1)} XR1 recording attachment(s) remain; preserve/clean before recording')


def live_batch(root, *, candidate_id, repeatability, env_file, ready_note, settings_note, appium_port=4723,
               defer_admission=False):
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
        # A continuation stage starts after the predecessor's repeats; none are replaced.
        first_repeat = 1 + len((manifest.get('prior_setup') or {}).get('repeats', []))
    else:
        if (root / 'approval.json').exists() or experiment.setup_attempt_count(root) >= experiment.SETUP_LIMIT:
            raise ValueError('setup is frozen or has reached its attempt limit')
        candidate = next((c for c in manifest['candidates'] if c['id'] == candidate_id), None)
        if candidate is None:
            raise ValueError('unknown candidate')
        for trial in experiment.setup_trials(root):
            if not (trial / 'admission.json').exists() or not experiment.read_json(trial / 'admission.json')['accepted']:
                raise ValueError('previous technical failure requires resolution; no replacement')
    context = dict(device='iPhone_XR', park='Workshop', appium_port=appium_port, settings_note=settings_note,
                   park_source='operator-confirmed readiness statement', operator_ready_note=ready_note,
                   allow_idle_navigation=True, training_admission=False)
    if (root / 'context.json').exists():
        recorded = experiment.read_json(root / 'context.json')
        # Readiness may be restated; fixed scene/settings must not change.
        if recorded['settings_note'] != settings_note or recorded.get('appium_port', 4723) != appium_port:
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
    from trueskate_ai.collection.gameplay_filter import is_menu_frame, is_editor_frame, _to_rgb01
    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    tunnel = subprocess.check_output(['launchctl', 'print', 'system/com.trueskate.remotexpc-tunnel'], text=True)
    if 'state = running' not in tunnel:
        raise RuntimeError('root recording tunnel must be running; no start attempted')
    recording_cleanup_ready(os.environ['IPHONE_XR_UDID'])
    cfg = dict(next(d for d in DEVICES if d['name'] == 'iPhone_XR'), appium_port=appium_port)
    base = f"http://127.0.0.1:{cfg['wda_port']}"
    if _http_json(base + '/status').get('sessionId'):
        raise RuntimeError('XR1 WDA session is already owned; do not disturb it')
    if _http_json(base + '/wda/activeAppInfo')['value']['bundleId'] != BUNDLE_ID:
        raise RuntimeError('True Skate must already be frontmost')
    if _http_json(base + '/wda/video').get('value') is not None:
        raise RuntimeError('recorder is not idle; no start attempted')
    appium_base = f'http://127.0.0.1:{appium_port}'
    def sessions():
        value = _http_json(appium_base + '/appium/sessions').get('value')
        if not isinstance(value, list) or any(not isinstance(s, dict) or not s.get('id') for s in value):
            raise RuntimeError('Appium session discovery must be enabled and return a session list')
        return [s['id'] for s in value]
    if sessions():
        raise RuntimeError('XR1 Appium session is already owned; do not disturb it')
    worker = DeviceSession(cfg, calibrate_touch_on_connect=False)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    trial = None
    try:
        worker.connect()
        driver = worker.driver
        owned_wda_session = _http_json(base + '/status').get('sessionId')
        actual_udid = driver.capabilities.get('udid') or driver.capabilities.get('appium:udid')
        if actual_udid != os.environ['IPHONE_XR_UDID']:
            raise RuntimeError('Appium session does not confirm configured XR1 UDID')
        def timing_preflight():
            metadata = _http_json(base + '/session/' + owned_wda_session + '/wda/actionTiming').get('value')
            if not isinstance(metadata, dict) or metadata.get('schema_version') != 1 or metadata.get('build_revision') != manifest['wda_revision'] or metadata.get('enabled'):
                raise RuntimeError('instrumented WDA identity/idle check failed before attempt reservation')
        timing_preflight()
        def screenshot():
            return base64.b64decode(_http_json(base + '/screenshot')['value'])
        def foreground_guard():
            if sessions() != [driver.session_id] or not owned_wda_session or _http_json(base + '/status').get('sessionId') != owned_wda_session:
                raise RuntimeError('Appium/WDA session ownership changed')
            if driver.query_app_state(BUNDLE_ID) != 4 or worker._active_bundle_id() != BUNDLE_ID:
                raise RuntimeError('True Skate foreground lost')
        def guard():
            foreground_guard()
            rgb = _to_rgb01(screenshot())
            if is_editor_frame(rgb) or is_menu_frame(rgb, allow_idle_navigation=True):
                raise RuntimeError('gameplay contamination')
        def settle(max_wait_s):
            if max_wait_s <= 0:
                raise RuntimeError('no time left to settle')
            return wait_for_centre_settle(screenshot, threshold=2., max_wait_s=max_wait_s)
        def perform(payload):
            return driver.execute('actions', payload)
        def perform_direct(payload):
            # One stroke = one XCTest record via WDA's direct endpoint (no Appium hop).
            request = urllib.request.Request(base + '/wda/perform_trick_gestures', method='POST',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())
        guard()
        for index in range(first_repeat, experiment.REPEATS + 1) if repeatability else (1,):
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
                perform=perform, perform_direct=perform_direct, guard=guard, foreground_guard=foreground_guard,
                settle=settle, revision=manifest['wda_revision'], context=context)
            # Disconnect before lengthy native decoding; never leave a stale session.
            worker.disconnect()
            if defer_admission:
                # Native-frame admission takes ~17 min on the rig; run `admit` elsewhere and copy it back.
                print(f"{'repeat' if repeatability else 'setup'} {index}: recording complete; admission deferred: {trial}", flush=True)
            else:
                experiment.admit_trial(trial, manifest['wda_revision'])
                print(f"{'repeat' if repeatability else 'setup'} {index}: recording complete and validated: {trial}", flush=True)
            if repeatability and index < experiment.REPEATS:
                if sessions() or _http_json(base + '/status').get('sessionId'):
                    raise RuntimeError('session ownership changed between repeats')
                recording_cleanup_ready(os.environ['IPHONE_XR_UDID'])
                worker.connect()
                driver = worker.driver
                owned_wda_session = _http_json(base + '/status').get('sessionId')
                actual_udid = driver.capabilities.get('udid') or driver.capabilities.get('appium:udid')
                if actual_udid != os.environ['IPHONE_XR_UDID']:
                    raise RuntimeError('reconnected Appium session does not confirm configured XR1 UDID')
                timing_preflight()
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
    preparation = sub.add_parser('prepare')
    preparation.add_argument('--previous-setup', type=Path, help='preserve pre-gameplay failures and carry their setup budget into a new manifest')
    preparation.add_argument('--superseded-reason', help='required when the candidate procedure changed; also carries assessed gameplay and technical failures')
    for name in ('run-setup', 'run-repeatability'):
        command = sub.add_parser(name)
        if name == 'run-setup':
            command.add_argument('--candidate', required=True)
        command.add_argument('--env-file', type=Path, required=True)
        command.add_argument('--operator-ready-note', required=True)
        command.add_argument('--settings-note', required=True, help='fixed waypoint, stance, camera and physics settings')
        command.add_argument('--appium-port', type=int, default=4723, help='session-discovery-enabled Appium; temporary diagnostic server may use a separate port')
        command.add_argument('--defer-admission', action='store_true', help='skip on-rig admission; run `admit` on a faster machine and copy admission.json back')
    admit = sub.add_parser('admit')
    admit.add_argument('--trial', required=True, help='relative setup/trial_NN or repeats/trial_NN to validate (same code, any machine)')
    assessment = sub.add_parser('assess')
    assessment.add_argument('--trial', required=True, help='relative setup/trial_NN or repeats/trial_NN')
    assessment.add_argument('--trick', required=True)
    assessment.add_argument('--status', choices=['landed', 'failed', 'unknown'], required=True)
    assessment.add_argument('--evidence-note', required=True)
    preview = sub.add_parser('preview')
    preview.add_argument('--candidate', required=True)
    preview.add_argument('--no-open', action='store_true', help='render only on rig; open returned movie on laptop')
    preview.add_argument('--operator-minimum', type=int, default=3, help='operator gate override: review the first N identical-recipe attempts (N < 3 needs --override-reason)')
    preview.add_argument('--override-reason', help='explicit operator reason, sealed into the review')
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
        if args.previous_setup is not None and not args.previous_setup.is_absolute():
            parser.error('--previous-setup must be absolute')
        manifest = experiment.prepare_manifest(REPO, args.previous_setup, args.superseded_reason)
        root.mkdir(parents=True, exist_ok=False)
        if args.previous_setup is not None:
            shutil.copytree(args.previous_setup, root / 'history/predecessor')
            if (args.previous_setup / 'context.json').exists():
                shutil.copyfile(args.previous_setup / 'context.json', root / 'context.json')
        save_new(root / 'manifest.json', manifest)
        experiment.load_manifest(root)
        print(root / 'manifest.json')
    elif args.command.startswith('run-'):
        live_batch(root, candidate_id=getattr(args, 'candidate', None),
                   repeatability=args.command == 'run-repeatability', env_file=args.env_file,
                   ready_note=args.operator_ready_note, settings_note=args.settings_note, appium_port=args.appium_port,
                   defer_admission=getattr(args, 'defer_admission', False))
    elif args.command == 'admit':
        trial = (root / args.trial).resolve()
        result = experiment.admit_trial(trial, experiment.load_manifest(root)['wda_revision'])
        print(json.dumps(dict(accepted=result['accepted'], heldout_middle_error_s=result['heldout_middle_error_s'],
                              gestures=[(g['role'], round(g['video_start_s'], 3)) for g in result['gestures']])))
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
        movie = experiment.review_candidate(root, args.candidate, minimum=args.operator_minimum,
                                            override_reason=args.override_reason)
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
        print(json.dumps(dict(setup_attempts=experiment.setup_attempt_count(root), setup_limit=experiment.SETUP_LIMIT,
            review_approved=(root / 'approval.json').exists(), candidates=[dict(id=c['id'], varied=c['varied'],
            strokes=[(x['name'], x['points'], x['encoded_duration_s']) for x in c['contacts']],
            catch_wait_s=c['catch_wait_s']) for c in manifest['candidates']],
            rejected=manifest['rejected'], repeats=experiment.report(root)), indent=2))


if __name__ == '__main__':
    main()
