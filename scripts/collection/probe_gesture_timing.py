"""One isolated XR2 timing sequence. No collector jobs or training admission.

Three 50 ms holds followed by three one-second linear gestures (original),
or six linear gestures of 0.3, 0.6, 1.0 s repeated (duration-repeat). One-second
waits: before, between returned calls, and after. The measured loop has no
screenshots, checks, resets, logging writes, or other device requests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sequence(profile="original"):
    items = [
        {'kind': 'calibration', 'points': [[.40, .35]], 'duration': .05},
        {'kind': 'calibration', 'points': [[.60, .35]], 'duration': .05},
        {'kind': 'calibration', 'points': [[.50, .65]], 'duration': .05},
        {'kind': 'linear', 'points': [[.38, .65], [.62, .45]], 'duration': 1.0},
        {'kind': 'linear', 'points': [[.62, .65], [.38, .45]], 'duration': 1.0},
        {'kind': 'linear', 'points': [[.38, .55], [.62, .55]], 'duration': 1.0},
    ]
    if profile == 'duration-repeat':
        linears = items[3:]
        items = items[:3] + [dict(g, duration=d) for _ in range(2)
                             for g, d in zip(linears, (.3, .6, 1.0))]
    elif profile != 'original':
        raise ValueError('Unknown sequence profile')
    return items


def run_sequence(commands, *, sleep=time.sleep, epoch=time.time, monotonic=time.monotonic):
    """Only waits, timestamps, prepared touch calls and in-memory bookkeeping."""
    events, waits = [], []
    a = monotonic()
    sleep(1.0)
    waits.append(monotonic() - a)
    for command in commands:
        t0, m0 = epoch(), monotonic()
        command.perform()
        m1, t1 = monotonic(), epoch()
        events.append({'t_call_start_epoch_s': t0, 't_call_end_epoch_s': t1,
                       't_call_start_monotonic_s': m0, 't_call_end_monotonic_s': m1})
        a = monotonic()
        sleep(1.0)
        waits.append(monotonic() - a)
    return events, waits


def build_bundle(driver, specs):
    """One pointer source, released during pauses; returns requested onset schedule."""
    from selenium.webdriver.common.action_chains import ActionChains
    from trueskate_ai.sim.touch_actions import make_touch_pointer
    finger = make_touch_pointer('bundle')
    command = ActionChains(driver, devices=[finger])
    x, y = specs[0]['points'][0]
    finger.create_pointer_move(x=x * 414, y=y * 896, duration=0)
    finger.create_pause(1.0)
    elapsed, starts = 1.0, []
    for spec in specs:
        x, y = spec['points'][0]
        finger.create_pointer_move(x=x * 414, y=y * 896, duration=0)
        finger.create_pointer_down()
        starts.append(elapsed)
        if spec['kind'] == 'calibration':
            finger.create_pause(spec['duration'])
        else:
            x, y = spec['points'][1]
            finger.create_pointer_move(x=x * 414, y=y * 896,
                                       duration=round(spec['duration'] * 1000))
        finger.create_pointer_up(0)
        finger.create_pause(1.0)
        elapsed += spec['duration'] + 1.0
    return command, starts, elapsed


def timing_http(url, payload=None):
    from urllib.request import Request, urlopen
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def validate_wda_timings(report, expected_revision, count):
    if report.get('schema_version') != 1 or report.get('build_revision') != expected_revision:
        raise ValueError('Unexpected WDA timing schema or build revision')
    if report.get('dropped_records') != 0 or len(report.get('records', [])) != count:
        raise ValueError('Missing or overflowed WDA timing records')
    boundaries = ('request_entered', 'preparation_started', 'preparation_finished',
                  'submitted_to_ios', 'ios_completion_callback', 'stability_wait_started',
                  'stability_wait_finished', 'request_finished')
    session = None
    for i, record in enumerate(report['records']):
        if record.get('sequence') != i or record.get('outcome') != 'success':
            raise ValueError('Unordered or failed WDA action')
        if record.get('missing_ios_callback') is not False or record.get('ios_callback_result') is not True:
            raise ValueError('Missing or unsuccessful iOS callback')
        if not record.get('session_id') or (session is not None and record['session_id'] != session):
            raise ValueError('WDA session changed')
        session = record['session_id']
        stamps = [record[b]['monotonic_s'] for b in boundaries]
        import math
        if not all(math.isfinite(t) for t in stamps) or any(b < a for a, b in zip(stamps, stamps[1:])):
            raise ValueError('Invalid WDA timestamp ordering')
    return True


def main():
    from selenium.webdriver.common.action_chains import ActionChains
    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    from trueskate_ai.sim.touch_actions import make_touch_pointer
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    from trueskate_ai.collection.gameplay_filter import is_menu_frame, is_editor_frame

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wda-timing-revision', help='Require opt-in timing from this exact WDA build SHA')
    parser.add_argument('--bundled', action='store_true', help='One request with device-side pauses')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--park', required=True, help='Observed park provenance, including uncertainty')
    parser.add_argument('--profile', choices=('original', 'duration-repeat'), default='original')
    parser.add_argument('--appium-port', type=int, default=4726, help='Optional isolated logging Appium; WDA remains on 8103')
    args = parser.parse_args()
    if args.wda_timing_revision and args.bundled:
        parser.error('WDA internal timing experiment requires separate requests')
    specs = sequence(args.profile)
    if args.out.exists():
        raise SystemExit('Use a new output directory; never overwrite an experiment.')
    tunnel = subprocess.check_output(['launchctl', 'print', 'system/com.trueskate.remotexpc-tunnel'], text=True)
    if 'state = running' not in tunnel:
        raise SystemExit('Required recording tunnel is not running.')
    args.out.mkdir(parents=True)
    cfg = dict(next(d for d in DEVICES if d['name'] == 'iPhone_XR2'))
    cfg['appium_port'] = args.appium_port
    worker = DeviceSession(cfg)
    recorder = None
    timing_url = None
    try:
        worker.connect()
        driver = worker.driver
        if driver.query_app_state(BUNDLE_ID) != 4:
            raise RuntimeError('True Skate must be foregrounded before recording.')
        before = driver.get_screenshot_as_png()
        (args.out / 'before.png').write_bytes(before)
        if is_editor_frame(before) or is_menu_frame(before, allow_idle_navigation=True):
            raise RuntimeError('Preflight detected blocking menu/editor; no recording started.')
        settings = driver.get_settings()
        commands = []
        for spec in specs:
            finger = make_touch_pointer('finger')
            action = ActionChains(driver, devices=[finger])
            x, y = spec['points'][0]
            finger.create_pointer_move(x=x * 414, y=y * 896, duration=0)
            finger.create_pointer_down()
            if spec['kind'] == 'calibration':
                finger.create_pause(spec['duration'])
            else:
                x, y = spec['points'][1]
                finger.create_pointer_move(x=x * 414, y=y * 896, duration=round(spec['duration'] * 1000))
            finger.create_pointer_up(0)
            commands.append(action)
        if args.bundled:
            bundle, planned_starts, planned_duration = build_bundle(driver, specs)
            encoded_bundle = bundle.w3c_actions.pointer_action.source.encode()
        if args.wda_timing_revision:
            status = timing_http('http://127.0.0.1:8103/status')
            session_id = status.get('sessionId')
            if not session_id:
                raise RuntimeError('WDA status has no active session')
            from urllib.parse import quote
            candidate = 'http://127.0.0.1:8103/session/' + quote(session_id, safe='') + '/wda/actionTiming'
            current = timing_http(candidate)['value']
            if current.get('build_revision') != args.wda_timing_revision or current.get('schema_version') != 1:
                raise RuntimeError('Instrumented WDA build identity mismatch; no recording started')
            timing_url = candidate
            enabled = timing_http(timing_url, {'enabled': True})['value']
            if enabled.get('enabled') is not True or enabled.get('records') != []:
                raise RuntimeError('WDA timing capture failed to initialize')
        recorder = XCTestScreenRecorder(driver, fps=30)
        recorder.start()  # Single attempt. No retries or WDA restarts.
        try:
            if args.bundled:
                t0, m0 = time.time(), time.monotonic()
                bundle.perform()
                m1, t1 = time.monotonic(), time.time()
                batch_times = dict(t_call_start_epoch_s=t0, t_call_end_epoch_s=t1,
                                   t_call_start_monotonic_s=m0, t_call_end_monotonic_s=m1)
                events = [{'planned_onset_from_batch_start_s': t} for t in planned_starts]
                waits = []
            else:
                events, waits = run_sequence(commands)
        finally:
            try:
                result = recorder.stop_and_save(args.out / 'segment_00000.mov')
            finally:
                if timing_url:
                    report = timing_http(timing_url, {'enabled': False})['value']
                    (args.out / 'wda-action-timings.json').write_text(json.dumps(report, indent=2))
                    timing_url = None
        manifest = {
            'experiment': 'isolated-touch-timing', 'profile': args.profile, 'device': 'iPhone_XR2',
            'park': args.park, 'started_at_epoch_s': result.started_at_epoch_s,
            'host_start_epoch_s': result.host_start_epoch_s, 'host_stop_epoch_s': result.host_stop_epoch_s,
            'fps': result.fps, 'allow_idle_navigation': True,
            'appium_port': args.appium_port, 'wda_port': 8103,
            'wait_requested_s': 1.0, 'wait_measured_monotonic_s': waits,
            'settings_before_recording': settings, 'gestures': [],
            'training_admission': 'diagnostic only; pending per-recording human calibration',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if args.bundled:
            manifest.update(execution_mode='single-bundled-request', batch_call=batch_times,
                            planned_batch_duration_s=planned_duration,
                            encoded_actions=encoded_bundle)
        for i, (spec, times) in enumerate(zip(specs, events)):
            manifest['gestures'].append({'gesture_index': i, **times,
                'gesture_distribution': 'tap' if spec['kind'] == 'calibration' else 'linear',
                'waypoints': spec['points'], 'duration': spec['duration'], 'easing_power': 1.0,
                'calibration_execution': 'short_hold' if spec['kind'] == 'calibration' else None})
        (args.out / 'segment_00000.json').write_text(json.dumps(manifest, indent=2))
        if args.wda_timing_revision:
            validate_wda_timings(report, args.wda_timing_revision, len(specs))
        after = driver.get_screenshot_as_png()
        (args.out / 'after.png').write_bytes(after)
        validation = {'foreground': driver.query_app_state(BUNDLE_ID) == 4,
                      'menu': is_menu_frame(after, allow_idle_navigation=True),
                      'editor': is_editor_frame(after)}
        (args.out / 'postflight.json').write_text(json.dumps(validation, indent=2))
        print(json.dumps({'out': str(args.out), 'gestures': len(events), 'waits': waits,
                          'postflight': validation}), flush=True)
    finally:
        if timing_url:
            try:
                report = timing_http(timing_url, {'enabled': False})['value']
                (args.out / 'wda-action-timings.json').write_text(json.dumps(report, indent=2))
            except Exception as error:
                print(f'WDA timing cleanup failed: {error}', flush=True)
        if recorder is not None and recorder.is_recording:
            recorder.abort()
        worker.disconnect()


if __name__ == '__main__':
    main()
