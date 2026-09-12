"""One isolated XR2 timing sequence. No collector jobs or training admission.

Three 50 ms holds followed by three one-second linear gestures. Seven one-second
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


def sequence():
    return [
        {'kind': 'calibration', 'points': [[.40, .35]], 'duration': .05},
        {'kind': 'calibration', 'points': [[.60, .35]], 'duration': .05},
        {'kind': 'calibration', 'points': [[.50, .65]], 'duration': .05},
        {'kind': 'linear', 'points': [[.38, .65], [.62, .45]], 'duration': 1.0},
        {'kind': 'linear', 'points': [[.62, .65], [.38, .45]], 'duration': 1.0},
        {'kind': 'linear', 'points': [[.38, .55], [.62, .55]], 'duration': 1.0},
    ]


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


def main():
    from selenium.webdriver.common.action_chains import ActionChains
    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    from trueskate_ai.sim.touch_actions import make_touch_pointer
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    from trueskate_ai.collection.gameplay_filter import is_menu_frame, is_editor_frame

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--park', required=True, help='Observed park provenance, including uncertainty')
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Use a new output directory; never overwrite an experiment.')
    tunnel = subprocess.check_output(['launchctl', 'print', 'system/com.trueskate.remotexpc-tunnel'], text=True)
    if 'state = running' not in tunnel:
        raise SystemExit('Required recording tunnel is not running.')
    args.out.mkdir(parents=True)
    worker = DeviceSession(next(d for d in DEVICES if d['name'] == 'iPhone_XR2'))
    recorder = None
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
        for spec in sequence():
            finger = make_touch_pointer('finger')
            action = ActionChains(driver, devices=[finger])
            x, y = spec['points'][0]
            finger.create_pointer_move(x=x * 414, y=y * 896, duration=0)
            finger.create_pointer_down()
            if spec['kind'] == 'calibration':
                finger.create_pause(spec['duration'])
            else:
                x, y = spec['points'][1]
                finger.create_pointer_move(x=x * 414, y=y * 896, duration=1000)
            finger.create_pointer_up(0)
            commands.append(action)
        recorder = XCTestScreenRecorder(driver, fps=30)
        recorder.start()  # Single attempt. No retries or WDA restarts.
        try:
            events, waits = run_sequence(commands)
        finally:
            result = recorder.stop_and_save(args.out / 'segment_00000.mov')
        manifest = {
            'experiment': 'isolated-six-touch-timing', 'device': 'iPhone_XR2',
            'park': args.park, 'started_at_epoch_s': result.started_at_epoch_s,
            'host_start_epoch_s': result.host_start_epoch_s, 'host_stop_epoch_s': result.host_stop_epoch_s,
            'fps': result.fps, 'allow_idle_navigation': True,
            'wait_requested_s': 1.0, 'wait_measured_monotonic_s': waits,
            'settings_before_recording': settings, 'gestures': [],
            'training_admission': 'diagnostic only; pending per-recording human calibration',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        for i, (spec, times) in enumerate(zip(sequence(), events)):
            manifest['gestures'].append({'gesture_index': i, **times,
                'gesture_distribution': 'tap' if spec['kind'] == 'calibration' else 'linear',
                'waypoints': spec['points'], 'duration': spec['duration'], 'easing_power': 1.0,
                'calibration_execution': 'short_hold' if spec['kind'] == 'calibration' else None})
        (args.out / 'segment_00000.json').write_text(json.dumps(manifest, indent=2))
        after = driver.get_screenshot_as_png()
        (args.out / 'after.png').write_bytes(after)
        validation = {'foreground': driver.query_app_state(BUNDLE_ID) == 4,
                      'menu': is_menu_frame(after, allow_idle_navigation=True),
                      'editor': is_editor_frame(after)}
        (args.out / 'postflight.json').write_text(json.dumps(validation, indent=2))
        print(json.dumps({'out': str(args.out), 'gestures': len(events), 'waits': waits,
                          'postflight': validation}), flush=True)
    finally:
        if recorder is not None and recorder.is_recording:
            recorder.abort()
        worker.disconnect()


if __name__ == '__main__':
    main()
