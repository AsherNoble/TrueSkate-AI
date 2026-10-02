"""Freeze or run exactly two bounded XR1/Inbound linear speed recordings."""
import argparse
import json
import signal
import socket
import subprocess
from pathlib import Path

from trueskate_ai.research.linear_speed_probe import manifest, verify_manifest, run_recording
from trueskate_ai.research.curve_protocol import save_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--wda-revision')
    args = parser.parse_args()
    if args.freeze:
        save_new(args.manifest, manifest())
        return
    if not args.out or not args.wda_revision:
        parser.error('execution requires --out and --wda-revision')
    if 'training-server' not in socket.gethostname():
        parser.error('run on training-server')
    if args.out.exists():
        parser.error('preserve attempts; output must be new')
    frozen = json.loads(args.manifest.read_text())
    verify_manifest(frozen)
    tunnel = subprocess.check_output(['launchctl', 'print', 'system/com.trueskate.remotexpc-tunnel'], text=True)
    if 'state = running' not in tunnel:
        parser.error('root recording tunnel is not running')
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    from trueskate_ai.sim.touch_actions import reset_position
    from trueskate_ai.collection.scene_settle import wait_for_centre_settle
    from trueskate_ai.collection.gameplay_filter import is_menu_frame, is_editor_frame
    from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture, _http_json
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    from trueskate_ai.research.curve_measurement import admit_recording
    worker = DeviceSession(next(d for d in DEVICES if d['name'] == 'iPhone_XR'))
    args.out.mkdir(parents=True)
    try:
        worker.connect()
        driver = worker.driver
        def guard():
            if driver.query_app_state(BUNDLE_ID) != 4 or worker._active_bundle_id() not in (None, BUNDLE_ID):
                raise RuntimeError('True Skate foreground lost')
            png = driver.get_screenshot_as_png()
            if is_editor_frame(png) or is_menu_frame(png, allow_idle_navigation=True):
                raise RuntimeError('gameplay contamination')
        def settle(max_wait_s):
            return wait_for_centre_settle(driver.get_screenshot_as_png, threshold=2., max_wait_s=max_wait_s)
        for repeat, commands in enumerate(frozen['recordings'], 1):
            guard()
            if _http_json('http://127.0.0.1:8100/wda/video').get('value') is not None:
                raise RuntimeError('recorder is not idle; no start attempted')
            reset_position(driver, worker.device_w, worker.device_h)
            initial = settle(10.)
            if not initial.settled:
                raise RuntimeError('pre-recording scene failed to settle')
            guard()
            out = args.out/f'recording_{repeat}'
            timing = WDAActionTimingCapture(wda_port=8100, expected_revision=args.wda_revision)
            metadata = dict(experiment=frozen['experiment'], manifest_sha256=frozen['sha256'],
                            device='iPhone_XR', park='Inbound', park_source='operator-confirmed in task',
                            repeat=repeat, allow_idle_navigation=True, initial_settle=initial.summary(),
                            device_size=[414,896], resets_labelled=True)
            run_recording(recorder=XCTestScreenRecorder(driver, fps=30), timing=timing,
                          commands=commands, perform=lambda s: driver.execute('actions', s['payload']),
                          guard=guard, settle=settle, out=out, revision=args.wda_revision, metadata=metadata)
            # Same source decode, gameplay, start/end fit and held-out middle
            # checks as the existing diagnostic. Resets are labelled requests.
            admit_recording(out, args.wda_revision)
            print(f'recording {repeat} admitted', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        save_new(args.out/'failure.json', {'error': f'{type(exc).__name__}: {exc}', 'no_replacements': True})
        raise
    finally:
        worker.disconnect()


if __name__ == '__main__':
    main()
