"""Freeze or run exactly two bounded XR1/Inbound linear speed recordings."""
import argparse
import base64
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
    parser.add_argument('--profile', choices=('broad','fine','length'), default='broad', help='Profile to freeze')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--human-gameplay-review', action='store_true',
                        help='Operator assesses menus/editor; skip automatic image guards and admission scan')
    parser.add_argument('--wda-revision')
    parser.add_argument('--repeat', type=int, choices=range(1,16),
                        help='One explicitly authorized diagnostic repeat; independent admission result')
    args = parser.parse_args()
    if args.freeze:
        if args.profile == 'length':
            from trueskate_ai.research.linear_length_probe import manifest as length_manifest
            save_new(args.manifest, length_manifest())
        else:
            save_new(args.manifest, manifest(args.profile))
        return
    if not args.out or not args.wda_revision:
        parser.error('execution requires --out and --wda-revision')
    if 'training-server' not in socket.gethostname():
        parser.error('run on training-server')
    if args.out.exists():
        parser.error('preserve attempts; output must be new')
    frozen = json.loads(args.manifest.read_text())
    verify_manifest(frozen)
    if args.repeat is not None and args.repeat > len(frozen["recordings"]):
        parser.error("recording index outside frozen workload")
    if frozen["experiment"] == "LINEAR-LENGTH-20261003" and not args.human_gameplay_review:
        parser.error("length experiment requires --human-gameplay-review")
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
        last_settle_png = None
        def screenshot():
            # Same WDA PNG as Appium's screenshot; bypass the proxy round trip.
            return base64.b64decode(_http_json('http://127.0.0.1:8100/screenshot')['value'])
        def guard():
            nonlocal last_settle_png
            if driver.query_app_state(BUNDLE_ID) != 4 or worker._active_bundle_id() != BUNDLE_ID:
                raise RuntimeError('True Skate foreground lost')
            png = None if args.human_gameplay_review else (last_settle_png if last_settle_png is not None else screenshot())
            last_settle_png = None
            if not args.human_gameplay_review and (is_editor_frame(png) or is_menu_frame(png, allow_idle_navigation=True)):
                raise RuntimeError('gameplay contamination')
        def settle(max_wait_s):
            def capture():
                nonlocal last_settle_png
                # Read one newly delivered full-resolution frame from the
                # existing WDA stream; no cached frames or recorder restart.
                import requests
                with requests.get('http://127.0.0.1:9100', stream=True, timeout=2) as response:
                    response.raise_for_status()
                    buffer = bytearray()
                    for chunk in response.iter_content(4096):
                        buffer.extend(chunk)
                        start = buffer.find(b'\xff\xd8')
                        end = buffer.find(b'\xff\xd9', start+2)
                        if start >= 0 and end >= 0:
                            last_settle_png = bytes(buffer[start:end+2])
                            # The guard evaluates this just-captured image rather
                            # than making a redundant screenshot request.
                            return last_settle_png
                        if len(buffer) > 4_000_000:
                            raise RuntimeError('invalid WDA frame stream')
                raise RuntimeError('WDA stream ended without a frame')
            return wait_for_centre_settle(capture, threshold=2., max_wait_s=max_wait_s)
        for repeat, commands in enumerate(frozen['recordings'], 1):
            if args.repeat is not None and repeat != args.repeat:
                continue
            if worker.driver is None:
                worker.connect()
                driver = worker.driver
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
                            device_size=[414,896], resets_labelled=True,
                            settle_source='fresh full-resolution WDA MJPEG frame; same threshold, interval, consecutive count',
                            authorization='operator requested this bounded duration sweep; no automatic replacements',
                            gameplay_review='operator visual review' if args.human_gameplay_review else 'automated')
            run_recording(recorder=XCTestScreenRecorder(driver, fps=30), timing=timing,
                          commands=commands, perform=lambda s: driver.execute('actions', s['payload']),
                          guard=guard, settle=settle, out=out, revision=args.wda_revision, metadata=metadata)
            worker.disconnect()  # Offline decode must not leave a session to expire.
            # Human review bypasses the image gate only. Separate offline timing
            # and native decode checks still apply; this is never training data.
            if args.human_gameplay_review:
                save_new(out/'review-policy.json', dict(gameplay_review='operator visual review',
                         automated_gameplay_scan=False, training_admission=False))
                print(f'recording {repeat} complete; operator visual review pending', flush=True)
            else:
                admit_recording(out, args.wda_revision)
                print(f'recording {repeat} admitted', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        save_new(args.out/'failure.json', {'error': f'{type(exc).__name__}: {exc}', 'no_replacements': True})
        raise
    finally:
        worker.disconnect()


if __name__ == '__main__':
    main()
