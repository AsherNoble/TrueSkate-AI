#!/usr/bin/env python3
"""Isolated XR2 paired-IPv4 lifecycle diagnostic; never admits training data.

Run --prepare first. Administrator authentication is needed only for the native
TUN helper. Production code, services, signing and collection jobs are untouched.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

# Instrumented WDA fork build on both XRs since CURVE-AUDIT-20261003.
REVISION = 'ae50404aac12d9f8c41f6c3fa8776e97975eaef5'
BUNDLE = 'com.trueaxis.skate'
REGISTRY_API = '/remotexpc/tunnels'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def http(url, payload=None, timeout=10):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    # Do not send phone traffic through an ambient HTTP proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def command(argv, timeout=45):
    result = subprocess.run([str(x) for x in argv], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        detail = (result.stderr.strip() or result.stdout.strip()).splitlines()[-3:]
        raise RuntimeError(f'{Path(str(argv[1] if len(argv) > 1 else argv[0])).name} exited {result.returncode}: '
                           + ' | '.join(detail)[:600])
    return result.stdout


class RegistryLease:
    """Exclusive ownership, exact byte restoration; never clobber another writer."""
    def __init__(self, path):
        self.path = Path(path)
        self.lock = self.path.with_name(self.path.name + '.ipv4-diagnostic-lock')
        self.changed = False
        self.locked = False
        self.token = str(uuid.uuid4())

    def __enter__(self):
        self.lock.mkdir()
        self.locked = True
        try:
            save(self.lock/'owner.json', {'pid': os.getpid(), 'token': self.token})
            self.original = self.path.read_bytes()
            if self.original.strip() != b'42314':
                raise RuntimeError('Unexpected shared tunnel registry port')
            self.path.write_bytes(b'42315')
            self.changed = True
            return self
        except BaseException:
            (self.lock/'owner.json').unlink(missing_ok=True)
            self.lock.rmdir()
            self.locked = False
            raise

    def __exit__(self, *_):
        try:
            if self.changed:
                if json.loads((self.lock/'owner.json').read_text()).get('token') != self.token:
                    raise RuntimeError('Registry lease ownership changed')
                if self.path.read_bytes() != b'42315':
                    raise RuntimeError('Registry changed externally; original value not overwritten')
                self.path.write_bytes(self.original)
        finally:
            if self.locked:
                owner = self.lock/'owner.json'
                if owner.exists() and json.loads(owner.read_text()).get('token') == self.token:
                    owner.unlink()
                    self.lock.rmdir()
                self.locked = False


def lend_registry(helper_cmd, portfile, backup, uid):
    """Root wrapper: lend the root-owned registry file/dir for the lease, then restore.

    The root daemon recreates the port file as root on boot. Ownership is returned
    after the helper exits; a still-redirected value is restored from a root backup.
    """
    f, d, b = (shlex.quote(str(x)) for x in (portfile, Path(portfile).parent, backup))
    return (f'set -u; fileown=$(stat -f %u:%g {f}); dirown=$(stat -f %u:%g {d}); cp -p {f} {b}; '
            f'chown {int(uid)} {f} {d}; {helper_cmd}; rc=$?; '
            f'if [ "$(cat {f})" = 42315 ]; then cat {b} > {f}; echo registry-restored-by-root >&2; fi; '
            f'chown "$fileown" {f}; chown "$dirown" {d}; exit $rc')


def guardian(state_path):
    """Recover our registry and owned runner if coordinator is killed outright."""
    state_path = Path(state_path)
    while state_path.exists():
        state = json.loads(state_path.read_text())
        try:
            os.kill(state['pid'], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        if not alive:
            errors = []
            if state.get('session_id'):
                try:
                    req = urllib.request.Request('http://127.0.0.1:4726/session/'+state['session_id'], method='DELETE')
                    urllib.request.urlopen(req, timeout=10).close()
                except Exception as exc:
                    errors.append('session: '+str(exc))
            if state.get('wda_pid'):
                try:
                    os.kill(state['wda_pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if state.get('wake_pid'):
                try:
                    os.kill(state['wake_pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            port = Path(state['portfile'])
            original = base64.b64decode(state['original'])
            lock = port.with_name(port.name+'.ipv4-diagnostic-lock')
            owner = lock/'owner.json'
            owns_lock = owner.exists() and json.loads(owner.read_text()).get('token') == state.get('registry_token')
            if owns_lock:
                if port.read_bytes() == b'42315':
                    port.write_bytes(original)
                elif port.read_bytes() != original:
                    errors.append('Registry changed externally; did not overwrite')
                owner.unlink()
                lock.rmdir()
            Path(state['lease']).unlink(missing_ok=True)
            save(state_path.with_suffix('.result.json'), {'coordinator_died': True, 'errors': errors})
            return
        time.sleep(1)


def movie_metrics(stream, pts, decoded, packet_count, requested_fps, minute):
    if stream.get('codec_name') != 'h264' or (stream.get('width'), stream.get('height')) != (828, 1792):
        raise ValueError('Unexpected video codec or geometry')
    if not pts or not all(math.isfinite(x) for x in pts):
        raise ValueError('Missing or nonfinite PTS')
    gaps = [b-a for a, b in zip(pts, pts[1:])]
    if any(x <= 0 for x in gaps):
        raise ValueError('PTS are not strictly increasing')
    count = int(stream['nb_read_frames'])
    if not count == len(pts) == decoded == packet_count:
        raise ValueError('Decoded frame, packet and PTS counts disagree')
    duration = float(stream['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid video duration')
    rate = count / duration
    if minute and (duration < 59 or rate < .98 * requested_fps or max(gaps, default=0) > .100001):
        raise ValueError('Minute duration or cadence gate failed')
    return {'frames': count, 'duration_s': duration, 'effective_fps': rate,
            'pts_span_fps': (count-1)/(pts[-1]-pts[0]) if count > 1 else None,
            'max_gap_s': max(gaps, default=0),
            'estimated_missing_slots': sum(max(0, round(x*requested_fps)-1) for x in gaps)}


def audit_movie(mov, fps, minute):
    probe = json.loads(command(['ffprobe', '-v', 'error', '-count_frames', '-count_packets',
                               '-select_streams', 'v:0', '-show_streams', '-of', 'json', mov], 600))
    frames = json.loads(command(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                                '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', mov], 600))
    pts = [float(f['best_effort_timestamp_time']) for f in frames['frames']]
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(mov), '-map', '0:v:0',
                              '-fps_mode', 'passthrough', '-progress', 'pipe:1', '-f', 'null', '-'],
                             capture_output=True, text=True, timeout=600, check=True)
    # ffmpeg may report recoverable corruption despite an exit code of zero.
    if decoded.stderr.strip():
        raise ValueError('Full movie decode reported errors')
    counts = re.findall(r'^frame=(\d+)$', decoded.stdout, re.M)
    if not counts:
        raise ValueError('No full-decode frame count')
    stream = probe['streams'][0]
    save(mov.with_suffix('.pts.json'), pts)
    save(mov.with_suffix('.ffprobe.json'), probe)
    metrics = movie_metrics(stream, pts, int(counts[-1]), int(stream['nb_read_packets']), fps, minute)
    metrics.update(bytes=mov.stat().st_size, sha256=hashlib.sha256(mov.read_bytes()).hexdigest())
    return metrics, pts


class Recording:
    """One start attempt, one stop attempt, including uncertain-start recovery."""
    def __init__(self, driver, folder, fps, wifi_pull=None):
        self.driver, self.folder, self.fps = driver, folder, fps
        self.wifi_pull = wifi_pull
        self.attempted = self.stopped = False

    def start(self):
        self.attempted = True
        self.host_start = time.time()
        self.info = self.driver.execute_script('mobile: startXCTestScreenRecording', {'fps': self.fps})
        save(self.folder / 'start.json', self.info)
        return self.info

    def stop(self, recovery=False):
        if self.stopped:
            raise RuntimeError('Stop already attempted; do not hammer recorder')
        self.stopped = True
        mov = self.folder / ('recovery.mov' if recovery else 'original.mov')
        # Off USB the stop succeeds in WDA but Appium's devicectl pull cannot find the movie.
        active = self.driver.execute_script('mobile: getXCTestScreenRecordingInfo', {}) if self.wifi_pull else None
        began = time.monotonic()
        try:
            res = self.driver.execute_script('mobile: stopXCTestScreenRecording', {})
        except Exception as exc:
            if not (active and f"recording identified by '{active['uuid']}'" in str(exc)):
                raise
            meta = dict(active, recovery=recovery, host_start_epoch_s=self.host_start, appium_pull_error=str(exc)[:300])
            save(self.folder / 'stop.json', meta)
            meta.update(self.wifi_pull(active['uuid'], mov), retrieval_s=time.monotonic()-began)
            save(self.folder / 'stop.json', meta)
            return mov, meta
        if not isinstance(res, dict) or not res.get('payload'):
            raise ValueError('Stop returned no movie payload')
        # Preserve UUID even if base64 decoding subsequently fails.
        meta = {k: v for k, v in res.items() if k != 'payload'}
        meta.update(recovery=recovery, retrieval_s=time.monotonic()-began, host_start_epoch_s=self.host_start)
        save(self.folder / 'stop.json', meta)
        data = base64.b64decode(res['payload'], validate=True)
        if not data:
            raise ValueError('Empty movie')
        mov.write_bytes(data)
        return mov, meta

    def recover(self):
        if self.attempted and not self.stopped:
            # A failed HTTP start can have started on-device. Stop once, no retry.
            self.stop(recovery=True)


class Coordinator:
    def __init__(self, args):
        self.a = args
        self.out = args.out_dir
        self.out.mkdir(parents=True, exist_ok=False)
        self.events = self.out / 'events.jsonl'
        self.lease = self.out / 'coordinator-lease.json'
        self.token = str(uuid.uuid4())
        self.ending = threading.Event()
        self.driver = self.wda = self.admin = self.registry = self.guardian_process = self.wake = None
        self.timing = None
        self.pending = []
        self.report = {'park': 'unverified current gameplay scene', 'corpus_admission': False,
                       'cases': [], 'cleanup_errors': [], 'pico_acceptance': 'not tested'}
        self.udid = None

    def event(self, event, **details):
        value = {'event': event, 'epoch_s': time.time(), **details}
        with self.events.open('a') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)

    def heartbeat(self):
        while not self.ending.is_set():
            tmp = self.lease.with_suffix('.new')
            save(tmp, {'pid': os.getpid(), 'token': self.token, 'updated': time.time()})
            tmp.replace(self.lease)
            self.ending.wait(2)

    def node(self, mode):
        return [self.a.node, str(Path(__file__).with_name('ios_ipv4_tunnel.mjs')), mode,
                '--modules-root', str(self.a.modules_root), '--udid', self.udid, '--host', self.a.host]

    def root_alive(self):
        output = command(['launchctl', 'print', 'system/com.trueskate.remotexpc-tunnel'])
        if 'state = running' not in output:
            raise RuntimeError('Required root tunnel daemon is not running')

    def usb_absent(self):
        if self.udid in command(['idevice_id', '-l']):
            raise RuntimeError('XR2 ordinary USB is still connected')

    def preflight(self):
        from dotenv import dotenv_values
        self.udid = dotenv_values(self.a.env_file).get('IPHONE_XR2_UDID')
        if not self.udid:
            raise RuntimeError('XR2 UDID missing')
        self.usb_absent()
        self.root_alive()
        disabled = command(['launchctl', 'print-disabled', f'gui/{os.getuid()}'])
        for label in ('com.trueskate.collect.xr1', 'com.trueskate.collect.xr2',
                      'com.trueskate.watchdog.xr1', 'com.trueskate.watchdog.xr2'):
            if not re.search(re.escape('"'+label+'"')+r'\s*=>\s*(?:true|disabled)', disabled):
                raise RuntimeError(f'Collection-off prerequisite not verified: {label}')
        metadata = http('http://127.0.0.1:42314'+REGISTRY_API+'/metadata')
        save(self.out/'original-registry-metadata.json', metadata)
        # Accept only explicit, empty registry; do not redirect another device.
        catalog = http('http://127.0.0.1:42314'+REGISTRY_API)
        save(self.out/'original-registry.json', catalog)
        if catalog.get('tunnels') != {} or metadata.get('activeTunnels') != 0:
            raise RuntimeError('Default registry contains another tunnel; refusing shared redirection')
        result = command(self.node('probe'), 20)
        save(self.out/'paired-probe.json', [json.loads(x) for x in result.splitlines() if x.startswith('{')])
        details = [json.loads(x) for x in result.splitlines() if x.startswith('{') and json.loads(x).get('event') == 'paired'][0]
        self.phone = 'http://' + details['host'] + ':8100'
        # Off USB, Appium cannot query the version through usbmux; supply the paired-IPv4 value.
        self.platform_version = details['ios']
        info = command(self.node('registry-info'), 15)
        self.portfile = Path(json.loads(info.splitlines()[-1])['file'])
        if self.portfile.read_bytes().strip() != b'42314':
            raise RuntimeError('Unexpected default registry file')
        lend = not (os.access(self.portfile, os.W_OK) and os.access(self.portfile.parent, os.W_OK))
        if lend and not self.a.tunnel_python:
            raise RuntimeError('Coordinator cannot lease the registry file and lock; refusing administrator launch')
        root = self.node('tunnel') + ['--lifetime', '900', '--lease-file', str(self.lease), '--lease-token', self.token]
        if self.a.tunnel_python:
            if not (self.a.pair_dir/f'remote_{self.udid}.plist').is_file():
                raise RuntimeError('RemotePairing record missing; pair over USB first')
            root += ['--tunnel-python', str(self.a.tunnel_python), '--pair-dir', str(self.a.pair_dir)]
        self.root_cmd = shlex.join(root) + ' > ' + shlex.quote(str(self.out/'helper.log')) + ' 2>&1'
        if lend:
            self.root_cmd = lend_registry(self.root_cmd, self.portfile, self.out/'registry-backup', os.getuid())
        (self.out/'administrator-command.txt').write_text(self.root_cmd+'\n')
        self.event('prepared', administrator_command=str(self.out/'administrator-command.txt'))

    def establish(self):
        self.heartbeat_thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.heartbeat_thread.start()
        # Precreate privately owned log; root writes through shell redirection.
        (self.out/'helper.log').touch(mode=0o600)
        if self.a.admin_prompt:
            script = 'on run argv\n do shell script (item 1 of argv) with administrator privileges\nend run'
            self.admin = subprocess.Popen(['osascript', '-e', script, self.root_cmd],
                                          stdout=(self.out/'admin.log').open('w'), stderr=subprocess.STDOUT)
        else:
            self.admin = subprocess.Popen(['sudo', '-n', '/bin/sh', '-c', self.root_cmd],
                                          stdout=(self.out/'admin.log').open('w'), stderr=subprocess.STDOUT)
        deadline = time.monotonic()+300
        while time.monotonic() < deadline:
            lines = (self.out/'helper.log').read_text().splitlines()
            events = [json.loads(x) for x in lines if x.startswith('{')]
            if any(e['event'] == 'helper-error' for e in events):
                raise RuntimeError('Tunnel helper failed; inspect helper.log')
            ready = next((e for e in events if e['event'] == 'helper-ready'), None)
            if ready:
                self.helper_ready = ready
                break
            if self.admin.poll() is not None:
                raise RuntimeError('Administrator launch failed; inspect admin.log')
            time.sleep(.5)
        else:
            raise RuntimeError('Administrator/tunnel readiness deadline exceeded')
        if http('http://127.0.0.1:42314'+REGISTRY_API).get('tunnels') != {}:
            raise RuntimeError('Default registry acquired a tunnel while awaiting authentication')
        self.registry = RegistryLease(self.portfile)
        self.guardian_state = self.out/'guardian-state.json'
        self.guardian_data = {'pid': os.getpid(), 'portfile': str(self.portfile),
                              'original': base64.b64encode(self.portfile.read_bytes()).decode(),
                              'lease': str(self.lease), 'registry_token': self.registry.token,
                              'wake_pid': self.wake.pid if self.wake else None}
        save(self.guardian_state, self.guardian_data)
        self.guardian_process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            '--guardian-state', str(self.guardian_state)], start_new_session=True,
            stdout=(self.out/'guardian.log').open('w'), stderr=subprocess.STDOUT)
        self.registry.__enter__()
        self.event('tunnel-ready', services=self.helper_ready['service_names'])

    def wda_ready(self):
        try:
            return http(self.phone+'/status')['value']['ready'] is True
        except (Exception,):
            return False

    def connect(self):
        if not self.wda_ready():
            self.wda = subprocess.Popen(self.node('wda'), stdout=(self.out/'wda.log').open('w'), stderr=subprocess.STDOUT)
            self.update_guardian(wda_pid=self.wda.pid)
            deadline = time.monotonic()+90
            while not self.wda_ready():
                if self.wda.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Installed WDA runner did not become ready; no rebuild attempted')
                time.sleep(1)
        from appium import webdriver
        from appium.options.ios import XCUITestOptions
        from appium.webdriver.client_config import AppiumClientConfig
        options = XCUITestOptions().load_capabilities({'platformName': 'iOS', 'appium:automationName': 'XCUITest',
                    'appium:udid': self.udid, 'appium:webDriverAgentUrl': self.phone,
                    'appium:platformVersion': self.platform_version,
                    # Silent Pico/analysis stretches must not end the session (default 60 s).
                    'appium:newCommandTimeout': 600,
                    'appium:noReset': True, 'appium:autoLaunch': False, 'appium:skipLogCapture': True})
        self.driver = webdriver.Remote(options=options, client_config=AppiumClientConfig('http://127.0.0.1:4726', timeout=90))
        self.update_guardian(session_id=self.driver.session_id)
        # Launching the runner backgrounds True Skate; resume it (never cold-launch).
        if self.wda and self.driver.query_app_state(BUNDLE) == 2:
            self.driver.activate_app(BUNDLE)
            time.sleep(3)
            self.event('app-resumed-after-wda-launch')
        self.guard(self.out/'initial.png')
        self.event('session-ready')

    def update_guardian(self, **details):
        self.guardian_data.update(details)
        tmp = self.guardian_state.with_suffix('.new')
        save(tmp, self.guardian_data)
        tmp.replace(self.guardian_state)

    def wait_until(self, deadline):
        while time.monotonic() < deadline:
            if not http('http://127.0.0.1:42315'+REGISTRY_API+'/'+self.udid):
                raise RuntimeError('Tunnel lost during recording')
            time.sleep(min(2, max(0, deadline-time.monotonic())))

    def guard(self, screenshot_path=None):
        from PIL import Image
        import io
        from trueskate_ai.collection.gameplay_filter import is_menu_frame, is_editor_frame, is_bolt_modal_frame
        if self.driver.query_app_state(BUNDLE) != 4:
            raise RuntimeError('True Skate is not foreground')
        if http(self.phone+'/wda/activeAppInfo')['value']['bundleId'] != BUNDLE:
            raise RuntimeError('WDA foreground bundle mismatch')
        if self.driver.get_window_size() != {'width': 414, 'height': 896}:
            raise RuntimeError('XR logical geometry mismatch')
        png = self.driver.get_screenshot_as_png()
        if screenshot_path:
            screenshot_path.write_bytes(png)
        if Image.open(io.BytesIO(png)).size != (828, 1792):
            raise RuntimeError('Screenshot geometry mismatch')
        if is_menu_frame(png) or is_editor_frame(png) or is_bolt_modal_frame(png):
            raise RuntimeError('Gameplay guard rejected current scene')

    def prerequisites(self, folder, phase):
        self.usb_absent()
        self.root_alive()
        if not self.wda_ready():
            raise RuntimeError('WDA no longer ready')
        entry = http('http://127.0.0.1:42315'+REGISTRY_API+'/'+self.udid)
        if not entry:
            raise RuntimeError('Diagnostic tunnel no longer active')
        self.guard(folder/(phase+'.png'))
        active = self.driver.execute_script('mobile: getXCTestScreenRecordingInfo', {})
        save(folder/(phase+'-recorder.json'), active)
        if active:
            raise RuntimeError('Recorder is not idle')
        if self.a.tunnel_python:
            output = command(self.attachments('list'), 240)
            summary = json.loads(output.splitlines()[-1])['summary']
        else:
            output = command(['appium', 'driver', 'run', 'xcuitest', 'cleanup-videos', '--', '--udid', self.udid, '--dry-run'], 45)
            summary = output
        (folder/(phase+'-attachments.txt')).write_text(output)
        if 'Found 0 UUID-shaped attachment' not in summary:
            raise RuntimeError('Zero recording attachments not verified')

    def attachments(self, mode, *extra):
        return [self.a.tunnel_python, str(Path(__file__).with_name('ios_wifi_attachments.py')), mode,
                '--address', self.helper_ready['host'], '--port', str(self.helper_ready['rsd_port']),
                '--udid', self.udid, *extra]

    def delete_leftovers(self):
        """Delete operator-named attachments left by an earlier run, only once preserved locally."""
        for uuid_, preserved in self.a.delete_leftover:
            if not (preserved.is_absolute() and preserved.is_file() and preserved.stat().st_size > 0):
                raise RuntimeError(f'Leftover {uuid_} has no preserved local copy; not deleting')
            listed = json.loads(command(self.attachments('list'), 240).splitlines()[-1])
            if uuid_.upper() not in (u.upper() for u in listed['uuids']):
                raise RuntimeError(f'Leftover {uuid_} not present; refusing other deletions')
            command(self.node('delete-attachment') + ['--uuid', uuid_], 60)
            self.event('leftover-attachment-deleted', uuid=uuid_, preserved=str(preserved))

    def wifi_pull(self, uuid_, mov):
        """Pull the stopped recording over the tunnel, then delete it as Appium would."""
        pulled = json.loads(command(self.attachments('pull', '--uuid', uuid_, '--out', str(mov)), 600).splitlines()[-1])
        command(self.node('delete-attachment') + ['--uuid', uuid_], 60)
        return {'wifi_pull': pulled, 'attachment_deleted': True}

    def case(self, name, fps, seconds, samples=None):
        from trueskate_ai.sim.touch_actions import reset_position, long_press, curved_drag
        from trueskate_ai.collection.scene_settle import wait_for_centre_settle
        from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture, validate_action_timing_report
        folder = self.out/name
        folder.mkdir()
        result = {'name': name, 'fps_requested': fps, 'lifecycle': 'failed',
                  'calibration': 'pending' if samples else 'not applicable'}
        self.report['cases'].append(result)
        recording = Recording(self.driver, folder, fps, self.wifi_pull if self.a.tunnel_python else None)
        try:
            self.prerequisites(folder, 'before')
            if samples:
                reset_position(self.driver, 414, 896)
                settle = wait_for_centre_settle(self.driver.get_screenshot_as_png, threshold=1.5, max_wait_s=15)
                save(folder/'settle.json', settle.summary())
                if not settle.settled:
                    raise RuntimeError('Scene did not settle before calibration')
                self.guard()
                self.timing = WDAActionTimingCapture(wda_port=8103, expected_revision=REVISION,
                    request_json=lambda url, payload=None: http(url.replace('http://127.0.0.1:8103', self.phone), payload))
                self.timing.start()
            recording.start()
            began = time.monotonic()
            self.event('recording-started', case=name, seconds=seconds)
            events = []
            if samples:
                schedule = [(2, None, 'start')] + [(6+5*i, g, None) for i, g in enumerate(samples)] + [(53, None, 'end')]
                for offset, gesture, role in schedule:
                    self.wait_until(began+offset)
                    self.guard()
                    epoch = time.time()
                    if role:
                        long_press(self.driver, 207, 448, duration=.05)
                        event = {'gesture_distribution': 'tap', 'point': [.5, .5], 'hold_duration_s': 0,
                                 'calibration_control': True, 'calibration_role': role,
                                 'calibration_execution': 'short_hold', 'calibration_tap_hold_s': .05}
                    else:
                        curved_drag(self.driver, [(x*414, y*896) for x,y in gesture.waypoints], total_duration=gesture.duration, easing=None)
                        event = gesture.meta()
                    event.update(gesture_index=len(events), wda_action_sequence=len(events),
                                 t_call_start_epoch_s=epoch, t_call_end_epoch_s=time.time())
                    events.append(event)
                    self.guard()
                timing_report = self.timing.stop()
                self.timing = None
                save(folder/'wda-action-timing.json', timing_report)
                validate_action_timing_report(timing_report, expected_revision=REVISION, expected_count=10)
            self.wait_until(began+seconds)
            mov, stop = recording.stop()
            self.prerequisites(folder, 'after')
            result['lifecycle'] = 'passed'
            # Movie audit and calibration are local and slow on this rig; run them after the
            # tunnel and WDA are released so capture stays within the helper lifetime.
            self.pending.append({'result': result, 'folder': folder, 'mov': mov, 'stop': stop,
                                 'fps': fps, 'samples': samples, 'events': events})
            self.event('case-recorded', case=name)
        except BaseException as exc:
            result['error'] = str(exc)
            if samples and result['calibration'] == 'pending':
                result['calibration'] = 'not completed'
            try:
                recording.recover()
            except Exception as recovery:
                result['recovery_error'] = str(recovery)
            raise
        finally:
            if self.timing:
                self.timing.cleanup()
                self.timing = None
            save(self.out/'report.json', self.report)

    def analyse(self, item):
        result, folder, mov, fps, samples = (item[k] for k in ('result', 'folder', 'mov', 'fps', 'samples'))
        metrics, pts = audit_movie(mov, fps, bool(samples))
        result['video'] = metrics
        if samples:
            started_at = float(item['stop']['startedAt'])
            manifest = {'gestures': item['events'], 'wda_action_timing_report': 'wda-action-timing.json',
                        'wda_timing_revision': REVISION, 'wda_action_count': 10}
            save(folder/'diagnostic-manifest.json', manifest)
            spec = importlib.util.spec_from_file_location('diagnostic_alignment', self.a.repo/'scripts/collection/align_xctest_traces.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            info, _ = module._wda_two_anchor_calibration(manifest=manifest, manifest_path=folder/'diagnostic-manifest.json',
                mov=mov, started_at=started_at, fps=fps, search_after_s=4, resize_width=256, source_frame_times=pts)
            save(folder/'calibration.json', info)
            result['calibration'] = 'passed'
        return metrics

    def analyse_pending(self):
        """Audit every recorded movie; one failure does not hide the others' results."""
        for item in self.pending:
            result = item['result']
            try:
                metrics = self.analyse(item)
                self.event('case-passed', case=result['name'], metrics=metrics)
            except Exception as exc:
                result['analysis_error'] = str(exc)[:600]
                if item['samples'] and result['calibration'] == 'pending':
                    result['calibration'] = 'not completed'
                self.event('case-analysis-failed', case=result['name'], error=result['analysis_error'])
            save(self.out/'report.json', self.report)

    def cleanup(self):
        for label, action in [('session', lambda: self.driver.quit() if self.driver else None),
                              ('owned-wda', self.stop_wda), ('registry', lambda: self.registry.__exit__(None,None,None) if self.registry else None)]:
            try:
                action()
            except Exception as exc:
                self.report['cleanup_errors'].append({'component': label, 'error': str(exc)})
        self.ending.set()
        # Wait for heartbeat thread to finish before revoking lease.
        if hasattr(self, 'heartbeat_thread'):
            self.heartbeat_thread.join(timeout=3)
        self.lease.unlink(missing_ok=True)
        if self.admin:
            try:
                if not hasattr(self, 'helper_ready') and self.admin.poll() is None:
                    self.admin.terminate()
                self.admin.wait(timeout=12)
                text = (self.out/'helper.log').read_text()
                if '"event":"helper-ready"' in text and '"event":"helper-stopped"' not in text:
                    raise RuntimeError('Native tunnel shutdown not verified')
            except Exception as exc:
                self.report['cleanup_errors'].append({'component': 'native-tunnel', 'error': str(exc)})
        try:
            self.root_alive()
            if self.registry and self.portfile.read_bytes() != self.registry.original:
                raise RuntimeError('Original registry value not restored')
            http('http://127.0.0.1:42314'+REGISTRY_API)
        except Exception as exc:
            self.report['cleanup_errors'].append({'component': 'postflight', 'error': str(exc)})
        save(self.out/'report.json', self.report)
        if self.guardian_process:
            self.guardian_state.unlink(missing_ok=True)
            self.guardian_process.wait(timeout=3)
        self.event('capture-released', cleanup_errors=self.report['cleanup_errors'])

    def stop_wda(self):
        if self.wda and self.wda.poll() is None:
            self.wda.terminate()
            self.wda.wait(timeout=12)

    def run(self):
        try:
            self.preflight()
            if self.a.prepare:
                return
            # Hold a bounded idle-sleep assertion through capture and deferred analysis.
            self.wake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-t', '1800'])
            self.establish()
            self.delete_leftovers()
            self.connect()
            if not self.a.pico_only:
                self.case('01-short-30', 30, 5)
                self.case('02-short-60', 60, 5)
                import numpy as np
                from trueskate_ai.data.gesture_sampling import sample_basic_linear_mixture
                rng = np.random.default_rng(20261006)
                samples = [sample_basic_linear_mixture(rng, tap_fraction=0) for _ in range(8)]
                save(self.out/'fixed-gestures.json', [g.meta() for g in samples])
                self.case('03-minute-30', 30, 60, samples)
                self.case('04-minute-60', 60, 60, samples)
            if self.a.pico_hover:
                self.event('waiting-pico', instruction='Keep Pico disconnected; create pico-ready when operator is ready to plug after recording-started')
                deadline = time.monotonic()+120
                while not (self.out/'pico-ready').exists():
                    if time.monotonic()>deadline:
                        raise RuntimeError('Pico readiness deadline; no recording retry')
                    time.sleep(.5)
                self.case('05-pico-hover-60', 60, self.a.pico_seconds)
                self.report['pico_acceptance'] = 'requires visual review of original movie and insertion timing'
        except BaseException as exc:
            self.report['error'] = str(exc)
            self.event('aborted', error=str(exc))
            raise
        finally:
            self.cleanup()
            self.analyse_pending()
            if self.wake and self.wake.poll() is None:
                self.wake.terminate()
                self.wake.wait(timeout=3)
            self.event('finished', report=str(self.out/'report.json'), cleanup_errors=self.report['cleanup_errors'])


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--guardian-state':
        guardian(sys.argv[2])
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--env-file', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--modules-root', type=Path, default=Path('/Users/training-server/.appium/node_modules/appium-xcuitest-driver/node_modules'))
    p.add_argument('--host', default='Test-XR-2.local')
    p.add_argument('--node', default='/usr/local/bin/node')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--admin-prompt', action='store_true')
    p.add_argument('--pico-hover', action='store_true')
    p.add_argument('--pico-only', action='store_true', help='skip cases 1-4; record only the Pico case')
    p.add_argument('--pico-seconds', type=int, default=45, choices=range(10, 181), metavar='10..180')
    p.add_argument('--tunnel-python', type=Path,
                   help='Python 3.13+ with pymobiledevice3: use the Wi-Fi RemotePairing tunnel')
    p.add_argument('--pair-dir', type=Path, default=Path('/Users/training-server/.pymobiledevice3'))
    p.add_argument('--delete-leftover', nargs=2, action='append', default=[], metavar=('UUID', 'PRESERVED_MOV'),
                   help='Wi-Fi mode: delete one earlier diagnostic attachment already copied to PRESERVED_MOV')
    a = p.parse_args()
    a.delete_leftover = [(u, Path(m)) for u, m in a.delete_leftover]
    if a.pico_only and not a.pico_hover:
        p.error('--pico-only needs --pico-hover')
    if a.delete_leftover and not a.tunnel_python:
        p.error('--delete-leftover needs --tunnel-python')
    for path in (a.repo, a.env_file, a.out_dir, a.modules_root, a.pair_dir, *([a.tunnel_python] if a.tunnel_python else [])):
        if not path.is_absolute():
            p.error('All paths must be absolute')
    if 'tmp' not in a.out_dir.parts:
        p.error('Diagnostic output must be isolated under tmp')
    sys.path.insert(0, str(a.repo/'src'))
    def interrupted(signum, _):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Coordinator(a).run()


if __name__ == '__main__':
    main()
