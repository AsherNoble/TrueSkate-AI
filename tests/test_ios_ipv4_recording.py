import base64
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('ipv4_diagnostic', Path(__file__).parents[1]/'scripts/ops/validate_ios_ipv4_recording.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_registry_restores_exact_bytes_on_interruption(tmp_path):
    port = tmp_path/'port'
    port.write_bytes(b'42314\n')
    port.chmod(0o600)
    with pytest.raises(KeyboardInterrupt):
        with diag.RegistryLease(port):
            assert port.read_bytes() == b'42315'
            with pytest.raises(FileExistsError):
                with diag.RegistryLease(port):
                    pass
            raise KeyboardInterrupt()
    assert port.read_bytes() == b'42314\n'
    assert port.stat().st_mode & 0o777 == 0o600
    assert not port.with_name('port.ipv4-diagnostic-lock').exists()


def test_registry_does_not_overwrite_external_owner(tmp_path):
    port = tmp_path/'port'
    port.write_bytes(b'42314')
    with pytest.raises(RuntimeError, match='changed externally'):
        with diag.RegistryLease(port):
            port.write_bytes(b'42316')
    assert port.read_bytes() == b'42316'


class Driver:
    def __init__(self, fail_start=False, fail_stop=False):
        self.calls = []
        self.fail_start, self.fail_stop = fail_start, fail_stop

    def execute_script(self, cmd, opts):
        self.calls.append(cmd)
        if 'startXCTest' in cmd:
            if self.fail_start:
                raise TimeoutError('response lost after device start')
            return {'uuid': 'diagnostic'}
        if self.fail_stop:
            raise TimeoutError('transport lost on stop')
        return {'uuid': 'diagnostic', 'payload': base64.b64encode(b'fixture').decode()}


def test_uncertain_start_stops_once_and_preserves_recovery(tmp_path):
    driver = Driver(fail_start=True)
    rec = diag.Recording(driver, tmp_path, 30)
    with pytest.raises(TimeoutError):
        rec.start()
    rec.recover()
    rec.recover()
    assert len(driver.calls) == 2
    assert (tmp_path/'recovery.mov').read_bytes() == b'fixture'
    assert json.loads((tmp_path/'stop.json').read_text())['recovery']


def test_failed_stop_is_not_retried(tmp_path):
    driver = Driver(fail_stop=True)
    rec = diag.Recording(driver, tmp_path, 30)
    rec.start()
    with pytest.raises(TimeoutError):
        rec.stop()
    rec.recover()
    assert len(driver.calls) == 2


def test_metrics_reject_bad_pts_truncated_counts_and_low_cadence():
    stream = {'codec_name':'h264', 'width':828, 'height':1792, 'duration':60, 'nb_read_frames':1800}
    pts = [i/30 for i in range(1800)]
    assert diag.movie_metrics(stream, pts, 1800, 1800, 30, True)['effective_fps'] == 30
    for bad_pts, decoded, packets in [(pts, 1799, 1800), (pts, 1800, 1799), ([0, float('nan')], 1800, 1800), (list(reversed(pts)),1800,1800)]:
        with pytest.raises(ValueError):
            diag.movie_metrics(stream, bad_pts, decoded, packets, 30, True)
    with pytest.raises(ValueError, match='cadence'):
        diag.movie_metrics(stream, pts, 1800, 1800, 60, True)


def test_full_decode_rejects_truncated_movie(tmp_path):
    import shutil
    if not shutil.which('ffprobe'):
        pytest.skip('ffprobe unavailable')
    mov = tmp_path/'truncated.mov'
    mov.write_bytes(b'\x00\x00\x00\x14ftypqt  ')
    with pytest.raises(Exception):
        diag.audit_movie(mov, 30, False)


def test_guardian_restores_registry_and_terminates_only_owned_runner(tmp_path, monkeypatch):
    port, lease, state = tmp_path/'port', tmp_path/'lease', tmp_path/'state.json'
    port.write_bytes(b'42315')
    port.with_name('port.ipv4-diagnostic-lock').mkdir()
    diag.save(port.with_name('port.ipv4-diagnostic-lock')/'owner.json', {'token':'test-owned'})
    lease.write_text('fixture')
    diag.save(state, {'pid': 999999, 'wda_pid': 12345, 'portfile': str(port),
                      'original': base64.b64encode(b'42314\n').decode(), 'lease': str(lease), 'registry_token':'test-owned'})
    calls = []
    def kill(pid, sig):
        calls.append((pid, sig))
        if pid == 999999:
            raise ProcessLookupError()
    monkeypatch.setattr(diag.os, 'kill', kill)
    diag.guardian(state)
    assert calls == [(999999, 0), (12345, diag.signal.SIGTERM)]
    assert port.read_bytes() == b'42314\n'
    assert not lease.exists()
    assert json.loads(state.with_suffix('.result.json').read_text())['errors'] == []


def test_guardian_does_not_restore_another_registry_owner(tmp_path, monkeypatch):
    port, lease, state = tmp_path/'port', tmp_path/'lease', tmp_path/'state.json'
    port.write_bytes(b'42315')
    lock = port.with_name('port.ipv4-diagnostic-lock')
    lock.mkdir()
    diag.save(lock/'owner.json', {'token':'another-owner'})
    lease.write_text('fixture')
    diag.save(state, {'pid': 999999, 'portfile': str(port), 'original': base64.b64encode(b'42314').decode(),
                      'lease': str(lease), 'registry_token':'our-owner'})
    def dead(*_):
        raise ProcessLookupError()
    monkeypatch.setattr(diag.os, 'kill', dead)
    diag.guardian(state)
    assert port.read_bytes() == b'42315'
    assert json.loads((lock/'owner.json').read_text())['token'] == 'another-owner'


def test_registry_owner_change_prevents_restore_and_lock_removal(tmp_path):
    port = tmp_path/'port'
    port.write_bytes(b'42314')
    with pytest.raises(RuntimeError, match='ownership changed'):
        with diag.RegistryLease(port) as registry:
            diag.save(registry.lock/'owner.json', {'token':'another-owner'})
    assert port.read_bytes() == b'42315'
    assert registry.lock.exists()


@pytest.mark.parametrize('helper, expected', [('printf 42315 > {f}', b'42314\n'), ('true', b'42314\n')])
def test_root_lend_wrapper_restores_redirect_and_ownership(tmp_path, helper, expected):
    import os
    import shlex
    import subprocess
    box = tmp_path/'strongbox'
    box.mkdir()
    port = box/'tunnelRegistryPort'
    port.write_bytes(b'42314\n')
    helper = helper.format(f=shlex.quote(str(port)))
    cmd = diag.lend_registry(helper, port, tmp_path/'backup', os.getuid())
    result = subprocess.run(['/bin/sh', '-c', cmd], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert port.read_bytes() == expected
    assert port.stat().st_uid == os.getuid() and box.stat().st_uid == os.getuid()


class WifiDriver:
    def __init__(self, error):
        self.calls, self.error = [], error

    def execute_script(self, cmd, opts):
        self.calls.append(cmd)
        if 'getXCTest' in cmd:
            return {'uuid': 'ABC', 'startedAt': 1.5, 'fps': 30}
        if 'startXCTest' in cmd:
            return {'uuid': 'ABC'}
        raise RuntimeError(self.error)


def test_wifi_stop_pulls_once_when_appium_cannot_locate_movie(tmp_path):
    driver = WifiDriver("Unable to locate XCTest screen recording identified by 'ABC' for the device x")
    pulls = []
    def pull(uuid_, mov):
        pulls.append(uuid_)
        mov.write_bytes(b'movie')
        return {'attachment_deleted': True}
    rec = diag.Recording(driver, tmp_path, 30, pull)
    rec.start()
    mov, meta = rec.stop()
    rec.recover()
    assert pulls == ['ABC'] and mov.read_bytes() == b'movie'
    assert meta['startedAt'] == 1.5 and meta['attachment_deleted']
    assert driver.calls.count('mobile: stopXCTestScreenRecording') == 1


def test_wifi_stop_other_errors_propagate_without_pull(tmp_path):
    rec = diag.Recording(WifiDriver('transport lost on stop'), tmp_path, 30, lambda *_: pytest.fail('pulled'))
    rec.start()
    with pytest.raises(RuntimeError, match='transport lost'):
        rec.stop()


def test_wifi_attachment_summary_requires_one_listing():
    spec = importlib.util.spec_from_file_location('wifi_attachments', Path(__file__).parents[1]/'scripts/ops/ios_wifi_attachments.py')
    att = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(att)
    uid = '0A0B0C0D-1111-2222-3333-444455556666'
    assert att.summarise({'Attachments': [], 'tmp/Attachments': {'error': 'x'}})['summary'].startswith('Found 0 UUID-shaped attachment')
    assert att.summarise({'Attachments': [uid, 'notes.txt']})['uuids'] == [uid]
    with pytest.raises(RuntimeError):
        att.summarise({'Attachments': {'error': 'x'}, 'tmp/Attachments': {'error': 'y'}})


def test_wifi_file_service_uses_odd_ids_and_strict_data_headers():
    import struct
    spec = importlib.util.spec_from_file_location('wifi_attachments', Path(__file__).parents[1]/'scripts/ops/ios_wifi_attachments.py')
    att = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(att)
    class Service:
        next_message_id = {1: 2, 3: 0}
    class FS:
        service = Service()
    att.odd_id(FS)
    assert FS.service.next_message_id[1] == 3
    att.odd_id(FS)
    assert FS.service.next_message_id[1] == 3
    header = b'rwb!FILE' + struct.pack('>QQQQ', 1, 0, 2, 2894193)
    assert att.data_header(header) == (1, 2, 2894193)
    for bad in (header[:39], b'xxxxFILE' + header[8:]):
        with pytest.raises(RuntimeError):
            att.data_header(bad)


def test_wda_keepalive_queries_app_state_and_screenshot(monkeypatch):
    calls = []
    def fake_http(url, payload=None, timeout=10):
        calls.append((url.rsplit('/', 1)[-1], payload))
        return {'sessionId': 'S1'} if url.endswith('/status') else {}
    monkeypatch.setattr(diag, 'http', fake_http)
    keepalive = diag.WdaKeepAlive('http://phone:8100', period=0.01)
    with keepalive:
        import time as _t
        _t.sleep(0.05)
    assert ('state', {'bundleId': diag.BUNDLE}) in calls and ('screenshot', None) in calls
    assert keepalive.errors == []
