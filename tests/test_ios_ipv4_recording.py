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
    lease.write_text('fixture')
    diag.save(state, {'pid': 999999, 'wda_pid': 12345, 'portfile': str(port),
                      'original': base64.b64encode(b'42314\n').decode(), 'lease': str(lease)})
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
