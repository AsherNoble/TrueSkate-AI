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
    with pytest.raises(ValueError, match='duration'):
        diag.movie_metrics(dict(stream, duration=80), pts, 1800, 1800, 30, True)


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


def test_deferred_analysis_reports_each_case_and_isolates_failures(tmp_path, monkeypatch):
    coord = object.__new__(diag.Coordinator)
    coord.out, coord.events = tmp_path, tmp_path/'events.jsonl'
    ok = {'name': 'ok', 'lifecycle': 'passed', 'calibration': 'not applicable'}
    bad = {'name': 'bad', 'lifecycle': 'passed', 'calibration': 'pending'}
    coord.report = {'cases': [ok, bad]}
    coord.pending = [{'result': ok, 'folder': tmp_path, 'mov': tmp_path/'ok.mov', 'stop': {}, 'fps': 30,
                      'samples': None, 'events': []},
                     {'result': bad, 'folder': tmp_path, 'mov': tmp_path/'bad.mov', 'stop': {'startedAt': 1},
                      'fps': 60, 'samples': ['g'], 'events': []}]
    def audit(mov, fps, minute):
        if mov.name == 'bad.mov':
            raise ValueError('Minute duration or cadence gate failed')
        return {'frames': 151}, []
    monkeypatch.setattr(diag, 'audit_movie', audit)
    coord.analyse_pending()
    assert ok['video'] == {'frames': 151} and 'analysis_error' not in ok
    assert bad['analysis_error'] == 'Minute duration or cadence gate failed' and bad['calibration'] == 'not completed'
    events = [json.loads(x)['event'] for x in coord.events.read_text().splitlines()]
    assert events == ['case-passed', 'case-analysis-failed']
    assert json.loads((tmp_path/'report.json').read_text())['cases'][1]['analysis_error']


def test_operator_ntfy_disables_dedupe_and_reports_failed_sends(monkeypatch):
    import logging
    from trueskate_ai.utils import notify as ntfy
    calls = []
    monkeypatch.setattr(ntfy, 'is_configured', lambda: True)
    def failing(message, **kw):
        calls.append(kw)
        logging.warning('ntfy notification failed: %s', 'timed out')
    monkeypatch.setattr(ntfy, 'notify', failing)
    op = diag.Operator(lambda *a, **k: None, True)
    assert 'timed out' in op.ntfy('PLUG THE PICO IN NOW', 'urgent')
    assert calls[0]['dedupe_s'] == 0 and calls[0]['block'] is True
    monkeypatch.setattr(ntfy, 'notify', lambda message, **kw: None)
    assert op.ntfy('again', 'urgent') is None
    monkeypatch.setattr(ntfy, 'is_configured', lambda: False)
    assert op.ntfy('again', 'urgent') == 'ntfy not configured'


def test_operator_alert_uses_all_channels_only_when_enabled(monkeypatch, capsys):
    events, spoken = [], []
    monkeypatch.setattr(diag.subprocess, 'Popen', lambda argv, **kw: spoken.append(argv))
    op = diag.Operator(lambda name, **kw: events.append((name, kw)), False)
    op.alert('PLUG THE PICO IN NOW', 'urgent', 'Plug the Pico in now')
    assert not events and not spoken and capsys.readouterr().out == ''
    op = diag.Operator(lambda name, **kw: events.append((name, kw)), True)
    monkeypatch.setattr(op, 'ntfy', lambda message, priority: None)
    op.alert('PLUG THE PICO IN NOW', 'urgent', 'Plug the Pico in now')
    assert 'PLUG THE PICO IN NOW' in capsys.readouterr().out
    assert spoken == [['/usr/bin/say', 'Plug the Pico in now']]
    assert events == [('operator-alert', {'message': 'PLUG THE PICO IN NOW', 'ntfy_error': None})]


def test_operator_ready_by_file_and_times_out(tmp_path):
    op = diag.Operator(lambda *a, **k: None, False)
    ready = tmp_path/'pico-ready'
    assert op.prompt_ready('prompt', None, ready, 0.6) is False
    ready.touch()
    assert op.prompt_ready('prompt', None, ready, 5) is True


def test_operator_enter_counts_only_after_the_prompt_even_while_ntfy_sends(tmp_path, monkeypatch, capsys):
    import os
    import threading
    import time as _time
    master, slave = os.openpty()
    monkeypatch.setattr(diag, 'open', lambda path, *a, **k: os.fdopen(os.dup(slave), 'r'), raising=False)
    monkeypatch.setattr(diag.subprocess, 'Popen', lambda *a, **k: None)
    events = []
    op = diag.Operator(lambda name, **kw: events.append(name), True)
    sending = threading.Event()
    def slow_ntfy(message, priority):
        sending.set()
        _time.sleep(1.5)
        return None
    monkeypatch.setattr(op, 'ntfy', slow_ntfy)
    os.write(master, b'\n')                                 # stray Enter before the prompt
    def operator():
        sending.wait(5)
        os.write(master, b'\n')                             # Enter while ntfy is still sending
    threading.Thread(target=operator, daemon=True).start()
    began = _time.monotonic()
    assert op.prompt_ready('PICO STEP NEXT', None, tmp_path/'never', 10) is True
    assert _time.monotonic() - began < 1.4                  # did not wait for the slow ntfy
    out = capsys.readouterr().out
    assert 'press Enter' in out and str(tmp_path/'never') in out and 'Ready received.' in out
    os.close(master); os.close(slave)


def test_operator_stray_enter_alone_does_not_count(tmp_path, monkeypatch):
    import os
    master, slave = os.openpty()
    monkeypatch.setattr(diag, 'open', lambda path, *a, **k: os.fdopen(os.dup(slave), 'r'), raising=False)
    monkeypatch.setattr(diag.subprocess, 'Popen', lambda *a, **k: None)
    op = diag.Operator(lambda *a, **k: None, True)
    monkeypatch.setattr(op, 'ntfy', lambda message, priority: None)
    os.write(master, b'\n')
    assert op.prompt_ready('PICO STEP NEXT', None, tmp_path/'never', 1.0) is False
    os.close(master); os.close(slave)


def test_failed_after_check_still_queues_the_recorded_movie(tmp_path, monkeypatch):
    from types import SimpleNamespace
    coord = object.__new__(diag.Coordinator)
    coord.out, coord.events = tmp_path, tmp_path/'events.jsonl'
    coord.a = SimpleNamespace(tunnel_python=None)
    coord.report, coord.pending, coord.timing = {'cases': []}, [], None
    coord.driver = Driver()
    def prerequisites(folder, phase):
        if phase == 'after':
            raise RuntimeError('Gameplay guard rejected current scene')
    monkeypatch.setattr(coord, 'prerequisites', prerequisites)
    monkeypatch.setattr(coord, 'wait_until', lambda deadline: None)
    with pytest.raises(RuntimeError, match='Gameplay guard'):
        coord.case('05-pico-hover-60', 60, 1)
    assert len(coord.pending) == 1 and coord.pending[0]['mov'].read_bytes() == b'fixture'
    assert coord.report['cases'][0]['lifecycle'] == 'failed'


def test_completion_speech_finishes_or_is_terminated_before_volume_restore(monkeypatch):
    events = []
    class Speech:
        def __init__(self, hung):
            self.hung, self.calls = hung, []
        def wait(self, timeout):
            self.calls.append(('wait', timeout))
            if self.hung:
                raise diag.subprocess.TimeoutExpired('say', timeout)
        def terminate(self):
            self.calls.append(('terminate',))
            self.hung = False
    good, hung = Speech(False), Speech(True)
    op = diag.Operator(lambda name, **kw: events.append(name), True)
    op.speech = [good, hung]
    op.finish_speech(.1)
    assert good.calls[0][0] == 'wait' and hung.calls[1][0] == 'terminate'
    assert op.speech == [] and events == ['speech-timeout']


def test_preloaded_pico_capture_uses_minute_gate_without_wda_calibration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    coord = object.__new__(diag.Coordinator)
    coord.a = SimpleNamespace(pico_program=tmp_path/'program.json')
    calls = []
    monkeypatch.setattr(diag, 'audit_movie', lambda mov, fps, minute: (calls.append((fps, minute)) or {}, []))
    result = {}
    coord.analyse(dict(result=result, folder=tmp_path, mov=tmp_path/'original.mov', fps=60, samples=None))
    assert calls == [(60, True)] and result['video'] == {}


def test_insertion_alert_network_delay_does_not_hold_the_capture_clock(monkeypatch):
    import threading
    import time as _time
    pending, release, done = threading.Event(), threading.Event(), threading.Event()
    op = diag.Operator(lambda *a, **kw: done.set(), True)
    monkeypatch.setattr(op, 'show', lambda *a: None)
    def slow_send(*a):
        pending.set()
        release.wait(5)
    monkeypatch.setattr(op, 'ntfy', slow_send)
    started = _time.monotonic()
    op.alert('PLUG NOW', 'urgent', background=True)
    assert _time.monotonic()-started < .2
    assert pending.wait(1) and not done.is_set()
    release.set()
    assert done.wait(1)


def test_deferred_analysis_failure_is_a_failed_run_after_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace
    coord = object.__new__(diag.Coordinator)
    coord.a = SimpleNamespace(pico_program=None, prepare=True)
    coord.out, coord.wake = tmp_path, None
    coord.report = dict(cases=[dict(lifecycle='passed')], cleanup_errors=[])
    order = []
    coord.operator = SimpleNamespace(alert=lambda *a: order.append(a[-1]), finish_speech=lambda: order.append('speech-done'))
    monkeypatch.setattr(coord, 'preflight', lambda: None)
    monkeypatch.setattr(coord, 'cleanup', lambda: order.append('cleanup'))
    def analyse():
        coord.report['cases'][0]['analysis_error'] = 'cadence failed'
        order.append('analyse')
    monkeypatch.setattr(coord, 'analyse_pending', analyse)
    monkeypatch.setattr(coord, 'event', lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match='analysis or cleanup'):
        coord.run()
    assert order == ['cleanup', 'analyse', 'Run aborted', 'speech-done']
    assert json.loads((tmp_path/'report.json').read_text())['outcome'] == 'failed'
