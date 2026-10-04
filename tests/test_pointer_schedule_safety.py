import importlib.util
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hardware/hid_pointer'))
from hid_bridge import Bridge
from hid_client import PointerClient
from trueskate_ai.control import hid_pointer as hp

spec = importlib.util.spec_from_file_location('pointer_runner', ROOT / 'scripts/collection/run_pointer_schedule.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
EVENTS = [(0, 0, 0, 1), (15_000, 1, -1, 1), (16_000, 0, 0, 0)]
RECEIPTS = ['SENT 0 0', 'SENT 1 15000', 'SENT 2 16000', 'DONE 3']


@pytest.mark.parametrize('lines', [
    RECEIPTS[:2] + ['DONE 2'], ['SENT 1 0'] + RECEIPTS[1:],
    RECEIPTS[:1] + RECEIPTS, RECEIPTS[:-1], RECEIPTS + ['DONE 3'],
    RECEIPTS + ['ERR lost'], RECEIPTS + ['ABORT 3 disconnect'],
    ['SENT 0 -1'] + RECEIPTS[1:], ['SENT 0 0 extra'] + RECEIPTS[1:],
    ['SENT 0 0', 'SENT 1 14999'] + RECEIPTS[2:],
    ['DONE 3'] + RECEIPTS, RECEIPTS + ['SENT 3 17000'],
])
def test_corrupt_receipts_fail_closed(lines):
    with pytest.raises(RuntimeError):
        runner.validate_receipts(lines, EVENTS)


def test_valid_receipts_are_attempts_with_exact_order():
    assert runner.validate_receipts(RECEIPTS, EVENTS) == [0, 15000, 16000]


@pytest.mark.parametrize('events', [
    [(-1, 0, 0, 0)], [(2**32, 0, 0, 0)], [(0, 128, 0, 0)],
    [(0, 0, 0, 2)], [(0, 0, 0, 1)], [(0, 0, 0, False)],
    [(0, 0, 0, 0), (0, 0, 0, 0)], [(0, 0, 0, 0), (58_000_000, 0, 0, 0)],
    [(0, 0, 0, 0)] * 2049,
])
def test_schedule_parser_and_sparse_budget(events):
    with pytest.raises(ValueError):
        runner.validate_schedule(events)


@pytest.mark.parametrize('samples', [
    [(-1, 20, 200), (0, 21, 201)], [(0, 20, 200), (1e10, 21, 201)],
    [(0, 20, 200), (0, 21, 201)], [(0, 20, 200), (1, float('nan'), 201)],
])
def test_invalid_strokes_refused_before_allocating_schedule(samples):
    with pytest.raises(ValueError):
        hp.build_schedule([hp.Stroke('bad', samples)])


@pytest.mark.parametrize('state,active', [(4, None), (4, 'com.apple.springboard'), (3, 'skate'), (None, 'skate')])
def test_foreground_unavailable_or_contradictory(monkeypatch, state, active):
    driver = type('Driver', (), {'query_app_state': lambda self, bundle: state})()
    monkeypatch.setattr(runner, 'http_json', lambda *a, **k: {'value': {'bundleId': active}})
    with pytest.raises(RuntimeError, match='foreground'):
        runner.guard_foreground(driver, 'http://unused', 'skate')


class Clock:
    now = 0.0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class Recorder:
    def __init__(self, clock, fail_stop=False, start_delay=0):
        self.clock, self.fail_stop, self.start_delay = clock, fail_stop, start_delay
        self.starts = self.stops = self.aborts = 0
    def start(self):
        self.starts += 1
        self.clock.now += self.start_delay
    def stop_and_save(self, path):
        self.stops += 1
        if self.fail_stop:
            raise RuntimeError('retrieval disconnected')
        path.write_bytes(b'partial video')
        return 'recording-result'


class Client:
    def __init__(self):
        self.cancelled = self.go = 0
    def cancel(self):
        self.cancelled += 1


def run_fixture(monkeypatch, tmp_path, *, lost=False, fail_stop=False, start_delay=0, bad_receipts=False):
    clock, client = Clock(), Client()
    recorder = Recorder(clock, fail_stop, start_delay)
    monkeypatch.setattr(runner, 'upload', lambda *a: None)
    def guard():
        if lost and clock.now >= 4:
            raise RuntimeError('foreground changed during recording lead sleep')
    def play(*a, **k):
        client.go += 1
        if bad_receipts:
            return runner.validate_receipts(RECEIPTS + ['ABORT 3 link'], EVENTS)
        clock.now += .1
        return {'receipts': RECEIPTS}
    monkeypatch.setattr(runner, 'play', play)
    invoke = lambda: runner.run_once(client=client, events=EVENTS, recorder=recorder, guard=guard,
                                    reset=lambda: None, out=tmp_path / 'run.mov', clock=clock, sleep=clock.sleep)
    return invoke, recorder, client


def test_valid_bounded_run(monkeypatch, tmp_path):
    invoke, recorder, client = run_fixture(monkeypatch, tmp_path)
    assert invoke()[0] == 'recording-result'
    assert (recorder.starts, recorder.stops, client.go, client.cancelled) == (1, 1, 1, 0)


@pytest.mark.parametrize('options', [dict(lost=True), dict(start_delay=59), dict(bad_receipts=True)])
def test_failure_after_start_retains_partial_without_retry(monkeypatch, tmp_path, options):
    invoke, recorder, client = run_fixture(monkeypatch, tmp_path, **options)
    with pytest.raises(RuntimeError):
        invoke()
    assert recorder.starts == recorder.stops == 1
    assert client.cancelled == 1
    assert (tmp_path / 'run.mov').read_bytes() == b'partial video'
    assert json.loads((tmp_path / 'run.failure.json').read_text())['partial_video']
    if not options.get('bad_receipts'):
        assert client.go == 0


def test_failed_stop_is_never_retried(monkeypatch, tmp_path):
    invoke, recorder, client = run_fixture(monkeypatch, tmp_path, fail_stop=True)
    with pytest.raises(RuntimeError, match='retrieval disconnected'):
        invoke()
    assert recorder.starts == recorder.stops == 1
    assert recorder.aborts == 0
    assert json.loads((tmp_path / 'run.retrieval-error.json').read_text())['error'] == 'retrieval disconnected'


def test_older_firmware_rejected_before_commands():
    class Old:
        def ask(self, command, wait):
            assert command == 'HELLO 2'
            return ['ERR unknown']
    with pytest.raises(RuntimeError, match='protocol v2'):
        runner.board_ready(Old())


def test_error_after_done_in_same_batch_is_not_discarded():
    client = object.__new__(PointerClient)
    client.read_lines = lambda wait: ['DONE 3', 'ABORT 3 lost']
    with pytest.raises(RuntimeError, match='aborted'):
        client.wait_for('DONE', timeout=1)


def test_client_eof_is_failure():
    left, right = socket.socketpair()
    client = object.__new__(PointerClient)
    client.sock, client.pending = left, b''
    right.close()
    try:
        with pytest.raises(ConnectionError):
            client.read_lines(.2)
    finally:
        left.close()


@pytest.mark.parametrize('bad', [b'X' * 64, b'GO\x00\n', b'\n'])
def test_bridge_owner_disconnect_and_malformed_lines_cancel(bad):
    class Serial:
        def __init__(self): self.writes = []
        def write(self, line): self.writes.append(line)
    serial = Serial()
    bridge = Bridge(serial, io.StringIO())
    owner, peer = socket.socketpair()
    assert bridge.claim(owner)
    competitor, rejected = socket.socketpair()
    assert not bridge.claim(competitor)
    assert rejected.recv(100) == b'ERR busy\n'
    rejected.close()
    thread = threading.Thread(target=bridge.serve, args=(owner,))
    thread.start()
    peer.sendall(bad)
    peer.close()
    thread.join(2)
    assert not thread.is_alive()
    assert bridge.owner is None
    assert serial.writes == [b'CANCEL\n']


def test_portable_firmware_parser_and_abort_engine(tmp_path):
    compiler = shutil.which('g++') or shutil.which('clang++')
    assert compiler, 'native C++ compiler required for firmware protocol regression'
    source = tmp_path / 'protocol.cpp'
    source.write_text(r'''
#include "schedule_protocol.h"
#include <cassert>
#include <cstring>
using namespace pointer_protocol;
int main() {
  int64_t v[4];
  assert(numbers("0 -127 127 1", v, 4));
  for (const char *text : {"0 1 2 0 extra", "0 1 2", "0 1 2 0x1", "18446744073709551616 0 0 0", "+0 0 0 0"})
    assert(!numbers(text, v, 4));
  Schedule s;
  for (int64_t t : {-1LL, 4294967296LL, 60000001LL}) { int64_t e[] = {t, 0, 0, 0}; assert(!s.add(e)); }
  for (int64_t b : {-1LL, 2LL}) { int64_t e[] = {0, 0, 0, b}; assert(!s.add(e)); }
  int64_t a[] = {0, 0, 0, 1}, b[] = {15000, 0, 0, 0};
  assert(s.add(a) && s.add(b));
  assert(!s.begin(0, true)); s.neutral(true); assert(s.begin(0, true));
  int attempts = 0;
  auto notify = [&](const Event &) { ++attempts; return true; };
  assert(s.tick(0, true, false, notify) == Tick::sent);
  assert(s.tick(100, false, false, notify) == Tick::aborted);
  assert(attempts == 1 && s.needsNeutral && !s.playing);
  assert(!s.begin(100, true)); s.neutral(false); assert(!s.begin(100, true));
  s.neutral(true); assert(s.begin(100, true));
  assert(s.tick(100, true, true, notify) == Tick::aborted); // latched disconnect/auth/subscription failure after quick recovery
  s.neutral(true); assert(s.begin(100, true));
  s.abort(); assert(!s.begin(100, true)); // CANCEL requires neutral
  s.neutral(true); assert(s.begin(100, true));
  assert(s.tick(100, true, false, [](const Event &) { return false; }) == Tick::aborted);
  s.neutral(true); assert(s.begin(100, true));
  assert(s.tick(100, true, false, notify) == Tick::sent);
  assert(s.tick(15100, true, false, notify) == Tick::done);
  assert(s.next == 2 && s.attemptedAt[0] == 0 && s.attemptedAt[1] == 15000);
  LineBuffer line;
  for (int i = 0; i < 63; ++i) assert(line.push('X') == LineResult::pending);
  assert(line.push('X') == LineResult::invalid);
  for (char c : "GO") assert(line.push(c) == LineResult::pending);
  assert(line.push('\n') == LineResult::pending); // no truncated prefix executes
  assert(line.push('G') == LineResult::pending && line.push('O') == LineResult::pending);
  assert(line.push('\n') == LineResult::complete && strcmp(line.text, "GO") == 0);
  assert(line.push(0) == LineResult::invalid);
}
'''.replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>'))
    executable = tmp_path / 'protocol'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'hardware/hid_pointer'),
                    str(source), '-o', str(executable)], check=True, capture_output=True)
    subprocess.run([str(executable)], check=True, capture_output=True)
