"""Appium startup must preserve the private control ownership contract."""
import importlib.util
import plistlib
from pathlib import Path
from unittest.mock import Mock


def test_appium_start_is_private_and_allows_ownership_discovery(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'launch_services', Path(__file__).parents[1] / 'scripts/launch_services.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    health = iter([False, True, True])
    monkeypatch.setattr(launcher, '_is_service_responding', lambda *a, **k: next(health))
    monkeypatch.setattr(launcher, '_is_port_in_use', lambda *a: False)
    monkeypatch.setattr(launcher, '_drain_pipe', lambda *a: None)
    monkeypatch.setattr(launcher.time, 'sleep', lambda *a: None)
    proc = Mock()
    proc.poll.return_value = None
    spawn = Mock(return_value=proc)
    monkeypatch.setattr(launcher.subprocess, 'Popen', spawn)
    launcher._processes['synthetic'] = {}
    assert launcher._start_appium({'name': 'synthetic', 'appium_port': 4723})
    args = spawn.call_args.args[0]
    assert args[args.index('--address') + 1] == '127.0.0.1'
    assert args[args.index('--allow-insecure') + 1] == 'xcuitest:xctest_screen_record,*:session_discovery'


def test_managed_appium_jobs_preserve_private_discovery():
    root = Path(__file__).parents[1] / 'scripts/ops'
    for name, port in [('xr1', '4723'), ('xr2', '4726')]:
        with (root / f'com.trueskate.appium.{name}.plist').open('rb') as stream:
            job = plistlib.load(stream)
        args = job['ProgramArguments']
        assert args[args.index('--port') + 1] == port
        assert args[args.index('--address') + 1] == '127.0.0.1'
        assert args[args.index('--allow-insecure') + 1] == 'xcuitest:xctest_screen_record,*:session_discovery'
