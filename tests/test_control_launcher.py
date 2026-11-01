"""Menu launcher checks use no SSH, phone, or private launch token."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest


spec = importlib.util.spec_from_file_location(
    'menu_open', Path(__file__).parents[1] / 'scripts/control/menu_open.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
TOKEN = 'synthetic-launch-token-for-tests'
URL = launcher.ORIGIN + '/#' + TOKEN


@pytest.fixture
def launch(monkeypatch):
    commands = []
    monkeypatch.setattr(launcher, 'load_config', lambda _: ('training-server@training-server', '/private/control.log'))
    monkeypatch.setattr(launcher.shutil, 'which', lambda *a, **k: '/usr/local/bin/tailscale')
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='Open ' + URL + '\n')
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    monkeypatch.setattr(launcher, 'ensure_tunnel', lambda *a: None)
    return commands


def test_latest_launch_after_service_restart(launch, monkeypatch):
    monkeypatch.setattr(launcher, 'read_status', lambda *a: {})
    assert launcher.running_url(Path('/unused')) == URL
    assert "grep '^Open http://127.0.0.1:8401/#'" in launch[0][-1]
    assert launch[0][-1].endswith(' | tail -n 1')


@pytest.mark.parametrize('error, message', [
    (URLError('private upstream detail'), 'not responding on rig port 8401'),
    (TimeoutError('private upstream detail'), 'not responding on rig port 8401'),
    (HTTPError(URL, 403, 'private upstream detail', {}, None), 'rejected the saved launch token'),
], ids=['unreachable', 'timeout', 'stale-token'])
def test_actionable_error_without_credentials(launch, monkeypatch, error, message):
    def fail(*args):
        raise error
    monkeypatch.setattr(launcher, 'read_status', fail)
    with pytest.raises(RuntimeError, match=message) as caught:
        launcher.running_url(Path('/unused'))
    assert TOKEN not in str(caught.value)
    assert 'private upstream detail' not in str(caught.value)
    assert 'docs/XR_CONTROL.md' in str(caught.value)


@pytest.mark.parametrize('url', ['https://example.org/#' + TOKEN, launcher.ORIGIN + '/#short'])
def test_rejects_untrusted_launch_url(url):
    with pytest.raises(RuntimeError):
        launcher.parse_launch_url(url)
