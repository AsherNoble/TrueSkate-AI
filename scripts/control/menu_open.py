#!/usr/bin/env python3
"""Retrieve the running rig's private URL and ensure its loopback SSH forward."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from urllib.error import HTTPError, URLError

PORT = 8401
ORIGIN = 'http://127.0.0.1:8401'


def load_config(path):
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError('XR control is not configured. See the project README for .trueskate_control.json.') from exc
    host, log = config.get('ssh_host'), config.get('launch_log')
    if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9_.@-]+', host) or host.startswith('-'):
        raise RuntimeError('Invalid ssh_host in XR control configuration.')
    if not isinstance(log, str) or not log.startswith('/') or '\n' in log or '\x00' in log:
        raise RuntimeError('launch_log must be an absolute path on the rig.')
    return host, log


def parse_launch_url(line):
    url = line.strip()
    if url.startswith('Open '):
        url = url[5:]
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.netloc != '127.0.0.1:8401'
            or parsed.path != '/' or parsed.query
            or not re.fullmatch(r'[A-Za-z0-9_-]{24,128}', parsed.fragment)):
        raise RuntimeError('The running control service did not provide a valid private URL.')
    return url, parsed.fragment


def read_status(token, name):
    request = Request(f'{ORIGIN}/api/{name}/status', headers={'X-Control-Token':token})
    with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
        return json.load(response)


def tunnel_open():
    try:
        with socket.create_connection(('127.0.0.1', PORT), timeout=.3):
            return True
    except OSError:
        return False


def ensure_tunnel(tailscale, host):
    if tunnel_open():
        return
    # No session takeover, service startup or permissive host-key settings.
    result = subprocess.run([
        '/usr/bin/ssh', '-f', '-N', '-T', '-o',
        'ProxyCommand=' + shlex.quote(tailscale) + ' nc %h %p',
        '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
        '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15',
        '-o', 'ServerAliveCountMax=2', '-L',
        '127.0.0.1:8401:127.0.0.1:8401', host,
    ], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=25)
    if result.returncode:
        raise RuntimeError('Could not open the private SSH forward. Verify Tailscale SSH access and host-key trust.')
    for _ in range(20):
        if tunnel_open():
            return
        time.sleep(.1)
    raise RuntimeError('The private SSH forward did not become ready.')


def running_url(config_path):
    host, log = load_config(config_path)
    # Menu-bar apps have a minimal PATH, so include standard macOS install paths.
    tailscale = shutil.which('tailscale', path=os.pathsep.join([
        '/usr/local/bin', '/opt/homebrew/bin', os.environ.get('PATH', '')]))
    if not tailscale:
        raise RuntimeError('Tailscale is not installed.')
    result = subprocess.run([tailscale, 'ssh', host,
                            "grep '^Open http://127.0.0.1:8401/#' " + shlex.quote(log) + ' | tail -n 1'],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=25)
    if result.returncode:
        raise RuntimeError('Could not read the running XR control URL. Verify the service and Tailscale SSH access.')
    url, token = parse_launch_url(result.stdout)
    ensure_tunnel(tailscale, host)
    # Authentication check prevents opening a stale token or unrelated listener.
    try:
        read_status(token, 'XR1')
    except HTTPError as exc:
        if exc.code == 403:
            raise RuntimeError('XR control rejected the saved launch token. Check launch_log in ~/.trueskate_control.json against the running service. See docs/XR_CONTROL.md.') from exc
        raise RuntimeError('The XR control service returned an HTTP error. See docs/XR_CONTROL.md for recovery.') from exc
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise RuntimeError('SSH connected, but the XR control service is not responding on rig port 8401. Start com.trueskate.control on training-server; see docs/XR_CONTROL.md. Reconnecting the phones alone will not start this web service.') from exc
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path.home()/'.trueskate_control.json')
    parser.add_argument('--check', action='store_true', help='Verify both panels without printing the token')
    args = parser.parse_args()
    try:
        url = running_url(args.config)
        if args.check:
            token = urlsplit(url).fragment
            for name in ('XR1','XR2'):
                status = read_status(token, name)
                print(f"{name}: connected={status['connected']}, fresh={status['fresh']}")
        else:
            print(url)  # Private IPC to the menu app; never persist this output in logs.
    except Exception as exc:
        # Do not include upstream URLs or payloads, which can contain credentials.
        message = str(exc) if isinstance(exc, RuntimeError) else 'XR control is unavailable. Verify the running service and SSH connection.'
        print(message, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
