#!/usr/bin/env python3
"""Forward the already-running rig UI over Tailscale SSH; Ctrl-C closes tunnel."""
import argparse
import shutil
import socket
import subprocess
import time
import webbrowser
from getpass import getpass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    tailscale = shutil.which('tailscale')
    if not tailscale:
        raise SystemExit('Install/enable Tailscale first.')
    subprocess.run([tailscale, 'status'], check=True, stdout=subprocess.DEVNULL)
    token = getpass('Paste the token after # from the rig launch URL: ').strip()
    if not token or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in token):
        raise SystemExit('Invalid launch token')
    # tailscale ssh wraps OpenSSH but does not expose forwarding flags. Use its
    # TCP proxy with OpenSSH to keep the transport on the tailnet.
    proxy = f'ProxyCommand={tailscale} nc %h %p'
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', 8401))
        except OSError:
            raise SystemExit('Local port 8401 is occupied; close the previous tunnel.')
    tunnel = subprocess.Popen(['ssh', '-N', '-T', '-o', proxy, '-o', 'ExitOnForwardFailure=yes',
                               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
                               '-L', '127.0.0.1:8401:127.0.0.1:8401', 'training-server@training-server'])
    try:
        for _ in range(100):
            if tunnel.poll() is not None:
                raise SystemExit('SSH forwarding failed; verify tailscale ssh access.')
            try:
                with socket.create_connection(('127.0.0.1', 8401), timeout=.2):
                    break
            except OSError:
                time.sleep(.2)
        else:
            raise SystemExit('SSH forwarding did not become ready.')
        webbrowser.open(f'http://127.0.0.1:8401/#{token}')
        print('XR control open. Disconnect both devices before Ctrl-C to close the tunnel.')
        tunnel.wait()
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait()


if __name__ == '__main__':
    main()
