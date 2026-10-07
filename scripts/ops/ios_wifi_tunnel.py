#!/usr/bin/env python3
"""Bounded Wi-Fi RemotePairing tunnel for the XR2 IPv4 diagnostic.

Run as root by ios_ipv4_tunnel.mjs under an isolated Python 3.13+ environment
with pymobiledevice3. iOS 18.2+ accepts only the TCP (TLS-PSK) tunnel, which
pymobiledevice3 supports on Python 3.13+. The RemotePairing record must already
exist (created promptlessly over USB with `lockdown remotepairing --pair`);
this helper never pairs. Prints one JSON ready line, then holds the kernel TUN
until stdin closes, SIGTERM/SIGINT, the lifetime cap, or transport loss.
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import signal
import socket
import sys

REMOTE_PAIRING_PORT = 49152


def emit(event, **details):
    print(json.dumps({'event': event, **details}), flush=True)


def ipv4(host):
    return socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]


def use_pair_dir(pair_dir):
    # Under administrator launch HOME is root's; read the rig user's records instead.
    import pymobiledevice3.common as common
    common._HOMEFOLDER = Path(pair_dir)


async def run(a):
    if sys.version_info < (3, 13):
        raise RuntimeError('TCP tunnel requires Python 3.13+')
    use_pair_dir(a.pair_dir)
    from pymobiledevice3.remote import tunnel_service
    record = Path(a.pair_dir)/f'remote_{a.udid}.plist'
    if not record.is_file():
        raise RuntimeError('RemotePairing record missing; pair over USB first')
    host = ipv4(a.host)
    service = await asyncio.wait_for(tunnel_service.create_core_device_tunnel_service_using_remotepairing(
        a.udid, host, REMOTE_PAIRING_PORT, autopair=False), 20)
    try:
        async with service.start_tcp_tunnel() as tunnel:
            emit('pmd3-tunnel-ready', host=host, address=tunnel.address, rsd_port=tunnel.port,
                 interface=tunnel.interface)
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, stop.set)
            loop.add_reader(sys.stdin.fileno(), lambda: sys.stdin.buffer.read1() or stop.set())
            waits = [asyncio.ensure_future(stop.wait()),
                     asyncio.ensure_future(tunnel.client.wait_closed())]
            done, pending = await asyncio.wait(waits, timeout=a.lifetime, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            reason = 'stopped' if waits[0] in done else 'transport-lost' if waits[1] in done else 'lifetime-limit'
            emit('pmd3-tunnel-stopping', reason=reason)
            return 0 if reason == 'stopped' else 1
    finally:
        await service.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--udid', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--pair-dir', required=True)
    p.add_argument('--lifetime', type=int, default=900, choices=range(1, 901), metavar='1..900')
    a = p.parse_args(argv)
    try:
        return asyncio.run(run(a))
    except Exception as error:
        emit('pmd3-tunnel-error', error=f'{type(error).__name__}: {str(error)[:200]}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
