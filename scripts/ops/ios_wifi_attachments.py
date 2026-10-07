#!/usr/bin/env python3
"""List or pull XCTest attachments over the diagnostic Wi-Fi tunnel (read-only).

Off USB, Appium's devicectl listing/pull cannot see XR2. This reads the same
testmanagerd appDataContainer folders through the kernel tunnel's RSD with
pymobiledevice3 (Python 3.13+). No root and no deletion. XR2 advertises only
transferFiles for the file service, so pymobiledevice3's listFiles
pre-check is skipped and the device stays the authority.

The device closes the control connection on any request with an even XPC
message id (it numbers its own messages 2, 4, ...), so requests use odd ids as
devicectl does. File bytes use the data channel, opened before RetrieveFile.
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import re
import struct
import sys
import uuid

DOMAIN_IDENTIFIER = 'com.apple.testmanagerd'
SUBDIRECTORIES = ('Attachments', 'tmp/Attachments')
UUID_NAME_RE = re.compile(r'^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$', re.I)
ATTEMPTS = 3


def uuid_names(names):
    return sorted(n for n in names if isinstance(n, str) and UUID_NAME_RE.match(n.strip()))


def summarise(listings):
    """Mirror cleanup-videos --dry-run: one successful listing is required."""
    ok = {k: v for k, v in listings.items() if isinstance(v, list)}
    if not ok:
        raise RuntimeError('No attachment folder could be listed')
    found = uuid_names({n for names in ok.values() for n in names})
    return {'listings': listings, 'uuids': found,
            'summary': f'Found {len(found)} UUID-shaped attachment(s)'}


DATA_MAGIC = b'rwb!FILE'
FILE_DATA, TRANSFER_COMPLETE = 1, 0x63
ROOT_CHANNEL = 1


def odd_id(fs):
    ids = fs.service.next_message_id
    if ids[ROOT_CHANNEL] % 2 == 0:
        ids[ROOT_CHANNEL] += 1


def data_header(buffer):
    if len(buffer) != 40 or buffer[:8] != DATA_MAGIC:
        raise RuntimeError('Invalid file service data header')
    kind, _reserved, file_id, size = struct.unpack('>QQQQ', buffer[8:])
    return kind, file_id, size


async def list_directory(fs, path):
    odd_id(fs)
    return await fs.retrieve_directory_list(path)


async def retrieve(rsd, fs, path, out):
    port = rsd.get_service_port('com.apple.coredevice.fileservice.data')
    reader, writer = await asyncio.open_connection(fs.service.address[0], port)
    try:
        odd_id(fs)
        reply = await fs.send_receive_request({'MessageUUID': str(uuid.uuid4()).upper(), 'Cmd': 'RetrieveFile',
                                               'Path': path, 'SessionID': fs.session})
        file_id = reply['NewFileID']
        writer.write(DATA_MAGIC + struct.pack('>QQQQ', FILE_DATA, 0, file_id, 0))
        await writer.drain()
        kind, got_id, size = data_header(await reader.readexactly(40))
        if (kind, got_id) != (FILE_DATA, file_id) or size == 0:
            raise RuntimeError(f'Unexpected file data header {kind}/{got_id}/{size}')
        received = 0
        with open(out, 'xb') as f:
            while received < size:
                chunk = await reader.read(min(1 << 20, size - received))
                if not chunk:
                    raise RuntimeError(f'Transfer ended at {received}/{size} bytes')
                f.write(chunk)
                received += len(chunk)
        kind, got_id, _ = data_header(await reader.readexactly(40))
        if (kind, got_id) != (TRANSFER_COMPLETE, file_id):
            raise RuntimeError('Missing transfer confirmation')
        return size
    finally:
        writer.close()


async def with_file_service(rsd, action):
    from pymobiledevice3.remote.core_device.file_service import FileServiceService, Domain
    last = None
    for _ in range(ATTEMPTS):
        try:
            async with FileServiceService(rsd, Domain.APP_DATA_CONTAINER, DOMAIN_IDENTIFIER) as fs:
                return await asyncio.wait_for(action(fs), 60)
        except (TimeoutError, asyncio.TimeoutError) as error:
            last = error
    raise last


async def run(a):
    from pymobiledevice3.remote.remote_service_discovery import RemoteServiceDiscoveryService
    rsd = RemoteServiceDiscoveryService((a.address, a.port))
    await asyncio.wait_for(rsd.connect(), 30)
    try:
        if rsd.udid != a.udid:
            raise RuntimeError('Tunnel device identity mismatch')
        rsd.require_feature = lambda *_args, **_kwargs: None
        listings = {}
        for sub in SUBDIRECTORIES:
            try:
                listings[sub] = await with_file_service(rsd, lambda fs, sub=sub: list_directory(fs, sub))
            except Exception as error:
                listings[sub] = {'error': f'{type(error).__name__}: {str(error)[:160]}'}
        if a.mode == 'list':
            return summarise(listings)
        matches = [f'{sub}/{name}' for sub, names in listings.items() if isinstance(names, list)
                   for name in names if name.upper() == a.uuid.upper()]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one attachment for {a.uuid}, found {len(matches)}')
        size = await with_file_service(rsd, lambda fs: retrieve(rsd, fs, matches[0], a.out))
        return {'pulled': matches[0], 'bytes': size, 'out': a.out}
    finally:
        await rsd.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('list', 'pull'))
    p.add_argument('--address', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--udid', required=True)
    p.add_argument('--uuid')
    p.add_argument('--out')
    a = p.parse_args(argv)
    if a.mode == 'pull' and not (a.uuid and a.out and Path(a.out).is_absolute() and not Path(a.out).exists()):
        p.error('pull needs --uuid and a new absolute --out')
    print(json.dumps(asyncio.run(run(a))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
