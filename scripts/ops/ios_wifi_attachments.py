#!/usr/bin/env python3
"""List or pull XCTest attachments over the diagnostic Wi-Fi tunnel (read-only).

Off USB, Appium's devicectl listing/pull cannot see XR2. This reads the same
testmanagerd appDataContainer folders through the kernel tunnel's RSD with
pymobiledevice3 (Python 3.13+). No root and no deletion. XR2 advertises only
transferFiles for the file service, so pymobiledevice3's listFiles
pre-check is skipped and the device stays the authority.
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import re
import sys

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
                listings[sub] = await with_file_service(rsd, lambda fs, sub=sub: fs.retrieve_directory_list(sub))
            except Exception as error:
                listings[sub] = {'error': f'{type(error).__name__}: {str(error)[:160]}'}
        if a.mode == 'list':
            return summarise(listings)
        matches = [f'{sub}/{name}' for sub, names in listings.items() if isinstance(names, list)
                   for name in names if name.upper() == a.uuid.upper()]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one attachment for {a.uuid}, found {len(matches)}')
        data = await with_file_service(rsd, lambda fs: fs.retrieve_file(matches[0]))
        if not data:
            raise RuntimeError('Empty attachment')
        out = Path(a.out)
        out.write_bytes(data)
        return {'pulled': matches[0], 'bytes': len(data), 'out': str(out)}
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
