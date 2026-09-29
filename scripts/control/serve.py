#!/usr/bin/env python3
"""Run explicitly on training-server; never launches rig services."""
import argparse
import os
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from dotenv import load_dotenv
from trueskate_ai.control.service import CONFIGS, ControlServer, Device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    load_dotenv(args.env_file)
    devices = [Device(c, os.environ.get(c.env_key)) for c in CONFIGS]
    server = ControlServer(devices, Path(__file__).parent / 'web')
    # This line is a private credential; do not send it to logs or shared chat.
    print(f'Open {server.origin}/#{server.token}', flush=True)
    for device in devices:
        device.relay.start()
    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    finally:
        for device in devices:
            device.relay.stop.set()
            # Only close our known, idle session; uncertain operations stay quarantined.
            if device.session and not device.uncertain and not device.lock.locked():
                try:
                    device.command('disconnect', {'epoch':device.epoch, 'sequence':device.sequence+1})
                except Exception:
                    pass
        server.server_close()


if __name__ == '__main__':
    main()
