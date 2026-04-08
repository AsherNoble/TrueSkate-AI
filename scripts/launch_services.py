"""Launch WDA and Appium for all configured devices.

Starts one WebDriverAgent (xcodebuild) and one Appium server per device
defined in DEVICES. Monitors all processes and cleans up on Ctrl+C.

Usage:
    python scripts/launch_services.py
"""
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.rl.device_worker import DEVICES

load_dotenv(_REPO_ROOT / ".env")

WDA_PROJECT_PATH = Path.home() / "Projects" / "WebDriverAgent"
WDA_STARTUP_TIMEOUT = 60

# Per-device process tracking:
# {device_name: {"wda": Process|None, "appium": Process|None, "appium_was_running": bool}}
_processes: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True


def _is_service_responding(url: str, timeout: int = 2) -> bool:
    try:
        return requests.get(url, timeout=timeout).status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# Signal / cleanup
# ---------------------------------------------------------------------------

def _signal_handler(sig, frame):
    print("\nShutting down...")
    _cleanup()
    sys.exit(0)


def _cleanup():
    for name, procs in _processes.items():
        if procs.get("appium") and not procs.get("appium_was_running"):
            print(f"[{name}] Stopping Appium...")
            procs["appium"].terminate()
            try:
                procs["appium"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                procs["appium"].kill()
        elif procs.get("appium_was_running"):
            print(f"[{name}] Leaving existing Appium instance running...")

        if procs.get("wda"):
            print(f"[{name}] Stopping WebDriverAgent...")
            procs["wda"].terminate()
            try:
                procs["wda"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                procs["wda"].kill()

    print("Cleanup complete")


# ---------------------------------------------------------------------------
# Per-device launchers
# ---------------------------------------------------------------------------

def _check_device_connected(device: dict) -> bool:
    name = device["name"]
    udid = os.environ.get(device["env_key"])
    if not udid:
        print(f"[{name}] {device['env_key']} not set in .env")
        return False

    print(f"[{name}] Checking device connection...")
    result = subprocess.run(["idevice_id", "-l"], capture_output=True, text=True)

    if udid in result.stdout:
        print(f"[{name}] Device found")
        return True

    print(f"[{name}] Device not found")
    return False


def _start_appium(device: dict) -> bool:
    name = device["name"]
    port = device["appium_port"]
    url = f"http://localhost:{port}/status"

    if _is_service_responding(url):
        print(f"[{name}] Appium already running on port {port}")
        _processes[name]["appium_was_running"] = True
        return True

    if _is_port_in_use(port):
        print(f"[{name}] Port {port} in use but Appium not responding")
        print(f"  Kill the process: lsof -ti :{port} | xargs kill")
        return False

    print(f"[{name}] Starting Appium on port {port}...")

    try:
        proc = subprocess.Popen(
            ["appium", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(3)

        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"[{name}] Appium failed to start")
            print(f"  Error: {stderr if stderr else stdout}")
            return False

        if not _is_service_responding(url, timeout=5):
            print(f"[{name}] Appium process started but not responding")
            return False

        _processes[name]["appium"] = proc
        print(f"[{name}] Appium running (PID: {proc.pid})")
        return True

    except FileNotFoundError:
        print(f"[{name}] Appium not found. Install with: npm install -g appium")
        return False


def _start_wda(device: dict) -> bool:
    name = device["name"]
    wda_port = device["wda_port"]
    health_url = f"http://localhost:{wda_port}/status"

    if _is_service_responding(health_url):
        print(f"[{name}] WebDriverAgent already running on port {wda_port}")
        return True

    udid = os.environ.get(device["env_key"])

    print(f"[{name}] Starting WebDriverAgent on port {wda_port}...")

    if not WDA_PROJECT_PATH.exists():
        print(f"[{name}] WebDriverAgent not found at {WDA_PROJECT_PATH}")
        return False

    wda_cmd = [
        "xcodebuild",
        "-project", str(WDA_PROJECT_PATH / "WebDriverAgent.xcodeproj"),
        "-scheme", "WebDriverAgentRunner",
        "-destination", f"id={udid}",
        "-allowProvisioningUpdates",
        "test",
    ]

    try:
        proc = subprocess.Popen(
            wda_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(WDA_PROJECT_PATH),
        )
        _processes[name]["wda"] = proc

        print(f"[{name}] WDA process started (PID: {proc.pid})")
        print(f"[{name}] Waiting up to {WDA_STARTUP_TIMEOUT}s for WDA...")

        wda_ready = False
        wda_url = None

        def read_output():
            nonlocal wda_ready, wda_url
            for line in proc.stdout:
                # Don't print raw xcodebuild output — it contains UDIDs
                if "ServerURLHere->" in line:
                    match = re.search(r"ServerURLHere->(.+?)<-ServerURLHere", line)
                    if match:
                        wda_url = match.group(1)
                        wda_ready = True

        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()

        start_time = time.time()
        while time.time() - start_time < WDA_STARTUP_TIMEOUT:
            if proc.poll() is not None:
                print(f"[{name}] WDA process terminated unexpectedly")
                return False
            if wda_ready:
                print(f"[{name}] WDA ready at {wda_url}")
                return True
            time.sleep(1)

        print(f"[{name}] WDA did not start within {WDA_STARTUP_TIMEOUT}s")
        return False

    except FileNotFoundError:
        print(f"[{name}] xcodebuild not found. Is Xcode installed?")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("True Skate ML Environment Launcher")
    print(f"Devices: {[d['name'] for d in DEVICES]}")
    print("=" * 60)

    signal.signal(signal.SIGINT, _signal_handler)

    for device in DEVICES:
        _processes[device["name"]] = {
            "wda": None,
            "appium": None,
            "appium_was_running": False,
        }

    # Check all devices connected
    for device in DEVICES:
        if not _check_device_connected(device):
            print(f"\nStartup failed: {device['name']} not connected")
            sys.exit(1)

    print()

    # Start Appium for each device
    for device in DEVICES:
        if not _start_appium(device):
            print(f"\nStartup failed: Appium for {device['name']}")
            _cleanup()
            sys.exit(1)
        print()

    # Start WDA for each device
    for device in DEVICES:
        if not _start_wda(device):
            print(f"\nStartup failed: WDA for {device['name']}")
            _cleanup()
            sys.exit(1)
        print()

    print("=" * 60)
    print("Environment ready!")
    print("=" * 60)
    for device in DEVICES:
        name = device["name"]
        print(f"  {name}:")
        print(f"    Appium: http://localhost:{device['appium_port']}")
        print(f"    WDA:    http://localhost:{device['wda_port']}")
        print(f"    MJPEG:  http://localhost:{device['mjpeg_port']}")
    print()
    print("Run training with: python scripts/train_cmaes.py")
    print("Press Ctrl+C to stop all services")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)
            for device in DEVICES:
                name = device["name"]
                procs = _processes[name]

                if (procs["appium"] and not procs["appium_was_running"]
                        and procs["appium"].poll() is not None):
                    print(f"\n[{name}] Appium process died unexpectedly")
                    _cleanup()
                    sys.exit(1)

                if procs["wda"] and procs["wda"].poll() is not None:
                    print(f"\n[{name}] WDA process died unexpectedly")
                    _cleanup()
                    sys.exit(1)

    except KeyboardInterrupt:
        _signal_handler(None, None)


if __name__ == "__main__":
    main()
