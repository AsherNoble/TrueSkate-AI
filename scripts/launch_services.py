"""Launch WDA and Appium for all configured devices.

Starts one WebDriverAgent (xcodebuild) and one Appium server per device
defined in DEVICES. Monitors all processes and cleans up on Ctrl+C.

Usage:
    python scripts/launch_services.py
    python scripts/launch_services.py --device iPhone_XR
"""
import argparse
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
from trueskate_ai.sim.touch_actions import skip_loading_screen

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
        if procs.get("iproxy_wda"):
            print(f"[{name}] Stopping WDA iproxy...")
            procs["iproxy_wda"].terminate()
            try:
                procs["iproxy_wda"].wait(timeout=2)
            except subprocess.TimeoutExpired:
                procs["iproxy_wda"].kill()

        if procs.get("iproxy_mjpeg"):
            print(f"[{name}] Stopping MJPEG iproxy...")
            procs["iproxy_mjpeg"].terminate()
            try:
                procs["iproxy_mjpeg"].wait(timeout=2)
            except subprocess.TimeoutExpired:
                procs["iproxy_mjpeg"].kill()

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

    def _run_wda_command(action: str) -> bool:
        """Try to start WDA with the given xcodebuild action."""
        cmd = [
            "xcodebuild",
            "-project", str(WDA_PROJECT_PATH / "WebDriverAgent.xcodeproj"),
            "-scheme", "WebDriverAgentRunner",
            "-destination", f"id={udid}",
            "-allowProvisioningUpdates",
            action,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(WDA_PROJECT_PATH),
            )
            _processes[name]["wda"] = proc

            print(f"[{name}] WDA process started with '{action}' (PID: {proc.pid})")
            print(f"[{name}] Waiting up to {WDA_STARTUP_TIMEOUT}s for WDA...")

            wda_ready = False
            wda_url = None

            def read_output():
                nonlocal wda_ready, wda_url
                for line in proc.stdout:
                    if "ServerURLHere->" in line:
                        match = re.search(
                            r"ServerURLHere->(.+?)<-ServerURLHere", line
                        )
                        if match:
                            wda_url = match.group(1)
                            wda_ready = True

            output_thread = threading.Thread(target=read_output, daemon=True)
            output_thread.start()

            start_time = time.time()
            while time.time() - start_time < WDA_STARTUP_TIMEOUT:
                exit_code = proc.poll()
                if exit_code is not None:
                    if exit_code == 0 and wda_ready:
                        print(f"[{name}] WDA ready at {wda_url}")
                        return True
                    else:
                        print(
                            f"[{name}] WDA with '{action}' exited with code {exit_code}"
                        )
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

    # Try test-without-building first (fast path if already built)
    print(f"[{name}] Attempting test-without-building...")
    if _run_wda_command("test-without-building"):
        return True

    # Fall back to build-for-testing then test-without-building
    print(f"[{name}] test-without-building failed, trying build-for-testing...")
    if not _run_wda_command("build-for-testing"):
        return False

    print(f"[{name}] Running test-without-building after build...")
    return _run_wda_command("test-without-building")


def _start_iproxy(device: dict) -> bool:
    """Start iproxy port forwarding from Mac to iOS device for WDA and MJPEG.

    WDA is forwarded because we use webDriverAgentUrl to bypass Appium's WDA
    management. MJPEG is forwarded manually because WDA always runs its MJPEG
    server on device port 9100 regardless of the mjpegServerPort capability —
    so we forward each device's local MJPEG port to device port 9100 directly.
    """
    name = device["name"]
    wda_local_port = device["wda_port"]
    mjpeg_local_port = device["mjpeg_port"]
    wda_device_port = 8100   # WDA always runs on 8100 on the device
    mjpeg_device_port = 9100  # WDA MJPEG always runs on 9100 on the device
    udid = os.environ.get(device["env_key"])

    try:
        # Forward WDA port
        print(f"[{name}] Starting iproxy for WDA: localhost:{wda_local_port} <-> device:{wda_device_port}...")
        wda_proc = subprocess.Popen(
            ["iproxy", str(wda_local_port), str(wda_device_port), "-u", udid],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _processes[name]["iproxy_wda"] = wda_proc
        time.sleep(0.5)
        print(f"[{name}] WDA iproxy running (PID: {wda_proc.pid})")

        # Forward MJPEG port (device always uses 9100 regardless of capability)
        print(f"[{name}] Starting iproxy for MJPEG: localhost:{mjpeg_local_port} <-> device:{mjpeg_device_port}...")
        mjpeg_proc = subprocess.Popen(
            ["iproxy", str(mjpeg_local_port), str(mjpeg_device_port), "-u", udid],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _processes[name]["iproxy_mjpeg"] = mjpeg_proc
        time.sleep(0.5)
        print(f"[{name}] MJPEG iproxy running (PID: {mjpeg_proc.pid})")

        return True
    except FileNotFoundError:
        print(f"[{name}] iproxy not found. Install with: brew install libimobiledevice")
        return False


# ---------------------------------------------------------------------------
# Launch TrueSkate
# ---------------------------------------------------------------------------

def _launch_trueskate_on_devices(devices: list[dict]) -> None:
    """Open True Skate app on all devices via Appium.
    
    Attempts to connect to Appium and activate the True Skate app on each device.
    Uses device-specific coordinates to skip the loading screen.
    Errors are logged but don't prevent other devices from trying.
    """
    from appium import webdriver
    from appium.options.ios import XCUITestOptions
    
    _BUNDLE_ID = "com.trueaxis.skate"
    
    for device in devices:
        name = device["name"]
        try:
            # print(f"[{name}] Opening True Skate...")
            
            udid = os.environ.get(device["env_key"])
            if not udid:
                print(f"[{name}] Skipping: {device['env_key']} not set in .env")
                continue
            
            options = XCUITestOptions()
            options.platform_name = "iOS"
            options.automation_name = "XCUITest"
            options.bundle_id = _BUNDLE_ID
            options.udid = udid
            options.wda_local_port = device["wda_port"]
            options.use_prebuilt_wda = True
            options.skip_log_capture = True
            options.no_reset = True
            options.set_capability("webDriverAgentUrl", f"http://127.0.0.1:{device['wda_port']}")
            
            appium_url = f"http://127.0.0.1:{device['appium_port']}"
            driver = webdriver.Remote(appium_url, options=options)

            actual = driver.get_window_size()
            exp_w, exp_h = int(device["logical_w"]), int(device["logical_h"])
            if actual["width"] != exp_w or actual["height"] != exp_h:
                print(
                    f"[{name}] ERROR: screen dimensions mismatch — "
                    f"expected {exp_w}×{exp_h}, got {actual['width']}×{actual['height']}. "
                    f"Check Display Zoom: Settings → Display & Brightness → Display Zoom → Default."
                )
                driver.quit()
                _cleanup()
                sys.exit(1)
            print(f"[{name}] Screen dimensions OK: {actual['width']}×{actual['height']} pts")

            # Always re-activate before dismissing the loading screen so the
            # "open" and "skip" steps stay coupled, even if Appium reports the
            # app as already foreground.
            state = driver.query_app_state(_BUNDLE_ID)
            if state == 4:
                print(f"[{name}] True Skate already in foreground")
            else:
                print(f"[{name}] True Skate not in foreground (state={state})")

            driver.activate_app(_BUNDLE_ID)
            time.sleep(1.5)

            try:
                logical_w = float(device.get("logical_w", 414))
                logical_h = float(device.get("logical_h", 896))
                # Primer touch: execute_trick runs skip after a prior gesture.
                # Reproduce that ordering to stabilize first-touch dispatch.
                driver.execute_script('mobile: tap', {'x': 0.8454 * logical_w, 'y': 0.8393 * logical_h})
                time.sleep(0.25)
                # skip_loading_screen(driver, logical_w, logical_h, duration=1.0)
                time.sleep(0.3)
            except Exception as e:
                print(f"[{name}] Warning: Could not skip loading screen: {e}")
            time.sleep(1.0)
            
            driver.quit()
            print(f"[{name}] True Skate opened successfully")
            
        except Exception as e:
            print(f"[{name}] Failed to open True Skate: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Launch WDA/Appium services for configured devices."
    )
    parser.add_argument(
        "--device",
        help="Device name to launch (e.g., iPhone_XR). Defaults to all devices.",
    )
    args = parser.parse_args()

    if args.device:
        selected_devices = [d for d in DEVICES if d["name"] == args.device]
        if not selected_devices:
            valid_names = ", ".join(d["name"] for d in DEVICES)
            print(f"Unknown device name: {args.device}")
            print(f"Valid values: {valid_names}")
            sys.exit(1)
    else:
        selected_devices = DEVICES

    print("=" * 60)
    print("True Skate ML Environment Launcher")
    print(f"Devices: {[d['name'] for d in selected_devices]}")
    print("=" * 60)

    signal.signal(signal.SIGINT, _signal_handler)

    for device in selected_devices:
        _processes[device["name"]] = {
            "wda": None,
            "appium": None,
            "appium_was_running": False,
            "iproxy_wda": None,
            "iproxy_mjpeg": None,
        }

    # Check all devices connected
    for device in selected_devices:
        if not _check_device_connected(device):
            print(f"\nStartup failed: {device['name']} not connected")
            sys.exit(1)

    print()

    # Start Appium for each device
    for device in selected_devices:
        if not _start_appium(device):
            print(f"\nStartup failed: Appium for {device['name']}")
            _cleanup()
            sys.exit(1)
        print()

    # Start iproxy tunnels first so WDA detection works properly
    for device in selected_devices:
        if not _start_iproxy(device):
            print(f"\nStartup failed: iproxy for {device['name']}")
            _cleanup()
            sys.exit(1)
        print()

    # Start WDA for each device (after iproxy so it can detect already-running WDA)
    for device in selected_devices:
        if not _start_wda(device):
            print(f"\nStartup failed: WDA for {device['name']}")
            _cleanup()
            sys.exit(1)
        print()

    print("=" * 60)
    print("Environment ready!")
    print("=" * 60)
    for device in selected_devices:
        name = device["name"]
        print(f"  {name}:")
        print(f"    Appium: http://localhost:{device['appium_port']}")
        print(f"    WDA:    http://localhost:{device['wda_port']}")
        print(f"    MJPEG:  http://localhost:{device['mjpeg_port']}")
    print()
    
    # Open True Skate on each device
    print("Opening True Skate on all devices...")
    _launch_trueskate_on_devices(selected_devices)
    print()
    
    print("Press Ctrl+C to stop all services")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)
            for device in selected_devices:
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

                if procs["iproxy_wda"] and procs["iproxy_wda"].poll() is not None:
                    print(f"\n[{name}] WDA iproxy process died unexpectedly")
                    _cleanup()
                    sys.exit(1)

                if procs["iproxy_mjpeg"] and procs["iproxy_mjpeg"].poll() is not None:
                    print(f"\n[{name}] MJPEG iproxy process died unexpectedly")
                    _cleanup()
                    sys.exit(1)

    except KeyboardInterrupt:
        _signal_handler(None, None)


if __name__ == "__main__":
    main()
