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

from trueskate_ai.rl.device_worker import (
    BUNDLE_ID,
    DEVICES,
    add_device_selection_args,
    resolve_devices,
    select_devices,
)
from trueskate_ai.utils.notify import notify

load_dotenv(_REPO_ROOT / ".env")

def _resolve_wda_project_path() -> Path:
    """Locate the WebDriverAgent Xcode project.

    Honours the WDA_PROJECT_PATH env var (e.g. in .env); otherwise looks
    beside this repo (WDA is checked out as a sibling), then falls back to
    the legacy ~/Projects/WebDriverAgent location.
    """
    override = os.getenv("WDA_PROJECT_PATH")
    if override:
        return Path(override).expanduser()
    candidates = [
        _REPO_ROOT.parent / "WebDriverAgent",
        Path.home() / "Projects" / "WebDriverAgent",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


WDA_PROJECT_PATH = _resolve_wda_project_path()
WDA_STARTUP_TIMEOUT = 240  # first WDA build compiles ~2-3 min

# Per-device restart backoff (seconds), indexed by consecutive-failure count.
# The last value repeats. Backoff decays back to 0 after a device stays healthy
# for _RESTART_DECAY_S, so an occasional blip doesn't permanently slow restarts.
_RESTART_BACKOFF_S = [5, 15, 30, 60, 120]
_RESTART_DECAY_S = 600
_DEVICE_USB_WAIT_S = 60  # how long to wait for a browned-out device to reappear

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


def _drain_pipe(proc: subprocess.Popen) -> None:
    """Discard a long-running child's stdout on a daemon thread.

    Appium and iproxy run for the whole session with nobody reading their
    output; once the ~64KB OS pipe buffer fills, the child blocks on write()
    and the rig wedges silently. A read-and-discard reader keeps it flowing.
    Mirrors WDA's reader thread, minus the line parsing. The thread exits when
    the pipe closes on terminate()/kill(), so it needs no join.
    """
    def _drain():
        try:
            for _ in proc.stdout:
                pass  # discard; we only care that the buffer never fills
        except Exception:
            pass  # pipe torn down mid-read during shutdown — best-effort

    threading.Thread(target=_drain, daemon=True).start()


# ---------------------------------------------------------------------------
# Signal / cleanup
# ---------------------------------------------------------------------------

def _signal_handler(sig, frame):
    print("\nShutting down...")
    _cleanup()
    sys.exit(0)


def _stop_proc(name: str, key: str, label: str, timeout: int) -> None:
    """Terminate one tracked subprocess (if present) and clear its slot."""
    procs = _processes.get(name, {})
    proc = procs.get(key)
    if not proc:
        return
    print(f"[{name}] Stopping {label}...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
    procs[key] = None


def _cleanup_device(name: str) -> None:
    """Stop all subprocesses owned for a single device.

    Leaves a pre-existing (externally started) Appium instance running, matching
    the original shutdown contract.
    """
    procs = _processes.get(name, {})
    _stop_proc(name, "iproxy_wda", "WDA iproxy", timeout=2)
    _stop_proc(name, "iproxy_mjpeg", "MJPEG iproxy", timeout=2)
    if procs.get("appium_was_running"):
        print(f"[{name}] Leaving existing Appium instance running...")
    else:
        _stop_proc(name, "appium", "Appium", timeout=5)
    _stop_proc(name, "wda", "WebDriverAgent", timeout=5)


def _cleanup():
    for name in _processes:
        _cleanup_device(name)
    print("Cleanup complete")


# ---------------------------------------------------------------------------
# Per-device launchers
# ---------------------------------------------------------------------------

def _coredevice_available(udid: str) -> bool:
    """CoreDevice (xcrun devicectl) view of the device — the transport xcodebuild/WDA
    actually use. libimobiledevice's ``idevice_id`` flakes empty intermittently on
    this rig (esp. after a reboot/usbmux hiccup), which would falsely fail the gate and
    stop WDA from ever starting; CoreDevice stays reliable, so we trust it as a fallback."""
    try:
        r = subprocess.run(["xcrun", "devicectl", "list", "devices"],
                           capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return False
    # NB: a plain substring test for "available" also matches "unavailable" (a
    # disconnected-but-remembered device), which made the launcher burn 2x240s
    # WDA builds (exit 70) per cycle against an absent phone instead of
    # reporting "not connected". (?<!un) keeps every other "available" variant.
    return any(
        udid in line and re.search(r"(?<!un)available", line.lower())
        for line in r.stdout.splitlines()
    )


def _check_device_connected(device: dict) -> bool:
    name = device["name"]
    udid = os.environ.get(device["env_key"])
    if not udid:
        print(f"[{name}] {device['env_key']} not set in .env")
        return False

    print(f"[{name}] Checking device connection...")
    result = subprocess.run(["idevice_id", "-l"], capture_output=True, text=True)

    if udid in result.stdout or _coredevice_available(udid):
        print(f"[{name}] Device found")
        return True

    print(f"[{name}] Device not found")
    return False


def _kill_port(port: int) -> None:
    """Kill whatever TCP-listens on ``port`` — a wedged Appium/iproxy holding it."""
    try:
        r = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        for pid in r.stdout.split():
            subprocess.run(["kill", "-9", pid], capture_output=True)
    except Exception:  # noqa: BLE001
        pass


def _start_appium(device: dict) -> bool:
    name = device["name"]
    port = device["appium_port"]
    url = f"http://localhost:{port}/status"

    # Tolerate a momentarily BUSY Appium (one serving a live collector): retry the
    # health check a few times before concluding it's dead, so we never kill a working
    # server out from under a collector.
    for _ in range(5):
        if _is_service_responding(url, timeout=4):
            print(f"[{name}] Appium already running on port {port}")
            _processes[name]["appium_was_running"] = True
            return True
        if not _is_port_in_use(port):
            break  # port free → start a fresh Appium below
        time.sleep(2)

    if _is_port_in_use(port):
        # Held but never answered across retries → a WEDGED Appium. Don't abort (that
        # strands this device AND used to block the other device's WDA) — kill the
        # holder and start fresh.
        print(f"[{name}] Port {port} wedged (held, not responding) — killing the holder")
        _kill_port(port)
        time.sleep(2)

    print(f"[{name}] Starting Appium on port {port}...")

    try:
        proc = subprocess.Popen(
            ["appium", "--port", str(port), "--allow-insecure", "xcuitest:xctest_screen_record"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Cold start on this rig can take ~10s; poll instead of a fixed 3s wait.
        for _ in range(30):
            if proc.poll() is None and _is_service_responding(url, timeout=2):
                break
            time.sleep(1)

        if proc.poll() is not None:
            # Crashed during startup — surface its output (stderr merged in).
            output, _ = proc.communicate()
            print(f"[{name}] Appium failed to start")
            print(f"  Error: {output}")
            return False

        if not _is_service_responding(url, timeout=5):
            print(f"[{name}] Appium process started but not responding")
            return False

        _processes[name]["appium"] = proc
        # Drain stdout so a chatty Appium can't fill the pipe buffer and hang.
        _drain_pipe(proc)
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
                    if wda_ready:
                        print(f"[{name}] WDA ready at {wda_url}")
                        return True
                    # build-for-testing just compiles and exits 0 — it never
                    # starts the on-device server, so it has no ServerURLHere.
                    # Treat a clean build as success so the caller can retry
                    # test-without-building. For the test actions, exiting
                    # before ServerURLHere is always a real failure.
                    if action == "build-for-testing" and exit_code == 0:
                        print(f"[{name}] build-for-testing completed")
                        return True
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
            stderr=subprocess.STDOUT,
            text=True,
        )
        _processes[name]["iproxy_wda"] = wda_proc
        _drain_pipe(wda_proc)  # undrained pipe wedges the tunnel
        time.sleep(0.5)
        print(f"[{name}] WDA iproxy running (PID: {wda_proc.pid})")

        # Forward MJPEG port (device always uses 9100 regardless of capability)
        print(f"[{name}] Starting iproxy for MJPEG: localhost:{mjpeg_local_port} <-> device:{mjpeg_device_port}...")
        mjpeg_proc = subprocess.Popen(
            ["iproxy", str(mjpeg_local_port), str(mjpeg_device_port), "-u", udid],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _processes[name]["iproxy_mjpeg"] = mjpeg_proc
        _drain_pipe(mjpeg_proc)  # undrained pipe wedges the tunnel
        time.sleep(0.5)
        print(f"[{name}] MJPEG iproxy running (PID: {mjpeg_proc.pid})")

        return True
    except FileNotFoundError:
        print(f"[{name}] iproxy not found. Install with: brew install libimobiledevice")
        return False


# ---------------------------------------------------------------------------
# Per-device (re)start
# ---------------------------------------------------------------------------

def _wait_for_device(device: dict, timeout: int = _DEVICE_USB_WAIT_S) -> bool:
    """Poll `idevice_id -l` until this device's UDID reappears on USB.

    Covers the brown-out / loose-port case (old XS especially): the phone drops
    off USB, then comes back — we wait for it before restarting services.
    """
    name = device["name"]
    udid = os.environ.get(device["env_key"])
    if not udid:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(["idevice_id", "-l"], capture_output=True, text=True)
        if udid in result.stdout or _coredevice_available(udid):
            return True
        print(f"[{name}] Waiting for device to reappear on USB...")
        time.sleep(3)
    return False


def _start_device(device: dict) -> bool:
    """Bring up Appium + iproxy + WDA for a single device, in order."""
    return (
        _start_appium(device)
        and _start_iproxy(device)
        and _start_wda(device)
    )


def _restart_device(device: dict) -> bool:
    """Tear down and restart one device's service stack, with decaying backoff.

    Other devices keep running. Returns True if the device came back up.
    """
    name = device["name"]
    procs = _processes[name]
    now = time.time()
    if now - procs.get("last_restart", 0.0) > _RESTART_DECAY_S:
        procs["restart_count"] = 0  # device was stable a while; reset escalation
    attempt = procs.get("restart_count", 0)
    backoff = _RESTART_BACKOFF_S[min(attempt, len(_RESTART_BACKOFF_S) - 1)]
    procs["restart_count"] = attempt + 1
    procs["last_restart"] = now

    print(f"[{name}] Restarting service stack (attempt {attempt + 1}); backoff {backoff}s.")
    _cleanup_device(name)
    procs["appium_was_running"] = False  # we own a fresh Appium after a restart
    time.sleep(backoff)

    if not _wait_for_device(device):
        print(f"[{name}] Device not back on USB; will retry on next monitor cycle.")
        return False

    if _start_device(device):
        procs["restart_count"] = 0  # full recovery; clear escalation
        _launch_trueskate_on_devices([device])
        print(f"[{name}] Recovered.")
        # One incident produces one failure alert and one recovery alert.  The
        # monitor can retry many times while a phone is unplugged or WDA is
        # unhealthy; emitting an alert for every retry turns one actionable
        # incident into hundreds of ntfy notifications.
        if procs.pop("incident_open", False):
            notify(f"{name} recovered and back collecting.",
                   title="TrueSkate rig", tags=["white_check_mark"])
        return True

    print(f"[{name}] Restart failed (attempt {attempt + 1}).")
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

    for device in devices:
        name = device["name"]
        try:
            udid = os.environ.get(device["env_key"])
            if not udid:
                print(f"[{name}] Skipping: {device['env_key']} not set in .env")
                continue
            
            options = XCUITestOptions()
            options.platform_name = "iOS"
            options.automation_name = "XCUITest"
            options.bundle_id = BUNDLE_ID
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
            state = driver.query_app_state(BUNDLE_ID)
            if state == 4:
                print(f"[{name}] True Skate already in foreground")
            else:
                print(f"[{name}] True Skate not in foreground (state={state})")

            driver.activate_app(BUNDLE_ID)
            time.sleep(1.5)

            try:
                logical_w = float(device.get("logical_w", 414))
                logical_h = float(device.get("logical_h", 896))
                # Primer touch: execute_trick runs skip after a prior gesture.
                # Reproduce that ordering to stabilize first-touch dispatch.
                driver.execute_script('mobile: tap', {'x': 0.8454 * logical_w, 'y': 0.8393 * logical_h})
                time.sleep(0.25)
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
        description="Launch WDA/Appium services for selected devices."
    )
    add_device_selection_args(parser)
    parser.add_argument(
        "--device",
        help="[deprecated] Single device name. Use --devices instead.",
    )
    args = parser.parse_args()

    try:
        if args.device:
            selected_devices = select_devices(names=[args.device])
        else:
            selected_devices = resolve_devices(
                devices_arg=args.devices,
                personal=args.personal,
                all_devices=args.all_devices,
            )
    except ValueError as exc:
        print(exc)
        sys.exit(1)

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

    # Bring up each device INDEPENDENTLY: one device's wedged Appium/WDA must NOT abort
    # the launcher and strand the other (the bug that kept XR2's WDA down while XR1's
    # Appium flaked). Skip a device that won't come up — the per-device monitor below
    # restarts whatever it's tracking. Only give up entirely if NO device comes up.
    ready = []
    for device in selected_devices:
        name = device["name"]
        if not _check_device_connected(device):
            print(f"[{name}] not connected — skipping\n")
            continue
        if _start_appium(device) and _start_iproxy(device) and _start_wda(device):
            ready.append(device)
            print(f"[{name}] services up\n")
        else:
            print(f"[{name}] startup incomplete — skipping this device\n")
            _cleanup_device(name)

    if not ready:
        print("\nStartup failed: no device came up")
        _cleanup()
        sys.exit(1)
    selected_devices = ready

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

    # Monitor: if any device's process dies, restart ONLY that device's stack.
    # One munted phone/port no longer takes down the whole rig.
    _PROC_LABELS = {
        "appium": "Appium",
        "wda": "WebDriverAgent",
        "iproxy_wda": "WDA iproxy",
        "iproxy_mjpeg": "MJPEG iproxy",
    }
    # Health-aware WDA check: a wedged-but-ALIVE xcodebuild (sleeping, 0% CPU, NOT
    # serving its port) passes proc.poll() forever, so liveness alone let a dead WDA
    # sit unrecovered for 12.5h. Probe the actual port too; only act after sustained
    # unresponsiveness (~30s) so a normal busy stall — e.g. a collector retrieving a
    # ~77 MB segment over WDA — isn't mistaken for a wedge.
    _WDA_HEALTH_FAIL_LIMIT = 15  # consecutive 2s ticks of /status failure => wedged
    try:
        while True:
            time.sleep(2)
            for device in selected_devices:
                name = device["name"]
                procs = _processes[name]

                died = None
                for key, label in _PROC_LABELS.items():
                    # A pre-existing external Appium isn't ours to monitor.
                    if key == "appium" and procs.get("appium_was_running"):
                        continue
                    proc = procs.get(key)
                    if proc and proc.poll() is not None:
                        died = label
                        break

                # Wedged-but-alive WDA: proc lives but the port stops answering.
                if died is None:
                    wda_url = f"http://localhost:{device['wda_port']}/status"
                    if _is_service_responding(wda_url, timeout=3):
                        procs["wda_health_fails"] = 0
                    else:
                        procs["wda_health_fails"] = procs.get("wda_health_fails", 0) + 1
                        if procs["wda_health_fails"] >= _WDA_HEALTH_FAIL_LIMIT:
                            died = "WebDriverAgent (wedged: alive but not serving)"
                            procs["wda_health_fails"] = 0

                if died:
                    msg = f"{name}: {died} died — restarting device services."
                    print(f"\n[{name}] {died} — restarting")
                    if not procs.get("incident_open", False):
                        procs["incident_open"] = True
                        notify(msg, title="TrueSkate rig", priority="high", tags=["warning"])
                    _restart_device(device)
                    procs["wda_health_fails"] = 0  # fresh stack; don't carry stale fails

    except KeyboardInterrupt:
        _signal_handler(None, None)


if __name__ == "__main__":
    main()
