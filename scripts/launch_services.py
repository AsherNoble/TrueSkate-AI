import subprocess
import time
import signal
import sys
import requests
import socket
from pathlib import Path

DEVICE_UDID = "00008030-0015490C0E52202E"
WDA_PROJECT_PATH = Path.home() / "Projects" / "WebDriverAgent"
WDA_HEALTH_CHECK_URL = "http://localhost:8100/status"
WDA_STARTUP_TIMEOUT = 60  # Increased timeout for WDA startup
APPIUM_PORT = 4723
APPIUM_URL = f"http://localhost:{APPIUM_PORT}/status"

appium_process = None
wda_process = None
appium_was_running = False
wda_url = None


def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def is_service_responding(url, timeout=2):
    """Check if a service is responding at the given URL."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def signal_handler(sig, frame):
    print("\nShutting down...")
    cleanup()
    sys.exit(0)


def cleanup():
    global appium_process, wda_process, appium_was_running

    if appium_process and not appium_was_running:
        print("Stopping Appium...")
        appium_process.terminate()
        try:
            appium_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            appium_process.kill()
    elif appium_was_running:
        print("Leaving existing Appium instance running...")

    if wda_process:
        print("Stopping WebDriverAgent...")
        wda_process.terminate()
        try:
            wda_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            wda_process.kill()

    print("Cleanup complete")


def check_device_connected():
    print("Checking device connection...")
    result = subprocess.run(["idevice_id", "-l"], capture_output=True, text=True)

    if DEVICE_UDID in result.stdout:
        print(f"Device found: {DEVICE_UDID}")
        return True
    else:
        print(f"Device {DEVICE_UDID} not found")
        print(f"Connected devices:\n{result.stdout}")
        return False


def start_appium():
    global appium_process, appium_was_running

    # Check if Appium is already running
    if is_service_responding(APPIUM_URL):
        print(f"Appium already running on port {APPIUM_PORT}")
        appium_was_running = True
        return True

    # Check if port is in use but service not responding
    if is_port_in_use(APPIUM_PORT):
        print(f"Port {APPIUM_PORT} is in use but Appium is not responding")
        print("Kill the process using the port and try again:")
        print(f"  lsof -ti :{APPIUM_PORT} | xargs kill")
        return False

    print(f"Starting Appium on port {APPIUM_PORT}...")

    try:
        appium_process = subprocess.Popen(
            ["appium", "--port", str(APPIUM_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(3)

        if appium_process.poll() is not None:
            stdout, stderr = appium_process.communicate()
            print("Appium failed to start")
            print("\nError output:")
            print(stderr if stderr else stdout)
            return False

        # Verify Appium is responding
        if not is_service_responding(APPIUM_URL, timeout=5):
            print("Appium process started but not responding")
            return False

        print(f"Appium running (PID: {appium_process.pid})")
        return True

    except FileNotFoundError:
        print("Appium not found. Install with: npm install -g appium")
        return False


def start_wda():
    global wda_process, wda_url

    # Check if WDA is already running
    if is_service_responding(WDA_HEALTH_CHECK_URL):
        print("WebDriverAgent already running on port 8100")
        return True

    print("Starting WebDriverAgent...")

    if not WDA_PROJECT_PATH.exists():
        print(f"WebDriverAgent not found at {WDA_PROJECT_PATH}")
        return False

    wda_cmd = [
        "xcodebuild",
        "-project", str(WDA_PROJECT_PATH / "WebDriverAgent.xcodeproj"),
        "-scheme", "WebDriverAgentRunner",
        "-destination", f"id={DEVICE_UDID}",
        "test"
    ]

    try:
        print("Starting xcodebuild...")
        print("-" * 60)

        # Capture output so we can parse it for the ServerURLHere message
        import io
        import threading

        wda_process = subprocess.Popen(
            wda_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(WDA_PROJECT_PATH)
        )

        print(f"WDA process started (PID: {wda_process.pid})")
        print(f"Waiting up to {WDA_STARTUP_TIMEOUT} seconds for WDA to start...")
        print("-" * 60)

        wda_ready = False
        local_wda_url = None

        def read_output():
            nonlocal wda_ready, local_wda_url
            for line in wda_process.stdout:
                print(line, end='')  # Print in real-time
                if 'ServerURLHere->' in line:
                    # Extract URL from ServerURLHere->http://IP:PORT<-ServerURLHere
                    import re
                    match = re.search(r'ServerURLHere->(.+?)<-ServerURLHere', line)
                    if match:
                        local_wda_url = match.group(1)
                        wda_ready = True

        # Start reading output in a background thread
        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()

        start_time = time.time()
        while time.time() - start_time < WDA_STARTUP_TIMEOUT:
            if wda_process.poll() is not None:
                print("\n" + "=" * 60)
                print("WDA process terminated unexpectedly")
                print("=" * 60)
                return False

            if wda_ready:
                global wda_url
                wda_url = local_wda_url
                print("\n" + "=" * 60)
                print(f"✓ WDA is ready at {local_wda_url}")
                print("=" * 60)
                return True

            time.sleep(1)

        print("\n" + "=" * 60)
        print(f"✗ WDA did not start within {WDA_STARTUP_TIMEOUT} seconds")
        print("=" * 60)
        return False

    except FileNotFoundError:
        print("xcodebuild not found. Is Xcode installed?")
        return False


def main():
    print("=" * 60)
    print("True Skate ML Environment Launcher")
    print("=" * 60)

    signal.signal(signal.SIGINT, signal_handler)

    if not check_device_connected():
        print("\nStartup failed: Device not connected")
        sys.exit(1)

    print()

    if not start_appium():
        print("\nStartup failed: Appium could not start")
        cleanup()
        sys.exit(1)

    print()

    if not start_wda():
        print("\nStartup failed: WebDriverAgent could not start")
        cleanup()
        sys.exit(1)

    print()
    print("=" * 60)
    print("Environment ready!")
    print("=" * 60)
    print(f"Device: {DEVICE_UDID}")
    print(f"Appium: http://localhost:{APPIUM_PORT}")
    if wda_url:
        print(f"WDA: {wda_url}")
    else:
        print(f"WDA: http://localhost:8100")
    print()
    print("Run your ML control scripts in another terminal")
    print("Press Ctrl+C to stop all services")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)

            # Only monitor Appium if we started it
            if appium_process and not appium_was_running and appium_process.poll() is not None:
                print("\nAppium process died unexpectedly")
                cleanup()
                sys.exit(1)

            if wda_process and wda_process.poll() is not None:
                print("\nWDA process died unexpectedly")
                cleanup()
                sys.exit(1)

    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()