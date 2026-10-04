"""Loopback-only XR control. No collection, calibration, or WDA lifecycle calls."""
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time
from urllib.request import urlopen

from PIL import Image
import requests

from trueskate_ai.sim.gestures import scale_to_device

MAX_BODY = 48_000
MAX_POINTS = 512
MAX_DURATION_MS = 5000
STALE_SECONDS = 2.0
BUNDLE = "com.trueaxis.skate"
# Fixed system-edge swipes (normalized x, y, ms). Control Centre starts at the
# top-right status-bar edge; App Switcher swipes up from the home indicator and
# pauses mid-screen before release. Home uses WDA's native home-button press.
SYSTEM_PATHS = {
    "control_center": ((0.9, 0.0, 0), (0.9, 0.15, 100), (0.9, 0.35, 250), (0.9, 0.5, 400)),
    "app_switcher": ((0.5, 0.995, 0), (0.5, 0.85, 120), (0.5, 0.7, 300),
                     (0.5, 0.62, 450), (0.5, 0.62, 1250)),
}
SYSTEM_COMMANDS = ("home", *SYSTEM_PATHS)
COMMAND_MESSAGES = {"gesture": "Executing gesture", "activate": "Opening True Skate",
                    "home": "Going to Home Screen", "control_center": "Opening Control Centre",
                    "app_switcher": "Opening App Switcher"}


class ControlError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    env_key: str
    appium: int
    wda: int
    mjpeg: int


CONFIGS = (
    DeviceConfig("XR1", "IPHONE_XR_UDID", 4723, 8100, 9100),
    DeviceConfig("XR2", "IPHONE_XR2_UDID", 4726, 8103, 9103),
)


def gesture_actions(points):
    """Validate elapsed milliseconds and preserve each curved path segment."""
    if not isinstance(points, list) or not 2 <= len(points) <= MAX_POINTS:
        raise ControlError("Expected 2–512 path points", 400)
    previous = -1
    converted = []
    for point in points:
        if not isinstance(point, dict) or set(point) != {"x", "y", "t"}:
            raise ControlError("Invalid path point", 400)
        x, y, t = (point[k] for k in ("x", "y", "t"))
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y, t)):
            raise ControlError("Coordinates and times must be finite numbers", 400)
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= t <= MAX_DURATION_MS):
            raise ControlError("Path outside screen or five-second limit", 400)
        if t <= previous or (not converted and t != 0):
            raise ControlError("Times must begin at zero and strictly increase", 400)
        px, py = scale_to_device(x, y, 414, 896)
        converted.append((round(px), round(py), round(t)))
        previous = t
    if converted[-1][2] < 30:
        raise ControlError("Minimum gesture duration is 30 ms", 400)
    x, y, _ = converted[0]
    steps = [{"type": "pointerMove", "duration": 0, "origin": "viewport", "x": x, "y": y},
             {"type": "pointerDown", "button": 0}]
    for (px, py, pt), (x, y, t) in zip(converted, converted[1:]):
        if (x, y) == (px, py):
            steps.append({"type": "pause", "duration": t - pt})
        else:
            steps.append({"type": "pointerMove", "duration": t - pt,
                          "origin": "viewport", "x": x, "y": y})
    steps.append({"type": "pointerUp", "button": 0})
    return {"actions": [{"type": "pointer", "id": "menu-finger",
                         "parameters": {"pointerType": "touch"}, "actions": steps}]}


def collector_running():
    """Conservative: any collector/wrapper reserves both devices, even between segments.

    Session checks cover other Appium clients. Never use stale heartbeat files as
    ownership evidence. An unreadable process table fails closed.
    """
    result = subprocess.run(["/bin/ps", "-axo", "command="], capture_output=True,
                            text=True, timeout=3, check=True)
    patterns = (r"(?:^|[/\s])collect_[\w-]+\.py(?:\s|$)",
                r"(?:^|[/\s])(?:mvp_collect_[\w-]+|rig_collect|rig_run)\.sh(?:\s|$)")
    return any(re.search(pattern, line) for line in result.stdout.splitlines() for pattern in patterns)


class FrameRelay:
    """Exactly one MJPEG upstream reader; all browsers share the newest JPEG."""
    def __init__(self, port):
        self.port = port
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0
        self.received = 0.0
        self.history = deque(maxlen=256)
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def publish(self, jpeg):
        # Reject corrupt/landscape streams: coordinates assume portrait XR geometry.
        with Image.open(io.BytesIO(jpeg)) as image:
            w, h = image.size
            if abs(w / h - 414 / 896) > 0.005:
                raise ValueError("Not a portrait XR frame")
            image.verify()
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.received = time.monotonic()
            self.history.append((self.sequence, self.received))
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return self.jpeg, self.sequence, time.monotonic() - self.received

    def fresh(self, sequence):
        if type(sequence) is not int:
            return False
        with self.condition:
            now = time.monotonic()
            return any(seq == sequence and now - at < STALE_SECONDS for seq, at in self.history)

    def run(self):
        while not self.stop.is_set():
            try:
                with urlopen(f"http://127.0.0.1:{self.port}/", timeout=3) as stream:
                    buffer = b""
                    while not self.stop.is_set():
                        chunk = stream.read1(65536)
                        if not chunk:
                            break
                        buffer += chunk
                        while True:
                            start = buffer.find(b"\xff\xd8")
                            end = buffer.find(b"\xff\xd9", max(0, start))
                            if start < 0 or end < 0:
                                break
                            self.publish(buffer[start:end + 2])
                            buffer = buffer[end + 2:]
                        if len(buffer) > 4_000_000:
                            raise ValueError("Oversized MJPEG frame")
            except Exception:
                # Staleness is derived from last received frame, never reconnect time.
                pass
            self.stop.wait(1)


class Device:
    def __init__(self, config, udid, relay=None, transport=None, ownership=collector_running):
        self.config, self.udid = config, udid
        self.relay = relay or FrameRelay(config.mjpeg)
        self.transport = transport or requests.Session()
        self.transport.trust_env = False
        self.ownership = ownership
        self.lock = threading.Lock()
        self.session = None
        self.epoch = None
        self.sequence = 0
        self.uncertain = False
        self.available = True
        self.last_probe = 0.0
        self.message = "Disconnected"

    def rpc(self, method, path, body=None, *, wda=False):
        port = self.config.wda if wda else self.config.appium
        response = self.transport.request(method, f"http://127.0.0.1:{port}{path}",
                                          json=body, timeout=(3, 15))
        response.raise_for_status()
        value = response.json()["value"]
        if isinstance(value, dict) and value.get("error"):
            raise RuntimeError("Device command failed")
        return value

    def sessions(self):
        try:
            value = self.rpc("GET", "/appium/sessions")
        except requests.HTTPError as exc:
            if exc.response.status_code != 404:
                raise ControlError("Appium session discovery unavailable; enable *:session_discovery", 503) from exc
            # Appium 2 before 2.19 used /sessions. Never fall back on permission failures.
            value = self.rpc("GET", "/sessions")
        if not isinstance(value, list) or any(not isinstance(s, dict) or not s.get("id") for s in value):
            raise RuntimeError("Cannot verify Appium ownership")
        return [s["id"] for s in value]

    def guard(self, connected=False):
        if self.ownership():
            raise ControlError("Collector is running; control refused")
        sessions = self.sessions()
        expected = [self.session] if connected else []
        if sessions != expected:
            if connected and self.session not in sessions:
                self.session, self.epoch = None, None
            raise ControlError("Appium ownership changed or device is busy")

    def poll_status(self):
        # Read-only ownership probes expose expiry/outages even before a gesture.
        # Never compete with a command or keep an idle Appium session alive.
        if (not self.session or self.uncertain or time.monotonic() - self.last_probe < 3
                or not self.lock.acquire(blocking=False)):
            return
        try:
            self.last_probe = time.monotonic()
            self.guard(connected=True)
            if not self.available:
                self.message = "Connected"
            self.available = True
        except ControlError as exc:
            self.available = False
            self.message = str(exc)
        except Exception:
            self.available = False
            self.message = "Device ownership unavailable; input disabled"
        finally:
            self.lock.release()

    def status(self):
        _, frame, age = self.relay.snapshot()
        return {"device": self.config.name, "connected": bool(self.session),
                "busy": self.lock.locked(), "uncertain": self.uncertain, "available": self.available,
                "message": self.message, "epoch": self.epoch, "next_sequence": self.sequence + 1,
                "frame": frame, "frame_age_ms": round(age * 1000) if frame else None,
                "fresh": bool(frame and age < STALE_SECONDS)}

    def command(self, kind, body):
        if not self.lock.acquire(blocking=False):
            raise ControlError("Device command already executing")
        mutating = False
        try:
            if self.uncertain:
                raise ControlError("Command outcome unknown; operator recovery required (no retry)")
            if kind == "connect":
                if self.session:
                    raise ControlError("Already connected")
                if not self.udid:
                    raise ControlError("Device UDID missing from server environment", 503)
                self.guard()
                status = self.rpc("GET", "/status", wda=True)
                if not isinstance(status, dict):
                    raise ControlError("Existing WDA unavailable", 503)
                caps = {"platformName": "iOS", "appium:automationName": "XCUITest",
                        "appium:udid": self.udid, "appium:deviceName": self.config.name,
                        "appium:bundleId": BUNDLE,
                        "appium:webDriverAgentUrl": f"http://127.0.0.1:{self.config.wda}",
                        "appium:autoLaunch": False, "appium:noReset": True,
                        "appium:fullReset": False, "appium:forceAppLaunch": False,
                        "appium:shouldTerminateApp": False, "appium:useNewWDA": False,
                        "appium:newCommandTimeout": 300, "appium:wdaConnectionTimeout": 10000}
                mutating = True
                self.message = "Connecting"
                value = self.rpc("POST", "/session", {"capabilities": {"alwaysMatch": caps}})
                sid = value["sessionId"]
                if not isinstance(sid, str) or not re.fullmatch(r"[\w-]+", sid):
                    raise RuntimeError("Invalid session identity")
                self.session = sid
                size = self.rpc("GET", f"/session/{sid}/window/rect")
                if (size.get("width"), size.get("height")) != (414, 896):
                    self.rpc("DELETE", f"/session/{sid}")
                    self.session = None
                    raise ControlError("Device is not 414×896 portrait; connection closed")
                self.epoch, self.sequence = secrets.token_hex(16), 0
                self.available = True
                self.message = "Connected"
                return self.status()
            if not self.session:
                raise ControlError("Connect first")
            if body.get("epoch") != self.epoch:
                raise ControlError("Connection changed; discarded command")
            if type(body.get("sequence")) is not int or body["sequence"] != self.sequence + 1:
                raise ControlError("Duplicate or out-of-order command discarded")
            # Consume before validation/dispatch: a failed submission can never replay.
            self.sequence += 1
            self.guard(connected=True)
            path = f"/session/{self.session}"
            if kind == "disconnect":
                mutating = True
                self.rpc("DELETE", path)
                self.session, self.epoch = None, None
                self.message = "Disconnected"
            elif kind in ("gesture", "activate", *SYSTEM_COMMANDS):
                if kind == "gesture":
                    actions = gesture_actions(body.get("points"))
                elif kind in SYSTEM_PATHS:
                    actions = gesture_actions([{"x": x, "y": y, "t": t} for x, y, t in SYSTEM_PATHS[kind]])
                else:
                    actions = None
                if not self.relay.fresh(body.get("frame")):
                    raise ControlError("Displayed video is stale; command discarded")
                size = self.rpc("GET", path + "/window/rect")
                if (size.get("width"), size.get("height")) != (414, 896):
                    raise ControlError("Device geometry changed; command discarded")
                if not self.relay.fresh(body.get("frame")):
                    raise ControlError("Displayed video became stale; command discarded")
                self.message = COMMAND_MESSAGES[kind]
                mutating = True
                if actions:
                    self.rpc("POST", path + "/actions", actions)
                elif kind == "home":
                    self.rpc("POST", path + "/execute/sync",
                             {"script": "mobile: pressButton", "args": [{"name": "home"}]})
                else:
                    self.rpc("POST", path + "/execute/sync",
                             {"script": "mobile: activateApp", "args": [{"bundleId": BUNDLE}]})
                self.message = "Connected — command complete"
            else:
                raise ControlError("Unknown command", 404)
            return self.status()
        except ControlError as exc:
            self.message = str(exc)
            raise
        except Exception as exc:
            if mutating:
                self.uncertain = True
                self.message = "Command outcome unknown; no retry. Operator recovery required."
            else:
                self.message = "Device unavailable; command not sent"
            raise ControlError(self.message, 503) from exc
        finally:
            self.lock.release()


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, devices, assets, port=8401, token=None):
        self.devices = {d.config.name: d for d in devices}
        self.assets = Path(assets)
        self.token = token or secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), Handler)
        self.host = f"127.0.0.1:{self.server_port}"
        self.origin = "http://" + self.host


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        pass  # Never log tokens, device identifiers or request payloads.

    def reply(self, code, body, content_type="application/json", headers=None):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def check(self, mutation=False):
        if self.headers.get("Host") != self.server.host:
            raise ControlError("Invalid host", 403)
        if mutation and self.headers.get("Origin") != self.server.origin:
            raise ControlError("Same origin required", 403)
        if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            raise ControlError("Cross-origin request refused", 403)
        if not secrets.compare_digest(self.headers.get("X-Control-Token", ""), self.server.token):
            raise ControlError("Launch token required", 403)

    def route(self):
        parts = self.path.split("/")
        if len(parts) != 4 or parts[1] != "api" or parts[2] not in self.server.devices:
            raise ControlError("Unknown device or endpoint", 404)
        return self.server.devices[parts[2]], parts[3]

    def stream_frames(self, device):
        """Optional authenticated MJPEG relay; each client closes after 60 seconds.

        The UI uses /frame so it can acknowledge exactly which frame it decoded.
        This endpoint shares the same reader and never contacts WDA itself.
        """
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=xrframe")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        end, previous = time.monotonic() + 60, -1
        while time.monotonic() < end and not device.relay.stop.is_set():
            jpeg, sequence, age = device.relay.snapshot()
            if not jpeg or age >= STALE_SECONDS:
                break
            if sequence != previous:
                self.wfile.write(f"--xrframe\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\nX-Frame-Sequence: {sequence}\r\n\r\n".encode() + jpeg + b"\r\n")
                self.wfile.flush()
                previous = sequence
            with device.relay.condition:
                device.relay.condition.wait(timeout=0.2)
        self.wfile.write(b"--xrframe--\r\n")

    def do_GET(self):
        try:
            # Static assets contain no token. Token arrives only via URL fragment.
            if self.path in ("/", "/ui.js", "/pointer.js", "/style.css"):
                if self.headers.get("Host") != self.server.host:
                    raise ControlError("Invalid host", 403)
                name = "index.html" if self.path == "/" else self.path[1:]
                mime = {"html": "text/html", "js": "text/javascript", "css": "text/css"}[name.split(".")[-1]]
                self.reply(200, (self.server.assets / name).read_bytes(), mime)
                return
            self.check()
            device, endpoint = self.route()
            if endpoint == "status":
                device.poll_status()
                self.reply(200, device.status())
            elif endpoint == "stream":
                self.stream_frames(device)
            elif endpoint == "frame":
                jpeg, sequence, age = device.relay.snapshot()
                if not jpeg or age >= STALE_SECONDS:
                    raise ControlError("Video unavailable or stale", 503)
                self.reply(200, jpeg, "image/jpeg", {"X-Frame-Sequence": sequence,
                                                    "X-Frame-Age-Ms": round(age * 1000)})
            else:
                raise ControlError("Unknown endpoint", 404)
        except ControlError as exc:
            self.reply(exc.status, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_POST(self):
        try:
            self.check(mutation=True)
            device, endpoint = self.route()
            if endpoint not in ("connect", "disconnect", "gesture", "activate", *SYSTEM_COMMANDS):
                raise ControlError("Unknown endpoint", 404)
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Type") != "application/json":
                raise ControlError("JSON with Content-Length required", 400)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ControlError("Invalid Content-Length", 400)
            if not 0 < length <= MAX_BODY:
                raise ControlError("Payload too large or empty", 413)
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise ControlError("Invalid JSON object", 400)
            self.reply(200, device.command(endpoint, body))
        except ControlError as exc:
            self.reply(exc.status, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
