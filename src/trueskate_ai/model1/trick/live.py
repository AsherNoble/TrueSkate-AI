"""Bounded XR capture of trick-speed runs on the rig; recordings only (admission is on the laptop).

Modelled on the kickflip runner (scripts/collection/probe_kickflip_repeatability.live_batch):
the same tunnel, attachment, session-ownership, recorder and timing-build checks. The
guard before each run checks only the foreground; there is no screenshot or menu check
(operator decision 2026-10-15). Every argument is checked before any device is touched.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.request

from trueskate_ai.model1.curved.contract import save_json
from .contract import EXPERIMENT, SCHEMA
from .schedule import plan_minute

WDA_REVISION = "ae50404aac12d9f8c41f6c3fa8776e97975eaef5"
# Both XRs share the 414×896 layout and control map; XR2 runs alongside an XR1 capture when its owner agrees.
PHONES = ("iPhone_XR", "iPhone_XR2")
TUNNEL_REGISTRY = "http://127.0.0.1:42314/remotexpc/tunnels/"
MIN_FREE_BYTES = 8 * 2**30
PRE_ROLL_SETTLE_S = 3.  # after the reset before each recording, as in curved v1
MENU_CHECKS = "none: protected hitboxes, foreground guard and calibration markers (operator decision 2026-10-15)"
FIXED_CONTEXT = ("device", "park", "settings_note")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@contextmanager
def busy_lock(path, *, owner: str, out):
    """Hold the rig's shared recording flag; refuse to start if anyone else holds it."""
    path = Path(path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"{path} exists: another recording holds the rig; no start attempted") from None
    with os.fdopen(fd, "w") as handle:
        json.dump(dict(owner=owner, pid=os.getpid(), started=_now(), out=str(out)), handle)
    try:
        yield
    finally:
        try:
            if json.loads(path.read_text()).get("pid") == os.getpid():
                path.unlink()
        except (OSError, ValueError):
            pass


def require_rig() -> None:
    if "training-server" not in socket.gethostname():
        raise ValueError("live runs belong on training-server, not laptop USB")


def check_arguments(root, *, minutes, env_file, ready_note, settings_note, park, device="iPhone_XR") -> None:
    if not Path(root).is_absolute():
        raise ValueError("absolute run directory required")
    if device not in PHONES:
        raise ValueError(f"device must be one of {', '.join(PHONES)}")
    if type(minutes) is not int or not 1 <= minutes <= 60:
        raise ValueError("minutes must be a whole number from 1 to 60")
    for name, value in (("operator readiness note", ready_note), ("settings note", settings_note), ("park", park)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} required")
    if not Path(env_file).is_absolute():
        raise ValueError("absolute rig .env path required")


def fix_context(root: Path, *, park, settings_note, device="iPhone_XR") -> dict:
    """Phone, park and settings are fixed per run directory; a change needs a new directory."""
    context = dict(experiment=EXPERIMENT, device=device, park=park, park_source="operator readiness statement",
                   settings_note=settings_note, menu_checks=MENU_CHECKS)
    path = root/"context.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if any(recorded.get(k) != context[k] for k in FIXED_CONTEXT):
            raise ValueError("phone, park or settings differ from this run directory; use a new --root")
        return recorded
    save_json(path, context)
    return context


def next_positions(root: Path) -> tuple[int, int]:
    """Next recording index and first unused run number, from what is already on disk."""
    index, run = 1, 0
    for folder in (root/"recordings").glob("m[0-9][0-9][0-9][0-9]"):
        index = max(index, int(folder.name[1:]) + 1)
        if (folder/"planned.json").exists():
            planned = json.loads((folder/"planned.json").read_text())
            numbers = [r["number"] for r in planned["runs"]] + planned.get("skipped_runs", [])
            run = max([run, *(n + 1 for n in numbers)])
    return index, run


def attachments_clear(udid: str, device: str = "iPhone_XR") -> None:
    """The phone's RemoteXPC tunnel is up and no XCTest attachment is left (KICKFLIP-REPEATABILITY-20261009)."""
    try:
        with urllib.request.urlopen(TUNNEL_REGISTRY + udid, timeout=5) as response:
            listed = response.read().decode()
    except OSError as exc:
        raise RuntimeError(f"{device} RemoteXPC tunnel unavailable; no start attempted ({exc})") from exc
    if udid not in listed:
        raise RuntimeError(f"tunnel registry did not return {device}; no start attempted")
    listing = subprocess.run(["appium", "driver", "run", "xcuitest", "cleanup-videos", "--", "--udid", udid,
                              "--dry-run"], capture_output=True, text=True, timeout=180)
    found = re.search(r"Found (\d+) UUID-shaped attachment", listing.stdout + listing.stderr)
    if listing.returncode or not found:
        raise RuntimeError(f"could not list {device} recording attachments; no start attempted")
    if int(found.group(1)):
        raise RuntimeError(f"{found.group(1)} {device} recording attachment(s) remain; preserve/clean before recording")


def collect(root, *, minutes, env_file, ready_note, settings_note, park, appium_port=4725, device="iPhone_XR",
            revision=WDA_REVISION, source_commit=None) -> None:
    """Record ``minutes`` one-minute recordings on one XR; stop at the first failure."""
    check_arguments(root, minutes=minutes, env_file=env_file, ready_note=ready_note,
                    settings_note=settings_note, park=park, device=device)
    require_rig()
    root, env_file = Path(root), Path(env_file)
    if not env_file.is_file():
        raise ValueError("rig .env file not found")
    root.mkdir(parents=True, exist_ok=True)
    context = fix_context(root, park=park, settings_note=settings_note, device=device)
    with (root/"sessions.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(started=_now(), operator_ready_note=ready_note, minutes=minutes, device=device,
                                     appium_port=appium_port, source_commit=source_commit, pid=os.getpid())) + "\n")
    from dotenv import load_dotenv
    load_dotenv(env_file, override=True)
    from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture, _http_json
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    from trueskate_ai.research.curved_audit import reset_payload
    from trueskate_ai.sim.device import BUNDLE_ID, DEVICES, DeviceSession
    from .recording import record_minute
    cfg = dict(next(d for d in DEVICES if d["name"] == device), appium_port=appium_port)
    udid = os.environ.get(cfg["env_key"])
    if not udid:
        raise ValueError(f"{device} UDID must come from the rig .env ({cfg['env_key']})")
    tunnel = subprocess.check_output(["launchctl", "print", "system/com.trueskate.remotexpc-tunnel"], text=True)
    if "state = running" not in tunnel:
        raise RuntimeError("root recording tunnel must be running; no start attempted")
    attachments_clear(udid, device)
    base = f"http://127.0.0.1:{cfg['wda_port']}"
    if _http_json(base + "/status").get("sessionId"):
        raise RuntimeError(f"{device} WDA session is already owned; do not disturb it")
    if _http_json(base + "/wda/activeAppInfo")["value"]["bundleId"] != BUNDLE_ID:
        raise RuntimeError("True Skate must already be frontmost")
    if _http_json(base + "/wda/video").get("value") is not None:
        raise RuntimeError("recorder is not idle; no start attempted")

    def sessions():
        value = _http_json(f"http://127.0.0.1:{appium_port}/appium/sessions").get("value")
        if not isinstance(value, list) or any(not isinstance(s, dict) or not s.get("id") for s in value):
            raise RuntimeError("Appium session discovery must be enabled and return a session list")
        return [s["id"] for s in value]

    if sessions():
        raise RuntimeError(f"{device} Appium session is already owned; do not disturb it")
    # Recordings issue commands continuously, but keep the session through slow stops.
    worker = DeviceSession(cfg, new_command_timeout_s=3600)
    try:
        worker.connect()
        driver = worker.driver
        owned = _http_json(base + "/status").get("sessionId")
        if (driver.capabilities.get("udid") or driver.capabilities.get("appium:udid")) != udid:
            raise RuntimeError(f"Appium session does not confirm the configured {device} UDID")
        timing = _http_json(f"{base}/session/{owned}/wda/actionTiming").get("value")
        if (not isinstance(timing, dict) or timing.get("schema_version") != 1
                or timing.get("build_revision") != revision or timing.get("enabled")):
            raise RuntimeError("instrumented WDA identity/idle check failed; no start attempted")

        def ownership():
            if sessions() != [driver.session_id] or not owned or _http_json(base + "/status").get("sessionId") != owned:
                raise RuntimeError("Appium/WDA session ownership changed")
            if driver.query_app_state(BUNDLE_ID) != 4 or worker._active_bundle_id() != BUNDLE_ID:
                raise RuntimeError("True Skate foreground lost")

        def guard():
            # Before every run: one WDA query (~0.1 s), as in curved v1. It also catches Control
            # Center over the game (DEPLOYMENT.md). The four ownership queries run once a minute;
            # before each run they could overrun the 0.4 s lead and fail the whole minute.
            if _http_json(base + "/wda/activeAppInfo").get("value", {}).get("bundleId") != BUNDLE_ID:
                raise RuntimeError("True Skate foreground lost")

        def perform(payload):
            return driver.execute("actions", payload)

        def perform_direct(payload):
            # One stroke = one XCTest record via WDA's direct endpoint (no Appium hop).
            request = urllib.request.Request(base + "/wda/perform_trick_gestures", method="POST",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())

        index, next_run = next_positions(root)
        for _ in range(minutes):
            if shutil.disk_usage(root).free < MIN_FREE_BYTES:
                raise RuntimeError("less than 8 GiB free; capture stopped")
            attachments_clear(udid, device)
            ownership()
            if _http_json(base + "/wda/video").get("value") is not None:
                raise RuntimeError("recorder is not idle")
            perform(reset_payload())
            time.sleep(PRE_ROLL_SETTLE_S)
            ownership()
            plan = plan_minute(next_run)
            out = root/"recordings"/f"m{index:04d}"
            metadata = dict(schema=SCHEMA, experiment=EXPERIMENT, recording=out.name, device=context["device"],
                            park=context["park"], settings_note=context["settings_note"], fps=60,
                            menu_checks=MENU_CHECKS, source_commit=source_commit)
            events = record_minute(out=out, plan=plan, recorder=XCTestScreenRecorder(driver, fps=60),
                                   timing=WDAActionTimingCapture(wda_port=cfg["wda_port"], expected_revision=revision),
                                   perform=perform, perform_direct=perform_direct, guard=guard, revision=revision,
                                   metadata=metadata)
            ownership()
            strokes = [e for e in events if e["kind"] == "stroke"]
            print(json.dumps(dict(recording=str(out), runs=len(plan["runs"]), runs_skipped=len(plan["skipped_runs"]),
                                  strokes_sent=sum(not e.get("skipped") for e in strokes),
                                  strokes_dropped=sum(bool(e.get("skipped")) for e in strokes),
                                  next_run=plan["next_run"])), flush=True)
            index, next_run = index + 1, plan["next_run"]
    finally:
        worker.disconnect()
