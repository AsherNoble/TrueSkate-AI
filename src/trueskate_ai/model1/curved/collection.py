"""Explicitly bounded collection on an existing healthy rig session."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import re
import time
from uuid import uuid4
from urllib.request import urlopen

from .contract import Gesture, SAMPLERS, save_json
from .ledger import pipeline_fingerprint

SLOTS = (5, 10, 15, 20, 24, 35, 40, 45, 49, 52)


def commands_for(ledger, session_id):
    from trueskate_ai.research.curved_audit import marker_payload, reset_payload
    commands = [dict(kind="control", role=role, slot_s=slot, payload=marker_payload())
                for role, slot in (("start", 1), ("middle", 30), ("end", 58))]
    commands += [dict(kind="reset", slot_s=slot, payload=reset_payload()) for slot in (27, 55)]
    candidates = []
    try:
        for slot in SLOTS:
            available = [s for s in SAMPLERS if ledger.remaining(session_id, s) > 0]
            if not available:
                break
            # Balance each session; each candidate has its own independent seed.
            sampler = max(available, key=lambda s: ledger.remaining(session_id, s))
            candidate = ledger.plan_candidate(session_id, sampler)
            candidates.append(candidate["id"])
            commands.append(dict(kind="sample", slot_s=slot, candidate_id=candidate["id"],
                                 gesture=candidate["gesture"],
                                 payload=Gesture.from_dict(candidate["gesture"]).payload()))
    except BaseException:
        ledger.cancel_planned(candidates)
        raise
    return sorted(commands, key=lambda c: c["slot_s"]), candidates


def attachment_guard(udid):
    """Read only; never delete recordings or attempt recorder recovery here."""
    if not udid:
        raise RuntimeError("device UDID unavailable")
    with urlopen("http://127.0.0.1:42314/remotexpc/tunnels/"+udid, timeout=5) as response:
        if udid not in response.read().decode():
            raise RuntimeError("device RemoteXPC tunnel unavailable")
    listing = subprocess.run(["appium", "driver", "run", "xcuitest", "cleanup-videos", "--",
                              "--udid", udid, "--dry-run"], capture_output=True, text=True, timeout=180)
    found = re.search(r"Found (\d+) UUID-shaped attachment", listing.stdout+listing.stderr)
    if listing.returncode or not found or int(found.group(1)):
        raise RuntimeError("recording attachments are not verified empty; use documented recovery")


def collect(ledger, *, session_id, park, device, revision, max_minutes, env_file, waypoint, settings_note):
    """One park/session per invocation, no launchd, retries, alerts or park guesses."""
    if "training-server" not in socket.gethostname():
        raise RuntimeError("live collection runs on training-server")
    if not 1 <= max_minutes <= 28:
        raise ValueError("explicit --max-minutes must be 1–28")
    session = ledger.session(session_id)
    if park != session["park"]:
        raise ValueError("operator's current park does not match the frozen session")
    if waypoint not in range(1, 6) or not settings_note.strip():
        raise ValueError("waypoint 1–5 and fixed settings note required")
    context = dict(device=device, waypoint=waypoint, settings_note=settings_note)
    previous = ledger.get("context:"+session_id)
    if previous is not None and previous != context:
        raise ValueError("session device/waypoint/settings changed; keep its frozen provenance")
    ledger.set("context:"+session_id, context)
    fingerprint = pipeline_fingerprint(revision)
    if session["phase"] == "production":
        ledger.production_sessions(fingerprint)
    else:
        row = ledger.db.execute("SELECT * FROM rounds WHERE number=?", (session["round"],)).fetchone()
        if row["status"] != "collecting" or row["fingerprint"] != fingerprint:
            raise ValueError("audit is sealed or pipeline differs from the current round")
    pending = ledger.db.execute("SELECT id FROM attempts WHERE status IN ('planned','dispatched')").fetchall()
    if pending:
        raise RuntimeError("interrupted attempts need explicit reconciliation before collecting")
    from dotenv import load_dotenv
    if not Path(env_file).is_file():
        raise ValueError("explicit rig environment file is missing")
    load_dotenv(Path(env_file), override=False)
    from trueskate_ai.sim.device import DeviceSession, DEVICES, BUNDLE_ID
    from trueskate_ai.collection.wda_action_timing import _http_json, WDAActionTimingCapture
    from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
    from trueskate_ai.collection.gameplay_filter import is_editor_frame, is_menu_frame
    from trueskate_ai.research.audit_execution import run_recording
    from trueskate_ai.research.curved_audit import reset_payload
    from .capture import extract_recording
    cfg = next(d for d in DEVICES if d["name"] == device)
    udid = os.environ.get("IPHONE_XR_UDID" if device == "iPhone_XR" else "IPHONE_XR2_UDID")
    base = f'http://127.0.0.1:{cfg["wda_port"]}'

    def guard():
        if _http_json(base+"/wda/activeAppInfo").get("value", {}).get("bundleId") != BUNDLE_ID:
            raise RuntimeError("True Skate foreground unavailable/lost")

    tunnel = subprocess.check_output(["launchctl", "print", "system/com.trueskate.remotexpc-tunnel"], text=True)
    if "state = running" not in tunnel:
        raise RuntimeError("root RemoteXPC tunnel unavailable")
    guard()  # Before connect, which may otherwise activate the app.
    if _http_json(base+"/wda/video").get("value") is not None:
        raise RuntimeError("recorder is not idle")
    attachment_guard(udid)
    worker = DeviceSession(cfg)
    try:
        worker.connect()
        driver = worker.driver
        attachment_guard(udid)

        def gameplay():
            guard()
            png = driver.get_screenshot_as_png()
            if is_menu_frame(png) or is_editor_frame(png):
                raise RuntimeError("menu/editor detected before touch")
            guard()

        for _ in range(max_minutes):
            if not any(ledger.remaining(session_id, s) for s in SAMPLERS):
                break
            # Keep existing free-space protection. Preserve all original recordings.
            if shutil.disk_usage(ledger.root).free < 8*2**30:
                raise RuntimeError("less than 8 GiB free; collection stopped")
            attachment_guard(udid)
            gameplay()
            driver.execute("actions", reset_payload())
            time.sleep(3)
            gameplay()
            commands, ids = commands_for(ledger, session_id)
            out = ledger.root/"recordings"/f"{session_id}-{uuid4().hex[:12]}"

            def perform(spec):
                gameplay()
                if spec["kind"] == "sample":
                    ledger.dispatch(spec["candidate_id"])  # Durable before RPC.
                return driver.execute("actions", spec["payload"])

            try:
                run_recording(recorder=XCTestScreenRecorder(driver, fps=60),
                              timing=WDAActionTimingCapture(wda_port=cfg["wda_port"], expected_revision=revision),
                              commands=commands, perform=perform, guard=guard, out=out,
                              revision=revision,
                              metadata=dict(schema="model1_curved_v1", seed=ledger.definition["seed"],
                                            session=session_id, device=device, park=park,
                                            park_source="operator verbatim confirmation", fingerprint=fingerprint,
                                            waypoint=waypoint, settings_note=settings_note))
                if _http_json(base+"/wda/video").get("value") is not None:
                    raise RuntimeError("recorder did not return idle")
                attachment_guard(udid)
                accepted = extract_recording(out, ledger, revision=revision)
                print(json.dumps(dict(recording=str(out), admitted=len(accepted), spent=ledger.spent)), flush=True)
            except BaseException as exc:
                # One failed recording stops this invocation. Unknown outcomes spend budget.
                ledger.seal_interrupted(ids, f"recording failed: {type(exc).__name__}: {exc}")
                if out.exists():
                    save_json(out/"collection-failure.json", dict(error=f"{type(exc).__name__}: {exc}"))
                raise
    finally:
        worker.disconnect()


def reconcile(ledger, *, abandon=False):
    """Recover complete admission evidence; never execute an interrupted command again."""
    pending = [r[0] for r in ledger.db.execute("SELECT id FROM attempts WHERE status IN ('planned','dispatched')")]
    recovered = []
    for path in ledger.root.glob("recordings/*/sample-*/meta.json"):
        meta = json.loads(path.read_text())
        if meta.get("candidate_id") in pending:
            ledger.settle(meta["candidate_id"], sample_path=path.parent)
            recovered.append(meta["candidate_id"])
    remaining = sorted(set(pending)-set(recovered))
    if abandon:
        ledger.seal_interrupted(remaining)
    return dict(recovered=recovered, unresolved=remaining, abandoned=abandon)
