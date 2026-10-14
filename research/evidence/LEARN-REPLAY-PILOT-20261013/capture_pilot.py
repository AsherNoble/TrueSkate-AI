"""Bounded pilot capture: one frozen session (five one-minute recordings) per invocation.

Runs on training-server from a staged checkout with the runtime .venv, through the
same path as the physical game cohort: run_recording with the 60 fps XCTest recorder,
WDA action timing, foreground/menu/editor guards and die-five calibration, then
native-frame clip extraction into a research ledger. No production source or
pipeline fingerprint changes. Sessions run in plan order. Every touch is recorded
as spent before its RPC. Nothing is retried, replaced or reordered.

Stop rules, as amended by the operator on 2026-10-14 after pilot-01-r1: a recording
whose only failure is die-five calibration loses its ten candidates and capture
continues, while total losses stay within LOSS_CAP. Any other failure or any
rejected candidate stops the pilot for diagnosis (Gate 2). A session stopped only by
calibration (pilot-01, before the amendment) resumes with --resume, which runs its
unattempted recordings in a new run directory.

Second amendment, 2026-10-14, after pilot-01+1: resets had drifted to a MACBA-wall
waypoint where the game turns the board around ~2.3 s after spawning, which defeated
calibration. The operator re-anchored XR1 at the street waypoint. Each pre-roll reset
is now followed by a stillness check that stops before any scientific touch if the
view is still moving. The loss cap counts only losses after the re-anchor, and the
operator's review closes pilot-01+1's cap stop.

Gate 2, 2026-10-15, after pilot-04-r0: mid-recording resets still drift to the wall
waypoint in about a third of recordings. The operator lifted the loss cap: a
calibration-only failure loses its recording's candidates and capture continues to
the end of the plan. Calibration, the stillness check and every other stop rule are
unchanged. The operator's decision closes pilot-04's cap stop.

Gate 2, 2026-10-15, after pilot-04+1: r3 ran every touch, but the recorder stop could
not retrieve its video, which stayed on XR1. The operator chose to copy it to host
storage, clean up with the documented wrapper and continue. r3 counts as a technical
failure (its candidates lost); pilot-04 resumes with r4.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
PHYSICAL = HERE.parent/"PHYSICAL-TRANSFER-20261013"
sys.path[:0] = [str(HERE), str(PHYSICAL)]
from bind_context import (GUARD_LEAD_S, PRE_ROLL_SETTLE_S, PREPARE_LEAD_S,  # noqa: E402
                          RECORDINGS_PER_SESSION, STILL_MAX_DIFF, STILL_WINDOW_S, load_plan, verify_bound)
from run_game import preparation  # noqa: E402
from run_helper import ENV_FILE, LOCK, TRUE_SKATE, append_durable, digest, rig_state  # noqa: E402

SESSIONS = [f"pilot-{i:02}" for i in range(12)]
SLOTS_PER_RECORDING = 10
SUPPORT_PER_RECORDING = 6  # Three markers, two in-recording resets, one pre-roll reset.
LOSSES_BEFORE_REANCHOR = 40  # pilot-01 r1-r4; reported apart from later losses (second amendment).
ACKNOWLEDGED_RUNS = {"pilot-01+1", "pilot-04", "pilot-04+1"}  # Stops the operator reviewed and closed.
CALIBRATION_FAILURE = "die-five calibration failed"


def recordings(plan, session):
    """Five schedules: the session's candidates in id order on the collector's slots."""
    from trueskate_ai.model1.curved.collection import CONTROL_SLOTS, RESET_SLOTS, SLOTS
    from trueskate_ai.research.curved_audit import marker_payload, reset_payload
    rows = sorted((r for r in plan["candidates"] if r["session"] == session), key=lambda r: r["id"])
    if len(rows) != RECORDINGS_PER_SESSION*SLOTS_PER_RECORDING or len(SLOTS) != SLOTS_PER_RECORDING:
        raise ValueError("frozen session size changed")
    out = []
    for index in range(RECORDINGS_PER_SESSION):
        chunk = rows[index*SLOTS_PER_RECORDING:(index+1)*SLOTS_PER_RECORDING]
        commands = [dict(kind="control", role=role, slot_s=slot, payload=marker_payload())
                    for role, slot in CONTROL_SLOTS]
        commands += [dict(kind="reset", slot_s=slot, payload=reset_payload()) for slot in RESET_SLOTS]
        commands += [dict(kind="sample", slot_s=slot, candidate_id=row["id"], gesture=row["gesture"],
                          payload=row["payload"]) for slot, row in zip(SLOTS, chunk)]
        out.append(dict(recording_id=f"{session}-r{index}", candidate_ids=[r["id"] for r in chunk],
                        commands=sorted(commands, key=lambda c: c["slot_s"])))
    return out


def spent(root):
    """Touches spent by every session so far, from their durable spend logs."""
    counts = Counter()
    for path in sorted(Path(root).glob("sessions/*/spend.jsonl")):
        for line in path.read_text().splitlines():
            counts["sample" if json.loads(line)["kind"] == "sample" else "support"] += 1
    return counts


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def runs(root, session):
    """A session's run directories: the original, then numbered resumptions."""
    base = Path(root)/"sessions"
    found = [base/session] if (base/session).exists() else []
    return found+sorted(base.glob(session+"+*"), key=lambda p: int(p.name.split("+")[1]))


def session_state(root, session):
    """Attempted recordings, outcomes and calibration losses across a session's runs."""
    dirs = runs(root, session)
    spent_ids = {int(r["request_id"].split("-")[1]) for d in dirs for r in rows(d/"spend.jsonl")
                 if r["kind"] == "sample"}
    settled = [s for d in dirs for s in rows(d/"settled.jsonl")]
    summaries = [json.loads((d/"run-summary.json").read_text()) for d in dirs if (d/"run-summary.json").exists()]
    errors = [s["error"] for s in summaries if s["error"] and s.get("run") not in ACKNOWLEDGED_RUNS]
    return dict(started=bool(dirs), finished=len(summaries) == len(dirs),
                attempted=sorted({p.name for d in dirs for p in (d/"recordings").glob(session+"-r*")}),
                admitted=sum(1 for s in settled if s["admitted"]),
                rejected=sorted(s["candidate_id"] for s in settled if not s["admitted"]),
                lost=sorted(spent_ids-{s["candidate_id"] for s in settled}),
                calibration_only=all(e.startswith("ValueError: "+CALIBRATION_FAILURE) for e in errors),
                runs=len(dirs))


def total_lost(root):
    return sum(len(session_state(root, s)["lost"]) for s in SESSIONS)


def new_losses(root):
    """Losses the cap counts: those after the re-anchor."""
    return total_lost(root)-LOSSES_BEFORE_REANCHOR


def grey(png):
    from PIL import Image
    import io
    import numpy as np
    image = Image.open(io.BytesIO(png)).convert("L").resize((64, 144), Image.BOX)
    return np.asarray(image, dtype=float)[24:120]


def assert_still(screenshot, *, start, clock=time.monotonic, sleep=time.sleep, window=STILL_WINDOW_S):
    """Sample screenshots across the post-reset window; refuse a view that is still moving."""
    import numpy as np
    sleep(max(0., start+window[0]-clock()))
    previous, diffs = grey(screenshot()), []
    while clock()-start < window[1]:
        current = grey(screenshot())
        diffs.append(float(np.mean(np.abs(current-previous))))
        previous = current
    if not diffs or max(diffs) > STILL_MAX_DIFF:
        raise RuntimeError(f"view still moving after the pre-roll reset (max diff "
                           f"{max(diffs) if diffs else 'n/a'}); waypoint drift? (Gate 2)")
    return diffs


def admissible(plan, root, session, *, resume=False):
    """Refuse a rerun, an out-of-order session, an open Gate 2 or the budget."""
    if session not in SESSIONS:
        raise ValueError("unknown pilot session")
    state = session_state(root, session)
    if resume:
        if not state["started"] or not state["finished"]:
            raise RuntimeError(f"{session} has no finished run to resume")
        if state["rejected"] or not state["calibration_only"]:
            raise RuntimeError(f"Gate 2 is open in {session}; only a calibration-only stop resumes")
        if len(state["attempted"]) == RECORDINGS_PER_SESSION:
            raise RuntimeError(f"{session} has no unattempted recording")
    elif state["started"]:
        raise RuntimeError(f"{session} already started; no retry or replacement")
    for previous in SESSIONS[:SESSIONS.index(session)]:
        prior = session_state(root, previous)
        if not prior["finished"] or len(prior["attempted"]) != RECORDINGS_PER_SESSION:
            raise RuntimeError(f"{previous} has not completed; sessions run in plan order")
        if prior["rejected"] or not prior["calibration_only"]:
            raise RuntimeError(f"Gate 2 is open after {previous}; the pilot stops for diagnosis")
    remaining = RECORDINGS_PER_SESSION-len(state["attempted"])
    used = spent(root)
    if (used["sample"]+remaining*SLOTS_PER_RECORDING > plan["candidate_dispatch_limit"]
            or used["support"]+remaining*SUPPORT_PER_RECORDING > plan["capture_support_limit"]):
        raise RuntimeError("frozen capture budget would be exceeded")
    return used


class PilotLedger:
    """What extract_recording needs from a ledger: session context and durable settlement."""

    def __init__(self, root, plan, *, park):
        self.root, self.park = Path(root), park
        self.split = {r["session"]: r["split"] for r in plan["candidates"]}

    def session(self, session_id):
        return dict(id=session_id, park=self.park, phase="learn-replay-pilot", split=self.split[session_id])

    def settle(self, candidate_id, *, sample_path=None, reason=None):
        append_durable(self.root/"settled.jsonl", dict(candidate_id=candidate_id, admitted=sample_path is not None,
                                                        sample_path=str(sample_path) if sample_path else None,
                                                        reason=reason, at=time.time()))


def request_id(spec):
    if spec["kind"] == "sample":
        return f"candidate-{spec['candidate_id']}"
    return spec.get("role") or f"reset-{spec['slot_s']}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE/"plan.json.gz")
    parser.add_argument("--bound", type=Path, required=True, help="bound context from bind_context.py")
    parser.add_argument("--session", required=True, choices=SESSIONS)
    parser.add_argument("--root", type=Path, required=True, help="pilot capture root on the rig")
    parser.add_argument("--resume", action="store_true",
                        help="run the unattempted recordings of a session stopped only by calibration")
    args = parser.parse_args(argv)
    if "training-server" not in socket.gethostname():
        raise SystemExit("pilot capture runs on training-server")
    plan = load_plan(args.plan)
    bound = verify_bound(json.loads(args.bound.read_text()), plan)
    admissible(plan, args.root, args.session, resume=args.resume)
    progress = session_state(args.root, args.session)
    out = args.root/"sessions"/(args.session+(f"+{progress['runs']}" if args.resume else ""))
    # A busy rig is refused before the session exists, so nothing marks it started.
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(dict(owner="learn-replay pilot capture "+args.session, pid=os.getpid(),
                       started=datetime.now(timezone.utc).isoformat(), out=str(out)), handle)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except BaseException:
        LOCK.unlink(missing_ok=True)
        raise
    context = bound["context"]
    fingerprint = "learn-replay-pilot-20261013:"+bound["sha256"]
    admitted, error = [], None
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
        from trueskate_ai.sim.device import DeviceSession, DEVICES
        from trueskate_ai.collection.wda_action_timing import _http_json, WDAActionTimingCapture
        from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
        from trueskate_ai.collection.gameplay_filter import is_editor_frame, is_menu_frame
        from trueskate_ai.research.audit_execution import run_recording
        from trueskate_ai.research.curved_audit import reset_payload
        from trueskate_ai.model1.curved.capture import extract_recording
        from trueskate_ai.model1.curved.collection import attachment_guard
        from trueskate_ai.model1.curved.contract import save_json
        cfg = next(d for d in DEVICES if d["name"] == bound["device"])
        udid = os.environ[bound["device_env_key"]]
        port = bound["wda_port"]
        base = f"http://127.0.0.1:{port}"
        save_json(out/"context.json", dict(session=args.session, plan_sha256=plan["sha256"],
                                           bound_sha256=bound["sha256"], fingerprint=fingerprint))

        def guard():
            if _http_json(base+"/wda/activeAppInfo").get("value", {}).get("bundleId") != TRUE_SKATE:
                raise RuntimeError("True Skate foreground unavailable/lost")

        state = rig_state(port, udid)
        save_json(out/"preflight.json", state)
        if not state["tunnel_running"] or state["recorder"].get("value") is not None:
            raise RuntimeError("preflight failed: tunnel and idle recorder required")
        guard()  # Before connect, which may otherwise activate the app.
        attachment_guard(udid)
        ledger = PilotLedger(out, plan, park=context["park"])
        worker = DeviceSession(cfg, new_command_timeout_s=3600)
        try:
            worker.connect()
            driver = worker.driver
            attachment_guard(udid)

            def gameplay():
                guard()
                png = driver.get_screenshot_as_png()
                # Idle navigation overlays usable gameplay (operator decision 2026-10-11).
                if is_menu_frame(png, allow_idle_navigation=True) or is_editor_frame(png):
                    raise RuntimeError("menu/editor detected before touch")
                guard()

            def touch(kind, rid, payload):
                append_durable(out/"spend.jsonl", dict(kind=kind, request_id=rid,
                                                       payload_sha256=digest(payload), spent_epoch_s=time.time()))
                return driver.execute("actions", payload)

            for recording in recordings(plan, args.session):
                rid = recording["recording_id"]
                if rid in progress["attempted"]:
                    continue  # Resumption: earlier runs already spent this recording.
                if shutil.disk_usage(out).free < 8*2**30:
                    raise RuntimeError("less than 8 GiB free; capture stopped")
                attachment_guard(udid)
                gameplay()
                touch("pre_roll_reset", rid+"-pre-roll", reset_payload())
                reset_at = time.monotonic()
                diffs = assert_still(driver.get_screenshot_as_png, start=reset_at)
                append_durable(out/"stillness.jsonl", dict(recording=rid, diffs=diffs, at=time.time()))
                time.sleep(max(0., reset_at+PRE_ROLL_SETTLE_S-time.monotonic()))
                gameplay()
                target = out/"recordings"/rid
                run_recording(
                    recorder=XCTestScreenRecorder(driver, fps=60),
                    timing=WDAActionTimingCapture(wda_port=port, expected_revision=bound["wda_revision"]),
                    commands=recording["commands"],
                    perform=lambda spec: touch(spec["kind"], request_id(spec), spec["payload"]),
                    guard=guard, out=target, prepare=preparation(gameplay),
                    prepare_lead_s=PREPARE_LEAD_S, guard_lead_s=GUARD_LEAD_S,
                    revision=bound["wda_revision"],
                    metadata=dict(schema="learn-replay-pilot-capture-v1", seed=plan["seed"],
                                  session=args.session, device=bound["device"], park=context["park"],
                                  park_source=context["source"], fingerprint=fingerprint,
                                  waypoint=context["waypoint"], settings_note=context["settings_note"],
                                  plan_sha256=plan["sha256"], bound_sha256=bound["sha256"]))
                if _http_json(base+"/wda/video").get("value") is not None:
                    raise RuntimeError("recorder did not return idle")
                attachment_guard(udid)
                try:
                    accepted = extract_recording(target, ledger, revision=bound["wda_revision"])
                except ValueError as exc:
                    if not str(exc).startswith(CALIBRATION_FAILURE):
                        raise
                    # Amended rule: the recording's candidates are lost, never replaced.
                    append_durable(out/"losses.jsonl", dict(recording=rid, candidate_ids=recording["candidate_ids"],
                                                            reason=str(exc), at=time.time()))
                    print(json.dumps(dict(recording=rid, admitted=0, lost=SLOTS_PER_RECORDING,
                                          reason=str(exc))), flush=True)
                    continue
                admitted += accepted
                print(json.dumps(dict(recording=rid, admitted=len(accepted))), flush=True)
                if len(accepted) != SLOTS_PER_RECORDING:
                    raise RuntimeError(f"{SLOTS_PER_RECORDING-len(accepted)} candidate(s) rejected in {rid}; "
                                       "the corpus is incomplete (Gate 2)")
        finally:
            worker.disconnect()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        print("PILOT_SESSION_STOPPED "+error, flush=True)
    finally:
        LOCK.unlink(missing_ok=True)
    kinds = Counter(json.loads(line)["kind"] for line in (out/"spend.jsonl").read_text().splitlines()) \
        if (out/"spend.jsonl").exists() else Counter()
    summary = dict(session=args.session, run=out.name, resume=args.resume, error=error, admitted=len(admitted),
                   lost=[r for line in rows(out/"losses.jsonl") for r in line["candidate_ids"]],
                   spent=dict(kinds), cumulative=dict(spent(args.root)), cumulative_lost=total_lost(args.root),
                   losses_after_reanchor=new_losses(args.root))
    (out/"run-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"PILOT_SESSION_DONE session={args.session} admitted={len(admitted)} "
          f"spent={sum(kinds.values())} error={'yes' if error else 'no'}", flush=True)


if __name__ == "__main__":
    main()
