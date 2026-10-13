"""Bounded XR1 game cohort: the two frozen one-minute True Skate recordings.

Runs on training-server from a staged checkout with the runtime .venv, through the
curved collector's own path: run_recording with the 60 fps XCTest recorder, WDA
action timing, foreground/menu/editor guards and die-five calibration, then
native-frame clip extraction. A small research ledger stands in for the curved
ledger, so no production source or pipeline fingerprint changes. Every touch is
recorded as spent before its RPC; the first failure stops the cohort; nothing is
retried or replaced. Afterwards a temporary review manifest lists the clips.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_helper import ENV_FILE, LOCK, TRUE_SKATE, append_durable, digest, rig_state, verify_plan  # noqa: E402

REVIEW_ROUND = 20261013  # Distinct from curved qualification rounds in exported file names.


def commands(plan, recording):
    """The frozen schedule in run_recording/extract_recording form: scientific -> sample."""
    paths = {p["path_id"]: p for p in plan["paths"]}
    out = []
    for spec in recording["commands"]:
        if spec["kind"] == "scientific":
            path = paths[spec["path_id"]]
            if digest(path["payload"]) != path["payload_sha256"]:
                raise ValueError("frozen payload changed: "+spec["path_id"])
            out.append(dict(spec, kind="sample", candidate_id=int(spec["request_id"].split("-")[1]),
                            gesture=path["gesture"], payload=path["payload"],
                            payload_sha256=path["payload_sha256"]))
        elif spec["kind"] in ("control", "reset"):
            out.append(dict(spec))
        else:
            raise ValueError("unknown frozen command kind")
    return out


class ResearchLedger:
    """What extract_recording needs from a ledger: session context and durable settlement."""

    def __init__(self, root, *, park, plan_sha256):
        self.root, self.park, self.plan_sha256 = Path(root), park, plan_sha256

    def session(self, session_id):
        return dict(id=session_id, park=self.park, phase="physical-transfer-game", split="review")

    def settle(self, candidate_id, *, sample_path=None, reason=None):
        append_durable(self.root/"settled.jsonl", dict(candidate_id=candidate_id, admitted=sample_path is not None,
                                                        sample_path=str(sample_path) if sample_path else None,
                                                        reason=reason, at=time.time()))


def review_manifest(root, admitted, *, fingerprint):
    """A temporary review manifest that review.py verifies and serves unchanged."""
    from trueskate_ai.model1.curved.capture import file_sha256, verify_sample
    from trueskate_ai.model1.curved.contract import SCHEMA, digest as curved_digest
    entries = []
    for path in sorted(admitted, key=lambda p: p.name):
        meta = verify_sample(path)
        entries.append(dict(id=meta["candidate_id"], park=meta["park"],
                            sample_dir=str(path.relative_to(root)),
                            meta_sha256=file_sha256(path/"meta.json"), media_sha256=file_sha256(path/"clip.mp4"),
                            frame_times_s=meta["frame_times_s"], onset_s=meta["onset_s"],
                            release_s=meta["release_s"], gesture=meta["gesture"], clock_rate=meta["clock_rate"]))
    manifest = dict(schema=SCHEMA, round=REVIEW_ROUND, fingerprint=fingerprint, samples=entries)
    manifest["sha256"] = curved_digest(manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE/"frozen-plan.json")
    parser.add_argument("--amended-gate", type=Path, required=True, help="helper-gate-amended.json")
    parser.add_argument("--wda-revision", required=True)
    parser.add_argument("--park", required=True, help="operator's verbatim park")
    parser.add_argument("--waypoint", required=True, help="operator's verbatim waypoint")
    parser.add_argument("--settings-note", required=True, help="operator's verbatim settings note")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if "training-server" not in socket.gethostname():
        raise SystemExit("the game cohort runs on training-server")
    plan = verify_plan(json.loads(args.plan.read_text()))
    if not json.loads(args.amended_gate.read_text())["amended_gate"]["passed"]:
        raise SystemExit("helper gate has not passed")
    args.out.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # Refuses a busy rig.
    with os.fdopen(descriptor, "w") as handle:
        json.dump(dict(owner="physical-transfer game cohort", pid=os.getpid(),
                       started=datetime.now(timezone.utc).isoformat(), out=str(args.out)), handle)
    context = dict(park=args.park, waypoint=args.waypoint, settings_note=args.settings_note,
                   source="operator verbatim confirmation")
    fingerprint = "physical-transfer-20261013:"+plan["sha256"]
    admitted, error = [], None
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
        from trueskate_ai.sim.device import DeviceSession, DEVICES
        from trueskate_ai.collection.wda_action_timing import _http_json, WDAActionTimingCapture
        from trueskate_ai.collection.xctest_capture import XCTestScreenRecorder
        from trueskate_ai.collection.gameplay_filter import is_editor_frame, is_menu_frame
        from trueskate_ai.research.audit_execution import run_recording
        from trueskate_ai.model1.curved.capture import extract_recording
        from trueskate_ai.model1.curved.collection import attachment_guard
        from trueskate_ai.model1.curved.contract import save_json
        cfg = next(d for d in DEVICES if d["name"] == plan["device"])
        udid = os.environ[plan["device_env_key"]]
        port = plan["wda_port"]
        base = f"http://127.0.0.1:{port}"
        save_json(args.out/"context.json", dict(context, plan_sha256=plan["sha256"], fingerprint=fingerprint))

        def guard():
            if _http_json(base+"/wda/activeAppInfo").get("value", {}).get("bundleId") != TRUE_SKATE:
                raise RuntimeError("True Skate foreground unavailable/lost")

        state = rig_state(port, udid)
        save_json(args.out/"preflight.json", state)
        if not state["tunnel_running"] or state["recorder"].get("value") is not None:
            raise RuntimeError("preflight failed: tunnel and idle recorder required")
        guard()  # Before connect, which may otherwise activate the app.
        attachment_guard(udid)
        ledger = ResearchLedger(args.out, park=args.park, plan_sha256=plan["sha256"])
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

            def touch(kind, request_id, payload):
                append_durable(args.out/"spend.jsonl", dict(kind=kind, request_id=request_id,
                                                            payload_sha256=digest(payload), spent_epoch_s=time.time()))
                return driver.execute("actions", payload)

            for recording in plan["game_recordings"]:
                rid = recording["recording_id"]
                if shutil.disk_usage(args.out).free < 8*2**30:
                    raise RuntimeError("less than 8 GiB free; cohort stopped")
                attachment_guard(udid)
                gameplay()
                reset = recording["pre_roll_reset"]
                touch("pre_roll_reset", rid+"-pre-roll", reset["payload"])
                time.sleep(reset["settle_s"])
                gameplay()
                out = args.out/"recordings"/rid
                run_recording(
                    recorder=XCTestScreenRecorder(driver, fps=60),
                    timing=WDAActionTimingCapture(wda_port=port, expected_revision=args.wda_revision),
                    commands=commands(plan, recording),
                    perform=lambda spec: touch(spec["kind"], spec.get("request_id", spec.get("role")), spec["payload"]),
                    guard=guard, out=out, prepare=lambda spec: gameplay(),
                    prepare_lead_s=recording["prepare_lead_s"], guard_lead_s=recording["guard_lead_s"],
                    revision=args.wda_revision,
                    metadata=dict(schema="physical-transfer-game-v1", seed=plan["seed"], session=rid,
                                  device=plan["device"], park=args.park, park_source=context["source"],
                                  fingerprint=fingerprint, waypoint=args.waypoint,
                                  settings_note=args.settings_note, plan_sha256=plan["sha256"]))
                if _http_json(base+"/wda/video").get("value") is not None:
                    raise RuntimeError("recorder did not return idle")
                attachment_guard(udid)
                accepted = extract_recording(out, ledger, revision=args.wda_revision)
                admitted += accepted
                print(json.dumps(dict(recording=rid, admitted=len(accepted))), flush=True)
        finally:
            worker.disconnect()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        print("GAME_COHORT_STOPPED "+error, flush=True)
    finally:
        LOCK.unlink(missing_ok=True)
    summary = dict(error=error, admitted=len(admitted),
                   spent=len((args.out/"spend.jsonl").read_text().splitlines())
                   if (args.out/"spend.jsonl").exists() else 0)
    if not error and admitted:
        manifest = review_manifest(args.out, admitted, fingerprint=fingerprint)
        from trueskate_ai.model1.curved.contract import save_json
        save_json(args.out/f"audit-{REVIEW_ROUND}"/"manifest.json", manifest)
        summary["review_manifest_sha256"] = manifest["sha256"]
    (args.out/"run-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"GAME_COHORT_DONE admitted={summary['admitted']} spent={summary['spent']} "
          f"error={'yes' if error else 'no'}", flush=True)


if __name__ == "__main__":
    main()
