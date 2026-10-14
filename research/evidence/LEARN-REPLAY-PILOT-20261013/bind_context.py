"""Bind verified XR1 context and the Gate 1 decision to the frozen pilot plan.

Runs on training-server, read-only: no device session, touch or recording. The
frozen plan.json.gz stays unchanged. The bound file pins its digest, the operator's
verbatim park/waypoint/settings, the device, iOS, True Skate and WDA versions, the
capture schedule and the curved pipeline fingerprint of the staged source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHYSICAL = HERE.parent/"PHYSICAL-TRANSFER-20261013"
sys.path[:0] = [str(HERE), str(PHYSICAL)]
from prepare import validate  # noqa: E402
from run_helper import ENV_FILE, TRUE_SKATE, digest, rig_state, run  # noqa: E402

KIND = "learn-replay-pilot-bound-context-v1"
DEVICE, DEVICE_ENV_KEY, WDA_PORT = "iPhone_XR", "IPHONE_XR_UDID", 8100
# Operator decision at Gate 1 (2026-10-14), made before any pilot touch: start
# preparation 2.25 s ahead of each slot instead of 1.75 s. Game attempt 4 needed up
# to 1.58 s; a later start cannot shorten any margin, and admission is unchanged.
PREPARE_LEAD_S, GUARD_LEAD_S, PRE_ROLL_SETTLE_S = 2.25, .25, 3
RECORDINGS_PER_SESSION = 5
# Second amendment (2026-10-14): stillness after each pre-roll reset. Mean |difference|
# of consecutive 64x144 grey screenshots, rows 24-120 (HUD and idle navigation bar
# cropped). Still post-reset views measured <= 3.4; the MACBA-wall waypoint's
# turnaround measured 17-26.
STILL_WINDOW_S, STILL_MAX_DIFF = (1.0, 4.5), 10.0
RUNNERS = ("bind_context.py", "capture_pilot.py", "prepare.py")
SHARED = ("run_helper.py", "run_game.py")


def load_plan(path):
    plan = json.loads(gzip.decompress(Path(path).read_bytes()))
    validate(plan)
    return plan


def schedule():
    from trueskate_ai.model1.curved.collection import CONTROL_SLOTS, RESET_SLOTS, SLOTS
    return dict(sample_slots_s=list(SLOTS), control_slots_s=[list(c) for c in CONTROL_SLOTS],
                reset_slots_s=list(RESET_SLOTS), prepare_lead_s=PREPARE_LEAD_S,
                guard_lead_s=GUARD_LEAD_S, pre_roll_settle_s=PRE_ROLL_SETTLE_S,
                recordings_per_session=RECORDINGS_PER_SESSION,
                first_marker_screenshot_check=False,
                pre_roll_stillness=dict(window_s=list(STILL_WINDOW_S), max_mean_abs_diff=STILL_MAX_DIFF,
                                        grey_size=[64, 144], rows=[24, 120]),
                stop_rules=["any recording failure other than die-five calibration stops the pilot (Gate 2)",
                            "a die-five calibration failure loses that recording's candidates without "
                            "replacement and capture continues; no loss cap (operator amendments "
                            "2026-10-14, cap lifted at Gate 2 on 2026-10-15)",
                            "a view still moving after a pre-roll reset stops the pilot before any "
                            "scientific touch (Gate 2)",
                            "any rejected candidate stops the pilot after its recording (Gate 2)",
                            "no retry, replacement or reordering; sessions run in plan order; a session "
                            "stopped only by calibration resumes with its unattempted recordings"])


def source_hashes():
    files = [HERE/name for name in RUNNERS]+[PHYSICAL/name for name in SHARED]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def device_facts(udid, scratch):
    """iOS and True Skate versions from devicectl's JSON output; nothing is installed."""
    details, apps = scratch/"details.json", scratch/"apps.json"
    for args, path in ((["details"], details), (["apps", "--include-all-apps"], apps)):
        result = run(["xcrun", "devicectl", "device", "info", *args, "--device", udid,
                      "--json-output", str(path)], timeout=120)
        if result["exit_code"]:
            raise RuntimeError("devicectl "+args[0]+" failed: "+result["stderr"][-300:])
    props = json.loads(details.read_text())["result"]["deviceProperties"]
    app = next(a for a in json.loads(apps.read_text())["result"]["apps"]
               if a.get("bundleIdentifier") == TRUE_SKATE)
    return dict(ios_version=props.get("osVersionNumber"), ios_build=props.get("osBuildUpdate"),
                true_skate_version=app.get("version"))


def bind(plan, *, context, gate1, facts, state, wda_revision, fingerprint, sources, amendment=None):
    bound = dict(kind=KIND, plan_sha256=plan["sha256"], device=DEVICE, device_env_key=DEVICE_ENV_KEY,
                 wda_port=WDA_PORT, wda_revision=wda_revision, context=context, gate1=gate1,
                 device_facts=facts, preflight=state, capture=schedule(),
                 curved_pipeline_fingerprint=fingerprint, runner_sources=sources)
    if amendment is not None:
        bound["amendment"] = amendment
    bound["sha256"] = digest(bound)
    return bound


def verify_bound(bound, plan):
    if bound.get("kind") != KIND or bound.get("plan_sha256") != plan["sha256"]:
        raise ValueError("bound context belongs to another plan")
    if bound.get("sha256") != digest({k: v for k, v in bound.items() if k != "sha256"}):
        raise ValueError("bound context digest changed")
    if bound["capture"] != schedule():
        raise ValueError("capture schedule differs from the bound one")
    return bound


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE/"plan.json.gz")
    parser.add_argument("--park", required=True, help="operator's verbatim park")
    parser.add_argument("--waypoint", required=True, help="operator's verbatim waypoint description")
    parser.add_argument("--settings-note", required=True, help="operator's verbatim settings note")
    parser.add_argument("--confirmed", required=True, help="when and how the operator confirmed the context")
    parser.add_argument("--gate1", type=Path, required=True, help="JSON: gate results and operator decision")
    parser.add_argument("--wda-revision", required=True)
    parser.add_argument("--amendment", type=Path, help="JSON: a post-hoc operator amendment to the stop rules")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE, override=False)
    from trueskate_ai.model1.curved.ledger import pipeline_fingerprint
    udid = os.environ[DEVICE_ENV_KEY]
    state = rig_state(WDA_PORT, udid)
    if not state["tunnel_running"] or state["recorder"].get("value") is not None:
        raise SystemExit("preflight failed: tunnel and idle recorder required")
    if state["foreground"].get("value", {}).get("bundleId") != TRUE_SKATE:
        raise SystemExit("True Skate is not frontmost")
    scratch = args.out.parent/(args.out.stem+"-devicectl")
    scratch.mkdir(parents=True, exist_ok=False)
    context = dict(park=args.park, waypoint=args.waypoint, settings_note=args.settings_note,
                   source="operator verbatim confirmation", confirmed=args.confirmed,
                   bound_utc=datetime.now(timezone.utc).isoformat())
    bound = bind(plan, context=context, gate1=json.loads(args.gate1.read_text()),
                 facts=device_facts(udid, scratch), state=state, wda_revision=args.wda_revision,
                 fingerprint=pipeline_fingerprint(args.wda_revision), sources=source_hashes(),
                 amendment=json.loads(args.amendment.read_text()) if args.amendment else None)
    with args.out.open("x") as handle:  # Never overwrite a binding.
        json.dump(bound, handle, indent=2)
        handle.write("\n")
    print("BOUND "+bound["sha256"], flush=True)


if __name__ == "__main__":
    main()
