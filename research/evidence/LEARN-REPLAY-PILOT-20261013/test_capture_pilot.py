"""Synthetic checks for the pilot binding and capture runner. No device, WDA or rig access."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bind_context  # noqa: E402
import capture_pilot  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return bind_context.load_plan(HERE/"plan.json.gz")


def bound_for(plan, **changes):
    return bind_context.bind(plan, context=dict(park="Synthetic Park", waypoint="w", settings_note="s",
                                                source="operator verbatim confirmation"),
                             gate1=dict(decision="go"), facts={}, state={}, wda_revision="rev",
                             fingerprint="fp", sources={}, **changes)


def test_sessions_cover_the_frozen_plan_within_budget(plan):
    ids, samples, support = [], 0, 0
    rows = {r["id"]: r for r in plan["candidates"]}
    for session in capture_pilot.SESSIONS:
        schedules = capture_pilot.recordings(plan, session)
        assert [s["recording_id"] for s in schedules] == [f"{session}-r{i}" for i in range(5)]
        for schedule in schedules:
            kinds = [c["kind"] for c in schedule["commands"]]
            assert kinds.count("sample") == 10 and kinds.count("control") == 3 and kinds.count("reset") == 2
            slots = [c["slot_s"] for c in schedule["commands"]]
            assert slots == sorted(slots) and len(set(slots)) == len(slots)
            for c in schedule["commands"]:
                if c["kind"] == "sample":
                    assert rows[c["candidate_id"]]["session"] == session
                    assert c["payload"] == rows[c["candidate_id"]]["payload"]
                    assert c["gesture"] == rows[c["candidate_id"]]["gesture"]
                    ids.append(c["candidate_id"])
            samples += 10
            support += capture_pilot.SUPPORT_PER_RECORDING
    assert ids == list(range(1, 601))
    assert samples == plan["candidate_dispatch_limit"] and support == plan["capture_support_limit"]


def test_preparation_lead_fits_after_the_longest_previous_touch(plan):
    from trueskate_ai.model1.curved.contract import MAX_DURATION_MS
    commands = capture_pilot.recordings(plan, "pilot-00")[0]["commands"]
    for previous, current in zip(commands, commands[1:]):
        busy = MAX_DURATION_MS/1000 if previous["kind"] == "sample" else .05
        assert current["slot_s"]-previous["slot_s"]-busy >= bind_context.PREPARE_LEAD_S


def write_session(root, session, *, error=None, admitted=50, kinds=None):
    out = root/"sessions"/session
    out.mkdir(parents=True)
    (out/"run-summary.json").write_text(json.dumps(dict(error=error, admitted=admitted)))
    rows = [dict(kind=k) for k, n in (kinds or dict(sample=50, control=15, reset=10, pre_roll_reset=5)).items()
            for _ in range(n)]
    (out/"spend.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))


def test_first_session_is_admissible(plan, tmp_path):
    assert capture_pilot.admissible(plan, tmp_path, "pilot-00") == {}


@pytest.mark.parametrize("setup,session,message", [
    (lambda r: write_session(r, "pilot-00"), "pilot-00", "already started"),
    (lambda r: None, "pilot-01", "pilot-00 has not completed"),
    (lambda r: write_session(r, "pilot-00", error="RuntimeError: x"), "pilot-01", "Gate 2"),
    (lambda r: write_session(r, "pilot-00", admitted=49), "pilot-01", "Gate 2"),
    (lambda r: write_session(r, "pilot-00", kinds=dict(sample=560)), "pilot-01", "budget"),
    (lambda r: write_session(r, "pilot-00", kinds=dict(support=331)), "pilot-01", "budget"),
])
def test_admission_refusals(plan, tmp_path, setup, session, message):
    setup(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        capture_pilot.admissible(plan, tmp_path, session)


def test_completed_sessions_allow_the_next_and_count_spend(plan, tmp_path):
    write_session(tmp_path, "pilot-00")
    assert capture_pilot.admissible(plan, tmp_path, "pilot-01") == {"sample": 50, "support": 30}


def test_bound_context_is_tamper_evident(plan):
    bound = bound_for(plan)
    assert bind_context.verify_bound(bound, plan) is bound
    changed = dict(bound, context=dict(bound["context"], park="Elsewhere"))
    with pytest.raises(ValueError, match="digest"):
        bind_context.verify_bound(changed, plan)
    other = dict(bound, plan_sha256="other")
    with pytest.raises(ValueError, match="another plan"):
        bind_context.verify_bound(other, plan)
    lead = json.loads(json.dumps(bound))
    lead["capture"]["prepare_lead_s"] = 1.75
    lead["sha256"] = bind_context.digest({k: v for k, v in lead.items() if k != "sha256"})
    with pytest.raises(ValueError, match="schedule"):
        bind_context.verify_bound(lead, plan)


def test_request_ids_name_every_touch(plan):
    commands = capture_pilot.recordings(plan, "pilot-03")[2]["commands"]
    names = [capture_pilot.request_id(c) for c in commands]
    assert len(set(names)) == len(names)
    assert "start" in names and "candidate-171" in names and "reset-26" in names


def test_pilot_ledger_extracts_clips_with_the_frozen_split(plan, tmp_path, monkeypatch):
    from trueskate_ai.collection.tap_timing_calibration import TwoAnchorTimingFit
    from trueskate_ai.model1.curved import capture
    session = next(r["session"] for r in plan["candidates"] if r["split"] == "test")
    source = tmp_path/"sessions"/session/"recordings"/f"{session}-r0"
    source.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x64:r=60",
                    "-frames:v", "600", "-c:v", "libx264", "-threads", "1", str(source/"original.mov")], check=True)
    samples = [c for c in capture_pilot.recordings(plan, session)[0]["commands"] if c["kind"] == "sample"][:2]
    planned = dict(commands=samples, seed=plan["seed"], session=session, device="iPhone_XR",
                   fingerprint="learn-replay-pilot-20261013:test", waypoint="w", settings_note="s")
    records = [dict(submitted_to_ios=dict(monotonic_s=t)) for t in (3., 7.)]
    pts = capture.probe_pts(source/"original.mov")
    monkeypatch.setattr(capture, "calibrate_recording", lambda *_: (
        planned, records, pts, TwoAnchorTimingFit(0, 1, 57), dict(accepted=True)))
    monkeypatch.setattr(capture, "is_editor_frame", lambda rgb: False)
    monkeypatch.setattr(capture, "is_menu_frame", lambda rgb, allow_idle_navigation=False: False)
    root = tmp_path/"sessions"/session
    ledger = capture_pilot.PilotLedger(root, plan, park="Synthetic Park")
    admitted = capture.extract_recording(source, ledger, revision="synthetic")
    assert len(admitted) == 2
    metas = [capture.verify_sample(path) for path in admitted]
    assert {(m["split"], m["phase"], m["session_id"], m["park"]) for m in metas} == {
        ("test", "learn-replay-pilot", session, "Synthetic Park")}
    assert [m["candidate_id"] for m in metas] == [c["candidate_id"] for c in samples]
    settled = [json.loads(line) for line in (root/"settled.jsonl").read_text().splitlines()]
    assert [s["admitted"] for s in settled] == [True, True]
