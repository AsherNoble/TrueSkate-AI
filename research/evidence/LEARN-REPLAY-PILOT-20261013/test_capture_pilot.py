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


CALIBRATION_STOP = "ValueError: die-five calibration failed: middle"


def write_run(root, session, *, run=None, recordings=range(5), lost=(), rejected=(), error=None, extra=()):
    """A finished run: spend and settlement per recording; lost recordings stay unsettled."""
    out = root/"sessions"/(run or session)
    out.mkdir(parents=True)
    base = 50*capture_pilot.SESSIONS.index(session)
    spend, settled = [], []
    for r in recordings:
        (out/"recordings"/f"{session}-r{r}").mkdir(parents=True)
        spend += [dict(kind="pre_roll_reset", request_id="p")]+[dict(kind="control", request_id="c")]*3
        spend += [dict(kind="reset", request_id="reset-26")]*2
        for c in range(base+10*r+1, base+10*r+11):
            spend.append(dict(kind="sample", request_id=f"candidate-{c}"))
            if r not in lost:
                settled.append(dict(candidate_id=c, admitted=c not in rejected))
    spend += list(extra)
    (out/"spend.jsonl").write_text("".join(json.dumps(r)+"\n" for r in spend))
    (out/"settled.jsonl").write_text("".join(json.dumps(s)+"\n" for s in settled))
    (out/"run-summary.json").write_text(json.dumps(dict(error=error, run=out.name)))


def test_first_session_is_admissible(plan, tmp_path):
    assert capture_pilot.admissible(plan, tmp_path, "pilot-00") == {}


@pytest.mark.parametrize("setup,session,message", [
    (lambda r: write_run(r, "pilot-00"), "pilot-00", "already started"),
    (lambda r: None, "pilot-01", "pilot-00 has not completed"),
    (lambda r: write_run(r, "pilot-00", error="RuntimeError: x"), "pilot-01", "Gate 2"),
    (lambda r: write_run(r, "pilot-00", rejected={7}), "pilot-01", "Gate 2"),
    (lambda r: write_run(r, "pilot-00", recordings=range(2), lost={1}, error=CALIBRATION_STOP),
     "pilot-01", "pilot-00 has not completed"),
    (lambda r: write_run(r, "pilot-00", extra=[dict(kind="sample", request_id="candidate-999")]*510),
     "pilot-01", "budget"),
    (lambda r: write_run(r, "pilot-00", extra=[dict(kind="control", request_id="c")]*301), "pilot-01", "budget"),
])
def test_admission_refusals(plan, tmp_path, setup, session, message):
    setup(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        capture_pilot.admissible(plan, tmp_path, session)


def test_completed_sessions_allow_the_next_and_count_spend(plan, tmp_path):
    write_run(tmp_path, "pilot-00")
    assert capture_pilot.admissible(plan, tmp_path, "pilot-01") == {"sample": 50, "support": 30}


def test_calibration_losses_complete_a_session(plan, tmp_path):
    write_run(tmp_path, "pilot-00", lost={1, 3})
    state = capture_pilot.session_state(tmp_path, "pilot-00")
    assert state["lost"] == list(range(11, 21))+list(range(31, 41)) and state["admitted"] == 30
    assert capture_pilot.admissible(plan, tmp_path, "pilot-01") == {"sample": 50, "support": 30}


def test_a_calibration_only_stop_resumes_its_unattempted_recordings(plan, tmp_path):
    write_run(tmp_path, "pilot-00")
    write_run(tmp_path, "pilot-01", recordings=range(2), lost={1}, error=CALIBRATION_STOP)
    with pytest.raises(RuntimeError, match="already started"):
        capture_pilot.admissible(plan, tmp_path, "pilot-01")
    assert capture_pilot.admissible(plan, tmp_path, "pilot-01", resume=True) == {"sample": 70, "support": 42}
    state = capture_pilot.session_state(tmp_path, "pilot-01")
    assert state["attempted"] == ["pilot-01-r0", "pilot-01-r1"] and state["runs"] == 1
    write_run(tmp_path, "pilot-01", run="pilot-01+1", recordings=range(2, 5))
    state = capture_pilot.session_state(tmp_path, "pilot-01")
    assert len(state["attempted"]) == 5 and state["lost"] == list(range(61, 71)) and state["admitted"] == 40
    assert capture_pilot.total_lost(tmp_path) == 10
    with pytest.raises(RuntimeError, match="no unattempted"):
        capture_pilot.admissible(plan, tmp_path, "pilot-01", resume=True)
    assert capture_pilot.admissible(plan, tmp_path, "pilot-02") == {"sample": 100, "support": 60}


@pytest.mark.parametrize("error,rejected,message", [
    ("RuntimeError: menu/editor detected before touch", (), "Gate 2 is open in pilot-01"),
    (CALIBRATION_STOP, {55}, "Gate 2 is open in pilot-01"),
])
def test_only_a_clean_calibration_stop_resumes(plan, tmp_path, error, rejected, message):
    write_run(tmp_path, "pilot-00")
    write_run(tmp_path, "pilot-01", recordings=range(2), lost={1}, rejected=rejected, error=error)
    with pytest.raises(RuntimeError, match=message):
        capture_pilot.admissible(plan, tmp_path, "pilot-01", resume=True)


def test_runs_are_ordered_original_then_resumptions(tmp_path):
    for name in ("pilot-03+2", "pilot-03", "pilot-03+10", "pilot-03+1", "pilot-030"):
        (tmp_path/"sessions"/name).mkdir(parents=True)
    assert [p.name for p in capture_pilot.runs(tmp_path, "pilot-03")] == [
        "pilot-03", "pilot-03+1", "pilot-03+2", "pilot-03+10"]


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


def test_the_reviewed_cap_stop_closes_pilot_01_and_the_cap_restarts(plan, tmp_path):
    write_run(tmp_path, "pilot-00")
    write_run(tmp_path, "pilot-01", recordings=range(2), lost={1}, error=CALIBRATION_STOP)
    write_run(tmp_path, "pilot-01", run="pilot-01+1", recordings=range(2, 5), lost={2, 3, 4},
              error="RuntimeError: calibration losses exceed 30 candidates (Gate 2)")
    assert capture_pilot.total_lost(tmp_path) == 40 and capture_pilot.new_losses(tmp_path) == 0
    assert capture_pilot.admissible(plan, tmp_path, "pilot-02") == {"sample": 100, "support": 60}


def test_the_lifted_cap_lets_heavy_losses_continue_and_resumes_pilot_04(plan, tmp_path):
    for session in capture_pilot.SESSIONS[:4]:
        write_run(tmp_path, session, lost={0, 1, 2})
    write_run(tmp_path, "pilot-04", recordings=range(1), lost={0},
              error="RuntimeError: calibration losses exceed 30 candidates (Gate 2)")
    assert capture_pilot.new_losses(tmp_path) == 90
    assert capture_pilot.admissible(plan, tmp_path, "pilot-04", resume=True) == {"sample": 210, "support": 126}
    write_run(tmp_path, "pilot-04", run="pilot-04+1", recordings=range(1, 5), lost={1, 2, 3, 4})
    assert capture_pilot.admissible(plan, tmp_path, "pilot-05") == {"sample": 250, "support": 150}


def png(seed, size=(828, 1792)):
    import io
    import numpy as np
    from PIL import Image
    pixels = np.random.default_rng(seed).integers(0, 255, (size[1]//16, size[0]//16), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).resize(size, Image.NEAREST).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeClock:
    def __init__(self):
        self.t = 100.

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def screenshots(frames, clock, step=.45):
    def take():
        clock.t += step
        return frames(clock.t)
    return take


def test_stillness_passes_a_still_view_after_the_nav_bar_appears():
    import io
    import numpy as np
    from PIL import Image
    still = png(1)
    image = np.asarray(Image.open(io.BytesIO(still))).copy()
    image[-200:] = 0  # Idle navigation bar: cropped out of the comparison.
    buffer = io.BytesIO(); Image.fromarray(image).save(buffer, format="PNG")
    clock = FakeClock()
    diffs = capture_pilot.assert_still(screenshots(lambda t: still if t < 102 else buffer.getvalue(), clock),
                                       start=100., clock=clock, sleep=clock.sleep)
    assert len(diffs) >= 6 and max(diffs) == 0


def test_stillness_refuses_a_turning_view():
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="still moving"):
        capture_pilot.assert_still(screenshots(lambda t: png(int(t*10)), clock), start=100., clock=clock,
                                   sleep=clock.sleep)


def test_stillness_is_in_the_bound_schedule():
    rule = bind_context.schedule()["pre_roll_stillness"]
    assert rule["window_s"] == [1.0, 4.5] and rule["max_mean_abs_diff"] == 10.0

