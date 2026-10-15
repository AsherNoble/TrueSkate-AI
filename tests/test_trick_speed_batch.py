"""Synthetic stroke-contract, sampler, minute-plan, recorder and admission tests (trick-speed batch)."""
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from trueskate_ai.collection.tap_timing_calibration import TwoAnchorTimingFit
from trueskate_ai.collection.wda_action_timing import BOUNDARIES
from trueskate_ai.data.control_hitboxes import segment_is_safe
from trueskate_ai.model1.curved.capture import probe_pts
from trueskate_ai.model1.curved.contract import save_json
from trueskate_ai.model1.trick import admission, recording, schedule
from trueskate_ai.model1.trick.contract import FAMILIES, MAX_STROKES, MIN_PIECE_MS, Stroke
from trueskate_ai.model1.trick.sampling import KICKFLIP_RUNS, MAX_GAP_MS, _split, sample_run
from trueskate_ai.research.curved_audit import marker_payload, reset_payload

# The approved candidate_19 pop and flick (KICKFLIP-REPEATABILITY-20261009).
POP = Stroke(((.505, .56), (.555, .815)), (0, 75), "pop")
FLICK = Stroke(((.537, .51), (.57, .54), (.715, .56)), (0, 60, 100), "flick")


def test_kickflip_strokes_match_the_kickflip_runners_direct_payload():
    from trueskate_ai.research.kickflip_repeatability import direct_stroke
    assert POP.payload() == direct_stroke("pop", [[.505, .56], [.555, .815]], [.075])["payload"]
    assert FLICK.payload() == direct_stroke("flick", [[.537, .51], [.57, .54], [.715, .56]], [.06, .04])["payload"]
    held = Stroke(((.537, .51), (.537, .51), (.715, .56)), (0, 70, 130), "flick")
    assert held.payload()["gestures"][0]["waypoints"][:2] == [dict(x=222, y=456), dict(x=222, y=456, duration_ms=70)]
    catch = Stroke(((.5, .505), (.5, .505)), (0, 520), "catch")
    assert catch.payload() == direct_stroke("catch", [[.5, .505]] * 2, [.52])["payload"]


def test_spin_crossing_path_has_safe_endpoints():
    # The rejection below is about the path, not its ends.
    assert segment_is_safe((.2, .3), (.2, .3)) and segment_is_safe((.2, .55), (.2, .55))
    assert not segment_is_safe((.2, .3), (.2, .55))


@pytest.mark.parametrize("points, times, family, message", [
    (((.5, .5), (.6, .6), (.7, .6)), (0, 39, 100), "random", "at least 40 ms"),
    (((.5, .5), (.6, .6)), (0, 45), "random", "duration outside"),
    (((.5, .5), (.6, .6), (.7, .6)), (0, 300, 601), "random", "duration outside"),
    (((.05, .5), (.3, .5)), (0, 100), "random", "usable gesture area"),
    (((.5, .5), (.5, .9)), (0, 100), "random", "usable gesture area"),
    (((.2, .3), (.2, .55)), (0, 100), "random", "protected control"),
    (((.5, .5), (.55, .5)), (0, 200), "catch", "catch holds still"),
    (((.5, .5), (.5, .5)), (0, 200), "pop", "catch holds still"),
    (((.5, .5), (.5001, .5)), (0, 200), "pop", "catch holds still"),  # quantizes to no movement
    (((.5, .5), (.6, .6)), (0, 75.), "pop", "integer millisecond"),
    (((.5, .5), (.6, .6)), (0, np.int64(75)), "pop", "integer millisecond"),
    (((.5, .5), (.6, .6)), (5, 80), "pop", "integer millisecond"),
    (((.5, .5), (.6, .6)), (0, 75), "kick", "unknown stroke family"),
    (((.5, .5), (.6, .6)), (0, 40, 80), "pop", "1–4 pieces"),
    (tuple((.5 + .01 * i, .5) for i in range(6)), tuple(range(0, 241, 48)), "random", "1–4 pieces"),
])
def test_stroke_contract_rejections(points, times, family, message):
    with pytest.raises(ValueError, match=message):
        Stroke(points, times, family)


def test_stroke_round_trip_fingerprint_and_positions():
    value = json.loads(json.dumps(POP.to_dict()))
    assert Stroke.from_dict(value) == POP and value["command_points"] == POP.command_points().tolist()
    with pytest.raises(ValueError, match="fingerprint"):
        Stroke.from_dict(dict(value, points=[[.505, .56], [.556, .815]]))
    with pytest.raises(ValueError, match="schema"):
        Stroke.from_dict(dict(value, schema="model1_curved_v1"))
    start, end = POP.command_points()
    assert np.allclose(POP.positions(0.), start) and np.allclose(POP.positions(.075), end)
    assert np.allclose(POP.positions(.0375), (start + end) / 2)
    assert np.allclose(POP.positions(.075 * 1.01, rate=1.01), end)
    with pytest.raises(ValueError):
        POP.positions(0., rate=0.)


def test_split_is_exact_and_respects_the_piece_minimum():
    rng = np.random.default_rng(0)
    for count in range(1, 5):
        for total in (MIN_PIECE_MS * count, MIN_PIECE_MS * count + 1, 333, 600):
            pieces = _split(rng, total, count)
            assert len(pieces) == count and sum(pieces) == total and min(pieces) >= MIN_PIECE_MS


def test_runs_are_reproducible_feasible_and_mostly_fast():
    assert sample_run(11) == sample_run(11) and sample_run(11) != sample_run(12)
    order = ("push", "pop", "flick", "catch")
    shapes = {shape for shape, _ in KICKFLIP_RUNS}
    durations, families, kinds = [], set(), set()
    for number in range(2000):
        run = sample_run(number)
        strokes = [Stroke.from_dict(s) for s in run["strokes"]]
        assert run["number"] == number and 1 <= len(strokes) <= MAX_STROKES
        assert len(run["gaps_ms"]) == len(strokes) - 1 and all(0 <= g <= MAX_GAP_MS for g in run["gaps_ms"])
        names = tuple(s.family for s in strokes)
        if run["kind"] == "kickflip_shaped":
            assert names in shapes and list(names) == sorted(names, key=order.index)
        durations += [s.times_ms[-1] for s in strokes]
        families.update(names)
        kinds.add(run["kind"])
    assert families == set(FAMILIES) and kinds == {"kickflip_shaped", "mixed"}
    assert np.mean(np.asarray(durations) < 200) > .55


def test_minute_plan_packs_runs_inside_spans_in_order():
    plan = schedule.plan_minute(0)
    assert plan == schedule.plan_minute(0)
    assert [(c["kind"], c.get("role"), c["slot_s"]) for c in plan["commands"]] == [
        ("control", "start", 1.5), ("reset", None, 26.), ("control", "middle", 29.),
        ("reset", None, 53.5), ("control", "end", 57.)]
    assert all(c["payload"] == (marker_payload() if c["kind"] == "control" else reset_payload())
               for c in plan["commands"])
    assert plan["runs"]
    previous_end = 0.
    for run in plan["runs"]:
        begin = run["slot_s"] - schedule.GUARD_S
        end = begin + run["budget_s"]
        assert any(lo - 1e-3 <= begin and end <= hi + 1e-3 for lo, hi in schedule.SPANS)
        assert begin >= previous_end - 1e-3
        previous_end = end
        assert abs(run["budget_s"] - schedule.run_budget_s(run)) < 1e-3
        assert schedule.nominal_span_s(run) <= schedule.MAX_RUN_SPAN_S
    assert all(schedule.nominal_span_s(sample_run(n)) > schedule.MAX_RUN_SPAN_S for n in plan["skipped_runs"])


def test_run_numbers_continue_across_minutes_without_reuse():
    first, seen = 0, []
    for _ in range(4):
        plan = schedule.plan_minute(first)
        used = [r["number"] for r in plan["runs"]] + plan["skipped_runs"]
        assert sorted(used) == list(range(first, plan["next_run"]))
        seen += used
        first = plan["next_run"]
    assert len(seen) == len(set(seen))


def test_spans_leave_room_for_marker_returns_guards_and_settling():
    slots = dict(schedule.CONTROL_SLOTS)
    (first_lo, first_hi), (second_lo, second_hi) = schedule.SPANS
    worst_tap_call = .05 + 1.22  # 50 ms tap plus the worst measured W3C return (curved rounds 3–5)
    assert slots["start"] + worst_tap_call <= first_lo and slots["middle"] + worst_tap_call <= second_lo
    assert first_hi <= schedule.RESET_SLOTS[0] - schedule.GUARD_S
    assert second_hi <= schedule.RESET_SLOTS[1] - schedule.GUARD_S
    assert slots["middle"] - schedule.RESET_SLOTS[0] >= 3 and slots["end"] - schedule.RESET_SLOTS[1] >= 3
    assert slots["end"] - slots["start"] >= 55 and slots["end"] + worst_tap_call <= schedule.STOP_S


@dataclass
class FakeResult:
    mov_path: Path
    started_at_epoch_s: float


class FakeRecorder:
    def __init__(self, rig):
        self.rig, self.started, self.stopped = rig, 0, 0

    def start(self):
        if self.rig.start_error:
            raise self.rig.start_error
        self.started += 1
        self.rig.t += self.rig.start_s

    def stop_and_save(self, path):
        self.stopped += 1
        Path(path).write_bytes(b"synthetic")
        return FakeResult(Path(path), 1.7e9)


class FakeTiming:
    def __init__(self, rig):
        self.rig, self.active = rig, False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False
        return dict(schema_version=1, build_revision="rev", dropped_records=0, records=self.rig.records)


class Rig:
    """A fake monotonic clock that sleeps, guards and WDA calls advance."""

    def __init__(self, *, command_s=.5, direct_return_s=.3, guard_s=.1, start_s=.2, start_error=None,
                 direct_response=None):
        self.t = 100.
        self.records, self.commands, self.strokes = [], [], []
        self.command_s, self.direct_return_s, self.guard_s = command_s, direct_return_s, guard_s
        self.start_s, self.start_error = start_s, start_error
        self.direct_response = direct_response or {"value": {}}
        self.recorder, self.timing = FakeRecorder(self), FakeTiming(self)

    def clock(self):
        return self.t

    def epoch(self):
        return 1.7e9 + self.t

    def sleep(self, seconds):
        self.t += max(0., seconds)

    def guard(self):
        self.t += self.guard_s

    def perform(self, payload):
        stamps = {b: dict(monotonic_s=self.t + i/1000, epoch_s=1.7e9 + self.t + i/1000)
                  for i, b in enumerate(BOUNDARIES)}
        self.records.append(dict(sequence=len(self.records), outcome="success", ios_callback_result=True,
                                 session_id="session", **stamps))
        self.commands.append(payload)
        self.t += self.command_s
        return {"value": None}

    def perform_direct(self, payload):
        self.strokes.append(payload)
        self.t += sum(w.get("duration_ms", 0) for w in payload["gestures"][0]["waypoints"]) / 1000
        self.t += self.direct_return_s
        return self.direct_response


def run_minute(tmp_path, rig, plan):
    return recording.record_minute(
        out=tmp_path/"m0001", plan=plan, recorder=rig.recorder, timing=rig.timing, perform=rig.perform,
        perform_direct=rig.perform_direct, guard=rig.guard, revision="rev",
        metadata=dict(device="iPhone_XR", park="Workshop", settings_note="synthetic"),
        clock=rig.clock, sleep=rig.sleep, epoch=rig.epoch)


def test_record_minute_sends_strokes_direct_and_markers_instrumented(tmp_path):
    rig, plan = Rig(), schedule.plan_minute(0)
    events = run_minute(tmp_path, rig, plan)
    execution = json.loads((tmp_path/"m0001"/"execution.json").read_text())
    assert execution["error"] is None and execution["failures"] == []
    assert rig.strokes == [Stroke.from_dict(s).payload() for r in plan["runs"] for s in r["strokes"]]
    assert rig.commands == [c["payload"] for c in plan["commands"]]
    strokes = [e for e in events if e["kind"] == "stroke"]
    assert all(not e["instrumented"] and e["success"] for e in strokes)
    assert [(e["run"], e["stroke"]) for e in strokes] == [
        (r["number"], j) for r in plan["runs"] for j in range(len(r["strokes"]))]
    assert sum(e["instrumented"] for e in events) == len(rig.records) == 5
    assert all(e["lateness_s"] <= recording.LATE_S for e in events if e.get("lateness_s") is not None)
    assert rig.recorder.started == rig.recorder.stopped == 1 and (tmp_path/"m0001"/"original.mov").exists()
    planned = json.loads((tmp_path/"m0001"/"planned.json").read_text())
    assert planned["wda_revision"] == "rev" and planned["park"] == "Workshop"
    assert planned["runs"] == json.loads(json.dumps(plan["runs"]))
    assert (tmp_path/"m0001"/"wda-timing.json").exists() and not rig.timing.active


@pytest.mark.parametrize("rig_options, message, recorded", [
    (dict(command_s=30.), "overran the next slot", True),  # the start marker returns after the first run's slot
    (dict(guard_s=.6), "schedule overrun", True),          # the foreground check outlasts its lead
    (dict(start_s=1.5), "recorder start exceeded", True),
    (dict(direct_response={"value": {"error": "boom"}}), "stroke response reports an error", True),
    (dict(start_error=RuntimeError("Already recording")), "Already recording", False),
])
def test_failed_minute_keeps_evidence_and_stops_the_recorder(tmp_path, rig_options, message, recorded):
    rig = Rig(**rig_options)
    with pytest.raises(RuntimeError, match=message):
        run_minute(tmp_path, rig, schedule.plan_minute(0))
    execution = json.loads((tmp_path/"m0001"/"execution.json").read_text())
    assert message in execution["error"] and execution["failures"][0]["phase"] == "execution"
    # One recorder start at most; a started recording is always retrieved, and timing is switched off.
    assert rig.recorder.stopped == int(recorded) == rig.recorder.started
    assert not rig.timing.active and (tmp_path/"m0001"/"wda-timing.json").exists()


def fixed_run(number, slot_s, durations_ms=(100, 100, 100), gaps_ms=None):
    strokes = [Stroke(((.5, .5), (.6, .6)), (0, d), "random").to_dict() for d in durations_ms]
    run = dict(number=number, seed=[0, number], kind="mixed", strokes=strokes,
               gaps_ms=list(gaps_ms if gaps_ms is not None else [0] * (len(strokes) - 1)), redraws=0)
    return dict(run, slot_s=slot_s, budget_s=round(schedule.run_budget_s(run), 3))


def test_slow_returns_drop_a_runs_remaining_strokes(tmp_path):
    first = fixed_run(1, 4.4)
    second = fixed_run(2, round(4.4 + first["budget_s"], 3))
    plan = dict(schedule.plan_minute(0), runs=[first, second], skipped_runs=[])
    events = run_minute(tmp_path, Rig(direct_return_s=1.4), plan)
    skipped = [e for e in events if e.get("skipped")]
    assert [(e["run"], e["stroke"]) for e in skipped] == [(1, 1)] and "next slot" in skipped[0]["reason"]
    assert [(e["run"], e["stroke"]) for e in events if e["kind"] == "stroke" and not e.get("skipped")] == [
        (1, 0), (2, 0), (2, 1), (2, 2)]
    assert json.loads((tmp_path/"m0001"/"execution.json").read_text())["error"] is None


def test_tampered_stroke_is_refused_before_any_touch(tmp_path):
    rig, plan = Rig(), copy.deepcopy(schedule.plan_minute(0))
    plan["runs"][0]["strokes"][0]["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        run_minute(tmp_path, rig, plan)
    assert not (tmp_path/"m0001").exists() and rig.recorder.started == 0 and not rig.strokes


def test_window_start_keeps_buffers_and_excludes_other_contacts():
    pts = np.arange(600) / 60
    others = [(.5, .55), (5., 5.2)]
    start = admission.window_start(pts, 2., 2.5, others, seed=[0, 1])
    assert start == admission.window_start(pts, 2., 2.5, others, seed=[0, 1])
    assert pts[start+4] < 2. and pts[start+115] > 2.5  # five frames either side
    assert pts[start] > .55 and pts[start+119] < 5.
    with pytest.raises(ValueError, match="no 120-frame window"):
        admission.window_start(pts, 9.6, 9.7, [(8.5, 8.55)], seed=[0, 3])
    gappy = pts.copy()
    gappy[150:] += .1  # a 117 ms hole in the source; every window holding the run crosses it
    with pytest.raises(ValueError, match="no 120-frame window"):
        admission.window_start(gappy, 2.45, 2.61, [], seed=[0, 4])


PHONE_OFFSET_S, RENDER_S = 1000., .15  # phone clock = rig clock + offset; marker taps show 150 ms after submission


def phone_record(sequence, rig_s):
    phone = rig_s + PHONE_OFFSET_S
    stamps = {b: dict(monotonic_s=phone + d, epoch_s=1.7e9 + phone + d)
              for b, d in zip(BOUNDARIES, (.005, .01, .09, .1, .2, .21, .28, .29))}
    return dict(sequence=sequence, outcome="success", ios_callback_result=True, session_id="session", **stamps)


def test_calibrate_maps_markers_and_direct_strokes_onto_the_video_clock(tmp_path, monkeypatch):
    from trueskate_ai.collection.die_five_calibration import DieFiveOnset
    out = synthetic_recording(tmp_path)
    run = fixed_run(7, 5., (100, 120))
    instrumented = [dict(kind="control", role="start", rig=10.5), dict(kind="reset", role=None, rig=35.),
                    dict(kind="control", role="middle", rig=38.), dict(kind="control", role="end", rig=66.)]
    events = [dict(kind=e["kind"], role=e["role"], instrumented=True, call_start_monotonic_s=e["rig"],
                   call_end_monotonic_s=e["rig"] + .6) for e in instrumented]
    stroke = dict(kind="stroke", instrumented=False, run=7, stroke=0, call_start_monotonic_s=14.,
                  call_end_monotonic_s=14.45)
    skipped = dict(kind="stroke", run=7, stroke=1, skipped=True, reason="would overrun the next slot")
    events = events[:1] + [stroke, skipped] + events[1:]
    # The recording starts at rig 9.0 s; the phone reports epoch = 1.7e9 + its monotonic clock.
    execution = dict(events=events, error=None, video=dict(started_at_epoch_s=1.7e9 + 9. + PHONE_OFFSET_S))
    planned = dict(wda_revision="rev", runs=[run])
    save_json(out/"wda-timing.json", dict(schema_version=1, build_revision="rev", dropped_records=0,
                                          records=[phone_record(i, e["rig"]) for i, e in enumerate(instrumented)]))
    seen = []

    def detect(frames, times, *, command_s, **_):
        seen.append(command_s)
        return DieFiveOnset(point_frames=(0,) * 5, onset_frame=0, onset_s=command_s + RENDER_S, votes=5)

    monkeypatch.setattr(admission, "detect_die_five_onset", detect)
    pts, fit, evidence, contacts = admission.calibrate(out, planned, execution)
    assert len(pts) == 600 and evidence["accepted"] and abs(evidence["heldout_middle_error_s"]) < 1e-9
    assert np.allclose(seen, [10.5 + .1 - 9., 38. + .1 - 9., 66. + .1 - 9.])  # submission times, video clock
    assert abs(fit.rate - 1.) < 1e-9 and abs(evidence["rig_to_wda_offset_s"] - (PHONE_OFFSET_S + .005)) < 1e-9
    assert [c[3]["kind"] for c in contacts] == ["control", "stroke", "reset", "control", "control"]
    tap_begin, tap_end, tap_run, _ = contacts[0]
    assert tap_run is None and np.isclose(tap_begin, 10.5 + .1 - 9. + RENDER_S) and np.isclose(tap_end - tap_begin, .05)
    begin, end, number, _ = contacts[1]
    # Rig call start + median rig-to-WDA offset (request entry), through the marker fit; then
    # whatever shift the visible trail gives (none can be read reliably in a test pattern).
    timing = evidence["stroke_timing"]["7:0"]
    assert number == 7 and np.isclose(timing["estimated_onset_s"], 14. + .005 - 9. + RENDER_S)
    assert np.isclose(begin, timing["estimated_onset_s"] + timing["shift_s"]) and np.isclose(end - begin, .1)

    def reject_middle(frames, times, *, command_s, **_):
        onset = None if 28 < command_s < 31 else command_s + RENDER_S  # the middle marker is at ~29.1 s
        return DieFiveOnset(point_frames=(None,) * 5, onset_frame=None if onset is None else 0, onset_s=onset, votes=0)

    monkeypatch.setattr(admission, "detect_die_five_onset", reject_middle)
    with pytest.raises(ValueError, match="die-five calibration failed: middle"):
        admission.calibrate(out, planned, execution)


def test_first_brightening_needs_a_persistent_step_after_the_search_start():
    times = np.arange(40) / 60
    step = np.r_[np.zeros(20), np.full(20, 50.)]
    assert admission.first_brightening(step, times, .1) == times[20]
    assert admission.first_brightening(step, times, times[21]) is None  # the step precedes the search
    spike = np.zeros(40)
    spike[20] = 50.
    assert admission.first_brightening(spike, times, 0.) is None  # one bright frame does not persist
    assert admission.first_brightening(np.random.default_rng(0).normal(0, 1, 40), times, 0.) is None


def test_probe_geometry_matches_the_tap_detector_and_clips_at_edges():
    crop, core, ring = admission.probe_geometry(828, 1792, (.5, .5))
    assert core.sum() and ring.sum() and not (core & ring).any()
    assert crop[1].stop - crop[1].start <= 2 * 40 + 2  # outer ring radius 40 px at XR resolution
    edge = admission.probe_geometry(828, 1792, (0., 0.))
    assert edge is not None and edge[0][0].start == 0 and edge[0][1].start == 0


def test_path_samples_give_each_pieces_speed_in_points_per_ms():
    hold = Stroke(((.5, .5), (.5, .5), (.6, .5)), (0, 50, 100), "random").to_dict()
    ms, xy, speed = admission.path_samples(hold)
    assert ms[0] == 0 and ms[-1] == 96 and np.allclose(xy[0], [.5, .5])
    moved = np.hypot(*((np.asarray(hold["command_points"][2]) - hold["command_points"][1])*[414, 896]))
    assert speed[0] == 0 and np.isclose(speed[list(ms).index(56)], moved/50)  # quantized points, per ms


def test_trail_shift_corrects_the_glow_and_drops_misreads():
    rows = [(0, .05, 0.), (8, .04, 2.), (16, .041, 2.), (24, .3, 2.), (32, .04, .2)]
    shift, used = admission.trail_shift(rows)  # the slow probe is skipped, the 0.3 s misread dropped
    assert used == 3 and np.isclose(shift, .041 + admission.GLOW_PT/2./1000)
    assert admission.trail_shift(rows[:2]) is None


def trail_video(path, strokes, size=(208, 448), seconds=10.):
    """Grey 60 fps video; each (points, duration_s, visible_from_s) draws a white trail along its path."""
    import cv2
    width, height = size
    frames = []
    for i in range(int(seconds*60)):
        frame = np.full((height, width), 100, np.uint8)
        for points, duration, start in strokes:
            u = min(1., (i/60 - start)/duration)
            if u < 0:
                continue
            (x0, y0), (x1, y1) = points
            head = (x0 + u*(x1 - x0), y0 + u*(y1 - y0))
            cv2.line(frame, (round(x0*(width-1)), round(y0*(height-1))),
                     (round(head[0]*(width-1)), round(head[1]*(height-1))), 255, 10)
        frames.append(frame)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
                    "-r", "60", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1", str(path)],
                   input=b"".join(f.tobytes() for f in frames), check=True)


def test_calibrate_moves_strokes_onto_their_visible_trail(tmp_path, monkeypatch):
    from trueskate_ai.collection.die_five_calibration import DieFiveOnset
    out = tmp_path/"recordings"/"m0001"
    out.mkdir(parents=True)
    path = ((.2, .3), (.8, .7))
    shown = Stroke(path, (0, 300), "random").to_dict()
    unseen = Stroke(((.7, .2), (.85, .3)), (0, 200), "random").to_dict()
    runs = [dict(number=7, strokes=[shown]), dict(number=8, strokes=[unseen])]
    # Estimates: 14 + 0.005 - 9 + 0.15 = 5.155 s and 7.155 s. Run 7's trail shows 60 ms late;
    # run 8 draws nothing, so it takes the recording's median shift.
    trail_video(out/"original.mov", [(path, .3, 5.155 + .06)])
    marks = [dict(kind="control", role="start", rig=10.5), dict(kind="control", role="middle", rig=38.),
             dict(kind="control", role="end", rig=66.)]
    events = [dict(kind=m["kind"], role=m["role"], instrumented=True, call_start_monotonic_s=m["rig"],
                   call_end_monotonic_s=m["rig"] + .6) for m in marks]
    events[1:1] = [dict(kind="stroke", instrumented=False, run=n, stroke=0, call_start_monotonic_s=rig,
                        call_end_monotonic_s=rig + .45) for n, rig in ((7, 14.), (8, 16.))]
    execution = dict(events=events, error=None, video=dict(started_at_epoch_s=1.7e9 + 9. + PHONE_OFFSET_S))
    save_json(out/"wda-timing.json", dict(schema_version=1, build_revision="rev", dropped_records=0,
                                          records=[phone_record(i, m["rig"]) for i, m in enumerate(marks)]))
    monkeypatch.setattr(admission, "detect_die_five_onset", lambda frames, times, *, command_s, **_: DieFiveOnset(
        point_frames=(0,) * 5, onset_frame=0, onset_s=command_s + RENDER_S, votes=5))
    _, _, evidence, contacts = admission.calibrate(out, dict(wda_revision="rev", runs=runs), execution)
    seen, unread = evidence["stroke_timing"]["7:0"], evidence["stroke_timing"]["8:0"]
    assert seen["source"] == "trail" and seen["probes"] >= 10 and abs(seen["shift_s"] - .06) < 1/60
    assert unread["source"] == "recording" and unread["shift_s"] == seen["shift_s"] and unread["probes"] == 0
    assert np.isclose(contacts[1][0], 5.155 + seen["shift_s"]) and np.isclose(contacts[2][0], 7.155 + seen["shift_s"])
    assert abs(evidence["stroke_visible_onsets_s"]["7:0"] - (5.155 + .06)) < 1/60 + 1e-6
    assert evidence["trail_timing"]["measured"] == 1 and evidence["trail_timing"]["strokes"] == 2


def synthetic_recording(root, name="m0001"):
    out = root/"recordings"/name
    out.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x64:r=60", "-frames:v", "600",
                    "-c:v", "libx264", "-threads", "1", str(out/"original.mov")], check=True)
    return out


def contact(run, index, begin):
    event = dict(kind="stroke", instrumented=False, run=run["number"], stroke=index,
                 call_start_monotonic_s=begin, call_end_monotonic_s=begin + .4)
    return (begin, begin + run["strokes"][index]["times_ms"][-1] / 1000, run["number"], event)


def tap(begin):
    return (begin, begin + .05, None, dict(kind="control", instrumented=True))


def synthetic_admission(root, monkeypatch, name="m0001", error=None):
    """Runs 1 and 2 fit; run 3 shares every possible window with a tap; run 4 was never sent."""
    out = synthetic_recording(root, name)
    runs = [fixed_run(1, 2., (100, 100)), fixed_run(2, 5., (200,)), fixed_run(3, 9.6, (100,)),
            fixed_run(4, 12., (100,))]
    save_json(out/"planned.json", dict(device="iPhone_XR", park="Workshop", settings_note="synthetic",
                                       wda_revision="rev", commands=[], runs=runs, skipped_runs=[]))
    save_json(out/"execution.json", dict(events=[], error=error))
    contacts = [tap(.5), contact(runs[0], 0, 2.), contact(runs[0], 1, 2.4), contact(runs[1], 0, 5.),
                tap(8.5), contact(runs[2], 0, 9.6)]
    pts = probe_pts(out/"original.mov")
    fit = TwoAnchorTimingFit(intercept_s=0., rate=1., anchor_span_s=57.)
    monkeypatch.setattr(admission, "calibrate", lambda *_: (pts, fit, dict(accepted=True), contacts))
    return out, pts, contacts


def test_admission_cuts_one_native_clip_per_run_and_explains_rejections(tmp_path, monkeypatch):
    out, pts, contacts = synthetic_admission(tmp_path, monkeypatch)
    result = admission.admit_recording(out)
    assert result["admitted"] == [1, 2]
    assert result["rejected"]["3"].startswith("no 120-frame window") and result["rejected"]["4"] == "not sent"
    assert not (out/"run-00003").exists() and not (out/"run-00004").exists()
    for number in (1, 2):
        folder = out/f"run-{number:05d}"
        meta = json.loads((folder/"meta.json").read_text())
        clip = probe_pts(folder/"clip.mp4", recording=False)
        start = meta["source_frame_numbers"][0]
        assert meta["source_frame_numbers"] == list(range(start, start + 120)) and len(clip) == 120
        assert np.allclose(meta["source_frame_pts_s"], pts[start:start+120])
        assert np.allclose(clip - clip[0], meta["frame_times_s"], atol=.001)
        times = meta["frame_times_s"]
        assert all(times[4] < s["onset_s"] < s["release_s"] < times[-5] for s in meta["strokes"])
        # No trail evidence here, so labels keep the rig-clock estimate.
        assert all(s["retimed_by_s"] == 0 and np.isclose(s["estimated_onset_s"], s["onset_s"]) for s in meta["strokes"])
        assert all(s["timing_source"] == admission.TIMING_SOURCE and abs(s["call_s"] - .4) < 1e-9
                   for s in meta["strokes"])
        assert Stroke.from_dict({k: v for k, v in meta["strokes"][0].items()
                                 if k not in ("onset_s", "release_s", "timing_source", "call_s")})
        first, last = meta["source_frame_pts_s"][0], meta["source_frame_pts_s"][-1]
        assert not any(b <= last and e >= first for b, e, run, _ in contacts if run != number)
        assert meta["sent_strokes"] == meta["planned_strokes"] == len(meta["strokes"])
        assert meta["park"] == "Workshop" and meta["admission"]["accepted"] is True


def test_admission_reruns_after_a_crashed_encode_but_never_redoes_a_settled_run(tmp_path, monkeypatch):
    out, _, _ = synthetic_admission(tmp_path, monkeypatch)
    real = admission.encode_clips

    def crash(original, jobs):
        jobs[0][1].write_bytes(b"partial")  # a half-written clip from the crashed attempt
        raise RuntimeError("ffmpeg crashed")

    monkeypatch.setattr(admission, "encode_clips", crash)
    assert admission.admit_all(tmp_path, jobs=1)["m0001"]["accepted"] is False
    (out/"admission.json").rename(out/"admission-failed-1.json")  # the documented way to rerun
    monkeypatch.setattr(admission, "encode_clips", real)
    assert admission.admit_all(tmp_path, jobs=1)["m0001"]["admitted"] == [1, 2]
    assert len(probe_pts(out/"run-00001"/"clip.mp4", recording=False)) == 120
    (out/"admission.json").rename(out/"admission-2.json")
    rerun = admission.admit_all(tmp_path, jobs=1)["m0001"]
    assert rerun["accepted"] is False and "already settled" in rerun["error"]
    assert json.loads((out/"run-00001"/"meta.json").read_text())["run"] == 1


def test_admit_all_inline_admits_each_recording_once(tmp_path, monkeypatch):
    synthetic_admission(tmp_path, monkeypatch)
    synthetic_recording(tmp_path, "m0002")
    failed = tmp_path/"recordings"/"m0002"
    save_json(failed/"planned.json", dict(runs=[]))
    save_json(failed/"execution.json", dict(events=[], error="RuntimeError: schedule overrun"))
    results = admission.admit_all(tmp_path, jobs=1)
    assert results["m0001"]["admitted"] == [1, 2]
    assert results["m0002"]["accepted"] is False and "schedule overrun" in results["m0002"]["error"]
    assert json.loads((failed/"admission.json").read_text()) == results["m0002"]
    assert admission.admit_all(tmp_path, jobs=1) == {}
