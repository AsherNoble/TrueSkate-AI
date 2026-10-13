"""Synthetic checks for the game cohort runner. No device, WDA or rig access."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_game  # noqa: E402
from run_helper import verify_plan  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return verify_plan(json.loads((HERE/"frozen-plan.json").read_text()))


def test_commands_follow_the_frozen_schedule(plan):
    paths = {p["path_id"]: p for p in plan["paths"]}
    ids = []
    for recording in plan["game_recordings"]:
        out = run_game.commands(plan, recording)
        assert [c["slot_s"] for c in out] == [c["slot_s"] for c in recording["commands"]]
        assert all(b > a for a, b in zip([c["slot_s"] for c in out], [c["slot_s"] for c in out][1:]))
        assert [c["kind"] for c in out] == [("sample" if c["kind"] == "scientific" else c["kind"])
                                           for c in recording["commands"]]
        for spec, frozen in zip(out, recording["commands"]):
            if frozen["kind"] == "scientific":
                assert spec["gesture"] == paths[frozen["path_id"]]["gesture"]
                assert spec["payload"] == paths[frozen["path_id"]]["payload"]
                ids.append(spec["candidate_id"])
            else:
                assert spec == frozen
    assert ids == list(range(1, 21))


def test_commands_refuse_a_changed_payload(plan):
    changed = json.loads(json.dumps(plan))
    recording = changed["game_recordings"][0]
    used = next(c["path_id"] for c in recording["commands"] if c["kind"] == "scientific")
    path = next(p for p in changed["paths"] if p["path_id"] == used)
    path["payload"]["actions"][0]["actions"][0]["x"] += 1
    with pytest.raises(ValueError, match="frozen payload changed"):
        run_game.commands(changed, recording)


def test_research_ledger_extracts_clips_review_py_serves(plan, tmp_path, monkeypatch):
    from trueskate_ai.collection.tap_timing_calibration import TwoAnchorTimingFit
    from trueskate_ai.model1.curved import capture, review
    from trueskate_ai.model1.curved.contract import save_json
    root = tmp_path/"cohort"
    source = root/"recordings"/"game-01"
    source.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x64:r=60",
                    "-frames:v", "600", "-c:v", "libx264", "-threads", "1", str(source/"original.mov")], check=True)
    samples = [c for c in run_game.commands(plan, plan["game_recordings"][0]) if c["kind"] == "sample"][:2]
    planned = dict(commands=samples, seed=plan["seed"], session="game-01", device=plan["device"],
                   fingerprint="physical-transfer-20261013:test", waypoint="2", settings_note="synthetic")
    records = [dict(submitted_to_ios=dict(monotonic_s=t)) for t in (3., 7.)]
    pts = capture.probe_pts(source/"original.mov")
    monkeypatch.setattr(capture, "calibrate_recording", lambda *_: (
        planned, records, pts, TwoAnchorTimingFit(0, 1, 57), dict(accepted=True)))
    monkeypatch.setattr(capture, "is_editor_frame", lambda rgb: False)
    monkeypatch.setattr(capture, "is_menu_frame", lambda rgb, allow_idle_navigation=False: False)
    ledger = run_game.ResearchLedger(root, park="Synthetic Park", plan_sha256=plan["sha256"])
    admitted = capture.extract_recording(source, ledger, revision="synthetic")
    assert len(admitted) == 2
    settled = [json.loads(line) for line in (root/"settled.jsonl").read_text().splitlines()]
    assert [(s["candidate_id"], s["admitted"]) for s in settled] == [(1, True), (2, True)]
    manifest = run_game.review_manifest(root, admitted, fingerprint="physical-transfer-20261013:test")
    save_json(root/f"audit-{run_game.REVIEW_ROUND}"/"manifest.json", manifest)
    loaded, paths, context = review.verified_samples(root.resolve(), run_game.REVIEW_ROUND)
    assert [s["id"] for s in loaded["samples"]] == [1, 2]
    assert {s["park"] for s in loaded["samples"]} == {"Synthetic Park"}
    assert context[1]["session"] == "game-01"


def game_cohort(tmp_path, *, admitted=range(1, 21), error=None):
    import analyze_game
    root = tmp_path/"cohort"
    (root/f"audit-{analyze_game.REVIEW_ROUND}").mkdir(parents=True)
    (root/"run-summary.json").write_text(json.dumps(dict(error=error)))
    (root/"settled.jsonl").write_text("".join(json.dumps(dict(candidate_id=i, admitted=i in admitted))+"\n"
                                              for i in range(1, 21)))
    times = [n/60 for n in range(120)]
    manifest = dict(sha256="m", samples=[dict(id=i, frame_times_s=times, onset_s=times[30]+.004)
                                         for i in range(1, 21)])
    (root/f"audit-{analyze_game.REVIEW_ROUND}"/"manifest.json").write_text(json.dumps(manifest))
    return root


def review_rows(onsets=None, ratings=None):
    import analyze_game
    return dict(manifest_sha256="m", round=analyze_game.REVIEW_ROUND, samples=[
        dict(id=i, rating=(ratings or {}).get(i, "meet"), systematic=False, reason="",
             onset_frame=(onsets or {}).get(i, 33)) for i in range(1, 21)])


def test_game_gate_passes_with_offsets_in_two_adjacent_frames(plan, tmp_path):
    import analyze_game
    result = analyze_game.gate(plan, game_cohort(tmp_path), review_rows(onsets={3: 32}))
    assert result["passed"], result["failures"]
    assert result["offset_range_frames"] == 1 and result["offset_counts"] == {"1": 1, "2": 19}


@pytest.mark.parametrize("kwargs,review,message", [
    (dict(), dict(onsets={3: 31}), "offset range 2"),
    (dict(), dict(ratings={1: "unclear", 8: "fail"}), "unequal_600: 3/5"),
    (dict(admitted=range(1, 20)), dict(), "19/20 admitted"),
    (dict(error="RuntimeError: x"), dict(), "technical admission"),
])
def test_game_gate_failures(plan, tmp_path, kwargs, review, message):
    import analyze_game
    result = analyze_game.gate(plan, game_cohort(tmp_path, **kwargs), review_rows(**review))
    assert not result["passed"] and any(message in f for f in result["failures"])


def test_game_gate_refuses_a_review_of_another_manifest(plan, tmp_path):
    import analyze_game
    review = review_rows()
    review["manifest_sha256"] = "other"
    with pytest.raises(ValueError, match="different manifest"):
        analyze_game.gate(plan, game_cohort(tmp_path), review)


def test_preparation_skips_only_the_first_marker(plan):
    calls = []
    prepare = run_game.preparation(lambda: calls.append(1))
    schedule = run_game.commands(plan, plan["game_recordings"][0])
    for spec in schedule:
        prepare(spec)
    assert len(calls) == len(schedule)-1
    assert schedule[0]["kind"] == "control" and schedule[0]["role"] == "start"
