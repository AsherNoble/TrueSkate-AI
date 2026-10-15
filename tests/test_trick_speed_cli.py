"""Trick-speed spot-check page, verdicts, summary, busy lock and offline refusals of the rig runner."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from trueskate_ai.model1.curved.capture import file_sha256
from trueskate_ai.model1.curved.contract import save_json
from trueskate_ai.model1.trick import live, review
from trueskate_ai.model1.trick.contract import Stroke

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO/"scripts/collection/trick_speed_batch.py"
POP = Stroke(((.505, .56), (.555, .8)), (0, 75), "pop")
FLICK = Stroke(((.537, .51), (.537, .51), (.57, .54), (.715, .56)), (0, 70, 130, 170), "flick")


def strokes_for(number):
    return [dict(POP.to_dict(), onset_s=.4, release_s=.475),
            dict(FLICK.to_dict(), onset_s=.85, release_s=1.02)][:1 + number % 2]


def recording(root, name, *, runs=(), admitted=(), rejected=None, error=None, lags=None):
    folder = root/"recordings"/name
    folder.mkdir(parents=True)
    save_json(folder/"planned.json", dict(runs=[dict(number=n, strokes=[]) for n in runs], skipped_runs=[]))
    events = [dict(kind="stroke", run=n, stroke=0, instrumented=False) for n in runs]
    save_json(folder/"execution.json", dict(events=events, error=error))
    for number in admitted:
        run = folder/f"run-{number:05d}"
        run.mkdir()
        (run/"clip.mp4").write_bytes(f"clip {name} {number}".encode())
        save_json(run/"meta.json", dict(recording=name, run=number, strokes=strokes_for(number), clock_rate=1.,
                                        frame_times_s=[i/60 for i in range(120)],
                                        clip_sha256=file_sha256(run/"clip.mp4")))
    if rejected is not None:
        save_json(folder/"admission.json", dict(accepted=True, admitted=list(admitted), rejected=rejected,
                                                calibration=dict(stroke_visible_lags_s=lags or {})))
    return folder


@pytest.fixture
def root(tmp_path):
    root = tmp_path/"run"
    recording(root, "m0001", runs=(1, 2, 3, 4), admitted=(1, 2, 3),
              rejected={"4": "no 120-frame window holds this run alone with five frames either side"},
              lags={"1:0": .02, "2:0": .04, "2:1": .06})
    recording(root, "m0002", runs=(5,), error="RuntimeError: schedule overrun")
    recording(root, "m0003", runs=(6, 7))  # fetched, not yet admitted
    return root


def page_clips(page: Path):
    text = page.read_text()
    start = text.index("const CLIPS = ") + len("const CLIPS = ")
    return json.loads(text[start:text.index(";\n", start)])


def test_review_page_keeps_its_seeded_pick(root):
    page = review.review_page(root, count=2)
    selection = review.load_selection(root)
    assert [c["code"] for c in selection["clips"]] == ["01", "02"] and selection["admitted"] == 3
    assert review.select(root, 2) == selection  # seeded
    clips = page_clips(page)
    assert [c["code"] for c in clips] == ["01", "02"]
    text = page.read_text()
    for row, clip in zip(selection["clips"], clips):
        assert f'src="../recordings/{row["recording"]}/run-{row["run"]:05d}/clip.mp4"' in text
        meta = json.loads((root/"recordings"/row["recording"]/f'run-{row["run"]:05d}'/"meta.json").read_text())
        assert len(clip["strokes"]) == len(meta["strokes"]) and len(clip["frame_times_s"]) == 120
        assert clip["strokes"][0]["times_s"] == [t/1000 for t in meta["strokes"][0]["times_ms"]]
    # A rebuild keeps the codes, whatever count is asked for.
    review.review_page(root, count=3)
    assert review.load_selection(root) == selection
    # A clip that changes after the pick is refused.
    row = selection["clips"][0]
    meta_path = root/"recordings"/row["recording"]/f'run-{row["run"]:05d}'/"meta.json"
    meta = json.loads(meta_path.read_text())
    save_json(meta_path, dict(meta, clip_sha256="0"*64), exclusive=False)
    with pytest.raises(ValueError, match="changed after it was picked"):
        review.review_page(root)


def test_verdicts_are_checked_before_writing_and_never_silently_replaced(root):
    review.review_page(root, count=2)
    for bad in ("", "hello", "01X", "03K", "01K 1F", "01K, oops"):
        with pytest.raises(ValueError):
            review.record_verdicts(root, bad)
    assert not (root/"review"/"verdicts.json").exists()
    assert review.record_verdicts(root, "01k, 02F") == 2
    recorded = json.loads((root/"review"/"verdicts.json").read_text())["verdicts"]
    assert {k: v["code"] for k, v in recorded.items()} == {"01": "K", "02": "F"}
    with pytest.raises(ValueError, match="already has a verdict"):
        review.record_verdicts(root, "02K")
    assert review.record_verdicts(root, "02U", replace=True) == 1
    with pytest.raises(ValueError, match="verdicts already recorded"):
        review.review_page(root, reselect=True)


def test_summary_counts_states_rejections_strokes_and_spot_check(root):
    review.review_page(root, count=3)
    review.record_verdicts(root, "01K 02F")
    summary = review.summarize(root)
    assert summary["recordings_by_state"] == {"admitted": 1, "recording failed": 1, "awaiting admission": 1}
    assert summary["totals"]["recordings"] == 3 and summary["totals"]["runs_planned"] == 7
    assert summary["totals"]["clips_admitted"] == 3 and summary["totals"]["strokes_sent"] == 7
    assert summary["admitted_share_of_sent_runs"] == 3/7
    assert list(summary["rejection_reasons"].values()) == [1]
    assert summary["stroke_families"] == {"pop": 3, "flick": 2}
    assert summary["touches_per_clip"] == {"1": 1, "2": 2}
    assert summary["spot_check"] == dict(picked=3, ok=1, fault=1, unsure=0, unjudged=1)
    assert summary["recordings"]["m0002"]["error"] == "RuntimeError: schedule overrun"
    assert summary["stroke_visible_lag_s"]["measured"] == 3 and summary["stroke_visible_lag_s"]["median"] == .04
    markdown = review.summary_markdown(summary)
    assert "Clips admitted: 3" in markdown and "1 OK, 1 fault" in markdown and "+40 ms" in markdown


def test_busy_lock_is_exclusive_and_only_releases_its_own_stamp(tmp_path):
    path = tmp_path/"RIG-BUSY"
    with live.busy_lock(path, owner="trick", out=tmp_path):
        stamp = json.loads(path.read_text())
        assert stamp["pid"] == os.getpid() and stamp["owner"] == "trick"
        with pytest.raises(RuntimeError, match="another recording holds the rig"):
            with live.busy_lock(path, owner="other", out=tmp_path):
                pass
        assert path.exists()
    assert not path.exists()
    with live.busy_lock(path, owner="trick", out=tmp_path):
        path.write_text(json.dumps(dict(owner="someone else", pid=1)))
    assert path.exists()  # another holder's stamp is never removed


def test_collect_refuses_bad_arguments_and_laptops_before_touching_anything(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no device, tunnel or Appium access expected")
    monkeypatch.setattr(live.subprocess, "check_output", forbidden)
    monkeypatch.setattr(live.subprocess, "run", forbidden)
    monkeypatch.setattr(live.urllib.request, "urlopen", forbidden)
    env = tmp_path/"rig.env"
    env.write_text("IPHONE_XR_UDID=\n")
    good = dict(minutes=2, env_file=env, ready_note="XR1 in Workshop", settings_note="flatground waypoint",
                park="The Workshop")
    root = tmp_path/"run"
    for change in (dict(minutes=0), dict(minutes=61), dict(minutes="2"), dict(ready_note=" "),
                   dict(settings_note=""), dict(park=""), dict(env_file=Path("rig.env"))):
        with pytest.raises(ValueError):
            live.collect(root, **dict(good, **change))
    with pytest.raises(ValueError, match="absolute run directory"):
        live.collect(Path("run"), **good)
    monkeypatch.setattr(live.socket, "gethostname", lambda: "ashers-laptop")
    with pytest.raises(ValueError, match="training-server"):
        live.collect(root, **good)
    assert not root.exists()


def test_cli_run_refuses_off_rig_before_taking_the_lock(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("trick_speed_batch", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(live.socket, "gethostname", lambda: "ashers-laptop")
    lock = tmp_path/"RIG-BUSY"
    args = argparse.Namespace(root=tmp_path/"run", minutes=1, env_file=tmp_path/"rig.env", operator_ready_note="ready",
                              settings_note="flatground", park="The Workshop", appium_port=4725, source_commit=None,
                              busy_lock=lock)
    with pytest.raises(ValueError, match="training-server"):
        cli.run(args)
    assert not lock.exists() and not (tmp_path/"run").exists()


def test_run_positions_and_fixed_context(tmp_path):
    root = tmp_path/"run"
    assert live.next_positions(root) == (1, 0)
    for name, runs, skipped in (("m0001", [0, 1, 3], [2]), ("m0002", [4, 5], [6, 7]), ("m0004", [], [])):
        save_json(root/"recordings"/name/"planned.json",
                  dict(runs=[dict(number=n) for n in runs], skipped_runs=skipped))
    (root/"recordings"/"m0005").mkdir()  # created, but stopped before its plan was written
    assert live.next_positions(root) == (6, 8)
    root.mkdir(exist_ok=True)
    context = live.fix_context(root, park="The Workshop", settings_note="flatground")
    assert live.fix_context(root, park="The Workshop", settings_note="flatground") == context
    with pytest.raises(ValueError, match="new --root"):
        live.fix_context(root, park="Inbound", settings_note="flatground")


def test_cli_help_status_and_summary(root):
    run = lambda *args: subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                                       check=True).stdout
    assert "run" in run("--help") and "verdicts" in run("--root", str(root), "verdicts", "--help")
    status = json.loads(run("--root", str(root), "status"))
    assert status["totals"]["recordings"] == 3 and list(status["failures"]) == ["m0002"]
    assert "Recordings: 3" in run("--root", str(root), "summary")
    assert json.loads((root/"summary.json").read_text())["totals"]["clips_admitted"] == 3
    assert (root/"summary.md").exists()
