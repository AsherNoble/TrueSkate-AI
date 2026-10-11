"""Synthetic contract, budget, provenance, native-frame and resume regression tests."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from trueskate_ai.model1.curved import audit, capture
from trueskate_ai.model1.curved.contract import Gesture, SCHEMA, digest, select_parks, save_json
from trueskate_ai.model1.curved.sampling import sample_gesture
from trueskate_ai.model1.curved.ledger import Ledger
from trueskate_ai.model1.curved.decoder import decode, evaluate_path
from trueskate_ai.model1.curved.dataset import targets, CurvedDataset, read_frames


@pytest.fixture
def native(tmp_path):
    path = tmp_path/"native.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x64:r=60",
                    "-frames:v", "120", "-c:v", "libx264", "-threads", "1", str(path)], check=True)
    return path


def meta_for(gesture=None, **extra):
    gesture = gesture or Gesture(((.5, .4), (.6, .6)), (0, 600), "arc")
    times = (np.arange(120)/60).tolist()
    return dict(schema=SCHEMA, candidate_id=1, session_id="session-1", park="park-1", phase="production",
                split="train", fingerprint="pipeline", gesture=gesture.to_dict(), clock_rate=1.,
                onset_s=.3, release_s=.3+gesture.times_ms[-1]/1000,
                frame_times_s=times, source_frame_pts_s=times, source_frame_numbers=list(range(120)),
                source_recording_id="recording-1", calibration=dict(accepted=True),
                admission=dict(accepted=True, frame_count=120, menu_checked_frames=120, native_frame_selection=True,
                               menu_policy=capture.MENU_POLICY),
                **extra)


def sample_dir(root, native, identity=1, split="train", gesture=None, phase="production"):
    folder = root/f"sample-{identity}"
    folder.mkdir()
    shutil.copy2(native, folder/"clip.mp4")
    meta = meta_for(gesture)
    meta.update(candidate_id=identity, session_id=f"session-{identity}", split=split, phase=phase,
                source_recording_id=f"recording-{identity}", clip_sha256=capture.file_sha256(folder/"clip.mp4"))
    save_json(folder/"meta.json", meta)
    row = dict(id=identity, sample_dir=folder.name, park=meta["park"], session=meta["session_id"],
               recording=meta["source_recording_id"], split=split,
               meta_sha256=capture.file_sha256(folder/"meta.json"), media_sha256=meta["clip_sha256"])
    return folder, meta, row


def manifest_for(rows):
    manifest = dict(schema=SCHEMA, kind="curved-training", samples=rows, qualified=dict(fingerprint="pipeline"))
    manifest["sha256"] = digest(manifest)
    return manifest


def audit_fixture():
    times = (np.arange(120)/60).tolist()
    manifest = dict(schema=SCHEMA, round=1, samples=[dict(id=i, frame_times_s=times, onset_s=.3, release_s=.9)
                                                   for i in range(100)])
    manifest["sha256"] = digest(manifest)
    review = dict(round=1, manifest_sha256=manifest["sha256"], samples=[dict(
        id=i, rating="meet", systematic=False, reason="", onset_frame=18, release_frame=54) for i in range(100)])
    return manifest, review


def test_generated_commands_arc_error_and_verbatim_parks():
    counts, durations, pause = set(), [], False
    for seed in range(2000):
        g = sample_gesture(seed, "arc" if seed % 2 else "random_chain")
        assert Gesture.from_dict(json.loads(json.dumps(g.to_dict()))).to_dict() == g.to_dict()
        assert min(np.diff(g.times_ms)) >= 100
        assert 300 <= g.times_ms[-1] <= 1200
        counts.add(len(g.times_ms)-1)
        durations.append(g.times_ms[-1])
        pause |= any(a == b for a, b in zip(g.points, g.points[1:]))
        if g.ideal and g.ideal["kind"] == "arc":
            radius, theta, n = g.ideal["radius_pt"], g.ideal["angle_rad"], len(g.times_ms)-1
            assert radius*(1-math.cos(theta/(2*n))) <= 2+1e-9
            assert n == 1 or radius*(1-math.cos(theta/(2*(n-1)))) > 2
    assert counts == set(range(1, 11)) and pause
    assert 730 < np.mean(durations) < 770
    parks = json.loads((Path(__file__).parents[1]/"config/parks.json").read_text())["parks"]
    assert len(parks) == 24
    assert select_parks(parks) == ['Skatepark: Inbound', 'SLS 2016 Newark', 'SLS 2015: Super Crown',
                                   'SLS 2015: LosAngeles', 'Macba']


def test_budget_persists_after_unknown_dispatch_and_rounds_are_bounded(tmp_path):
    root = tmp_path/"run"
    ledger = Ledger.initialize(root, [f"p{i}" for i in range(6)])
    ledger.start_round("v1")
    session = ledger.sessions()[0]["id"]
    candidate = ledger.plan_candidate(session, "arc")
    ledger.dispatch(candidate["id"])
    ledger.close()
    ledger = Ledger(root)
    assert ledger.spent == 1
    with pytest.raises(ValueError, match="already dispatched"):
        ledger.dispatch(candidate["id"])
    ledger.seal_interrupted([candidate["id"]])
    assert ledger.spent == 1
    with pytest.raises(ValueError, match="qualified"):
        ledger.production_sessions("v1")
    ledger.fail_round("technical failure")
    with pytest.raises(ValueError, match="revised pipeline"):
        ledger.start_round("v1")
    # Technical failures need a revised pipeline but do not consume review rounds.
    for number in range(2, 5):
        assert ledger.start_round(f"v{number}") == number
        ledger.fail_round("technical failure")
    assert ledger.reviewed_rounds() == 0
    for number in range(5, 10):
        assert ledger.start_round(f"v{number}") == number
        ledger.db.execute("UPDATE rounds SET status='failed',review_json=? WHERE number=?",
                          (json.dumps(dict(report={}, score=dict(passed=False))), number))
        ledger.db.commit()
    assert ledger.reviewed_rounds() == 5
    with pytest.raises(ValueError, match="five reviewed audit"):
        ledger.start_round("v10")
    ledger.close()


def test_dispatch_ceiling_atomic_and_planning_does_not_spend(tmp_path):
    ledger = Ledger.initialize(tmp_path/"run", [f"p{i}" for i in range(6)])
    definition = ledger.definition
    definition["attempt_limit"] = 1  # Small synthetic budget tests the same transaction.
    ledger.set("definition", definition)
    ledger.start_round("v1")
    session = ledger.sessions()[0]["id"]
    a = ledger.plan_candidate(session, "arc")
    b = ledger.plan_candidate(session, "arc")
    assert ledger.spent == 0
    ledger.dispatch(a["id"])
    with pytest.raises(ValueError, match="ceiling"):
        ledger.dispatch(b["id"])
    assert ledger.spent == 1
    ledger.close()


def test_review_95_threshold_unclear_boundaries_and_systemic_veto():
    manifest, review = audit_fixture()
    for row in review["samples"][:5]:
        row.update(rating="unclear", reason="contact obscured")
    assert audit.score_review(manifest, review)["passed"]
    review["samples"][5]["onset_frame"] = 20
    assert not audit.score_review(manifest, review)["passed"]
    review["samples"][5]["onset_frame"] = 19
    assert audit.score_review(manifest, review)["passed"]
    review["samples"][6].update(systematic=True, reason="consistent offset")
    assert not audit.score_review(manifest, review)["passed"]
    review["samples"][0]["id"] = 99
    with pytest.raises(ValueError, match="unique"):
        audit.score_review(manifest, review)


def test_review_import_retries_after_files_written(tmp_path):
    ledger = Ledger.initialize(tmp_path/"run", [f"p{i}" for i in range(6)])
    ledger.start_round("v1")
    manifest, review = audit_fixture()
    folder = ledger.root/"fixture"
    folder.mkdir()
    (folder/"meta.json").write_text('{}')
    (folder/"clip.mp4").write_bytes(b"synthetic reviewer binding fixture")
    for row in manifest["samples"]:
        row.update(sample_dir="fixture", meta_sha256=capture.file_sha256(folder/"meta.json"),
                   media_sha256=capture.file_sha256(folder/"clip.mp4"))
    manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
    review["manifest_sha256"] = manifest["sha256"]
    save_json(ledger.root/"audit-1/manifest.json", manifest)
    ledger.db.execute("UPDATE rounds SET status='review',manifest_hash=?", (manifest["sha256"],))
    ledger.db.commit()
    save_json(ledger.root/"audit-1/review.json", review)  # Crash before DB commit.
    assert ledger.import_review(review)["passed"]
    assert ledger.import_review(review)["passed"]
    ledger.production_sessions("v1")
    assert len(ledger.sessions(phase="production")) == 50
    assert sorted(s["split"] for s in ledger.sessions(phase="production", park=ledger.definition["parks"][0])) == ["test"]+["train"]*8+["validation"]
    with pytest.raises(ValueError, match="exact qualified"):
        ledger.production_sessions("v2")
    ledger.close()


def test_crops_span_timeline_and_avoid_controls_and_gaps():
    pts = np.arange(600)/60
    starts = {capture.window_start(pts, 3., 3.6, contacts=[(3., 3.6)], own=0, seed=s) for s in range(300)}
    assert len(starts) > 60
    assert min(starts) <= 107 and max(starts) >= 174
    for s in starts:
        t = pts[s:s+120]
        assert len(t[t < 3.]) >= 5 and len(t[t > 3.6]) >= 5
    for seed in range(20):
        start = capture.window_start(pts, 3., 3.6, contacts=[(3., 3.6), (2., 2.2)], own=0, seed=seed)
        assert pts[start] > 2.2
    with pytest.raises(ValueError, match="no 120-frame"):
        capture.window_start(pts, 3., 3.6, contacts=[(3., 3.6), (2.99, 3.01)], own=0, seed=0)


def test_native_loader_and_contamination_tampering(tmp_path, native):
    folder, meta, row = sample_dir(tmp_path, native)
    manifest = manifest_for([row])
    save_json(tmp_path/"dataset.json", manifest)
    dataset = CurvedDataset(tmp_path/"dataset.json", "train")
    assert dataset[0]["video"].shape == (120, 3, 144, 64)
    assert dataset[0]["onset"] == 18
    capture.verify_sample(folder)
    (folder/".menu").touch()
    with pytest.raises(ValueError, match="contamination"):
        CurvedDataset(tmp_path/"dataset.json", "train")
    (folder/".menu").unlink()
    (folder/"meta.json").write_text('{}')
    with pytest.raises(ValueError, match="changed"):
        CurvedDataset(tmp_path/"dataset.json", "train")


def test_loader_rejects_session_leakage_and_padding(tmp_path, native):
    _, _, row = sample_dir(tmp_path, native)
    duplicate = dict(row, id=2, split="test", sample_dir="elsewhere")
    save_json(tmp_path/"dataset.json", manifest_for([row, duplicate]))
    with pytest.raises(ValueError, match="leaks"):
        CurvedDataset(tmp_path/"dataset.json", "train")
    short = tmp_path/"short.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(native), "-frames:v", "119", str(short)], check=True)
    with pytest.raises(ValueError, match="120"):
        read_frames(short)


def test_decoder_recovers_late_kink_and_scores_without_alignment():
    meta = meta_for(Gesture(((.5, .4), (.7, .6), (.5, .6)), (0, 300, 600), "random_chain"))
    meta.update(onset_s=1.1, release_s=1.7)
    t = np.asarray(meta["frame_times_s"])
    xy = Gesture.from_dict(meta["gesture"]).positions(t-1.1)
    onset, release = np.full(120, -100.), np.full(120, -100.)
    onset[66], release[102] = 0, 0
    pred = decode(xy, onset, release, t)
    assert pred["onset_frame"] == 66 and pred["segment_count"] == 2
    assert pred["times_ms"] == [0, 300, 600]
    assert evaluate_path(pred, meta)["passed"]
    shifted = dict(pred, onset_s=1.2, release_s=1.8, onset_frame=72, release_frame=108)
    assert not evaluate_path(shifted, meta)["passed"]
    line = xy.copy()
    line[:, 1] = .5
    for phase in range(10):
        noisy = xy+np.random.default_rng(phase).normal(0, .15, xy.shape)
        result = decode(noisy, onset, release, t)
        assert 1 <= result["segment_count"] <= 6 and min(np.diff(result["times_ms"])) >= 100
        assert not result["fit_met_tolerance"]


def test_model_all_frames_have_finite_gradients():
    from trueskate_ai.model1.curved.model import CurvedModel, loss_function
    torch.set_num_threads(1)
    model = CurvedModel(4)
    video = torch.rand(1, 120, 3, 16, 8, requires_grad=True)
    times = torch.arange(120)[None]/60
    meta = meta_for()
    batch = {k: torch.as_tensor(v)[None] for k, v in targets(meta).items()}
    pred = model(video, times)
    loss = loss_function(pred, batch)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(video.grad).all()
    assert (video.grad.abs().sum((2, 3, 4)) > 0).all()


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.meta = [meta_for(), meta_for()]
        self.frames = torch.randint(0, 255, (2, 120, 3, 16, 8), generator=torch.Generator().manual_seed(14), dtype=torch.uint8)
    def __len__(self):
        return 2
    def __getitem__(self, i):
        return dict(video=self.frames[i], index=i, **{k: torch.as_tensor(v) for k, v in targets(self.meta[i]).items()})


def test_epoch_resume_matches_uninterrupted_weights(tmp_path):
    from trueskate_ai.model1.curved.training import fit, Recipe
    torch.set_num_threads(1)
    data = SyntheticDataset()
    recipe = Recipe(epochs=2, batch_size=1, base_channels=4)
    identity = dict(dataset="synthetic", human="synthetic")
    fit(data, data, tmp_path/"full", recipe=recipe, identity=identity)
    partial = fit(data, data, tmp_path/"resumed", recipe=recipe, identity=identity, stop_after_epochs=1)
    assert partial["epoch"] == 1 and not partial["complete"]
    resumed = fit(data, data, tmp_path/"resumed", recipe=recipe, identity=identity, resume=True)
    assert resumed["complete"]
    a = torch.load(tmp_path/"full/epoch-002.pt", weights_only=False)
    b = torch.load(tmp_path/"resumed/epoch-002.pt", weights_only=False)
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    with pytest.raises(ValueError, match="changed"):
        fit(data, data, tmp_path/"resumed", recipe=Recipe(epochs=3, base_channels=4), identity=identity, resume=True)


def test_bundle_roundtrip_and_hash_rejection(tmp_path, native):
    from trueskate_ai.model1.curved.cloud import bundle, stage, quote
    _, _, row = sample_dir(tmp_path, native)
    save_json(tmp_path/"dataset.json", manifest_for([row]))
    human = dict(schema=SCHEMA, kind="frozen-human-qualitative", samples=[])
    human["sha256"] = digest(human)
    save_json(tmp_path/"human.json", human)
    bundle(tmp_path/"dataset.json", tmp_path/"human.json", tmp_path/"bundle")
    manifest, _ = stage(tmp_path/"bundle", tmp_path/"staged")
    assert CurvedDataset(manifest, "train")[0]["video"].shape[0] == 120
    assert .3 < quote()["benchmark_compute_usd"] < .5
    (tmp_path/"bundle/part-0000.tar").write_bytes(b"changed")
    with pytest.raises(ValueError, match="shard changed"):
        stage(tmp_path/"bundle", tmp_path/"changed")
    (tmp_path/"sample-1/.menu").touch()
    with pytest.raises(ValueError, match="contamination"):
        bundle(tmp_path/"dataset.json", tmp_path/"human.json", tmp_path/"contaminated-bundle")


def test_cloud_budget_rejects_before_importing_modal():
    path = Path(__file__).parents[1]/"scripts/model1/curved_modal.py"
    spec = importlib.util.spec_from_file_location("curved_modal_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="budget"):
        module.approved_seconds("train", 0, 3600)
    with pytest.raises(ValueError, match="15 minutes"):
        module.approved_seconds("benchmark", 100, 901)
    assert module.approved_seconds("benchmark", 2, 900) == 900


def test_modal_definition_builds_without_starting_remote_work(monkeypatch):
    import contextlib
    import modal
    from types import SimpleNamespace
    path = Path(__file__).parents[1]/"scripts/model1/curved_modal.py"
    spec = importlib.util.spec_from_file_location("curved_modal_definition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    @contextlib.contextmanager
    def blocked(*args, **kwargs):
        raise RuntimeError("remote work blocked by test")
        yield
    monkeypatch.setattr(modal.App, "run", blocked)
    with pytest.raises(RuntimeError, match="remote work blocked by test"):
        module.execute(SimpleNamespace(command="benchmark", approved_budget_usd=2,
                       max_seconds=900, run="synthetic", bundle_hash="a"*64, resume=False))


def test_extract_native_recording_and_reject_menu(tmp_path, monkeypatch):
    from trueskate_ai.collection.tap_timing_calibration import TwoAnchorTimingFit
    source = tmp_path/"recording"
    source.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x64:r=60",
                    "-frames:v", "600", "-c:v", "libx264", "-threads", "1", str(source/"original.mov")], check=True)
    gesture = Gesture(((.5, .4), (.6, .6)), (0, 600), "arc")
    planned = dict(commands=[dict(kind="sample", candidate_id=1, gesture=gesture.to_dict()),
                             dict(kind="sample", candidate_id=2, gesture=gesture.to_dict())],
                   seed=12, session="s", device="synthetic", fingerprint="p", waypoint=1, settings_note="synthetic")
    records = [dict(submitted_to_ios=dict(monotonic_s=t)) for t in (3., 7.)]
    pts = capture.probe_pts(source/"original.mov")
    monkeypatch.setattr(capture, "calibrate_recording", lambda *_: (
        planned, records, pts, TwoAnchorTimingFit(0, 1, 57), dict(accepted=True)))
    monkeypatch.setattr(capture, "is_editor_frame", lambda rgb: False)
    # Only the idle-navigation signature: admitted under the curved menu policy.
    monkeypatch.setattr(capture, "is_menu_frame", lambda rgb, allow_idle_navigation=False: not allow_idle_navigation)
    class EvidenceLedger:
        def __init__(self): self.rows = {}
        def session(self, _): return dict(id="s", park="synthetic", phase="audit", split=None)
        def settle(self, identity, *, sample_path=None, reason=None):
            self.rows[identity] = (sample_path, reason)
    ledger = EvidenceLedger()
    result = capture.extract_recording(source, ledger, revision="synthetic")
    assert len(result) == 2
    for folder in result:
        meta = capture.verify_sample(folder)
        assert len(meta["source_frame_numbers"]) == 120
        assert meta["source_recording_id"] == "recording"
    # A fresh extraction of the same synthetic source must reject contamination.
    second = tmp_path/"menu-recording"
    second.mkdir()
    shutil.copy2(source/"original.mov", second/"original.mov")
    monkeypatch.setattr(capture, "is_menu_frame", lambda rgb, allow_idle_navigation=False: True)
    assert capture.extract_recording(second, ledger, revision="synthetic") == []
    assert all(path is None and "menu" in reason for path, reason in ledger.rows.values())


def test_extract_preserves_irregular_native_intervals(tmp_path, monkeypatch):
    """A native 15 ms interval must survive encoding (live round 3 regression)."""
    from trueskate_ai.collection.tap_timing_calibration import TwoAnchorTimingFit
    source = tmp_path/"recording"
    source.mkdir()
    # 1/600 time base like XCTest; frame 200 arrives one tick early, so 15 ms then 16.67 ms.
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x64:r=60",
                    "-frames:v", "600", "-vf", "settb=1/600,setpts=N*10-gte(N\\,200)",
                    "-fps_mode", "passthrough", "-enc_time_base", "1:600", "-video_track_timescale", "600",
                    "-c:v", "libx264", "-threads", "1", str(source/"original.mov")], check=True)
    pts = capture.probe_pts(source/"original.mov")
    assert np.isclose(np.diff(pts)[199], .015)
    assert capture.source_time_base(source/"original.mov") == "1:600"
    gesture = Gesture(((.5, .4), (.6, .6)), (0, 600), "arc")
    planned = dict(commands=[dict(kind="sample", candidate_id=1, gesture=gesture.to_dict())],
                   seed=12, session="s", device="synthetic", fingerprint="p", waypoint=1, settings_note="synthetic")
    records = [dict(submitted_to_ios=dict(monotonic_s=3.))]
    monkeypatch.setattr(capture, "calibrate_recording", lambda *_: (
        planned, records, pts, TwoAnchorTimingFit(0, 1, 57), dict(accepted=True)))
    monkeypatch.setattr(capture, "is_editor_frame", lambda rgb: False)
    monkeypatch.setattr(capture, "is_menu_frame", lambda rgb, allow_idle_navigation=False: False)
    class EvidenceLedger:
        def __init__(self): self.rows = {}
        def session(self, _): return dict(id="s", park="synthetic", phase="audit", split=None)
        def settle(self, identity, *, sample_path=None, reason=None):
            self.rows[identity] = (sample_path, reason)
    ledger = EvidenceLedger()
    (folder,) = capture.extract_recording(source, ledger, revision="synthetic")
    meta = capture.verify_sample(folder)
    numbers = meta["source_frame_numbers"]
    assert numbers[0] < 200 <= numbers[-1]
    assert ledger.rows[1] == (folder, None)


def test_selection_uses_all_park_quotas_and_whole_sessions(tmp_path, monkeypatch):
    # Lightweight metadata-only fake pool. Video/hash checking has separate real-media tests.
    class Pool:
        root = tmp_path
        definition = dict(parks=[f"park-{i}" for i in range(5)], seed=20261011)
        def get(self, key): return dict(fingerprint="pipeline")
        def sessions(self, *, phase, park):
            return [dict(id=f"{park}-s{i}", park=park, split="train" if i < 8 else "validation" if i == 8 else "test")
                    for i in range(10)]
        def admitted(self, session): return self.records[session]
        def remaining(self, *args, **kwargs): return 0
    pool = Pool()
    pool.records = {}
    identity = 0
    for park in pool.definition["parks"]:
        for session in pool.sessions(phase="production", park=park):
            rows = []
            for _ in range(280):
                identity += 1
                folder = tmp_path/str(identity)
                folder.mkdir()
                save_json(folder/"meta.json", dict(candidate_id=identity, fingerprint="pipeline", phase="production",
                          session_id=session["id"], split=session["split"], park=park,
                          source_recording_id=session["id"]+"-recording", admission=dict(accepted=True)))
                rows.append(dict(id=identity, sample_path=str(folder)))
            pool.records[session["id"]] = rows
    monkeypatch.setattr(audit, "file_sha256", lambda path: digest(str(path)))
    result = audit.freeze_dataset(pool)
    assert len(result["samples"]) == 12500 and result["pool_count"] == 14000
    assert audit.freeze_dataset(pool) == result
    for park in pool.definition["parks"]:
        assert sum(r["park"] == park for r in result["samples"]) == 2500
    for split, count in (("train", 10000), ("validation", 1250), ("test", 1250)):
        assert sum(r["split"] == split for r in result["samples"]) == count
    session_splits = {}
    for row in result["samples"]:
        session_splits.setdefault(row["session"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in session_splits.values())
