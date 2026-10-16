"""Trick Model 1 MVP v0: data contract, split, loader, decoder limits and a two-epoch smoke test."""
import contextlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from trueskate_ai.model1.curved.capture import file_sha256
from trueskate_ai.model1.curved.contract import save_json
from trueskate_ai.model1.trick.contract import SCHEMA, Gesture
from trueskate_ai.model1.trick import dataset as data
from trueskate_ai.model1.trick.decoder import decode, fit_segments, score_clip, gesture_spans

POP = ((.50, .57), (.53, .77))
FLICK = ((.40, .45), (.52, .43), (.60, .30))
PUSH = ((.45, .60), (.70, .62))


def row(points, times_ms, family, onset):
    gesture = Gesture(tuple(points), tuple(times_ms), family)
    return dict(gesture.to_dict(), onset_s=onset, release_s=onset + times_ms[-1]/1000)


def meta_for(*gestures, recording="m0001", run=1):
    gestures = gestures or (row(POP, (0, 75), "pop", .505), row(FLICK, (0, 60, 100), "flick", .90))
    times = (np.arange(120)/60).tolist()
    return dict(schema=SCHEMA, recording=recording, run=run, device="iPhone_XR", park="The Workshop", clock_rate=1.,
                frame_times_s=times, source_frame_pts_s=times, source_frame_numbers=list(range(120)),
                strokes=list(gestures), admission=dict(accepted=True, frame_count=120))


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    path = tmp_path_factory.mktemp("native")/"native.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=32x64:r=60",
                    "-frames:v", "120", "-c:v", "libx264", "-threads", "1", str(path)], check=True)
    return path


def clip(data_root, native, root="trial", recording="m0001", run=1, meta=None):
    folder = data_root/root/"recordings"/recording/f"run-{run:05d}"
    folder.mkdir(parents=True)
    shutil.copy2(native, folder/"clip.mp4")
    meta = dict(meta or meta_for(), recording=recording, run=run, clip_sha256=file_sha256(folder/"clip.mp4"))
    save_json(folder/"meta.json", meta)
    return folder


def test_targets_mark_each_gesture_and_its_path():
    meta = meta_for()
    label = data.targets(meta)
    assert label["count"] == 2 and label["contact"].sum() == (35 - 30 + 1) + (60 - 54 + 1)
    assert int(np.argmax(label["down"][:45])) == 30 and int(np.argmax(label["up"][:45])) == 35
    assert int(np.argmax(label["down"][45:])) + 45 == 54 and int(np.argmax(label["up"][45:])) + 45 == 60
    np.testing.assert_allclose(label["xy"][30], Gesture(POP, (0, 75), "pop").command_points()[0], atol=1e-6)
    np.testing.assert_allclose(label["xy"][60], Gesture(FLICK, (0, 60, 100), "flick").command_points()[-1], atol=1e-6)
    assert not label["mask"][36:54].any() and label["xy"][40].tolist() == [0, 0]


@pytest.mark.parametrize("change, message", [
    (lambda m: m["strokes"][1].update(onset_s=.55, release_s=.65), "overlap"),
    (lambda m: m["strokes"][0].update(onset_s=.05, release_s=.125), "buffers"),
    (lambda m: m["strokes"][0].update(release_s=.61), "disagree"),
    (lambda m: m.update(schema="model1_curved_v1"), "admitted"),
    (lambda m: m.update(strokes=m["strokes"]*3), "1–4"),
    (lambda m: m["source_frame_numbers"].__setitem__(5, 99), "consecutive"),
])
def test_contract_rejects_bad_clips(change, message):
    meta = meta_for()
    change(meta)
    with pytest.raises(ValueError, match=message):
        data.check_clip(meta)


def test_split_keeps_copies_of_a_seeded_run_together():
    rows = [dict(root=root, recording=f"m{minute:04d}", run=10*minute + k)
            for root in ("trial", "trial-xr2") for minute in range(1, 8) for k in range(2)]
    rows.append(dict(root="batch-xr1", recording="m0006", run=500))  # its own plan: its own minute rule
    rows.append(dict(root="batch-xr2", recording="m0002", run=60))  # repeats trial m0006's run 60
    data.assign_splits(rows, ["trial", "trial-xr2", "batch-xr1", "batch-xr2"])
    split = {(r["root"], r["recording"], r["run"]): r["split"] for r in rows}
    assert {split[(root, "m0001", 10)] for root in ("trial", "trial-xr2")} == {"train"}
    assert split[("trial", "m0006", 60)] == split[("trial-xr2", "m0006", 61)] == "validation"
    assert split[("trial", "m0007", 70)] == "test" and split[("batch-xr1", "m0006", 500)] == "validation"
    assert split[("batch-xr2", "m0002", 60)] == "validation"  # follows the group, not its own minute
    assert data.split_for("m0014") == "test"
    with pytest.raises(ValueError, match="minute"):
        data.split_for("minute-1")


def test_freeze_pack_and_loader_round_trip(tmp_path, native):
    root = tmp_path/"EVIDENCE"
    for name in ("trial", "trial-xr2"):
        clip(root, native, name, "m0001", 1)
        clip(root, native, name, "m0006", 2, meta_for(row(PUSH, (0, 300), "push", .4)))
    marked = clip(root, native, "trial", "m0002", 3)
    (marked/".menu").touch()
    manifest = data.freeze(root, ["trial", "trial-xr2"], tmp_path/"manifest.json")
    assert manifest["counts"]["train"] == dict(clips=2, strokes=4) and len(manifest["excluded"]) == 1
    assert manifest["counts"]["validation"]["clips"] == 2
    index = data.pack(tmp_path/"manifest.json", root, tmp_path/"bundle", jobs=2)
    train = data.TrickDataset(tmp_path/"bundle", "train")
    item = train[0]
    assert item["video"].shape == (120, 3, 144, 64) and item["video"].dtype == torch.uint8
    assert item["contact"].shape == (120,) and int(item["count"]) == 2
    np.testing.assert_array_equal(train.frames[0], data.read_frames(native))
    frames = tmp_path/"bundle"/index["files"]["trial/m0001/00001"]["frames"]
    frames.write_bytes(frames.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="changed"):
        data.TrickDataset(tmp_path/"bundle", "train")
    with pytest.raises(FileExistsError):
        data.pack(tmp_path/"manifest.json", root, tmp_path/"bundle")
    (root/"trial/recordings/m0001/run-00001/.contaminated").touch()
    with pytest.raises(ValueError, match="contamination"):
        data.pack(tmp_path/"manifest.json", root, tmp_path/"bundle-2")


def test_manifest_refuses_tampering_and_leaks(tmp_path, native):
    root = tmp_path/"EVIDENCE"
    clip(root, native, "trial", "m0001", 1)
    clip(root, native, "trial", "m0007", 2)
    manifest = data.freeze(root, ["trial"], tmp_path/"manifest.json")
    tampered = json.loads(json.dumps(manifest))
    tampered["samples"][0]["split"] = "test"
    with pytest.raises(ValueError, match="frozen"):
        data.verify_manifest(tampered)
    leaked = dict(manifest, samples=[dict(s, run=1) for s in manifest["samples"]])
    leaked["sha256"] = data.digest({k: v for k, v in leaked.items() if k != "sha256"})
    with pytest.raises(ValueError, match="leaks"):
        data.verify_manifest(leaked)


def logits_for(meta, sharp=8.):
    label = data.targets(meta)
    return {k: np.where(label[k] > .5, sharp, -sharp) + np.log(np.maximum(label[k], 1e-3)) * (k != "contact")
            for k in ("contact", "down", "up")}, label


def test_decoder_recovers_gestures_and_scores_them():
    meta = meta_for()
    signals, label = logits_for(meta)
    gestures = decode(label["xy"], signals["contact"], signals["down"], signals["up"], label["times"])
    assert [(s["onset_frame"], s["release_frame"]) for s in gestures] == [(30, 35), (54, 60)]
    assert gestures[1]["pieces"] == 2 and abs(gestures[1]["onset_s"] - .9) < .01
    score = score_clip(gestures, meta)
    assert score["count_ok"] and score["clip_pass_0.02"] and all(s["timing_ok"] for s in score["strokes"])
    late = [dict(s, onset_s=s["onset_s"] + .05, release_s=s["release_s"] + .05) for s in gestures]
    assert not any(s["timing_ok"] for s in score_clip(late, meta)["strokes"])
    moved = [dict(s, points=(np.asarray(s["points"]) + [.03, 0]).tolist()) for s in gestures]
    result = score_clip(moved, meta)
    assert result["clip_pass_0.05"] and not result["clip_pass_0.02"]
    missing = score_clip(gestures[:1], meta)
    assert not missing["count_ok"] and not missing["strokes"][1]["matched"]


def test_gesture_spans_limit_count_and_length():
    times = np.arange(120)/60
    contact, down, up = (np.full(120, -6.) for _ in range(3))
    assert gesture_spans(contact, down, up, times) == []
    for start in (10, 30, 50, 70, 90):  # five clear gestures: only four are kept
        contact[start:start + 6] = 6.
        down[start], up[start + 5] = 6., 6.
    assert len(gesture_spans(contact, down, up, times)) == 4
    long = np.full(120, -6.)
    long[10:60] = 6.  # 817 ms of contact cannot be one gesture
    spans = gesture_spans(long, np.where(np.arange(120) == 10, 6., -6.), np.where(np.arange(120) == 59, 6., -6.), times)
    assert all(37 >= b - a >= 2 for a, b in spans)
    single = np.full(120, -6.)
    single[20] = 6.  # one frame of contact is not a gesture
    assert gesture_spans(single, single, single, times) == []


def test_fit_segments_finds_a_kink_and_keeps_limits():
    times = np.arange(19)/60
    gesture = Gesture(((.4, .4), (.6, .4), (.6, .6)), (0, 150, 300), "random")
    xy = gesture.positions(times)
    indices, ms, residual = fit_segments(times, xy)
    assert len(indices) == 3 and residual < .01 and abs(ms[indices[1]] - 150) <= 17
    for seed in range(5):
        noisy = xy + np.random.default_rng(seed).normal(0, .05, xy.shape)
        indices, ms, _ = fit_segments(times, noisy)
        assert 1 <= len(indices) - 1 <= 4 and min(np.diff(ms[indices])) >= 40
    indices, ms, _ = fit_segments(times[:3], xy[:3])  # 33 ms: one segment
    assert indices == [0, 2]


def test_model_gradients_reach_every_frame():
    from trueskate_ai.model1.trick.model import TrickModel, loss_function
    torch.set_num_threads(1)
    model = TrickModel(4)
    video = torch.rand(1, 120, 3, 16, 8, requires_grad=True)
    batch = {k: torch.as_tensor(v)[None] for k, v in data.targets(meta_for()).items()}
    pred = model(video, batch["times"])
    assert {"xy", "contact", "down", "up"} <= pred.keys() and pred["contact"].shape == (1, 120)
    loss = loss_function(pred, batch)
    loss.backward()
    assert torch.isfinite(loss) and (video.grad.abs().sum((2, 3, 4)) > 0).all()


class Synthetic(torch.utils.data.Dataset):
    def __init__(self):
        self.meta = [meta_for(), meta_for(row(PUSH, (0, 300), "push", .4))]
        self.rows = [dict(id=f"s/{i}", device="iPhone_XR", recording="m0001", run=i) for i in range(2)]
        self.frames = torch.randint(0, 255, (2, 120, 3, 16, 8), generator=torch.Generator().manual_seed(3),
                                    dtype=torch.uint8)

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return dict(video=self.frames[i], index=i, **{k: torch.as_tensor(v) for k, v in data.targets(self.meta[i]).items()})


def test_two_epoch_smoke_resume_and_test_once(tmp_path, monkeypatch):
    from trueskate_ai.model1.trick import training
    torch.set_num_threads(1)
    synthetic = Synthetic()
    recipe = training.Recipe(epochs=2, batch_size=1, base_channels=4)
    identity = dict(bundle="synthetic", manifest="synthetic")
    full = training.fit(synthetic, synthetic, tmp_path/"full", recipe=recipe, identity=identity)
    assert full["complete"] and full["epoch"] == 2
    summary = json.loads((tmp_path/"full/validation-002.json").read_text())["summary"]
    assert summary["clips"] == 2 and summary["strokes"] == 3 and "family" in summary["strata"]
    partial = training.fit(synthetic, synthetic, tmp_path/"resumed", recipe=recipe, identity=identity,
                           stop_after_epochs=1)
    assert not partial["complete"]
    training.fit(synthetic, synthetic, tmp_path/"resumed", recipe=recipe, identity=identity, resume=True)
    a = torch.load(tmp_path/"full/epoch-002.pt", weights_only=False)["model"]
    b = torch.load(tmp_path/"resumed/epoch-002.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError, match="changed"):
        training.fit(synthetic, synthetic, tmp_path/"resumed", recipe=training.Recipe(epochs=3, base_channels=4),
                     identity=identity, resume=True)
    monkeypatch.setattr(training, "load_bundle", lambda bundle: (dict(sha256="synthetic"), {}))
    monkeypatch.setattr(training, "TrickDataset", lambda bundle, split: synthetic)
    assert training.test_once(tmp_path/"bundle", tmp_path/"full")["clips"] == 2
    with pytest.raises(FileExistsError):
        training.test_once(tmp_path/"bundle", tmp_path/"full")


def modal_script():
    path = Path(__file__).parents[1]/"scripts/model1/train_trick_modal.py"
    spec = importlib.util.spec_from_file_location("train_trick_modal_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_budget_rejects_before_importing_modal():
    module = modal_script()
    with pytest.raises(ValueError, match="budget"):
        module.approved_seconds(.5, 1800)
    with pytest.raises(ValueError, match="3600"):
        module.approved_seconds(10, 7200)
    assert module.approved_seconds(1.2, 1800) == 1800


def test_modal_definition_builds_without_starting_remote_work(monkeypatch):
    import modal
    module = modal_script()

    @contextlib.contextmanager
    def blocked(*args, **kwargs):
        raise RuntimeError("remote work blocked by test")
        yield

    monkeypatch.setattr(modal.App, "run", blocked)
    with pytest.raises(RuntimeError, match="blocked"):
        module.execute(SimpleNamespace(command="train", approved_budget_usd=2, max_seconds=900, run="synthetic",
                                       bundle_hash="a"*64, resume=False, epochs=2, base_channels=None))
