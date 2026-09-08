import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from trueskate_ai.bc.gesture_tokens import decode
from trueskate_ai.bc.infer import SequencePolicyRunner, TimestampedMjpegBuffer
from trueskate_ai.bc.model2 import SequencePolicyConfig, stroke_loss
from trueskate_ai.bc.sequence_dataset import SequenceDataset


def _stroke(start, end, value=0.5):
    params = [value] * 6 + [max(0.05, end - start), 1.0, 0.0]
    return {"params": params, "t_start": start, "t_end": end}


def _clip(root: Path, strokes, fps=10, frames=12):
    d = root / "clip"
    d.mkdir()
    (d / "clip.json").write_text(json.dumps({"fps": fps, "strokes": strokes}))
    for i in range(frames):
        cv2.imwrite(str(d / f"frame_{i:06d}.png"), np.full((20, 10, 3), i, np.uint8))
    return d


def _cfg(m_out=2):
    return SequencePolicyConfig(n_frames=2, m_past=4, m_out=m_out, img_h=20, img_w=10,
                                d_model=16, n_heads=2, n_layers=1)


def test_exact_decision_boundary_includes_equal_frame(tmp_path):
    _clip(tmp_path, [_stroke(0.2, 0.3), _stroke(0.5, 0.6)])
    ds = SequenceDataset(tmp_path, cfg=_cfg(1))
    _, gi, decision, indices = ds.index[1]
    assert gi == 1 and decision == pytest.approx(0.3)
    assert indices[-1] == 3  # equality included; frame 0.4 excluded


def test_mid_action_has_no_frame_zero_fallback_or_cold_start(tmp_path):
    _clip(tmp_path, [_stroke(-0.1, 0.1), _stroke(0.2, 0.3), _stroke(0.5, 0.6)])
    ds = SequenceDataset(tmp_path, cfg=_cfg(1))
    assert [(gi, t) for _, gi, t, _ in ds.index] == [(2, pytest.approx(0.3))]
    assert ds.exclusion_counts["group_before_clip"] == 1
    assert ds.exclusion_counts["cold_start_mid_action"] == 1


def test_sequential_decision_at_previous_completion_retains_wait(tmp_path):
    _clip(tmp_path, [_stroke(0.1, 0.2), _stroke(0.5, 0.6)])
    ds = SequenceDataset(tmp_path, cfg=_cfg(1))
    item = ds[1]
    native = decode(item["target"].numpy())
    assert ds.index[1][2] == pytest.approx(0.2)
    assert native[0, -1] == pytest.approx(0.3)


def test_overlap_group_mask_and_negative_internal_delay_round_trip(tmp_path):
    _clip(tmp_path, [_stroke(0.1, 0.4), _stroke(0.2, 0.3)])
    ds = SequenceDataset(tmp_path, cfg=_cfg(3))
    item = ds[0]
    assert item["target_mask"].tolist() == [True, True, False]
    native = decode(item["target"].numpy())[:2]
    assert native[0, -1] == pytest.approx(0.1)
    assert native[1, -1] == pytest.approx(-0.2)
    runner = SequencePolicyRunner(None, _cfg(3), torch.device("cpu"))
    vec, n, pre = runner.to_param_vector(native)
    assert n == 2 and pre == pytest.approx(0.1)
    assert vec[-1] == pytest.approx(-0.2)


def test_oversized_overlap_group_fails_with_identity(tmp_path):
    clip = _clip(tmp_path, [_stroke(0.1, 0.5), _stroke(0.2, 0.4), _stroke(0.3, 0.6)])
    with pytest.raises(ValueError, match=r"clip.*group 0 requires m_out=3"):
        SequenceDataset(tmp_path, cfg=_cfg(2))


def test_padded_slots_do_not_contribute_regression_but_activity_is_supervised():
    pred = torch.zeros(1, 2, 9)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, False]])
    logits = torch.zeros(1, 2)
    base = stroke_loss(pred, target, mask, logits)
    target[:, 1] = 1000
    assert stroke_loss(pred, target, mask, logits) == pytest.approx(base)
    assert stroke_loss(pred, target, mask, torch.tensor([[5.0, -5.0]])) < base


def test_live_window_resamples_recent_frames_and_replaces_old_decisions():
    buf = TimestampedMjpegBuffer(max_seconds=2)
    for t in [9.0, 9.82, 9.91, 10.0]:
        buf.add(t, np.full((2, 2, 3), int(t * 100) % 256, np.uint8))
    window = buf.recent_window(3, 0.2, now=10.0)
    assert [int(x[0, 0, 0]) for x in window] == [int(9.0 * 100) % 256,
                                                   int(9.91 * 100) % 256,
                                                   int(10.0 * 100) % 256]
    cfg = _cfg(1)
    runner = SequencePolicyRunner(None, cfg, torch.device("cpu"))
    runner.replace_window(window)
    runner.replace_window([np.full((2, 2, 3), 7, np.uint8)])
    assert len(runner._frames) == 1


class _FixedPolicy(torch.nn.Module):
    def forward(self, frames, past, past_mask=None):
        strokes = torch.full((1, 3, 9), 0.5)
        activity = torch.tensor([[5.0, 5.0, -5.0]])
        return strokes, activity


def test_end_to_end_active_prefix_to_param_vector_preserves_delays():
    cfg = _cfg(3)
    runner = SequencePolicyRunner(_FixedPolicy(), cfg, torch.device("cpu"))
    runner.observe(np.zeros((20, 10, 3), np.uint8))
    strokes = runner.act()
    strokes[0, -1], strokes[1, -1] = 0.25, -0.1
    vec, n, pre = runner.to_param_vector(strokes)
    assert n == 2 and pre == pytest.approx(0.25) and vec[-1] == pytest.approx(-0.1)


def test_smoke_never_defaults_to_a_durable_model_path():
    """A diagnostic must not be able to clobber a model artefact.

    On 2026-08-20 a `--smoke` run overwrote `notebooks/models/sequence_model.pth`
    because that was the `--out` default in both modes.  Model artefacts are
    gitignored, so the clobber was silent and unrecoverable.
    """
    import argparse
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "train" / "train_sequence_model.py").read_text()
    # The default must be resolved after parsing, not baked into add_argument.
    assert 'ap.add_argument("--out", type=Path, default=None' in source
    assert '"tmp" / "sequence_model_smoke.pth" if args.smoke' in source
    assert 'notebooks" / "models" / "sequence_model.pth"' in source

    # And the resolution itself must send smoke to tmp/ and a real run to models/.
    def resolve(smoke, explicit=None):
        args = argparse.Namespace(smoke=smoke, out=explicit)
        if args.out is None:
            args.out = (root / "tmp" / "sequence_model_smoke.pth" if args.smoke
                        else root / "notebooks" / "models" / "sequence_model.pth")
        return args.out

    assert resolve(smoke=True).parent.name == "tmp"
    assert resolve(smoke=False).parent.name == "models"
    # An explicit --out still wins in either mode.
    assert resolve(smoke=True, explicit=Path("/x/y.pth")) == Path("/x/y.pth")


def test_smoke_honours_the_epochs_flag():
    """`--epochs` used to be ignored under --smoke.

    It was floored at 3 (`max(3, args.epochs)`) and independently capped by an
    `ep >= 2` break inside train(), so every smoke run printed "epoch 3/3"
    whatever was asked for.  A flag that reports something other than what it did
    is the same hazard as `trail_frames_present` and the synthesised
    `frame_times`, both of which produced retracted conclusions.
    """
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1]
              / "scripts" / "train" / "train_sequence_model.py").read_text()
    assert "max(3, args.epochs)" not in source
    assert "if smoke and ep >= 2" not in source
    assert "epochs=args.epochs, batch_size=args.batch_size" in source
    # The real-run default must not have been altered while fixing the smoke path.
    assert 'ap.add_argument("--epochs", type=int, default=10' in source
