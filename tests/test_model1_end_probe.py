from pathlib import Path

import pytest
import torch

from trueskate_ai.model1.linear.end_probe import (
    VARIANTS, along_path_error, capture_end_scores, end_variants, frame_read,
)
from trueskate_ai.model1.linear.regressor import BasicLinearRegressor


def _model_and_frames(seed: int = 0):
    torch.manual_seed(seed)
    model = BasicLinearRegressor(base_channels=4, temporal_mixer=True).eval()
    return model, torch.rand(2, 32, 3, 72, 32)


def test_baseline_reread_reproduces_the_model_end_point():
    model, frames = _model_and_frames()
    with torch.no_grad():
        expected = model(frames)
        target = torch.tensor([[.2, .3, .6, .7, .5], [.5, .5, .4, .2, 1.1]])
        prediction, ends, diagnostics = end_variants(model, frames, target)
    assert torch.allclose(prediction, expected)
    assert torch.allclose(ends["baseline"], expected[:, 2:4], atol=1e-6)
    assert set(ends) == set(VARIANTS)
    assert diagnostics["attention_time"].shape == (2,)
    # Per-clip rows store diagnostics beside variants; a shared key overwrites one.
    assert not set(diagnostics) & set(VARIANTS)


def test_oracle_liftoff_equals_baseline_when_durations_agree():
    model, frames = _model_and_frames(1)
    with torch.no_grad():
        predicted = model(frames)
        target = torch.cat((torch.full((2, 4), .5), predicted[:, 4:5]), dim=1)
        _, ends, _ = end_variants(model, frames, target)
    assert torch.allclose(ends["oracle_liftoff"], ends["baseline"], atol=1e-6)
    assert not torch.allclose(ends["sigma_0.05"], ends["baseline"], atol=1e-6)


def test_capture_restores_the_read_even_when_forward_fails():
    model, _frames = _model_and_frames()
    with pytest.raises(ValueError):
        capture_end_scores(model, torch.rand(1, 32, 72, 32))
    assert "_read_xy" not in vars(model)


def test_frame_read_uses_the_frame_nearest_liftoff():
    scores = torch.zeros(1, 32, 9, 5)
    scores[0, 20, 6, 4] = 50.  # sharp peak in frame 20 at x=1, y=0.75
    scores[0, 5, 1, 0] = 50.   # distractor in another frame
    xy = frame_read(scores, torch.tensor([20 / 31]))
    assert torch.allclose(xy, torch.tensor([[1., .75]]), atol=1e-3)


def test_along_path_error_is_negative_when_short():
    target = torch.tensor([[0., 0., 1., 0., .5]])
    assert along_path_error(torch.tensor([[.9, 0.]]), target).item() == pytest.approx(-.1)
    assert along_path_error(torch.tensor([[1.1, .05]]), target).item() == pytest.approx(.1)


def test_endprobe_modal_entry_point_is_validation_only():
    source = Path("scripts/model1/train_basic_linear_modal.py").read_text()
    body = source[source.index("def probe_end_decoding("):]
    body = body[:body.index("\n@app.")]
    assert 'manifest_partition="validation"' in body
    assert '"test"' not in body and "test_indices" not in body
    assert "raise FileExistsError" in body and "_require_gpu(required_gpu)" in body
    assert "max_baseline_drift > 1e-4" in body
