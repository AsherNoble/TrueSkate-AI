"""EQ-001: validation-fit along-path end-bias correction."""
import numpy as np
import pytest
import torch

from trueskate_ai.vision.basic_linear_bias import (
    AlongPathBias, fit_along_path_bias, signed_along_path_error,
)
from trueskate_ai.vision.basic_linear_training import basic_linear_metrics


def _record(start, end, *, along=0.0, perpendicular=0.0, duration=0.5):
    """A synthetic record whose last knot is displaced by a known amount."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    unit = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-unit[1], unit[0]])
    predicted_end = end + along * unit + perpendicular * normal
    return {
        "predicted": [*start, *predicted_end, duration],
        "target": [*start, *end, duration],
    }


def test_signed_error_is_negative_for_an_undershoot():
    record = _record((0.2, 0.3), (0.7, 0.8), along=-0.012)
    assert signed_along_path_error(record) == pytest.approx(-0.012, abs=1e-9)


def test_perpendicular_displacement_does_not_enter_the_along_fit():
    record = _record((0.2, 0.3), (0.7, 0.8), along=0.0, perpendicular=0.02)
    assert signed_along_path_error(record) == pytest.approx(0.0, abs=1e-9)


def test_fit_recovers_a_known_injected_bias_within_ten_percent():
    rng = np.random.default_rng(0)
    injected = -0.0071  # the measured validation figure
    records = []
    for _ in range(400):
        start = rng.uniform(0.2, 0.4, size=2)
        end = start + rng.uniform(0.2, 0.4, size=2)
        records.append(_record(start, end, along=injected + rng.normal(0., 0.004),
                               perpendicular=rng.normal(0., 0.003)))
    bias = fit_along_path_bias(records)
    assert bias.samples == 400
    assert bias.shift == pytest.approx(injected, rel=0.10)


def test_applying_the_fit_removes_the_bias_from_held_out_records():
    rng = np.random.default_rng(1)
    injected = -0.010

    def draw(count, generator):
        rows = []
        for _ in range(count):
            start = generator.uniform(0.2, 0.4, size=2)
            end = start + generator.uniform(0.2, 0.4, size=2)
            rows.append(_record(start, end, along=injected + generator.normal(0., 0.003)))
        return rows

    bias = fit_along_path_bias(draw(300, rng))
    held_out = draw(300, np.random.default_rng(2))
    before = np.mean([signed_along_path_error(record) for record in held_out])
    corrected = []
    for record in held_out:
        prediction = bias.apply(torch.tensor([record["predicted"]], dtype=torch.float64))
        corrected.append(signed_along_path_error(
            {"predicted": prediction[0].tolist(), "target": record["target"]}))
    assert before == pytest.approx(injected, abs=0.002)
    assert abs(float(np.mean(corrected))) < abs(before) / 5


def test_a_zero_shift_is_an_exact_no_op():
    prediction = torch.tensor([[0.2, 0.3, 0.7, 0.8, 0.5]])
    corrected = AlongPathBias(shift=0., samples=0).apply(prediction)
    assert torch.equal(corrected, prediction)
    assert corrected is not prediction


def test_correction_leaves_every_other_component_untouched():
    prediction = torch.tensor([[0.2, 0.3, 0.5, 0.4, 0.7, 0.8, 0.55]])  # K=3
    corrected = AlongPathBias(shift=-0.01, samples=10).apply(prediction)
    assert torch.allclose(corrected[:, :4], prediction[:, :4])  # knots 0 and 1
    assert corrected[0, -1] == prediction[0, -1]                # duration
    assert not torch.allclose(corrected[:, 4:6], prediction[:, 4:6])


def test_correction_moves_the_last_knot_forward_along_the_predicted_path():
    prediction = torch.tensor([[0.2, 0.2, 0.6, 0.2, 0.5]])  # horizontal, +x
    corrected = AlongPathBias(shift=-0.01, samples=10).apply(prediction)
    assert corrected[0, 2] == pytest.approx(0.61, abs=1e-6)  # pushed forward
    assert corrected[0, 3] == pytest.approx(0.20, abs=1e-6)  # not sideways


def test_degenerate_paths_are_skipped_and_never_corrected():
    degenerate = {"predicted": [0.3, 0.3, 0.3, 0.3, 0.5], "target": [0.3, 0.3, 0.3, 0.3, 0.5]}
    assert signed_along_path_error(degenerate) is None
    assert fit_along_path_bias([degenerate]).shift == 0.
    prediction = torch.tensor([[0.3, 0.3, 0.3, 0.3, 0.5]])
    assert torch.allclose(AlongPathBias(shift=-0.01, samples=5).apply(prediction), prediction)


def test_metrics_accept_the_correction_and_never_fit_it():
    class _Model(torch.nn.Module):
        def forward(self, frames):
            return torch.tensor([[0.2, 0.2, 0.59, 0.2, 0.5]]).repeat(len(frames), 1)

    batch = {"frames": torch.zeros(1, 1), "target": torch.tensor([[0.2, 0.2, 0.62, 0.2, 0.5]])}
    device = torch.device("cpu")
    uncorrected = basic_linear_metrics(_Model(), [batch], device)
    corrected = basic_linear_metrics(_Model(), [batch], device,
                                     correction=AlongPathBias(shift=-0.03, samples=100))
    assert uncorrected["end_coordinate_median"] == pytest.approx(0.03, abs=1e-6)
    assert corrected["end_coordinate_median"] == pytest.approx(0.0, abs=1e-6)
