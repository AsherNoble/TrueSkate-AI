"""EQ-001: validation-fit along-path end-bias correction."""
import numpy as np
import pytest
import torch

from trueskate_ai.vision.basic_linear_bias import (
    AlongPathBias, discordant_pairs, fit_along_path_bias, mcnemar_exact_p,
    along_path_fit_key, perpendicular_error, signed_along_path_error,
)
from trueskate_ai.vision.basic_linear_training import basic_linear_metrics


def _record(start, end, *, along=0.0, perpendicular=0.0, duration=0.5,
            start_error=(0.0, 0.0)):
    """A synthetic record whose last knot is displaced by a known amount.

    ``start_error`` displaces the *predicted* first knot away from the commanded
    one, so the predicted and commanded chords genuinely differ.  The checkpoint
    EQ-002 will run (`basic_linear_linear_mixed_fresh_holdout_20260813`) has
    100.0% start recovery at median 0.00635, so the displacement is small but
    never exactly zero.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    unit = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-unit[1], unit[0]])
    predicted_end = end + along * unit + perpendicular * normal
    predicted_start = start + np.asarray(start_error, dtype=float)
    return {
        "predicted": [*predicted_start, *predicted_end, duration],
        "target": [*start, *end, duration],
    }


def test_signed_error_is_negative_for_an_undershoot():
    record = _record((0.2, 0.3), (0.7, 0.8), along=-0.012)
    assert signed_along_path_error(record) == pytest.approx(-0.012, abs=1e-9)


def test_perpendicular_displacement_does_not_enter_the_commanded_axis():
    record = _record((0.2, 0.3), (0.7, 0.8), along=0.0, perpendicular=0.02)
    assert signed_along_path_error(record, axis="commanded") == pytest.approx(0.0, abs=1e-9)


def test_perpendicular_displacement_leaks_into_the_predicted_axis_second_order():
    """The predicted chord is itself rotated by the perpendicular error.

    Exactly q**2 / sqrt(L**2 + q**2) for a perpendicular displacement q on a
    chord of length L (q**2/L to leading order) — 3.7e-5 at the autopsy's
    measured perpendicular sd (0.0032) over a typical 0.35 chord, i.e. ~0.1% of
    the 0.03 tolerance.  Note it is **strictly positive whatever the sign of q**,
    so it is a systematic bias of the estimator, not noise that averages out.
    """
    start, end, perpendicular = np.array([0.2, 0.3]), np.array([0.7, 0.8]), 0.02
    record = _record(start, end, along=0.0, perpendicular=perpendicular)
    chord = float(np.linalg.norm(end - start))
    exact = perpendicular ** 2 / np.hypot(chord, perpendicular)
    assert signed_along_path_error(record, axis="predicted") == pytest.approx(exact, abs=1e-12)
    assert exact == pytest.approx(perpendicular ** 2 / chord, rel=0.001)
    assert abs(exact) < 0.001
    # Strictly positive either way: flipping the sign of q gives the same leak.
    flipped = _record(start, end, along=0.0, perpendicular=-perpendicular)
    assert signed_along_path_error(flipped, axis="predicted") == pytest.approx(exact, abs=1e-12)


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


def test_the_two_axes_diverge_once_the_first_knot_is_wrong():
    """Every other test has predicted start == commanded start, where the axes
    agree to ~1e-8 rad and the choice is unobservable."""
    agreeing = _record((0.2, 0.3), (0.7, 0.8), along=-0.01)
    assert (signed_along_path_error(agreeing, axis="predicted")
            == pytest.approx(signed_along_path_error(agreeing, axis="commanded"), abs=1e-9))
    displaced = _record((0.2, 0.3), (0.7, 0.8), along=-0.01, start_error=(0.012, -0.009))
    predicted_axis = signed_along_path_error(displaced, axis="predicted")
    commanded_axis = signed_along_path_error(displaced, axis="commanded")
    assert predicted_axis != pytest.approx(commanded_axis, abs=1e-6)
    # Both still recover the injected undershoot to well inside tolerance.
    assert predicted_axis == pytest.approx(-0.01, abs=0.002)
    assert commanded_axis == pytest.approx(-0.01, abs=0.002)


def test_a_wrong_first_knot_does_not_break_the_fit():
    rng = np.random.default_rng(7)
    injected = -0.0095
    records = [
        _record(start := rng.uniform(0.2, 0.4, size=2),
                start + rng.uniform(0.2, 0.4, size=2),
                along=injected + rng.normal(0., 0.004),
                perpendicular=rng.normal(0., 0.0036),
                start_error=rng.normal(0., 0.0054, size=2))
        for _ in range(600)
    ]
    predicted_axis = fit_along_path_bias(records, axis="predicted")
    commanded_axis = fit_along_path_bias(records, axis="commanded")
    assert predicted_axis.axis == "predicted"
    assert predicted_axis.shift == pytest.approx(injected, rel=0.10)
    assert commanded_axis.shift == pytest.approx(injected, rel=0.10)
    assert abs(predicted_axis.shift - commanded_axis.shift) < 0.001


def test_axis_is_validated():
    with pytest.raises(ValueError):
        signed_along_path_error(_record((0.2, 0.3), (0.7, 0.8)), axis="nonsense")


def test_a_commanded_axis_fit_cannot_be_applied():
    """apply() corrects along the predicted chord and reads no axis field, so a
    commanded-axis fit would silently reintroduce the EQ-001 mismatch."""
    records = [_record((0.2, 0.3), (0.7, 0.8), along=-0.01)]
    commanded = fit_along_path_bias(records, axis="commanded")
    assert commanded.axis == "commanded"
    with pytest.raises(ValueError, match="cannot be applied"):
        commanded.apply(torch.tensor([[0.2, 0.3, 0.7, 0.8, 0.5]]))
    fit_along_path_bias(records).apply(torch.tensor([[0.2, 0.3, 0.7, 0.8, 0.5]]))


def test_discordant_pairs_counts_direction_not_totals():
    assert discordant_pairs([0., 0., 1., 1.], [1., 0., 1., 0.]) == (1, 1)
    assert discordant_pairs([0.] * 5, [0.] * 5) == (0, 0)
    with pytest.raises(ValueError):
        discordant_pairs([0., 1.], [1.])


def test_mcnemar_exact_matches_the_hand_computed_cases():
    # The EQ-001 red team's figures for the end-bias correction's own counts.
    assert mcnemar_exact_p(4, 1) == pytest.approx(0.375)
    assert mcnemar_exact_p(3, 0) == pytest.approx(0.25)
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(1, 1) == 1.0
    # Symmetric in its arguments: gaining 6 is as surprising as losing 6.
    assert mcnemar_exact_p(6, 0) == pytest.approx(mcnemar_exact_p(0, 6))
    # Only a much larger, one-sided imbalance clears the usual bar.
    assert mcnemar_exact_p(10, 0) < 0.05


def test_perpendicular_error_is_the_across_path_component():
    record = _record((0.2, 0.3), (0.7, 0.8), along=-0.02, perpendicular=0.004)
    assert perpendicular_error(record, axis="commanded") == pytest.approx(0.004, abs=1e-9)
    # Pure along-path error has no perpendicular component on either axis.
    pure = _record((0.2, 0.3), (0.7, 0.8), along=-0.02)
    assert perpendicular_error(pure, axis="commanded") == pytest.approx(0., abs=1e-9)
    assert perpendicular_error(pure, axis="predicted") == pytest.approx(0., abs=1e-9)


def test_the_fit_records_where_it_came_from():
    records = [_record((0.2, 0.3), (0.7, 0.8), along=-0.01)]
    key = along_path_fit_key("validation", [3, 1, 4])
    bias = fit_along_path_bias(records, fit_on=key)
    assert bias.fit_on == key
    assert fit_along_path_bias([]).fit_on == "unspecified"


def test_the_provenance_key_is_derived_from_the_indices_not_asserted():
    """A caller-written label can claim a provenance the artefact lacks; a hash
    of the index set can be re-derived and checked."""
    assert along_path_fit_key("validation", [1, 2, 3]).startswith("validation[3]:")
    assert along_path_fit_key("validation", [1, 2, 3]) == along_path_fit_key("validation", [1, 2, 3])
    assert along_path_fit_key("validation", [1, 2, 3]) != along_path_fit_key("validation", [1, 2, 4])
    # Order matters: a different split ordering is a different split.
    assert along_path_fit_key("validation", [1, 2, 3]) != along_path_fit_key("validation", [3, 2, 1])
    assert along_path_fit_key("test", [1, 2, 3]) != along_path_fit_key("validation", [1, 2, 3])
