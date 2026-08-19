"""MVP-3 fixed-time trajectory representation and K-knot polyline decoder."""
import numpy as np
import pytest
import torch

from trueskate_ai.data.trajectory_resample import (
    command_knot_times, resample_command_at_times,
)
from trueskate_ai.vision.basic_linear_regressor import BasicLinearRegressor


def test_straight_drag_resamples_to_its_own_midpoint():
    # The representation asks "where was the finger at half time", which is well
    # posed even for a straight gesture — unlike "where is the bend", which is
    # what forced the abandoned minimum-bend contract rule.
    sampled = resample_command_at_times([(.2, .7), (.8, .3)], .6, knots=3)
    assert sampled == pytest.approx(np.array([[.2, .7], [.5, .5], [.8, .3]]), abs=1e-6)


def test_constant_velocity_three_waypoint_knots_are_the_commanded_waypoints():
    # With easing 1.0 the executor splits time equally per segment, so evenly
    # spaced sample times land exactly on the waypoints.
    waypoints = [(.2, .7), (.5, .2), (.8, .6)]
    sampled = resample_command_at_times(waypoints, .6, knots=3)
    assert sampled == pytest.approx(np.array(waypoints), abs=1e-6)


def test_easing_is_absorbed_into_positions_rather_than_predicted():
    waypoints = [(.2, .7), (.5, .2), (.8, .6)]
    linear = resample_command_at_times(waypoints, .6, knots=3, easing_power=1.0)
    eased = resample_command_at_times(waypoints, .6, knots=3, easing_power=2.0)
    # Same path, different velocity profile -> the interior knot moves, and that
    # displacement is exactly what removes easing_power from the target vector.
    assert eased[1] != pytest.approx(linear[1], abs=1e-3)
    assert eased[0] == pytest.approx(linear[0], abs=1e-9)
    assert eased[-1] == pytest.approx(linear[-1], abs=1e-9)


def test_endpoints_are_preserved_exactly_for_an_arbitrary_shape():
    z = [(.2, .2), (.8, .2), (.2, .8), (.8, .8)]
    for knots in (3, 5, 9, 17):
        sampled = resample_command_at_times(z, 1.2, knots=knots)
        assert len(sampled) == knots
        assert sampled[0] == pytest.approx(np.array([.2, .2]), abs=1e-9)
        assert sampled[-1] == pytest.approx(np.array([.8, .8]), abs=1e-9)


def test_enough_knots_recover_a_z_shape_corners():
    # The roadmap claim: only K changes to cover richer shapes.
    z = [(.2, .2), (.8, .2), (.2, .8), (.8, .8)]
    sampled = resample_command_at_times(z, 1.2, knots=9)
    corners = np.array([[.65, .2], [.35, .8]])
    for corner in corners:
        assert np.abs(sampled - corner).sum(axis=1).min() < 1e-6


def test_knot_times_match_the_executors_segment_split():
    times = command_knot_times([(.1, .1), (.5, .5), (.9, .9)], 1.0, 1.0)
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0, abs=2e-3)
    assert times[1] == pytest.approx(0.5, abs=2e-3)


def test_resample_rejects_degenerate_commands():
    with pytest.raises(ValueError, match="knots"):
        resample_command_at_times([(.1, .1), (.9, .9)], .5, knots=1)
    with pytest.raises(ValueError, match="waypoints"):
        resample_command_at_times([(.1, .1)], .5, knots=3)
    with pytest.raises(ValueError, match="duration"):
        resample_command_at_times([(.1, .1), (.9, .9)], 0.0, knots=3)


def _track(knots_xy: torch.Tensor, steps: int = 40):
    fraction = torch.linspace(0., 1., steps)[None, :]
    basis = BasicLinearRegressor._hat_basis(fraction, knots_xy.shape[1])
    return torch.einsum("btk,bkc->btc", basis, knots_xy), fraction


def test_hat_basis_reduces_to_the_mvp2_line_at_two_knots():
    fraction = torch.linspace(0., 1., 7)[None, :]
    basis = BasicLinearRegressor._hat_basis(fraction, 2)
    assert torch.allclose(basis[..., 0], 1. - fraction)
    assert torch.allclose(basis[..., 1], fraction)
    assert torch.allclose(basis.sum(dim=-1), torch.ones_like(fraction))


def test_polyline_fit_recovers_curved_and_z_tracks_exactly():
    for knots_xy in (
        torch.tensor([[[.2, .7], [.5, .2], [.8, .6]]]),
        torch.tensor([[[.2, .2], [.425, .2], [.65, .2], [.725, .275], [.5, .5],
                       [.275, .725], [.35, .8], [.575, .8], [.8, .8]]]),
    ):
        positions, fraction = _track(knots_xy, steps=60)
        fitted = BasicLinearRegressor._fit_polyline(
            positions, fraction, torch.ones_like(fraction), knots=knots_xy.shape[1],
        )
        assert (fitted - knots_xy).abs().max() < 1e-3


def test_polyline_irls_downweights_an_outlier_frame_at_three_knots():
    knots_xy = torch.tensor([[[.2, .7], [.5, .2], [.8, .6]]])
    positions, fraction = _track(knots_xy)
    positions[0, 11] = torch.tensor([.02, .95])
    weights = torch.ones_like(fraction)
    plain = BasicLinearRegressor._fit_polyline(positions, fraction, weights, knots=3)
    assert (plain - knots_xy).abs().max() > 1e-2
    basis = BasicLinearRegressor._hat_basis(fraction, 3)
    for _ in range(3):
        fitted = BasicLinearRegressor._fit_polyline(positions, fraction, weights, knots=3)
        residual = torch.linalg.vector_norm(
            positions - torch.einsum("btk,bkc->btc", basis, fitted), dim=2,
        )
        weights = weights * (.02 / residual.clamp_min(1e-6)).clamp(max=1.)
    robust = BasicLinearRegressor._fit_polyline(positions, fraction, weights, knots=3)
    assert (robust - knots_xy).abs().max() < 1e-3


def test_polyline_fit_rejects_too_few_knots():
    positions, fraction = _track(torch.tensor([[[.2, .7], [.8, .3]]]))
    with pytest.raises(ValueError, match="knots"):
        BasicLinearRegressor._fit_polyline(positions, fraction, torch.ones_like(fraction), knots=1)


@pytest.mark.parametrize("knots", (2, 3, 5, 9))
def test_regressor_emits_two_k_plus_one_targets(knots):
    model = BasicLinearRegressor(base_channels=4, line_fit=True, temporal_mixer=True, knots=knots)
    frames = torch.rand(2, 16, 3, 60, 32)
    prediction = model(frames)
    assert prediction.shape == (2, 2 * knots + 1)
    assert torch.isfinite(prediction).all()
    prediction.sum().backward()
    assert model.trajectory_score.weight.grad is not None


def test_two_knots_stays_byte_compatible_with_mvp2():
    # The MVP-2 contract is [x0,y0,x1,y1,duration]; MVP-3 must not silently
    # change the meaning of existing checkpoints.
    model = BasicLinearRegressor(base_channels=4, knots=2)
    assert model.knots == 2
    assert model(torch.rand(1, 8, 3, 30, 18)).shape == (1, 5)


def test_extra_knots_require_the_line_fit_decoder():
    with pytest.raises(ValueError, match="line-fit"):
        BasicLinearRegressor(base_channels=4, knots=3)
    with pytest.raises(ValueError, match="knots"):
        BasicLinearRegressor(base_channels=4, line_fit=True, knots=1)
