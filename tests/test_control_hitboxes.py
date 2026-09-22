import numpy as np
import pytest

from trueskate_ai.data.control_hitboxes import (
    CONTROL_START_EXCLUSIONS,
    controls_along_segment,
    controls_at_point,
    controls_at_start,
    point_is_safe,
    move_point_out_of_controls,
    move_start_out_of_controls,
    segment_is_safe,
    start_is_safe,
)
from trueskate_ai.data.gesture_sampling import (
    GestureSample,
    clamp_in_bounds,
    sample_basic_linear_mixture,
    sample_mixture,
)
from trueskate_ai.sim.gesture_params import PARAMS_PER_SLOT, build_param_bounds


@pytest.mark.parametrize("hitbox", CONTROL_START_EXCLUSIONS, ids=lambda h: h.name)
def test_each_control_region_moves_a_touch_down_to_safety(hitbox):
    x0, y0, x1, y1 = hitbox.rect
    unsafe = ((x0 + x1) / 2, (y0 + y1) / 2)
    assert hitbox.name in controls_at_start(unsafe)
    moved = move_start_out_of_controls(unsafe)
    assert start_is_safe(moved)


def test_moving_gesture_protects_endpoints_but_not_intermediate_points():
    endpoint = (0.5, 0.125)  # reset safety region, within global bounds
    control = (0.15, 0.405)  # spin safety region, within global bounds
    sample = GestureSample(
        kind="flick",
        waypoints=[(0.14, 0.20), control, endpoint],  # start in Bolt safety region
        duration=0.5,
        easing_power=1.0,
    )

    clamp_in_bounds(sample)

    assert start_is_safe(sample.waypoints[0])
    assert sample.waypoints[1] == pytest.approx(control)
    assert point_is_safe(sample.waypoints[2])
    assert controls_at_start(sample.waypoints[1]) == ("spin",)
    assert sample.waypoints[2] != pytest.approx(endpoint)


@pytest.mark.parametrize("kind", ["tap", "hold"])
def test_stationary_touch_point_is_treated_as_its_start(kind):
    sample = GestureSample(kind=kind, point=(0.9, 0.95), hold_duration_s=0.5)
    clamp_in_bounds(sample)
    assert start_is_safe(sample.point)


def test_multislot_vectors_move_each_slots_start_and_end_only():
    num_gestures = 2
    params = np.mean(build_param_bounds(num_gestures), axis=1)
    starts = ((0.14, 0.20), (0.15, 0.405))
    waypoints = (
        ((0.15, 0.405), (0.5, 0.125)),
        ((0.14, 0.30), (0.9, 0.87)),
    )
    for slot in range(num_gestures):
        base = slot * PARAMS_PER_SLOT
        params[base:base + 2] = starts[slot]
        params[base + 2:base + 4] = waypoints[slot][0]
        params[base + 4:base + 6] = waypoints[slot][1]
    sample = GestureSample(
        kind="nslot", params=params.tolist(), num_gestures=num_gestures, use_spin=False,
    )

    clamp_in_bounds(sample)

    for slot in range(num_gestures):
        base = slot * PARAMS_PER_SLOT
        assert start_is_safe((sample.params[base], sample.params[base + 1]))
        assert sample.params[base + 2:base + 4] == pytest.approx(waypoints[slot][0])
        assert point_is_safe((sample.params[base + 4], sample.params[base + 5]))


def test_transient_bottom_row_is_still_strictly_excluded():
    assert controls_at_start((0.5, 0.95)) == ("community",)
    assert start_is_safe(move_start_out_of_controls((0.5, 0.95)))


def test_segment_intersection_detects_crossing_with_safe_endpoints():
    start = (0.10, 0.50)
    end = (0.30, 0.30)

    assert point_is_safe(start)
    assert point_is_safe(end)
    assert controls_along_segment(start, end) == ("spin",)
    assert not segment_is_safe(start, end)


def test_segment_intersection_treats_control_boundary_as_unsafe():
    reset = next(hitbox for hitbox in CONTROL_START_EXCLUSIONS if hitbox.name == "reset")
    _, _, _, lower_edge = reset.rect
    assert "reset" in controls_along_segment((0.45, lower_edge), (0.55, lower_edge))


def test_segment_clear_of_controls_is_safe():
    assert segment_is_safe((0.30, 0.40), (0.75, 0.65))


def test_linear_sampler_never_crosses_controls():
    rng = np.random.default_rng(20260917)
    for _ in range(2_000):
        sample = sample_basic_linear_mixture(rng, tap_fraction=0.0)
        assert start_is_safe(sample.waypoints[0])
        assert point_is_safe(sample.waypoints[-1])
        assert segment_is_safe(sample.waypoints[0], sample.waypoints[-1])


def test_generic_endpoint_helpers_share_the_versioned_regions():
    unsafe = (0.5, 0.125)
    assert controls_at_point(unsafe) == controls_at_start(unsafe) == ("reset",)
    moved = move_point_out_of_controls(unsafe)
    assert point_is_safe(moved)
    assert moved == move_start_out_of_controls(unsafe)


def test_random_multislot_sampler_resamples_every_touch_down():
    rng = np.random.default_rng(91017)
    for _ in range(500):
        sample = sample_mixture(
            rng,
            fracs=(0.0, 1.0, 0.0),
            num_gestures=3,
            use_spin=False,
        )
        for slot in range(sample.num_gestures):
            base = slot * PARAMS_PER_SLOT
            assert start_is_safe((sample.params[base], sample.params[base + 1]))
            assert point_is_safe((sample.params[base + 4], sample.params[base + 5]))
