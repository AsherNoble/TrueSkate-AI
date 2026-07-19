import torch

from trueskate_ai.vision.temporal_trace_predictor import (
    TemporalTracePredictor,
    TemporalTraceState,
)


def _model(**kwargs) -> TemporalTracePredictor:
    torch.manual_seed(7)
    defaults = {
        "base_channels": 4,
        "hidden_channels": 8,
        "downsample_stages": 2,
    }
    defaults.update(kwargs)
    return TemporalTracePredictor(**defaults).eval()


def test_step_and_sequence_shapes_include_activity_and_multi_peak_heatmap():
    model = _model()
    frame = torch.rand(2, 3, 31, 17)

    step = model.step(frame, delta_t=torch.tensor([0.03, 0.04]))
    assert step.heatmap.shape == (2, 1, 31, 17)
    assert step.active_logits.shape == (2,)
    assert step.active_probability.shape == (2,)
    assert step.state.hidden.shape == (2, 8, 8, 5)
    assert torch.all((step.heatmap >= 0) & (step.heatmap <= 1))

    sequence = model(torch.rand(2, 4, 3, 31, 17), delta_times=torch.rand(2, 4))
    assert sequence.heatmaps.shape == (2, 4, 1, 31, 17)
    assert sequence.active_logits.shape == (2, 4)
    assert sequence.active_probabilities.shape == (2, 4)
    assert sequence.teacher_forcing_mask.shape == (2, 4)

    # The spatial output is not a softmax: simultaneous high-valued pixels are
    # representable instead of being forced to share one unit of probability.
    with torch.no_grad():
        model.heatmap_head.weight.zero_()
        model.heatmap_head.bias.zero_()
    heatmap = model.step(frame[:1]).heatmap
    assert heatmap[0, 0, 0, 0] == 0.5
    assert heatmap.sum() > 1.0


def test_sequence_is_causal_when_future_rgb_frames_change():
    model = _model()
    prefix = torch.rand(1, 3, 3, 24, 16)
    future_a = torch.zeros(1, 2, 3, 24, 16)
    future_b = torch.ones(1, 2, 3, 24, 16)

    output_a = model(torch.cat((prefix, future_a), dim=1))
    output_b = model(torch.cat((prefix, future_b), dim=1))

    torch.testing.assert_close(output_a.heatmaps[:, :3], output_b.heatmaps[:, :3])
    torch.testing.assert_close(output_a.active_logits[:, :3], output_b.active_logits[:, :3])


def test_teacher_feedback_uses_previous_target_never_current_or_future_target():
    model = _model()
    frames = torch.rand(1, 4, 3, 24, 16)
    targets_a = torch.zeros(1, 4, 1, 24, 16)
    targets_b = targets_a.clone()
    targets_b[:, 2, :, 5:10, 4:8] = 1.0
    force_every_transition = torch.ones(1, 3, dtype=torch.bool)

    output_a = model(
        frames,
        teacher_heatmaps=targets_a,
        teacher_forcing_mask=force_every_transition,
    )
    output_b = model(
        frames,
        teacher_heatmaps=targets_b,
        teacher_forcing_mask=force_every_transition,
    )

    # Target 2 may first influence prediction 3.  Predictions 0..2 must match.
    torch.testing.assert_close(output_a.heatmaps[:, :3], output_b.heatmaps[:, :3])
    assert not torch.allclose(output_a.heatmaps[:, 3], output_b.heatmaps[:, 3])
    assert output_a.teacher_forcing_mask.tolist() == [[False, True, True, True]]


def test_state_is_explicit_and_none_resets_to_reproducible_cold_start():
    model = _model()
    frame = torch.rand(1, 3, 24, 16)

    cold = model.step(frame)
    continued = model.step(frame, cold.state)
    reset = model.step(frame, None)

    torch.testing.assert_close(cold.heatmap, reset.heatmap)
    torch.testing.assert_close(cold.state.hidden, reset.state.hidden)
    assert not torch.allclose(cold.state.hidden, continued.state.hidden)

    detached = continued.state.detach()
    assert isinstance(detached, TemporalTraceState)
    assert detached.hidden.grad_fn is None
    assert detached.previous_heatmap.grad_fn is None


def test_previous_heatmap_feedback_changes_the_current_prediction():
    model = _model()
    frame = torch.rand(1, 3, 24, 16)
    empty = torch.zeros(1, 1, 24, 16)
    peaked = empty.clone()
    peaked[:, :, 7:11, 10:14] = 1.0

    output_empty = model.step(frame, feedback_heatmap=empty)
    output_peaked = model.step(frame, feedback_heatmap=peaked)

    assert not torch.allclose(output_empty.heatmap, output_peaked.heatmap)
    assert not torch.allclose(output_empty.state.hidden, output_peaked.state.hidden)


def test_delta_time_is_an_optional_causal_input():
    model = _model()
    frame = torch.rand(1, 3, 24, 16)

    default_delta = model.step(frame)
    zero_delta = model.step(frame, delta_t=0.0)
    long_delta = model.step(frame, delta_t=0.2)

    torch.testing.assert_close(default_delta.heatmap, zero_delta.heatmap)
    assert not torch.allclose(zero_delta.state.hidden, long_delta.state.hidden)


def test_autoregressive_sequence_has_gradient_flow_across_time():
    model = _model().train()
    frames = torch.rand(1, 3, 3, 24, 16, requires_grad=True)

    output = model(frames, teacher_forcing_probability=0.0)
    loss = output.heatmaps[:, -1].mean() + output.active_logits[:, -1].mean()
    loss.backward()

    assert frames.grad is not None
    assert torch.count_nonzero(frames.grad[:, 0]).item() > 0
    assert model.recurrent.gates.weight.grad is not None
    assert torch.count_nonzero(model.recurrent.gates.weight.grad).item() > 0
    assert model.heatmap_head.weight.grad is not None


def test_sequence_forward_matches_manual_causal_steps():
    model = _model()
    frames = torch.rand(1, 3, 3, 25, 15)
    deltas = torch.tensor([[0.0, 0.033, 0.041]])

    rolled = model(frames, delta_times=deltas)
    state = None
    manual_heatmaps = []
    manual_activity = []
    for t in range(frames.shape[1]):
        step = model.step(frames[:, t], state, delta_t=deltas[:, t])
        state = step.state
        manual_heatmaps.append(step.heatmap)
        manual_activity.append(step.active_logits)

    torch.testing.assert_close(rolled.heatmaps, torch.stack(manual_heatmaps, dim=1))
    torch.testing.assert_close(rolled.active_logits, torch.stack(manual_activity, dim=1))
