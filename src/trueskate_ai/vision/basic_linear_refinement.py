"""Optional post-inference visual refinement for MVP-2 endpoints."""
from __future__ import annotations

import torch


def refine_linear_endpoints(frames: torch.Tensor, prediction: torch.Tensor, *,
                            blend: float, spatial_sigma: float, time_sigma: float) -> torch.Tensor:
    """Nudge a predicted endpoint toward local orange trace evidence.

    This deliberately runs *after* the neural regressor.  The model supplies a
    global proposal, making a local colour cue robust to unrelated orange game
    elements elsewhere on the screen.
    """
    if frames.ndim != 5 or prediction.ndim != 2 or prediction.shape[1] != 5:
        raise ValueError("expected frames [batch,time,3,height,width] and prediction [batch,5]")
    if not 0.0 <= blend <= 1.0 or spatial_sigma <= 0 or time_sigma <= 0:
        raise ValueError("blend must be in [0,1] and sigmas must be positive")
    batch, steps, _channels, height, width = frames.shape
    reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
    red, green, blue = frames.unbind(dim=2)
    motion = torch.abs(frames - reference).mean(dim=2)
    orange = ((red - green + .12).relu() * (green - blue + .12).relu()
              * (red - .20).relu() * motion)
    time = torch.linspace(0., 1., steps, dtype=frames.dtype, device=frames.device)
    xa = torch.linspace(0., 1., width, dtype=frames.dtype, device=frames.device)
    ya = torch.linspace(0., 1., height, dtype=frames.dtype, device=frames.device)
    onset = time.new_full((batch,), .24)
    liftoff = (onset + prediction[:, 4] / 2.27).clamp(max=.88)

    def refine(xy: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
        time_weight = torch.exp(-.5 * ((time[None, :, None, None] - centre[:, None, None, None])
                                       / time_sigma).square())
        x_weight = torch.exp(-.5 * ((xa[None, None, None, :] - xy[:, 0, None, None, None])
                                    / spatial_sigma).square())
        y_weight = torch.exp(-.5 * ((ya[None, None, :, None] - xy[:, 1, None, None, None])
                                    / spatial_sigma).square())
        weight = orange * time_weight * x_weight * y_weight
        normalizer = weight.sum((1, 2, 3)).clamp_min(1e-8)
        candidate = torch.stack(((weight * xa.view(1, 1, 1, width)).sum((1, 2, 3)) / normalizer,
                                 (weight * ya.view(1, 1, height, 1)).sum((1, 2, 3)) / normalizer), dim=1)
        return xy * (1. - blend) + candidate * blend

    result = prediction.clone()
    result[:, :2] = refine(prediction[:, :2], onset)
    result[:, 2:4] = refine(prediction[:, 2:4], liftoff)
    # Coordinates are normalised; duration remains in its native 0.30..1.20s
    # interval and must never be clipped to one second during evaluation.
    result[:, :4] = result[:, :4].clamp(0., 1.)
    return result
