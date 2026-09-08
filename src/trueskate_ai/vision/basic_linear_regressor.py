"""Spatial-temporal MVP-2 regressor for one finite-slope straight drag."""
from __future__ import annotations

import torch
from torch import nn

from trueskate_ai.data.gesture_sampling import BASIC_LINEAR_MAX_S, BASIC_LINEAR_MIN_S

# Aligned MVP-2 clips span a fixed [-0.5, 1.77]s response window, so one second
# of command time is 1/CLIP_WINDOW_S of normalised clip time.  This is a
# property of the aligner, not a swept hyper-parameter.
CLIP_WINDOW_S = 2.27


class BasicLinearRegressor(nn.Module):
    """Predict ``[x0,y0,x1,y1,duration]`` while retaining spatial evidence."""

    def __init__(self, base_channels: int = 16, *, start_onset: float = .24,
                 start_sigma: float = .05, end_onset: float = .24,
                 temporal_mixer: bool = False, trajectory_track: bool = False,
                 line_fit: bool = False, irls_iterations: int = 3,
                 huber_delta: float = .02, knots: int = 2):
        super().__init__()
        if start_sigma <= 0:
            raise ValueError("start_sigma must be positive")
        if irls_iterations < 0:
            raise ValueError("irls_iterations must be non-negative")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if knots < 2:
            raise ValueError("knots must be at least 2")
        if knots != 2 and not line_fit:
            raise ValueError("more than two knots requires the line-fit decoder")
        # MVP-3 predicts positions at `knots` fixed, evenly-spaced times.  At
        # knots=2 the output is byte-identical to MVP-2's [x0,y0,x1,y1,duration],
        # so existing checkpoints and metrics keep working unchanged.
        self.knots = int(knots)
        # The line fit reads endpoints off a trajectory consensus, so it needs
        # the per-frame contact map regardless of the standalone track flag.
        if line_fit:
            trajectory_track = True
        self.line_fit_enabled = bool(line_fit)
        self.irls_iterations = int(irls_iterations)
        self.huber_delta = float(huber_delta)
        c = base_channels
        self.start_onset = float(start_onset)
        self.start_sigma = float(start_sigma)
        self.end_onset = float(end_onset)
        self.temporal_mixer_enabled = bool(temporal_mixer)
        self.trajectory_track_enabled = bool(trajectory_track)
        self.encoder = nn.Sequential(
            # MVP-2 must distinguish both endpoints.  At 128px input width a
            # stride-four map has only 32 x-cells (one cell is ~0.031 in model
            # coordinates), already the whole recovery tolerance.  Keep the
            # first layer dense and downsample only once so soft-argmax has
            # genuine sub-tolerance endpoint evidence.
            nn.Conv2d(6, c, 5, stride=1, padding=2), nn.GroupNorm(1, c), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GroupNorm(2, c * 2), nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.GroupNorm(4, c * 4), nn.SiLU(),
        )
        channels = c * 4
        # Start and end are not interchangeable visual concepts.  A shared map
        # plus an externally imposed time prior can collapse to the trace's
        # midpoint; give each endpoint its own learned spatial evidence.
        self.start_score = nn.Conv2d(c * 4, 1, 1)
        self.end_score = nn.Conv2d(c * 4, 1, 1)
        # Endpoint maps must rank one particular moment of a rendered line.
        # A separate path map, supervised at every manifest-active frame,
        # teaches the shared encoder where the moving contact is without
        # forcing either endpoint-specific head to score the entire trail.
        # In the blended track decoder the path map starts as a near-zero
        # correction to the endpoint decoder, earning influence only once its
        # own per-frame map is useful.  That caution is also why it never
        # overtook the baseline, so the line fit replaces the gate outright
        # rather than blending past it.
        self.duration_head = nn.Sequential(
            nn.Conv1d(2, c, 3, padding=1), nn.SiLU(), nn.Conv1d(c, c, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(c * 8, c * 2), nn.SiLU(), nn.Linear(c * 2, 1),
        )
        # EVERY OPTIONAL MODULE IS BUILT BELOW THIS LINE, AND NOTHING UNCONDITIONAL
        # BELONGS AFTER IT.  Module construction draws from the global RNG, so an
        # optional module built earlier shifts the stream for everything after it:
        # enabling `trajectory_track` used to change all 8 `duration_head` tensors
        # and, through the shared stream, every epoch's shuffle -- which silently
        # destroyed seed-matched A/B comparisons (EQ-048).
        # One seed per optional module, drawn UNCONDITIONALLY so the global stream
        # advances by the same amount whichever optionals are enabled, and each
        # module is then built inside a forked RNG from its own seed.  Ordering
        # alone is not enough: the first-built optional would still shift the
        # stream for the rest, so arms differing in `temporal_mixer` would get
        # different `onset_head` weights at the same `--seed`.
        optional_seeds = torch.randint(0, 2 ** 31 - 1, (4,)).tolist()

        def optional(index: int, enabled: bool, factory):
            if not enabled:
                return None
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(optional_seeds[index])
                return factory()

        channels = c * 4
        # The baseline scores each frame independently.  A finite linear drag
        # is a trajectory, so this optional residual 3-D mixer lets an endpoint
        # score use adjacent trace positions without discarding spatial detail.
        # It is intentionally small (depthwise temporal filtering + pointwise
        # mixing) so the controlled ablation changes temporal context only.
        self.temporal_mixer = optional(0, temporal_mixer, lambda: nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), groups=channels),
            nn.GroupNorm(4, channels), nn.SiLU(), nn.Conv3d(channels, channels, 1),
        ))
        self.trajectory_score = optional(1, trajectory_track, lambda: nn.Conv2d(c * 4, 1, 1))
        self.trajectory_fusion = optional(2, trajectory_track and not line_fit,
                                          lambda: nn.Parameter(torch.tensor(-4.0)))
        # The line fit needs to know *where on the path* each frame sits, which
        # requires a touch onset.  Learn it per clip from the same evidence
        # series rather than re-imposing the swept 0.24 constant: onset and
        # duration together define the active window the fit regresses over.
        self.onset_head = optional(3, line_fit, lambda: nn.Sequential(
            nn.Conv1d(2, c, 3, padding=1), nn.SiLU(), nn.Conv1d(c, c, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(c * 8, c * 2), nn.SiLU(), nn.Linear(c * 2, 1),
        ))

    @staticmethod
    def _read_xy(scores: torch.Tensor, time_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, height, width = scores.shape
        if time_prior.ndim == 1:
            time_prior = time_prior.unsqueeze(0)
        if time_prior.shape not in {(1, steps), (batch, steps)}:
            raise ValueError("time_prior must have shape [time] or [batch,time]")
        logits = scores.flatten(1) / .15 + time_prior[:, :, None, None].expand_as(scores).flatten(1)
        attention = torch.softmax(logits, dim=1).reshape_as(scores)
        xa = torch.linspace(0., 1., width, dtype=scores.dtype, device=scores.device)
        ya = torch.linspace(0., 1., height, dtype=scores.dtype, device=scores.device)
        return ((attention * xa.view(1, 1, 1, width)).sum((1, 2, 3)),
                (attention * ya.view(1, 1, height, 1)).sum((1, 2, 3)))

    @staticmethod
    def _frame_positions(scores: torch.Tensor) -> torch.Tensor:
        """Decode one contact position per frame from a trajectory score map.

        One spatial softmax *per frame* (not one global space-time softmax),
        matching the per-frame manifest supervision in
        ``basic_linear_trajectory_map_loss``.
        """
        if scores.ndim != 4:
            raise ValueError("scores must have shape [batch,time,height,width]")
        _batch, _steps, height, width = scores.shape
        spatial = torch.softmax(scores.flatten(2) / .15, dim=2).reshape_as(scores)
        xa = torch.linspace(0., 1., width, dtype=scores.dtype, device=scores.device)
        ya = torch.linspace(0., 1., height, dtype=scores.dtype, device=scores.device)
        return torch.stack((
            (spatial * xa.view(1, 1, 1, width)).sum((2, 3)),
            (spatial * ya.view(1, 1, height, 1)).sum((2, 3)),
        ), dim=2)

    @staticmethod
    def _hat_basis(fraction: torch.Tensor, knots: int) -> torch.Tensor:
        """Piecewise-linear basis over ``knots`` evenly-spaced fixed knot times.

        Returns ``[batch,time,knots]``.  The knot *times* are fixed, so there are
        no free breakpoints to estimate: the fit only ever answers "where was the
        finger at time k/(K-1)", which is well posed even for a perfectly
        straight gesture.  At ``knots=2`` this reduces exactly to ``(1-s, s)``,
        the MVP-2 constant-velocity line.
        """
        if knots < 2:
            raise ValueError("knots must be at least 2")
        index = torch.arange(knots, dtype=fraction.dtype, device=fraction.device)
        scaled = fraction[..., None] * (knots - 1)
        return (1. - (scaled - index).abs()).clamp_min(0.)

    @classmethod
    def _fit_polyline(cls, positions: torch.Tensor, fraction: torch.Tensor,
                      weights: torch.Tensor, *, knots: int) -> torch.Tensor:
        """Weighted least-squares fit of a fixed-time-knot polyline.

        Closed form, so it stays a differentiable ``knots x knots`` solve rather
        than an inner optimisation.  Each knot becomes a consensus over every
        observed frame instead of a read at one moment, which is the point: no
        single occluded or mis-detected frame can carry a knot on its own.
        Returns ``[batch,knots,2]``.
        """
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("positions must have shape [batch,time,2]")
        if fraction.shape != positions.shape[:2] or weights.shape != positions.shape[:2]:
            raise ValueError("fraction and weights must have shape [batch,time]")
        basis = cls._hat_basis(fraction, knots)
        weighted = basis * weights[..., None]
        normal = torch.einsum("bti,btj->bij", weighted, basis)
        rhs = torch.einsum("bti,btc->bic", weighted, positions)
        # A clip whose evidence never covers one knot's neighbourhood leaves the
        # normal equations near-singular there.  A small ridge keeps the solve
        # finite and bounded instead of letting it explode along the null
        # direction; it biases an unsupported knot slightly toward the origin,
        # which the position loss then corrects.
        eye = torch.eye(knots, dtype=positions.dtype, device=positions.device)
        return torch.linalg.solve(normal + 1e-3 * eye, rhs)

    @classmethod
    def _fit_constant_velocity(cls, positions: torch.Tensor, fraction: torch.Tensor,
                               weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MVP-2 two-point special case of :meth:`_fit_polyline`."""
        fitted = cls._fit_polyline(positions, fraction, weights, knots=2)
        return fitted[:, 0], fitted[:, 1]

    def _line_fit_endpoints(self, scores: torch.Tensor, *, onset: torch.Tensor,
                            duration_norm: torch.Tensor, active: torch.Tensor,
                            ) -> torch.Tensor:
        """Robustly fit the commanded path; returns knot positions [batch,K,2]."""
        steps = scores.shape[1]
        positions = self._frame_positions(scores)
        time = torch.linspace(0., 1., steps, dtype=scores.dtype, device=scores.device)
        span = duration_norm.clamp_min(1e-3)[:, None]
        fraction = ((time[None, :] - onset[:, None]) / span).clamp(0., 1.)
        # Confidence: a frame showing a real contact has a sharply peaked map,
        # while a pre-touch or post-liftoff frame is flat.  Peak-minus-mean is
        # that sharpness, and the soft active window suppresses frames the
        # predicted timing says hold no contact at all.
        peak = scores.flatten(2).amax(dim=2)
        mean = scores.flatten(2).mean(dim=2)
        confidence = nn.functional.softplus(peak - mean)
        edge = .02
        window = (torch.sigmoid((time[None, :] - onset[:, None]) / edge)
                  * torch.sigmoid((onset[:, None] + span - time[None, :]) / edge))
        weights = confidence * window * active[None, :]
        basis = self._hat_basis(fraction, self.knots)
        for _ in range(self.irls_iterations):
            fitted = self._fit_polyline(positions, fraction, weights, knots=self.knots)
            path = torch.einsum("btk,bkc->btc", basis, fitted)
            residual = torch.linalg.vector_norm(positions - path, dim=2)
            # Standard IRLS: the reweighting is treated as a constant so
            # gradients flow through the solve, not through the weight update.
            huber = (self.huber_delta / residual.clamp_min(1e-6)).clamp(max=1.).detach()
            weights = weights * huber
        return self._fit_polyline(positions, fraction, weights, knots=self.knots)

    @staticmethod
    def _read_track_endpoints(scores: torch.Tensor, *, start_centre: torch.Tensor,
                              end_centre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode path-map contact positions, then select start and liftoff.

        The trajectory map has one spatial softmax per frame (not one global
        space-time softmax), matching its manifest supervision.  Known aligned
        timing windows then reduce those contact positions to the two command
        endpoints without asking the endpoint heads to represent every point
        on the path.
        """
        batch, steps = scores.shape[:2]
        positions = BasicLinearRegressor._frame_positions(scores)
        time = torch.linspace(0., 1., steps, dtype=scores.dtype, device=scores.device)

        def select(centre: torch.Tensor, sigma: float) -> torch.Tensor:
            temporal = torch.softmax(-((time[None, :] - centre[:, None]) / sigma).square(), dim=1)
            return (positions * temporal[:, :, None]).sum(dim=1)

        if start_centre.shape != (batch,) or end_centre.shape != (batch,):
            raise ValueError("track endpoint centres must have shape [batch]")
        return select(start_centre, .05), select(end_centre, .15)

    def _forward_scores(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [batch,time,3,height,width]")
        batch, steps = frames.shape[:2]
        reference = frames[:, :max(1, round(steps * .22))].mean(dim=1, keepdim=True)
        difference = torch.abs(frames - reference)
        encoded = self.encoder(torch.cat((frames, difference), dim=2).flatten(0, 1))
        h, w = encoded.shape[-2:]
        if self.temporal_mixer is not None:
            encoded_5d = encoded.reshape(batch, steps, encoded.shape[1], h, w).transpose(1, 2)
            encoded = (encoded_5d + self.temporal_mixer(encoded_5d)).transpose(1, 2).flatten(0, 1)
        start_scores = self.start_score(encoded).reshape(batch, steps, h, w)
        end_scores = self.end_score(encoded).reshape(batch, steps, h, w)
        trajectory_scores = (self.trajectory_score(encoded).reshape(batch, steps, h, w)
                             if self.trajectory_score is not None else None)
        # The pre-touch reference is the leading ~22% of every aligned clip.
        # Suppress that known no-gesture region, then make only a gentle learned
        # temporal preference.  Strong priors previously let background pixels in
        # the pre-touch prefix dominate early-endpoint attention.
        time = torch.linspace(0., 1., steps, dtype=frames.dtype, device=frames.device)
        active = torch.where(time < .18, torch.full_like(time, -12.0), torch.zeros_like(time))
        evidence = torch.maximum(start_scores, end_scores)
        series = torch.stack((evidence.amax((2, 3)), evidence.mean((2, 3))), dim=1)
        duration = BASIC_LINEAR_MIN_S + torch.sigmoid(self.duration_head(series)) * (BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S)
        # The trace is visible from command onset to liftoff, while later frames
        # contain only board/camera motion.  The old ``+time`` end prior was
        # therefore systematically attracted to post-liftoff pixels.  The
        # direct clip window is -0.5s..~1.77s, so onset is near 0.24 and an
        # estimated liftoff is onset + duration / 2.27.  A frozen-checkpoint
        # ablation showed the start location is sharply concentrated at onset;
        # use that information during training instead of asking the head to
        # choose a start among the entire accumulated trace.
        if self.line_fit_enabled:
            assert self.onset_head is not None and trajectory_scores is not None
            # Onset is learned per clip and bounded to the plausible first half
            # of the window; duration is shared with the head above, so the two
            # jointly define the interval the fit regresses over.
            fitted_onset = torch.sigmoid(self.onset_head(series))[:, 0] * .5
            fitted = self._line_fit_endpoints(
                trajectory_scores, onset=fitted_onset,
                duration_norm=duration[:, 0] / CLIP_WINDOW_S,
                active=(time >= .18).to(dtype=frames.dtype),
            )
            # The least-squares solve leaves non-standard strides; downstream
            # losses reshape the prediction, so hand back a contiguous tensor.
            prediction = torch.cat((fitted.flatten(1), duration), dim=1).contiguous()
            return prediction, start_scores, end_scores, trajectory_scores
        onset = time.new_full((batch,), self.start_onset)
        # Start and end timing use separate anchors.  A start-only held-out
        # sweep may select an early/broad prior; it must not shift the end map
        # into the known pre-touch region as a side effect.
        liftoff = (time.new_full((batch,), self.end_onset) + duration[:, 0] / 2.27).clamp(max=0.88)
        start_prior = active - ((time[None, :] - onset[:, None]) / self.start_sigma).square()
        end_prior = active - ((time[None, :] - liftoff[:, None]) / 0.15).square()
        x0, y0 = self._read_xy(start_scores, start_prior)
        x1, y1 = self._read_xy(end_scores, end_prior)
        if trajectory_scores is not None:
            track_start, track_end = self._read_track_endpoints(
                trajectory_scores, start_centre=onset, end_centre=liftoff,
            )
            blend = torch.sigmoid(self.trajectory_fusion)
            x0, y0 = ((1. - blend) * x0 + blend * track_start[:, 0],
                      (1. - blend) * y0 + blend * track_start[:, 1])
            x1, y1 = ((1. - blend) * x1 + blend * track_end[:, 0],
                      (1. - blend) * y1 + blend * track_end[:, 1])
        prediction = torch.cat((x0[:, None], y0[:, None], x1[:, None], y1[:, None], duration), dim=1)
        return prediction, start_scores, end_scores, trajectory_scores

    def forward_with_scores(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the gesture plus effective start/end score maps for inspection."""
        prediction, start_scores, end_scores, _trajectory_scores = self._forward_scores(frames)
        return prediction, start_scores, end_scores

    def forward_with_track_scores(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return endpoint maps plus a separately supervised moving-contact map."""
        prediction, start_scores, end_scores, trajectory_scores = self._forward_scores(frames)
        if trajectory_scores is None:
            raise RuntimeError("trajectory track was not enabled for this regressor")
        return prediction, start_scores, end_scores, trajectory_scores

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.forward_with_scores(frames)[0]

    @torch.no_grad()
    def predict_linear(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self(frames)
        # The line fit solves for endpoints and may extrapolate slightly past
        # the screen when a clip's evidence covers only part of the path.  Clamp
        # on the inference boundary only: projecting onto the feasible screen
        # can only reduce error against an in-range command, while clamping
        # inside ``forward`` would zero the gradient of a saturated endpoint
        # during training.
        coordinates = value[:, :4].clamp(0., 1.)
        return {"x0": coordinates[:, 0], "y0": coordinates[:, 1],
                "x1": coordinates[:, 2], "y1": coordinates[:, 3], "dur": value[:, 4]}
