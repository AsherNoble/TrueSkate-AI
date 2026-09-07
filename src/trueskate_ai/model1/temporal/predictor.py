"""Causal temporal touch-trace predictor for Model 1.

The rendered True Skate trace is a short history of the touch path, so a
single RGB frame is ambiguous: the current finger is normally at one end of
the trace, but crossings and multiple simultaneous touches make that endpoint
hard to identify.  This module treats Model 1 as a causal tracker instead of an
independent frame classifier.

At time ``t`` the model consumes only:

* RGB frame ``t``;
* the heatmap predicted (or teacher-forced) at ``t - 1``;
* recurrent state produced at ``t - 1``; and
* optionally, the elapsed time since the preceding frame.

The spatial heatmap uses independent sigmoid outputs rather than a spatial
softmax.  It can therefore represent multiple simultaneous touch peaks (for
example, a moving flick plus the held spin control).

The existing :class:`GaussianBumpPredictor` remains the non-temporal Model 1
baseline.  Checkpoints for that model are intentionally not compatible with
this architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest useful GroupNorm group count for ``channels``."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvNormAct(nn.Module):
    """A small convolutional block that is stable for batch size one."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvGRUCell(nn.Module):
    """A convolutional GRU cell that preserves the feature-map geometry."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        combined_channels = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            combined_channels, 2 * hidden_channels, kernel_size, padding=padding
        )
        self.candidate = nn.Conv2d(
            combined_channels, hidden_channels, kernel_size, padding=padding
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or hidden.ndim != 4:
            raise ValueError("ConvGRU inputs must be BCHW tensors")
        if hidden.shape[0] != x.shape[0] or hidden.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"hidden shape {tuple(hidden.shape)} is incompatible with input "
                f"shape {tuple(x.shape)}"
            )
        reset, update = torch.sigmoid(
            self.gates(torch.cat((x, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((x, reset * hidden), dim=1))
        )
        return update * hidden + (1.0 - update) * candidate


@dataclass(frozen=True)
class TemporalTraceConfig:
    """Architecture parameters stored alongside a temporal Model 1 checkpoint."""

    in_channels: int = 3
    base_channels: int = 16
    hidden_channels: int = 32
    downsample_stages: int = 2
    use_time_deltas: bool = True

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError("in_channels must be positive")
        if self.base_channels < 1:
            raise ValueError("base_channels must be positive")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if self.downsample_stages < 1:
            raise ValueError("downsample_stages must be at least 1")


@dataclass
class TemporalTraceState:
    """State carried from one causal prediction step to the next.

    ``previous_heatmap`` is deliberately part of the state rather than hidden
    inside the module.  Callers reset a gesture or clip by passing ``None`` to
    :meth:`TemporalTracePredictor.step`; separate streams can safely share one
    model instance.
    """

    hidden: torch.Tensor
    previous_heatmap: torch.Tensor

    def detach(self) -> "TemporalTraceState":
        """Detach state at a truncated-backpropagation boundary."""

        return TemporalTraceState(
            hidden=self.hidden.detach(),
            previous_heatmap=self.previous_heatmap.detach(),
        )


@dataclass
class TemporalTraceStepOutput:
    """Output of one causal timestep."""

    heatmap: torch.Tensor
    active_logits: torch.Tensor
    state: TemporalTraceState

    @property
    def active_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.active_logits)


@dataclass
class TemporalTraceSequenceOutput:
    """Outputs of an autoregressive sequence rollout."""

    heatmaps: torch.Tensor
    active_logits: torch.Tensor
    state: TemporalTraceState
    teacher_forcing_mask: torch.Tensor

    @property
    def active_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.active_logits)


class TemporalTracePredictor(nn.Module):
    """Lightweight causal RGB + previous-heatmap touch tracker.

    Args:
        in_channels: Number of current-frame image channels, normally three.
        base_channels: Width of the full-resolution encoder stage.  Eight is a
            useful smoke/edge configuration; 16 is the default training width.
        hidden_channels: Width of the spatial ConvGRU state.
        downsample_stages: Number of 2x encoder reductions.  The decoder always
            returns a heatmap at the exact input resolution, including odd sizes.
        use_time_deltas: Append a constant ``delta_t`` plane to each step.  This
            lets the tracker account for variable capture FPS.  Omitting a delta
            at call time supplies zero rather than a future-derived value.

    ``forward`` is the sequence API used for training.  Scheduled sampling can
    be controlled either by a probability or by a deterministic mask.  For the
    prediction at ``t``, a true mask value selects target heatmap ``t - 1``;
    it never selects target ``t`` or any future target.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        hidden_channels: int = 32,
        downsample_stages: int = 2,
        use_time_deltas: bool = True,
    ):
        super().__init__()
        self.config = TemporalTraceConfig(
            in_channels=in_channels,
            base_channels=base_channels,
            hidden_channels=hidden_channels,
            downsample_stages=downsample_stages,
            use_time_deltas=use_time_deltas,
        )

        # One channel is the previous heatmap.  A scalar time delta is expanded
        # into one additional image-sized plane when enabled.
        encoder_in = in_channels + 1 + int(use_time_deltas)
        stage_channels = [base_channels * (2**i) for i in range(downsample_stages + 1)]
        self._stage_channels = stage_channels

        self.stem = _ConvNormAct(encoder_in, stage_channels[0])
        self.down_blocks = nn.ModuleList(
            _ConvNormAct(stage_channels[i], stage_channels[i + 1], stride=2)
            for i in range(downsample_stages)
        )
        encoded_channels = stage_channels[-1]
        self.recurrent = ConvGRUCell(encoded_channels, hidden_channels)

        self.bottleneck_projection = _ConvNormAct(hidden_channels, encoded_channels)
        decoder_blocks = []
        current_channels = encoded_channels
        # The deepest encoded tensor has already entered the ConvGRU.  Decode
        # against the shallower encoder skips only.
        for skip_channels in reversed(stage_channels[:-1]):
            decoder_blocks.append(_ConvNormAct(current_channels + skip_channels, skip_channels))
            current_channels = skip_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.heatmap_head = nn.Conv2d(current_channels, 1, kernel_size=1)

        activity_hidden = max(4, hidden_channels // 2)
        self.activity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, activity_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(activity_hidden, 1),
        )

    @classmethod
    def from_config(cls, config: TemporalTraceConfig) -> "TemporalTracePredictor":
        """Construct a model from checkpointed architecture metadata."""

        return cls(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            hidden_channels=config.hidden_channels,
            downsample_stages=config.downsample_stages,
            use_time_deltas=config.use_time_deltas,
        )

    def _prepare_delta(
        self, frame: torch.Tensor, delta_t: Optional[torch.Tensor | float]
    ) -> Optional[torch.Tensor]:
        if not self.config.use_time_deltas:
            if delta_t is not None:
                raise ValueError("delta_t was supplied but use_time_deltas=False")
            return None
        batch, _, height, width = frame.shape
        if delta_t is None:
            values = frame.new_zeros((batch, 1, 1, 1))
        elif isinstance(delta_t, (float, int)):
            values = frame.new_full((batch, 1, 1, 1), float(delta_t))
        else:
            values = delta_t.to(device=frame.device, dtype=frame.dtype)
            if values.ndim == 0:
                values = values.expand(batch)
            if values.shape == (batch,):
                values = values[:, None, None, None]
            elif values.shape == (batch, 1):
                values = values[:, :, None, None]
            elif values.shape != (batch, 1, 1, 1):
                raise ValueError(
                    f"delta_t must be scalar, [B], [B,1], or [B,1,1,1], got "
                    f"{tuple(values.shape)}"
                )
        return values.expand(batch, 1, height, width)

    @staticmethod
    def _prepare_feedback(frame: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        if feedback.ndim != 4 or feedback.shape[1] != 1:
            raise ValueError(
                f"feedback heatmap must have shape [B,1,H,W], got {tuple(feedback.shape)}"
            )
        if feedback.shape[0] != frame.shape[0]:
            raise ValueError("feedback and frame batch sizes differ")
        feedback = feedback.to(device=frame.device, dtype=frame.dtype)
        if feedback.shape[-2:] != frame.shape[-2:]:
            feedback = F.interpolate(
                feedback, size=frame.shape[-2:], mode="bilinear", align_corners=False
            )
        return feedback

    def step(
        self,
        frame: torch.Tensor,
        state: Optional[TemporalTraceState] = None,
        *,
        feedback_heatmap: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor | float] = None,
    ) -> TemporalTraceStepOutput:
        """Predict one frame using only causal state and previous feedback.

        ``feedback_heatmap`` overrides ``state.previous_heatmap``.  This is the
        hook used by teacher forcing and scheduled sampling.  With neither a
        state nor explicit feedback, the history is an all-zero cold start.
        """

        if frame.ndim != 4:
            raise ValueError(f"frame must have shape [B,C,H,W], got {tuple(frame.shape)}")
        if frame.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} RGB channels, got {frame.shape[1]}"
            )
        batch, _, height, width = frame.shape
        if feedback_heatmap is None:
            feedback_heatmap = (
                state.previous_heatmap
                if state is not None
                else frame.new_zeros((batch, 1, height, width))
            )
        feedback = self._prepare_feedback(frame, feedback_heatmap)

        inputs = [frame, feedback]
        delta_plane = self._prepare_delta(frame, delta_t)
        if delta_plane is not None:
            inputs.append(delta_plane)

        encoded = self.stem(torch.cat(inputs, dim=1))
        skips = [encoded]
        for block in self.down_blocks:
            encoded = block(encoded)
            skips.append(encoded)

        if state is None:
            hidden = encoded.new_zeros(
                (batch, self.config.hidden_channels, *encoded.shape[-2:])
            )
        else:
            hidden = state.hidden
            expected = (batch, self.config.hidden_channels, *encoded.shape[-2:])
            if tuple(hidden.shape) != expected:
                raise ValueError(
                    f"state hidden shape {tuple(hidden.shape)} does not match expected {expected}; "
                    "reset state when batch size or frame geometry changes"
                )
            hidden = hidden.to(device=encoded.device, dtype=encoded.dtype)

        next_hidden = self.recurrent(encoded, hidden)
        decoded = self.bottleneck_projection(next_hidden)
        for block, skip in zip(self.decoder_blocks, reversed(skips[:-1])):
            decoded = F.interpolate(
                decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = block(torch.cat((decoded, skip), dim=1))
        # Independent sigmoid pixels allow zero, one, or several spatial peaks.
        heatmap = torch.sigmoid(self.heatmap_head(decoded))
        if heatmap.shape[-2:] != (height, width):
            heatmap = F.interpolate(
                heatmap, size=(height, width), mode="bilinear", align_corners=False
            )
        active_logits = self.activity_head(next_hidden).squeeze(-1)
        next_state = TemporalTraceState(hidden=next_hidden, previous_heatmap=heatmap)
        return TemporalTraceStepOutput(
            heatmap=heatmap,
            active_logits=active_logits,
            state=next_state,
        )

    @staticmethod
    def _normalise_teacher_mask(
        mask: torch.Tensor, batch: int, steps: int, device: torch.device
    ) -> torch.Tensor:
        mask = mask.to(device=device, dtype=torch.bool)
        if tuple(mask.shape) == (batch, max(0, steps - 1)):
            mask = torch.cat(
                (torch.zeros((batch, 1), dtype=torch.bool, device=device), mask), dim=1
            )
        elif tuple(mask.shape) != (batch, steps):
            raise ValueError(
                f"teacher_forcing_mask must be [B,T] or [B,T-1], got {tuple(mask.shape)}"
            )
        # There is no target at t=-1, so teacher forcing the first step is not
        # meaningful even if a caller accidentally marks it true.
        if steps:
            mask = torch.cat((torch.zeros_like(mask[:, :1]), mask[:, 1:]), dim=1)
        return mask

    def forward(
        self,
        frames: torch.Tensor,
        *,
        teacher_heatmaps: Optional[torch.Tensor] = None,
        teacher_forcing_probability: float = 0.0,
        teacher_forcing_mask: Optional[torch.Tensor] = None,
        initial_state: Optional[TemporalTraceState] = None,
        delta_times: Optional[torch.Tensor] = None,
        detach_feedback: bool = False,
    ) -> TemporalTraceSequenceOutput:
        """Roll out a causal sequence, optionally with scheduled feedback.

        Args:
            frames: RGB tensor ``[B,T,C,H,W]`` in chronological order.
            teacher_heatmaps: Ground-truth heatmaps ``[B,T,1,H,W]``.  At step
                ``t`` only element ``t - 1`` can be consumed.
            teacher_forcing_probability: Probability of ground-truth feedback
                independently sampled per batch item and transition.  A trainer
                can decay this value between epochs for scheduled sampling.
            teacher_forcing_mask: Deterministic bool mask ``[B,T]`` or
                ``[B,T-1]``.  It takes precedence over the probability and is
                useful for reproducible tests or custom schedules.
            initial_state: Optional history for clips that intentionally begin
                mid-gesture.  Pass ``None`` for a real cold start/reset.
            delta_times: Optional elapsed seconds ``[B,T]``; never derived from
                future frames inside this model.
            detach_feedback: Detach predicted heatmaps before feeding the next
                step, while retaining recurrent-state BPTT.
        """

        if frames.ndim != 5:
            raise ValueError(
                f"frames must have shape [B,T,C,H,W], got {tuple(frames.shape)}"
            )
        batch, steps, channels, height, width = frames.shape
        if steps < 1:
            raise ValueError("frames must contain at least one timestep")
        if channels != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} RGB channels, got {channels}"
            )
        if not 0.0 <= teacher_forcing_probability <= 1.0:
            raise ValueError("teacher_forcing_probability must be in [0,1]")
        if teacher_heatmaps is not None:
            expected = (batch, steps, 1, height, width)
            if tuple(teacher_heatmaps.shape) != expected:
                raise ValueError(
                    f"teacher_heatmaps must have shape {expected}, got "
                    f"{tuple(teacher_heatmaps.shape)}"
                )
        if delta_times is not None and tuple(delta_times.shape) != (batch, steps):
            raise ValueError(
                f"delta_times must have shape [B,T], got {tuple(delta_times.shape)}"
            )

        if teacher_forcing_mask is not None:
            selected_teacher = self._normalise_teacher_mask(
                teacher_forcing_mask, batch, steps, frames.device
            )
        elif teacher_forcing_probability == 0.0:
            selected_teacher = torch.zeros(
                (batch, steps), dtype=torch.bool, device=frames.device
            )
        elif teacher_forcing_probability == 1.0:
            selected_teacher = torch.ones(
                (batch, steps), dtype=torch.bool, device=frames.device
            )
            selected_teacher[:, 0] = False
        else:
            selected_teacher = (
                torch.rand((batch, steps), device=frames.device)
                < teacher_forcing_probability
            )
            selected_teacher[:, 0] = False
        if selected_teacher.any() and teacher_heatmaps is None:
            raise ValueError("teacher feedback was selected but teacher_heatmaps is None")

        state = initial_state
        predicted_heatmaps = []
        active_logits = []
        for t in range(steps):
            feedback = None
            if t > 0:
                assert state is not None
                predicted_feedback = state.previous_heatmap
                if detach_feedback:
                    predicted_feedback = predicted_feedback.detach()
                if teacher_heatmaps is not None:
                    use_teacher = selected_teacher[:, t, None, None, None]
                    feedback = torch.where(
                        use_teacher,
                        teacher_heatmaps[:, t - 1].to(
                            device=frames.device, dtype=frames.dtype
                        ),
                        predicted_feedback,
                    )
                else:
                    feedback = predicted_feedback
            delta = delta_times[:, t] if delta_times is not None else None
            output = self.step(
                frames[:, t], state, feedback_heatmap=feedback, delta_t=delta
            )
            state = output.state
            predicted_heatmaps.append(output.heatmap)
            active_logits.append(output.active_logits)

        assert state is not None
        return TemporalTraceSequenceOutput(
            heatmaps=torch.stack(predicted_heatmaps, dim=1),
            active_logits=torch.stack(active_logits, dim=1),
            state=state,
            teacher_forcing_mask=selected_teacher,
        )


__all__ = [
    "ConvGRUCell",
    "TemporalTraceConfig",
    "TemporalTracePredictor",
    "TemporalTraceSequenceOutput",
    "TemporalTraceState",
    "TemporalTraceStepOutput",
]
