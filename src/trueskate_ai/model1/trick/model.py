"""Model 1 MVP v0 for trick clips: curved v1's network with three per-frame contact signals."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from trueskate_ai.model1.curved.model import CurvedModel

SIGNALS = ("contact", "down", "up")
EVENT_POS_WEIGHT = 4.  # touch-down and lift frames are about 2% of all frames


class TrickModel(CurvedModel):
    """CurvedModel's picture encoder, time layer and finger heatmap, unchanged.

    Its one touch-down/lift choice per clip becomes three yes/no signals per frame (in
    contact, touch-down here, lift here), so a clip can hold several strokes. The signal
    head also sees each frame's strongest feature, because a new trail is small, and a
    dilated layer widens its view to about seven frames either side.
    """

    def __init__(self, base_channels=16):
        super().__init__(base_channels)
        c = base_channels
        del self.boundaries
        self.signals = nn.Sequential(
            nn.Conv1d(8*c+2, 4*c, 5, padding=2), nn.SiLU(),
            nn.Conv1d(4*c, 4*c, 5, padding=4, dilation=2), nn.SiLU(),
            nn.Conv1d(4*c, len(SIGNALS), 1))

    def forward(self, video, times):
        video = video.float()/255 if video.dtype == torch.uint8 else video
        b, t, _, h, w = video.shape
        difference = torch.cat([torch.zeros_like(video[:, :1]), (video[:, 1:]-video[:, :-1]).abs()], dim=1)
        features = self.encoder(torch.cat([video, difference], dim=2).reshape(b*t, 6, h, w))
        _, channels, fh, fw = features.shape
        features = features.reshape(b, t, channels, fh, fw).permute(0, 2, 1, 3, 4)
        features = features+self.temporal(features)
        heatmap = self.position(features).squeeze(1)/.15
        probabilities = heatmap.flatten(2).softmax(-1).reshape(b, t, fh, fw)
        x = torch.linspace(0, 1, fw, device=video.device)
        y = torch.linspace(0, 1, fh, device=video.device)
        xy = torch.stack([(probabilities.sum(2)*x).sum(-1),
                          (probabilities.sum(3)*y).sum(-1)], dim=-1)
        dt = torch.cat([torch.zeros_like(times[:, :1]), times[:, 1:]-times[:, :-1]], dim=1)
        context = torch.cat([features.mean((3, 4)), features.amax((3, 4)), times[:, None], dt[:, None]], dim=1)
        signals = self.signals(context)
        return dict(xy=xy, heatmap=heatmap, **{name: signals[:, i] for i, name in enumerate(SIGNALS)})


def loss_function(pred, batch):
    """Contact and event cross-entropy, plus curved v1's position loss on contact frames."""
    mask = batch["mask"]
    contact = F.binary_cross_entropy_with_logits(pred["contact"], batch["contact"])
    weight = torch.tensor(EVENT_POS_WEIGHT, device=mask.device)
    events = sum(F.binary_cross_entropy_with_logits(pred[k], batch[k], pos_weight=weight) for k in ("down", "up"))
    position = F.smooth_l1_loss(pred["xy"][mask]/.03, batch["xy"][mask]/.03)
    h, w = pred["heatmap"].shape[-2:]
    x = torch.linspace(0, 1, w, device=mask.device)
    y = torch.linspace(0, 1, h, device=mask.device)
    target = batch["xy"]
    squared = (x-target[..., 0, None, None]).square()+(y[:, None]-target[..., 1, None, None]).square()
    heat_target = (-squared/(2*.035**2)).flatten(2).softmax(-1)
    heat_ce = -(heat_target*pred["heatmap"].flatten(2).log_softmax(-1)).sum(-1)[mask].mean()
    return contact+events+position+.1*heat_ce
