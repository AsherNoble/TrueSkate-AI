"""Per-frame heatmaps and contact boundaries with unrestricted clip onset."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CurvedModel(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(6, c, 5, padding=2), nn.GroupNorm(1, c), nn.SiLU(),
            nn.Conv2d(c, 2*c, 3, stride=2, padding=1), nn.GroupNorm(2, 2*c), nn.SiLU(),
            nn.Conv2d(2*c, 4*c, 3, padding=1), nn.GroupNorm(4, 4*c), nn.SiLU())
        self.temporal = nn.Sequential(
            nn.Conv3d(4*c, 4*c, (3, 1, 1), padding=(1, 0, 0), groups=4*c),
            nn.GroupNorm(4, 4*c), nn.SiLU(), nn.Conv3d(4*c, 4*c, 1))
        self.position = nn.Conv3d(4*c, 1, 1)
        self.boundaries = nn.Sequential(nn.Conv1d(4*c+2, 4*c, 5, padding=2), nn.SiLU(),
                                        nn.Conv1d(4*c, 2, 1))

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
        context = torch.cat([features.mean((3, 4)), times[:, None], dt[:, None]], dim=1)
        boundaries = self.boundaries(context)
        return dict(xy=xy, heatmap=heatmap, onset=boundaries[:, 0], release=boundaries[:, 1])


def loss_function(pred, batch):
    mask = batch["mask"]
    position = F.smooth_l1_loss(pred["xy"][mask]/.03, batch["xy"][mask]/.03)
    h, w = pred["heatmap"].shape[-2:]
    x = torch.linspace(0, 1, w, device=mask.device)
    y = torch.linspace(0, 1, h, device=mask.device)
    target = batch["xy"]
    squared = (x-target[..., 0, None, None]).square()+(y[:, None]-target[..., 1, None, None]).square()
    heat_target = (-squared/(2*.035**2)).flatten(2).softmax(-1)
    heat_ce = -(heat_target*pred["heatmap"].flatten(2).log_softmax(-1)).sum(-1)[mask].mean()
    return (F.cross_entropy(pred["onset"], batch["onset"])
            + F.cross_entropy(pred["release"], batch["release"])+position+.1*heat_ce)
