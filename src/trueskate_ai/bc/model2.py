"""Model 2 — the vision-grounded gesture-sequence policy.

`(n recent frames + m past strokes) -> next strokes`. A CNN frame encoder + a
Transformer set-encoder with a readout query token (decision/behaviour-transformer
style), regressing a padded action group in normalised [0, 1] token space plus
a contiguous slot-activity prefix
(see `gesture_tokens.py`).

Why a *readout* encoder and not a per-step causal decoder: our action is already
a compact macro-stroke and we predict a short chunk per decision, then deploy
receding-horizon (predict m_out, execute a few, re-observe). So a single readout
step suffices for v1; swap in a causal decoder / CVAE / diffusion head later only
if mode-averaging shows up. Pure torch — no device deps. See the plan + journal.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from trueskate_ai.bc.gesture_tokens import STROKE_DIM


@dataclass
class SequencePolicyConfig:
    stroke_dim: int = STROKE_DIM
    n_frames: int = 6          # recent visual history (≈0.2 s at 30 fps)
    m_past: int = 4            # past strokes conditioned on
    m_out: int = 1             # maximum strokes in one overlapping action group
    img_ch: int = 3
    # Portrait, NOT square — the screen is ~2.16:1 (19.5:9) and must not be
    # squished to 1:1 (that distorts board/trace geometry the encoder reads).
    # 208×96 = 2.167:1, within 0.1% of the device 896/414 = 2.164:1; both ÷16.
    # FrameEncoder ends in AdaptiveAvgPool2d(1), so H≠W needs no arch change.
    img_h: int = 208
    img_w: int = 96
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1


class FrameEncoder(nn.Module):
    """Small CNN: (B, C, H, W) -> (B, d_model). Game UI is simple; keep it light."""

    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),   # H/2
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),      # H/4
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),     # H/8
            nn.Conv2d(128, d_model, 3, stride=2, padding=1), nn.ReLU(),  # H/16
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class SequencePolicy(nn.Module):
    """n frames + m past strokes -> next m_out strokes (normalised [0, 1])."""

    # token type ids
    _FRAME, _STROKE, _QUERY = 0, 1, 2

    def __init__(self, cfg: SequencePolicyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.frame_encoder = FrameEncoder(cfg.img_ch, d)
        self.stroke_embed = nn.Linear(cfg.stroke_dim, d)
        self.type_embed = nn.Embedding(3, d)
        self.frame_pos = nn.Embedding(max(1, cfg.n_frames), d)
        self.stroke_pos = nn.Embedding(max(1, cfg.m_past), d)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.query, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=d * cfg.ff_mult,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
        )
        # enable_nested_tensor=False: the padded-batch fast path calls
        # aten::_nested_tensor_from_mask_left_aligned, unimplemented on MPS (the
        # rig's device) — it only triggers in eval() with a key-padding mask, so
        # it crashed inference though training was fine. Disabling it is
        # numerically identical and removes that train/eval divergence.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.stroke_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, cfg.m_out * cfg.stroke_dim),
        )
        self.activity_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.m_out))

    def forward(
        self,
        frames: torch.Tensor,          # (B, n_frames, C, H, W)
        past_strokes: torch.Tensor,    # (B, m_past, stroke_dim) — normalised [0,1]
        past_mask: torch.Tensor | None = None,  # (B, m_past) True = real, False = pad
    ) -> tuple[torch.Tensor, torch.Tensor]:  # strokes [0,1], activity logits
        B, nf = frames.shape[0], frames.shape[1]
        d = self.cfg.d_model
        dev = frames.device

        # frame tokens
        fe = self.frame_encoder(frames.flatten(0, 1)).view(B, nf, d)
        fe = fe + self.type_embed(torch.full((1,), self._FRAME, device=dev)) \
                + self.frame_pos(torch.arange(nf, device=dev))[None]

        tokens = [fe]
        key_padding = [torch.zeros(B, nf, dtype=torch.bool, device=dev)]  # False = attend

        mp = past_strokes.shape[1]
        if mp > 0:
            se = self.stroke_embed(past_strokes) \
                + self.type_embed(torch.full((1,), self._STROKE, device=dev)) \
                + self.stroke_pos(torch.arange(mp, device=dev))[None]
            tokens.append(se)
            pad = (~past_mask) if past_mask is not None else torch.zeros(B, mp, dtype=torch.bool, device=dev)
            key_padding.append(pad)

        q = self.query.expand(B, 1, d) + self.type_embed(torch.full((1,), self._QUERY, device=dev))
        tokens.append(q)
        key_padding.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))

        seq = torch.cat(tokens, dim=1)
        kpm = torch.cat(key_padding, dim=1)
        out = self.encoder(seq, src_key_padding_mask=kpm)
        readout = out[:, -1]  # the query token
        pred = self.stroke_head(readout).view(B, self.cfg.m_out, self.cfg.stroke_dim)
        return torch.sigmoid(pred), self.activity_head(readout)


def build_policy(cfg: SequencePolicyConfig | None = None) -> SequencePolicy:
    return SequencePolicy(cfg or SequencePolicyConfig())


def stroke_loss(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor,
                activity_logits: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Masked stroke regression plus supervised contiguous slot activity."""
    error = (pred - target) ** 2
    if weights is not None:
        error = error * weights
    mask = target_mask.unsqueeze(-1).to(error.dtype)
    regression = (error * mask).sum() / (mask.sum() * pred.shape[-1]).clamp_min(1)
    activity = torch.nn.functional.binary_cross_entropy_with_logits(
        activity_logits, target_mask.to(activity_logits.dtype))
    return regression + activity


def config_to_dict(cfg: SequencePolicyConfig) -> dict:
    return asdict(cfg)
