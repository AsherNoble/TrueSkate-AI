"""Tiny CNN that answers: is this frame an active True Skate skatepark?

Used as a safety guard during 24h runs to catch the case where the agent has
left the park (home screen, pause menu, another app) and would otherwise keep
firing swipes into the void. See experiments/scene_classifier_journal.md.

Two entry points:
    SceneClassifier — the nn.Module (train with scripts/train/train_scene_classifier.py).
    SceneGuard      — runtime wrapper; loads a checkpoint if configured, else
                      stays disabled so callers pay zero cost until a model exists.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

INPUT_SIZE = 64  # square grayscale input


class SceneClassifier(nn.Module):
    """3-conv-block binary classifier over a 1×64×64 grayscale frame."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),  # single logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def preprocess(img) -> torch.Tensor:
    """Convert any frame to a normalized 1×1×64×64 tensor.

    Accepts a PIL.Image, a path/str, or a numpy array (H×W grayscale or H×W×C).
    """
    if isinstance(img, (str, Path)):
        pil = Image.open(img)
    elif isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil = img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    pil = pil.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None, :, :]  # (1,1,H,W)


class SceneGuard:
    """Runtime wrapper around a trained SceneClassifier.

    Disabled (a no-op returning None) unless a checkpoint is provided and loads.
    This keeps the eval hot-path free until a model is trained and enabled.
    """

    def __init__(
        self, model_path: str | os.PathLike | None = None,
        *, threshold: float = 0.5, device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.device = device
        self.model: SceneClassifier | None = None
        self.model_path = str(model_path) if model_path else None
        if self.model_path and Path(self.model_path).exists():
            try:
                model = SceneClassifier()
                state = torch.load(self.model_path, map_location=device)
                model.load_state_dict(state)
                model.eval().to(device)
                self.model = model
                logging.info("SceneGuard enabled (model=%s).", self.model_path)
            except Exception as exc:
                logging.warning("SceneGuard failed to load %s: %s", self.model_path, exc)
        elif self.model_path:
            logging.warning("SceneGuard model not found: %s", self.model_path)

    @classmethod
    def from_env(cls, *, threshold: float | None = None) -> "SceneGuard":
        """Build from SCENE_GUARD_MODEL / SCENE_GUARD_THRESHOLD env vars."""
        thr = threshold
        if thr is None:
            thr = float(os.environ.get("SCENE_GUARD_THRESHOLD", "0.5"))
        return cls(os.environ.get("SCENE_GUARD_MODEL"), threshold=thr)

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def probability(self, img) -> float | None:
        """P(in skatepark) for one frame, or None if disabled / on error."""
        if self.model is None:
            return None
        try:
            with torch.no_grad():
                x = preprocess(img).to(self.device)
                return float(torch.sigmoid(self.model(x)).item())
        except Exception as exc:
            logging.warning("SceneGuard inference failed: %s", exc)
            return None

    def in_skatepark(self, img) -> bool | None:
        """True/False if a verdict is available, else None (disabled/error)."""
        p = self.probability(img)
        if p is None:
            return None
        return p >= self.threshold
