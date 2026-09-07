"""Detect True Skate's replay / camera-settings menu vs live skatepark gameplay.

The random-gesture SLS collector can tap True Skate into REPLAY mode and get stuck
there for a long stretch — the park is still visible behind a replay, so the frames
*look* like gameplay at a glance, but the board isn't being driven and the gestures
just poke the replay UI. Those ``(frame, random-gesture)`` pairs are noise for a
frame->gesture model.

The reliable discriminator is the **bottom button bar** (red ``BACK`` + teal
``SHARE``/``HIDE``/``CAMERA``) that replay/menu shows and live gameplay never does.
We score the saturated red and teal fraction of the bottom ~10% strip. Validated on
known frames: live gameplay ~= (0.0, 0.0); replay/menu ~= (0.12, 0.12) — a clean,
resolution-independent split (works on the full Appium screenshot and on the 512px
aligned frames alike). This is a cheap heuristic, not the (untrained) SceneGuard CNN.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

# A button-bar button covers a few % of the bottom strip; gameplay is ~0. Require BOTH
# red AND teal present (a single coloured park element is one colour, not a bar).
_RED_THRESH = 0.02
_TEAL_THRESH = 0.02


def _to_rgb01(img) -> np.ndarray:
    """Normalise PNG bytes / path / PIL.Image / RGB ndarray to float RGB in [0, 1]."""
    if isinstance(img, np.ndarray):
        a = img.astype(np.float32)
        return (a / 255.0 if a.max() > 1.0 else a)[..., :3]
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    elif isinstance(img, (str, Path)):
        img = Image.open(img)
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def menu_bar_score(img) -> tuple[float, float]:
    """(red_fraction, teal_fraction) of the bottom strip — the replay/menu button bar."""
    a = _to_rgb01(img)
    h = a.shape[0]
    strip = a[int(0.90 * h):, :, :]
    r, g, b = strip[..., 0], strip[..., 1], strip[..., 2]
    sat = strip.max(-1) - strip.min(-1)
    red = (sat > 0.25) & (r > 0.5) & (r - g > 0.18) & (r - b > 0.18)
    teal = (sat > 0.18) & (g > 0.40) & (b > 0.40) & (g - r > 0.05) & (b - r > 0.05)
    return float(red.mean()), float(teal.mean())


def is_menu_frame(img) -> bool:
    """True if the frame is True Skate's replay/menu (NOT live skatepark gameplay)."""
    rf, tf = menu_bar_score(img)
    return rf > _RED_THRESH and tf > _TEAL_THRESH


def is_gameplay_frame(img) -> bool:
    """True if the frame is live skatepark gameplay (no replay/menu button bar)."""
    return not is_menu_frame(img)
