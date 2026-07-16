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

# Module-level configurable thresholds — tweak these when tuning the menu detector.
# Default values were validated on known frames (live gameplay ~= (0.0,0.0); replay/menu ~= (0.12,0.12)).
RED_THRESH: float = 0.02
"""Red fraction threshold for menu detection (module-level, configurable)."""

TEAL_THRESH: float = 0.02
"""Teal fraction threshold for menu detection (module-level, configurable)."""

# Backwards-compatible internal names used elsewhere in the module.
_RED_THRESH = RED_THRESH
_TEAL_THRESH = TEAL_THRESH


def _to_rgb01(img) -> np.ndarray:
    """Normalise PNG bytes / path / PIL.Image / RGB ndarray to float RGB in [0, 1].

    Validation and errors:
    - numpy arrays: must be at least 2-D and non-empty. Grayscale (2-D) arrays are
      promoted to RGB by stacking channels. Arrays with any zero dimension raise ValueError.
    - bytes / bytearray / path / str: IO errors are caught and reported as ValueError.
    - PIL Image inputs are accepted. Any other input types raise ValueError.

    Returns a (H, W, 3) float32 ndarray in range [0, 1].
    """
    # numpy input
    if isinstance(img, np.ndarray):
        if img.size == 0 or img.ndim < 2:
            raise ValueError("Empty or invalid numpy image array (expected ndim>=2 and non-zero size)")
        # Promote grayscale to RGB
        if img.ndim == 2:
            a = np.stack([img, img, img], axis=-1)
        else:
            a = img
        a = a.astype(np.float32)
        # Distinguish 0-1 floats vs 0-255 ints
        try:
            maxv = a.max()
        except ValueError:
            raise ValueError("Invalid numpy image array (could not compute max)")
        if maxv <= 1.0:
            out = a[..., :3]
        else:
            out = (a / 255.0)[..., :3]
        if out.size == 0:
            raise ValueError("Resulting image is empty after normalization")
        return out

    # bytes / path / PIL handling
    pil_img = None
    if isinstance(img, (bytes, bytearray)):
        try:
            pil_img = Image.open(io.BytesIO(img))
        except Exception as e:
            raise ValueError(f"Could not open image from bytes: {e}") from e
    elif isinstance(img, (str, Path)):
        try:
            pil_img = Image.open(img)
        except Exception as e:
            raise ValueError(f"Could not open image from path {img}: {e}") from e
    elif isinstance(img, Image.Image):
        pil_img = img
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    try:
        rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as e:
        raise ValueError(f"Error converting image to RGB: {e}") from e

    if rgb.size == 0 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        # Either raise or return a tiny safe fallback; raising lets callers decide.
        raise ValueError("Loaded image is empty or has zero width/height")

    return rgb


def menu_bar_score(img) -> tuple[float, float]:
    """(red_fraction, teal_fraction) of the bottom strip — the replay/menu button bar.

    For images with very small height (<10 pixels) this function treats the frame as
    non-menu and returns (0.0, 0.0) to avoid slicing errors and noisy statistics.
    """
    a = _to_rgb01(img)
    if a.ndim < 3 or a.shape[2] < 3:
        raise ValueError("Expected image with 3 colour channels (H, W, 3)")
    h = a.shape[0]
    # Guard tiny images: avoid slicing into an empty strip for very short images
    if h < 10:
        return 0.0, 0.0
    strip = a[int(0.90 * h):, :, :]
    if strip.size == 0:
        return 0.0, 0.0
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


def is_editor_frame(img) -> bool:
    """True if the frame is True Skate's park editor (NOT live gameplay)."""
    a = _to_rgb01(img)
    h, w = a.shape[:2]

    # Editor-unique stack near the bottom:
    # 1) toolbar row with dark buttons + red "SORT: ALL" / red icon accents
    # 2) obstacle carousel row with small white-heart badges on dark thumbnails
    toolbar = a[int(0.80 * h):int(0.88 * h), int(0.03 * w):int(0.97 * w), :]
    carousel = a[int(0.88 * h):int(0.96 * h), int(0.05 * w):int(0.95 * w), :]

    t_r, t_g, t_b = toolbar[..., 0], toolbar[..., 1], toolbar[..., 2]
    t_sat = toolbar.max(-1) - toolbar.min(-1)
    toolbar_red = (t_sat > 0.25) & (t_r > 0.50) & (t_r - t_g > 0.18) & (t_r - t_b > 0.18)
    toolbar_dark = toolbar.mean(-1) < 0.24

    c_r, c_g, c_b = carousel[..., 0], carousel[..., 1], carousel[..., 2]
    c_sat = carousel.max(-1) - carousel.min(-1)
    # Hearts are bright neutral white, unlike red ramps or park textures.
    carousel_white = (c_r > 0.83) & (c_g > 0.83) & (c_b > 0.83) & (c_sat < 0.18)
    carousel_dark = carousel.mean(-1) < 0.28

    return (
        float(toolbar_red.mean()) > 0.04
        and float(toolbar_dark.mean()) > 0.15
        and float(carousel_white.mean()) > 0.02
        and float(carousel_dark.mean()) > 0.30
    )


def _run_editor_self_test() -> int:
    base = Path("tmp/editor_detector/labeled")
    editor_files = sorted((base / "editor").glob("*.png"))
    gameplay_files = sorted((base / "gameplay").glob("*.png"))
    all_files = [(p, True) for p in editor_files] + [(p, False) for p in gameplay_files]

    if not all_files:
        print("No labeled frames found under tmp/editor_detector/labeled/")
        return 1

    correct = 0
    for p, expected in all_files:
        pred = is_editor_frame(p)
        ok = pred == expected
        correct += int(ok)
        print(
            f"{p.name:24s} expected={str(expected):5s} pred={str(pred):5s} "
            f"{'OK' if ok else 'FAIL'}"
        )

    total = len(all_files)
    acc = correct / total
    print(f"overall: {correct}/{total} = {acc:.3%}")
    return 0 if correct == total else 2


if __name__ == "__main__":
    raise SystemExit(_run_editor_self_test())
