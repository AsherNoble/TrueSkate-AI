"""Detect True Skate menus vs live skatepark gameplay.

The random-gesture SLS collector can tap True Skate into REPLAY mode and get stuck
there for a long stretch — the park is still visible behind a replay, so the frames
*look* like gameplay at a glance, but the board isn't being driven and the gestures
just poke the replay UI. Those ``(frame, random-gesture)`` pairs are noise for a
frame->gesture model.

Two bottom-bar signatures are reliable:

* replay/camera menus have saturated red ``BACK`` plus teal action buttons;
* the app hub has five repeated neutral-gray navigation cells (``ME``,
  ``SKATEPARKS``, ``COMMUNITY``, ``SHOP``, ``SETTINGS``) on a dark band.

Both checks are resolution-independent.  The hub check deliberately requires the
repeated signature in at least four fifths of the screen, so a dark park, the home
indicator, or the gameplay speedometer alone cannot trigger it.  These are cheap
heuristics, not the (untrained) SceneGuard CNN.
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

# Main-hub bottom navigation.  The strip is divided into the five equal-width UI
# cells.  A cell must contain both a mostly-dark background and enough neutral-gray
# icon/text pixels; requiring four cells makes the detector conservative on dark
# gameplay scenes with a home indicator or speedometer in only one/two cells.
_HUB_STRIP_Y = 0.90
_HUB_DARK_MAX = 0.22
_HUB_NEUTRAL_MIN = 0.45
_HUB_NEUTRAL_SAT_MAX = 0.20
_HUB_DARK_FRAC = 0.42
_HUB_NEUTRAL_FRAC = 0.08
_HUB_MIN_CELLS = 4

# Bolt Challenges center modal ("COMPLETE THE STREAKS" dialog). A random gesture
# opens it and, being a CENTER dialog, it is invisible to the bottom-bar menu
# detector, so it lingers and contaminates whole collection runs — leaking into
# Model 1 training with garbage labels.  The modal is a static light panel filling
# the frame's central band, so the near-white fraction there is cleanly bimodal:
# validated clean gameplay ~=0.00 (12 samples, 4 sessions <=0.016) vs a modal-stuck
# session ~=0.61 (every frame).  See memory bolt-challenges-modal-contamination.
_BOLT_ROI = (0.18, 0.82, 0.33, 0.74)  # x0, x1, y0, y1 (normalised) — the dialog body
_BOLT_WHITE_MIN = 0.855               # per-channel near-white (~218/255)
BOLT_MODAL_THRESH: float = 0.15       # central white-fraction above this => modal


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


def _menu_bar_score_rgb(a: np.ndarray) -> tuple[float, float]:
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


def menu_bar_score(img) -> tuple[float, float]:
    """(red_fraction, teal_fraction) of the bottom strip — the replay/menu button bar.

    For images with very small height (<10 pixels) this function treats the frame as
    non-menu and returns (0.0, 0.0) to avoid slicing errors and noisy statistics.
    """
    return _menu_bar_score_rgb(_to_rgb01(img))


def _hub_nav_score_rgb(a: np.ndarray) -> tuple[int, tuple[tuple[float, float], ...]]:
    """Score app-hub navigation cells in an already-normalized RGB frame."""
    if a.ndim < 3 or a.shape[2] < 3:
        raise ValueError("Expected image with 3 colour channels (H, W, 3)")
    h, w = a.shape[:2]
    if h < 10 or w < 5:
        return 0, ()

    strip = a[int(_HUB_STRIP_Y * h):, :, :3]
    scores: list[tuple[float, float]] = []
    hits = 0
    for cell in np.array_split(strip, 5, axis=1):
        high = cell.max(-1)
        low = cell.min(-1)
        dark_fraction = float((high < _HUB_DARK_MAX).mean())
        neutral_fraction = float(
            ((low > _HUB_NEUTRAL_MIN) &
             ((high - low) < _HUB_NEUTRAL_SAT_MAX)).mean()
        )
        scores.append((dark_fraction, neutral_fraction))
        if dark_fraction > _HUB_DARK_FRAC and neutral_fraction > _HUB_NEUTRAL_FRAC:
            hits += 1
    return hits, tuple(scores)


def hub_nav_score(img) -> tuple[int, tuple[tuple[float, float], ...]]:
    """Return the number and per-cell scores of app-hub navigation cells.

    Each per-cell tuple is ``(dark_fraction, neutral_gray_fraction)`` over the
    bottom 10% of the frame.  Public scores keep detector regressions inspectable
    without duplicating its pixel thresholds in diagnostics/tests.
    """
    return _hub_nav_score_rgb(_to_rgb01(img))


def is_menu_frame(img) -> bool:
    """True for replay/camera menus or the five-cell app hub, never gameplay."""
    a = _to_rgb01(img)
    rf, tf = _menu_bar_score_rgb(a)
    if rf > _RED_THRESH and tf > _TEAL_THRESH:
        return True
    hub_cells, _ = _hub_nav_score_rgb(a)
    return hub_cells >= _HUB_MIN_CELLS


def is_gameplay_frame(img) -> bool:
    """True if the frame is live skatepark gameplay (no recognized menu UI)."""
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


def bolt_modal_score(img) -> float:
    """Central near-white fraction — the Bolt Challenges modal-dialog signature.

    The modal is a static light panel filling the frame's central band. Clean
    gameplay leaves that band mid-toned (~0.00); the panel saturates it (~0.6).
    Complements ``is_menu_frame`` (which only reads the bottom bar and therefore
    never sees this center dialog). Kept public so detector regressions stay
    inspectable without duplicating the ROI/threshold in diagnostics/tests.
    """
    a = _to_rgb01(img)
    h, w = a.shape[:2]
    x0, x1, y0, y1 = _BOLT_ROI
    roi = a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w), :]
    if roi.size == 0:
        return 0.0
    return float((roi > _BOLT_WHITE_MIN).all(axis=-1).mean())


def is_bolt_modal_frame(img) -> bool:
    """True if the Bolt Challenges 'Complete the Streaks' modal covers gameplay.

    A center dialog the bottom-bar ``is_menu_frame`` cannot see; use both to
    exclude non-gameplay frames from collection and from Model 1 training.
    """
    return bolt_modal_score(img) > BOLT_MODAL_THRESH


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
