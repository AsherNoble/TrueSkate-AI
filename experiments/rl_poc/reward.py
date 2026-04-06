"""Reward function for the CMA-ES 360 flip experiment.

After each action attempt, captures a screenshot, runs OCR-based trick
detection, and maps the result to a scalar reward.

Reward tiers (for landed tricks):
    1.0   — "360 FLIP" (exact, no modifiers)
    0.75  — 360 FLIP with a modifier, or NIGHTMARE flip
    0.6   — Flip tricks (FLIP, HEEL, KICK, HARD, LASER, VARIAL, INWARD,
            IMPOSSIBLE, DOLPHIN, DRAGON)
    0.4   — Rotation tricks (SHOVE, SPIN, GAZELLE, 360, 540, 720)
    0.2   — Basic air tricks (OLLIE, NOLLIE, 180)
    0.1   — Any other recognized trick (grinds, slides, manuals, etc.)
    0.0   — None (no trick detected)

Failed tricks receive a 0.4× multiplier on the base tier reward.

For combo tricks (e.g. "KICKFLIP + CROOKED GRIND"), each component is
evaluated independently and the maximum reward is returned.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Resolve src/ onto the path so trick_info_reader can be imported from
# anywhere (e.g. running directly or from the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.sim.trick_info_reader import TrickResult, detect_trick  # noqa: E402


def capture_and_detect(driver) -> TrickResult | None:
    """Capture a screenshot and run trick OCR on it.

    Args:
        driver: Appium WebDriver instance.

    Returns:
        TrickResult(trick=..., status="landed"|"failed") or None.
    """
    png_bytes = driver.get_screenshot_as_png()
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return detect_trick(frame)


def compute_reward(result: TrickResult | None) -> float:
    """Map a TrickResult to a scalar reward.

    For combo tricks joined with " + ", each component is scored and the
    maximum reward across components is returned.

    Failed tricks receive a 0.4× multiplier on the base tier reward.

    Args:
        result: Output of detect_trick() — a TrickResult or None.

    Returns:
        Scalar reward in [0.0, 1.0].
    """
    if result is None:
        return 0.0

    components = [c.strip() for c in result.trick.split(" + ")]
    base_reward = max(_score_component(c) for c in components)

    # Apply 0.4× multiplier for failed tricks
    if result.status == "failed":
        return base_reward * 0.4

    return base_reward


def _score_component(trick: str) -> float:
    """Score a single (non-combo) trick string. First match wins."""
    trick = trick.replace("540", "360")
    _MODIFIERS = ("FAKIE", "SWITCH", "DOUBLE", "TRIPLE", "NOLLIE")

    # --- Tier 1.0: exact target, no modifiers ---
    if "360 FLIP" in trick and not any(m in trick for m in _MODIFIERS):
        return 1.0

    # --- Tier 0.75: 360 FLIP with a modifier, or nightmare flip ---
    # "360 DOUBLE FLIP" / "360 TRIPLE FLIP" don't contain the exact substring
    # "360 FLIP", so also check for "360" + "FLIP" co-occurring with a modifier.
    if "360 FLIP" in trick and any(m in trick for m in _MODIFIERS):
        return 0.75
    if "360" in trick and "FLIP" in trick and any(m in trick for m in _MODIFIERS):
        return 0.75
    if "NIGHTMARE" in trick:
        return 0.75

    # --- Tier 0.6: flip tricks (flip component is mechanically critical) ---
    _FLIP_KEYWORDS = (
        "FLIP", "HEEL", "KICK", "HARD", "LASER", "VARIAL", "INWARD",
        "IMPOSSIBLE", "DOLPHIN", "DRAGON",
    )
    if any(kw in trick for kw in _FLIP_KEYWORDS):
        return 0.6

    # --- Tier 0.4: rotation tricks (right rotation but no flip) ---
    _ROTATION_KEYWORDS = ("SHOVE", "SPIN", "GAZELLE", "360", "540", "720")
    if any(kw in trick for kw in _ROTATION_KEYWORDS):
        return 0.4

    # --- Tier 0.2: basic air tricks ---
    _AIR_KEYWORDS = ("OLLIE", "NOLLIE", "180")
    if any(kw in trick for kw in _AIR_KEYWORDS):
        return 0.2

    # --- Tier 0.1: any other recognized trick (grinds, slides, manuals, etc.) ---
    return 0.1


def get_reward(driver, wait_time: float = 1.5) -> tuple[float, TrickResult | None]:
    """Wait for the trick notification, capture, and return the reward.

    This is the main entry point called by the CMA-ES optimization loop.

    Args:
        driver: Appium WebDriver instance.
        wait_time: Seconds to wait after gestures finish before screenshotting.
            The game needs time to display the trick name notification.
            Default 1.5s — may need tuning.

    Returns:
        Tuple of (reward, result) where reward is a float in [0.0, 1.0]
        and result is a TrickResult(trick, status) or None.
    """
    time.sleep(wait_time)
    result = capture_and_detect(driver)
    reward = compute_reward(result)
    return reward, result


# ---------------------------------------------------------------------------
# Sanity-check entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # Landed tricks
        ("360 FLIP", "landed",               1.0),
        ("FAKIE 360 FLIP", "landed",         0.75),
        ("SWITCH 360 FLIP", "landed",        0.75),
        ("360 DOUBLE FLIP", "landed",        0.75),
        ("360 TRIPLE FLIP", "landed",        0.75),
        ("NOLLIE 360 FLIP", "landed",        0.75),
        ("540 FLIP", "landed",               1.0),
        ("540 DOUBLE FLIP", "landed",        0.75),
        ("NIGHTMARE FLIP", "landed",         0.75),
        ("KICKFLIP", "landed",               0.6),
        ("INWARD HEELFLIP", "landed",        0.6),
        ("HARD FLIP", "landed",              0.6),
        ("LASER FLIP", "landed",             0.6),
        ("VARIAL KICKFLIP", "landed",        0.6),
        ("IMPOSSIBLE", "landed",             0.6),
        ("360 POP SHOVE-IT", "landed",       0.4),
        ("540 POP SHOVE-IT", "landed",       0.4),
        ("FS POP SHOVE-IT", "landed",        0.4),
        ("POP SHOVE-IT", "landed",           0.4),
        ("BIG SPIN", "landed",               0.4),
        ("BACKSIDE 360", "landed",           0.4),
        ("OLLIE", "landed",                  0.2),
        ("NOLLIE", "landed",                 0.2),
        ("BACKSIDE 180", "landed",           0.2),
        ("KICKFLIP + 50-50 GRIND", "landed", 0.6),
        # Failed tricks (0.4× multiplier)
        ("360 FLIP", "failed",               0.4),
        ("540 DOUBLE FLIP", "failed",        0.3),
        ("KICKFLIP", "failed",               0.24),
        ("360 POP SHOVE-IT", "failed",       0.16),
        # No trick
        (None, None,                         0.0),
    ]

    all_passed = True
    for trick, status, expected in test_cases:
        result = TrickResult(trick=trick, status=status) if trick is not None else None
        actual = compute_reward(result)
        expected_rounded = round(expected, 2)
        actual_rounded = round(actual, 2)
        test_status = "PASS" if actual_rounded == expected_rounded else "FAIL"
        if test_status == "FAIL":
            all_passed = False
        label = f"{trick!r} ({status})" if trick is not None else "None"
        print(f"  [{test_status}] compute_reward({label:35s}) = {actual:.2f}  (expected {expected:.2f})")

    print()
    print("All tests passed." if all_passed else "FAILURES detected.")
