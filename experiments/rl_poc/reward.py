"""Reward function for the CMA-ES 360 flip experiment.

After each action attempt, captures a screenshot, runs OCR-based trick
detection, and maps the result to a scalar reward.

Reward tiers:
    1.0  — "360 FLIP" (exact, no modifiers)
    0.5  — Trick contains a flip/spin/shove keyword (FLIP, HEEL, KICK, HARD,
           LASER, VARIAL, INWARD, NIGHTMARE, IMPOSSIBLE, DOLPHIN, DRAGON,
           SHOVE, SPIN, GAZELLE, 360, 540, 720)
    0.3  — Trick contains OLLIE, NOLLIE, or 180
    0.1  — Any other recognized trick (grinds, slides, manuals, etc.)
    0.0  — None (no trick detected)

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

from trueskate_ai.sim.trick_info_reader import detect_trick  # noqa: E402


def capture_and_detect(driver) -> str | None:
    """Capture a screenshot and run trick OCR on it.

    Args:
        driver: Appium WebDriver instance.

    Returns:
        Detected trick string (e.g. "360 FLIP", "KICKFLIP + CROOKED GRIND")
        or None if no trick was found.
    """
    png_bytes = driver.get_screenshot_as_png()
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return detect_trick(frame)


def compute_reward(trick_name: str | None) -> float:
    """Map a detected trick string to a scalar reward.

    For combo tricks joined with " + ", each component is scored and the
    maximum reward across components is returned.

    Args:
        trick_name: Output of detect_trick() — e.g. "360 FLIP",
            "KICKFLIP + CROOKED GRIND", or None.

    Returns:
        Scalar reward in [0.0, 1.0].
    """
    if trick_name is None:
        return 0.0

    components = [c.strip() for c in trick_name.split(" + ")]
    return max(_score_component(c) for c in components)


def _score_component(trick: str) -> float:
    """Score a single (non-combo) trick string."""
    # --- Tier 1.0: exact target, no modifiers ---
    # Must contain "360 FLIP" but NOT any stance/multiplier modifier.
    _MODIFIERS = ("FAKIE", "SWITCH", "DOUBLE", "TRIPLE")
    if "360 FLIP" in trick and not any(m in trick for m in _MODIFIERS):
        return 1.0

    # --- Tier 0.5: any trick sharing flip/spin/shove mechanics ---
    _TIER_0_5_KEYWORDS = (
        "FLIP", "HEEL", "KICK", "HARD", "LASER", "VARIAL", "INWARD",
        "NIGHTMARE", "IMPOSSIBLE", "DOLPHIN", "DRAGON", "SHOVE", "SPIN",
        "GAZELLE", "360", "540", "720",
    )
    if any(kw in trick for kw in _TIER_0_5_KEYWORDS):
        return 0.5

    # --- Tier 0.3: basic air tricks ---
    _TIER_0_3_KEYWORDS = ("OLLIE", "NOLLIE", "180")
    if any(kw in trick for kw in _TIER_0_3_KEYWORDS):
        return 0.3

    # --- Tier 0.1: any other recognized trick (grinds, slides, manuals, etc.) ---
    return 0.1


def get_reward(driver, wait_time: float = 1.5) -> tuple[float, str | None]:
    """Wait for the trick notification, capture, and return the reward.

    This is the main entry point called by the CMA-ES optimization loop.

    Args:
        driver: Appium WebDriver instance.
        wait_time: Seconds to wait after gestures finish before screenshotting.
            The game needs time to display the trick name notification.
            Default 1.5s — may need tuning.

    Returns:
        Tuple of (reward, trick_name) where reward is a float in [0.0, 1.0]
        and trick_name is the raw OCR result (or None).
    """
    time.sleep(wait_time)
    trick_name = capture_and_detect(driver)
    reward = compute_reward(trick_name)
    return reward, trick_name


# ---------------------------------------------------------------------------
# Sanity-check entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("360 FLIP",               1.0),
        ("FAKIE 360 FLIP",         0.5),
        ("SWITCH 360 FLIP",        0.5),
        ("360 DOUBLE FLIP",        0.5),
        ("KICKFLIP",               0.5),
        ("INWARD HEELFLIP",        0.5),
        ("HARD FLIP",              0.5),
        ("LASER FLIP",             0.5),
        ("VARIAL KICKFLIP",        0.5),
        ("360 POP SHOVE-IT",       0.5),
        ("540 POP SHOVE-IT",       0.5),
        ("IMPOSSIBLE",             0.5),
        ("BACKSIDE 360",           0.5),
        ("BIG SPIN",               0.5),
        ("OLLIE",                  0.3),
        ("NOLLIE",                 0.3),
        ("BACKSIDE 180",           0.3),
        ("KICKFLIP + 50-50 GRIND", 0.5),
        (None,                     0.0),
    ]

    all_passed = True
    for trick, expected in test_cases:
        actual = compute_reward(trick)
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] compute_reward({trick!r:30s}) = {actual:.1f}  (expected {expected:.1f})")

    print()
    print("All tests passed." if all_passed else "FAILURES detected.")
