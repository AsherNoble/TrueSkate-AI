"""Reward function for the CMA-ES 360 flip experiment.

After each action attempt, captures a screenshot, runs OCR-based trick
detection, and maps the result to a scalar reward.

Reward tiers (for landed tricks):
    1.0   — "360 FLIP" (exact, no modifiers)
    0.75  — 360 FLIP with a modifier, or NIGHTMARE flip
    0.6   — Flip tricks (FLIP, HEEL, KICK, HARD, LASER, VARIAL, INWARD,
            IMPOSSIBLE, DOLPHIN, DRAGON)
    0.5   — 360+ rotation tricks (360, 540, 720, SPIN, GAZELLE)
    0.3   — Shove-it tricks (SHOVE)
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


class RepetitionPenalty:
    """Tracks landed trick counts and returns a multiplier that penalises repeats.

    Multiplier formula: 1 / (1 + count) — first landing gives 1.0 (no penalty),
    second gives 0.5, third 0.33, tenth 0.09. Base reward can only go down.

    "360 FLIP" and "BACKSIDE 360 FLIP" are exempt: they always return 1.0 so
    the target signal is never reduced.

    Only landed tricks are counted (failed tricks don't feed the counts).
    """

    _NO_PENALTY = frozenset({"360 FLIP", "BACKSIDE 360 FLIP"})

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def get_multiplier_and_record(self, trick_name: str) -> float:
        """Return penalty multiplier for this trick and increment its landed count.

        Args:
            trick_name: The trick string from TrickResult.trick.

        Returns:
            Multiplier in (0, 1]. 1.0 for exempt tricks or first landing.
        """
        if trick_name in self._NO_PENALTY:
            return 1.0
        count = self._counts.get(trick_name, 0)
        self._counts[trick_name] = count + 1
        return 1.0 / (1 + count)

    def count(self, trick_name: str) -> int:
        """Return how many times trick_name has been landed so far."""
        return self._counts.get(trick_name, 0)


def capture_and_detect(driver) -> TrickResult | None:
    """Capture 5 screenshots spaced 0.25s apart and run trick OCR on each.

    Takes screenshots at t=0, t=0.25, t=0.5, t=0.75, t=1.0 seconds. Returns
    the first non-None TrickResult found, or None if all 5 fail to detect a trick.

    Args:
        driver: Appium WebDriver instance.

    Returns:
        TrickResult(trick=..., status="landed"|"failed") or None.
    """
    for capture_idx in range(5):
        if capture_idx > 0:
            time.sleep(0.25)

        png_bytes = driver.get_screenshot_as_png()
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        result = detect_trick(frame)
        if result is not None:
            return result

    return None


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

    if result.status == "failed":
        return base_reward * (base_reward - 0.1)  # tiered (failed 360 flip yields 0.9)

    return base_reward


def _score_component(trick: str) -> float:
    """Score a single (non-combo) trick string. First match wins."""
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

    # --- Tier 0.5: flip tricks (flip component is mechanically critical) ---
    _FLIP_KEYWORDS = (
        "FLIP", "HEEL", "KICK", "HARD", "LASER", "VARIAL", "INWARD",
        "IMPOSSIBLE", "DOLPHIN", "DRAGON",
    )
    if any(kw in trick for kw in _FLIP_KEYWORDS):
        return 0.5

    # --- Tier 0.3: 360+ rotation tricks (no flip) ---
    _ROTATION_KEYWORDS = ("SPIN", "GAZELLE", "360", "540", "720")
    if any(kw in trick for kw in _ROTATION_KEYWORDS):
        return 0.3

    # --- Tier 0.2: 180 tricks ---
    if "180" in trick:
        return 0.2

    # --- Tier 0.1: any other recognized trick (shoves, ollies, grinds, etc.) ---
    return 0.1


def get_reward(
    driver,
    wait_time: float = 0.0,
    penalty: RepetitionPenalty | None = None,
) -> tuple[float, TrickResult | None, float]:
    """Wait for the trick notification, capture, score, and return the reward.

    This is the main entry point called by the CMA-ES optimization loop.
    Captures 5 screenshots spaced 0.25s apart (total ~1.0s after initial wait).

    Args:
        driver: Appium WebDriver instance.
        wait_time: Seconds to wait after gestures finish before first screenshot.
        penalty: Optional RepetitionPenalty. When provided, landed tricks receive
            a diminishing multiplier and their count is incremented.

    Returns:
        Tuple of (reward, result, multiplier):
            reward     — base reward * multiplier, float.
            result     — TrickResult(trick, status) or None.
            multiplier — factor applied to base reward (1.0 if no penalty or failed).
    """
    time.sleep(wait_time)
    result = capture_and_detect(driver)
    base = compute_reward(result)

    multiplier = 1.0
    if penalty is not None and result is not None and result.status == "landed":
        multiplier = penalty.get_multiplier_and_record(result.trick)

    return base * multiplier, result, multiplier


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
        ("540 FLIP", "landed",               0.5),
        ("540 DOUBLE FLIP", "landed",        0.5),
        ("NIGHTMARE FLIP", "landed",         0.75),
        ("KICKFLIP", "landed",               0.5),
        ("INWARD HEELFLIP", "landed",        0.5),
        ("HARD FLIP", "landed",              0.5),
        ("LASER FLIP", "landed",             0.5),
        ("VARIAL KICKFLIP", "landed",        0.5),
        ("IMPOSSIBLE", "landed",             0.5),
        ("360 POP SHOVE-IT", "landed",       0.3),
        ("540 POP SHOVE-IT", "landed",       0.3),
        ("FS POP SHOVE-IT", "landed",        0.1),
        ("POP SHOVE-IT", "landed",           0.1),
        ("BIG SPIN", "landed",               0.3),
        ("BACKSIDE 360", "landed",           0.3),
        ("OLLIE", "landed",                  0.1),
        ("NOLLIE", "landed",                 0.1),
        ("BACKSIDE 180", "landed",           0.2),
        ("KICKFLIP + 50-50 GRIND", "landed", 0.5),
        # Failed tricks — base * (base - 0.1)
        ("360 FLIP", "failed",               0.9),   # 1.0 * 0.9
        ("540 DOUBLE FLIP", "failed",        0.2),   # 0.5 * 0.4
        ("KICKFLIP", "failed",               0.2),   # 0.5 * 0.4
        ("360 POP SHOVE-IT", "failed",       0.06),  # 0.3 * 0.2
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
    print("All base reward tests passed." if all_passed else "FAILURES detected.")

    print("\n=== RepetitionPenalty multiplier tests ===")
    penalty = RepetitionPenalty()
    multiplier_cases = [
        # (trick, status, expected_multiplier, description)
        ("KICKFLIP",         "landed", round(1.0 / 1, 4), "first landing  → 1.0"),
        ("KICKFLIP",         "landed", round(1.0 / 2, 4), "second landing → 0.5"),
        ("KICKFLIP",         "landed", round(1.0 / 3, 4), "third landing  → 0.33"),
        ("OLLIE",            "landed", round(1.0 / 1, 4), "new trick      → 1.0"),
        ("KICKFLIP",         "failed", 1.0,               "failed         — no penalty, no count"),
        ("360 FLIP",         "landed", 1.0,               "exempt         — always 1.0"),
        ("BACKSIDE 360 FLIP","landed", 1.0,               "exempt         — always 1.0"),
    ]
    penalty_passed = True
    for trick, status, expected_mult, desc in multiplier_cases:
        result = TrickResult(trick=trick, status=status)
        mult = penalty.get_multiplier_and_record(result.trick) if result.status == "landed" else 1.0
        actual_mult = round(mult, 4)
        test_status = "PASS" if actual_mult == expected_mult else "FAIL"
        if test_status == "FAIL":
            penalty_passed = False
        print(f"  [{test_status}] {desc:40s} multiplier={actual_mult:.4f}  (expected {expected_mult:.4f})")

    print()
    print("All multiplier tests passed." if penalty_passed else "MULTIPLIER FAILURES detected.")
