"""Reward primitives for the CMA-ES + PPO training loops.

Reward shaping for CMA-ES is delegated to per-target Curriculum objects
(see `trueskate_ai.rl.cmaes.curriculum`). PPO reward stays binary via
`compute_conditioned_reward()` / `get_conditioned_reward()`.

This module owns the OCR capture window, frame merging, repetition penalty,
and shared utilities (`near_miss_multiplier`, `normalize_trick_name`).
"""
import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np
import requests

from trueskate_ai.sim.trick_info_reader import TrickResult, detect_trick

CaptureDiagnostics = dict[str, int | float | None]


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


def _merge_detected_results(results: list[TrickResult]) -> TrickResult:
    """Merge frame-level detections into one eval-level TrickResult.

    Duplicate trick names are removed while preserving first-seen order.
    Overlapping variants collapse to the most specific form
    (e.g. DOUBLE HARD FLIP wins over HARD FLIP).
    Status precedence is landed > failed > unknown.
    """
    def _is_more_specific_variant(candidate: str, current: str) -> bool:
        if candidate == current:
            return False
        if candidate.endswith(current):
            return True
        cand_tokens = set(candidate.split())
        curr_tokens = set(current.split())
        return curr_tokens.issubset(cand_tokens) and len(cand_tokens) > len(curr_tokens)

    trick_components: list[str] = []
    for result in results:
        for component in result.trick.split(" + "):
            normalized = normalize_trick_name(component)
            if normalized:
                trick_components.append(normalized)

    tricks: list[str] = []
    for candidate in trick_components:
        handled = False
        for idx, current in enumerate(tricks):
            if candidate == current:
                handled = True
                break
            if _is_more_specific_variant(candidate, current):
                tricks[idx] = candidate
                handled = True
                break
            if _is_more_specific_variant(current, candidate):
                handled = True
                break
        if not handled:
            tricks.append(candidate)

    if any(result.status == "landed" for result in results):
        status = "landed"
    elif any(result.status == "failed" for result in results):
        status = "failed"
    else:
        status = "unknown"

    return TrickResult(trick=" + ".join(tricks), status=status)


def merge_trick_results(*results: TrickResult | None) -> TrickResult | None:
    """Merge optional TrickResult values into one deduplicated result."""
    present = [result for result in results if result is not None]
    if not present:
        return None
    return _merge_detected_results(present)


class ContinuousTrickMonitor:
    """Reads MJPEG frames continuously and runs OCR trick detection."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_flag = False
        self._response: requests.Response | None = None
        self._detections: list[TrickResult] = []
        self._frames_checked = 0
        self._started_at = 0.0

    def start(self, mjpeg_url: str | None) -> None:
        if mjpeg_url is None or self._thread is not None:
            return
        self._stop_flag = False
        self._detections = []
        self._frames_checked = 0
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._capture_loop, args=(mjpeg_url,), daemon=True
        )
        self._thread.start()

    def _capture_loop(self, mjpeg_url: str) -> None:
        buf = b""
        try:
            resp = requests.get(mjpeg_url, stream=True, timeout=5)
            self._response = resp
            for chunk in resp.iter_content(chunk_size=4096):
                if self._stop_flag:
                    break
                buf += chunk
                while True:
                    start_idx = buf.find(b"\xff\xd8")
                    if start_idx == -1:
                        buf = b""
                        break
                    end_idx = buf.find(b"\xff\xd9", start_idx + 2)
                    if end_idx == -1:
                        buf = buf[start_idx:]
                        break
                    jpeg_bytes = buf[start_idx : end_idx + 2]
                    buf = buf[end_idx + 2 :]
                    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    self._frames_checked += 1
                    result = detect_trick(frame)
                    if result is not None:
                        self._detections.append(result)
        except Exception as exc:
            logging.warning("ContinuousTrickMonitor failed: %s", exc)
        finally:
            self._response = None

    def stop(self) -> tuple[TrickResult | None, CaptureDiagnostics]:
        self._stop_flag = True
        resp = self._response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        merged = _merge_detected_results(self._detections) if self._detections else None
        diagnostics: CaptureDiagnostics = {
            "frames_checked": self._frames_checked,
            "distinct_tricks_detected": len({result.trick for result in self._detections}),
            "monitor_elapsed_s": max(0.0, time.monotonic() - self._started_at),
        }
        return merged, diagnostics


def capture_and_detect_with_diagnostics(
    driver,
    *,
    capture_count: int = 14,
    capture_interval: float = 0.15,
    action_start_time: float | None = None,
) -> tuple[TrickResult | None, CaptureDiagnostics]:
    """Capture multiple screenshots and run trick OCR on each.

    Captures continuously (as fast as screenshots can be read) over a bounded
    detection window. The window length is capture_count * capture_interval.
    If multiple distinct trick names are seen across frames, they are merged
    using the same " + " concatenation convention.

    Args:
        driver: Appium WebDriver instance.

    Returns:
        TrickResult(trick=..., status="landed"|"failed") or None.
    """
    if capture_count <= 0:
        raise ValueError(f"capture_count must be positive, got {capture_count}")
    if capture_interval < 0.0:
        raise ValueError(f"capture_interval must be >= 0.0, got {capture_interval}")

    capture_started_at = time.monotonic()
    captures_attempted = 0
    detection_capture_idx: int | None = None
    detected_results: list[TrickResult] = []
    capture_idx = 0

    window_s = capture_count * capture_interval
    capture_anchor = action_start_time if action_start_time is not None else capture_started_at
    if action_start_time is not None:
        sleep_for = action_start_time - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    capture_deadline = capture_anchor + window_s

    while True:
        png_bytes = driver.get_screenshot_as_png()
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        captures_attempted += 1

        result = detect_trick(frame)
        if result is not None:
            if detection_capture_idx is None:
                detection_capture_idx = capture_idx
            detected_results.append(result)

        now = time.monotonic()
        if now >= capture_deadline:
            break

        capture_idx += 1

    merged_result = _merge_detected_results(detected_results) if detected_results else None
    diagnostics = {
        "captures_attempted": captures_attempted,
        "detection_capture_idx": detection_capture_idx,
        "capture_elapsed_s": time.monotonic() - capture_started_at,
        "skipped_captures": 0,
    }
    return merged_result, diagnostics


def capture_and_detect(
    driver,
    *,
    capture_count: int = 14,
    capture_interval: float = 0.15,
    action_start_time: float | None = None,
) -> TrickResult | None:
    result, _ = capture_and_detect_with_diagnostics(
        driver,
        capture_count=capture_count,
        capture_interval=capture_interval,
        action_start_time=action_start_time,
    )
    return result


def get_reward(
    driver,
    *,
    scorer: Callable[[TrickResult | None], float],
    wait_time: float = 0.0,
    penalty: RepetitionPenalty | None = None,
    capture_count: int = 14,
    capture_interval: float = 0.15,
    action_start_time: float | None = None,
    return_diagnostics: bool = False,
) -> tuple[float, TrickResult | None, float] | tuple[float, TrickResult | None, float, CaptureDiagnostics]:
    """Wait for the trick notification, capture, score via injected scorer, apply penalty.

    Args:
        driver: Appium WebDriver instance.
        scorer: Callable mapping TrickResult | None → float. Typically
            ``Curriculum.score`` for CMA-ES or a lambda wrapping
            ``compute_conditioned_reward`` for PPO.
        wait_time: Seconds to wait after gestures finish before first screenshot.
        penalty: Optional RepetitionPenalty. When provided, landed tricks receive
            a diminishing multiplier and their count is incremented.
        capture_count: Number of screenshots to capture per eval.
        capture_interval: Seconds between consecutive screenshots.
        action_start_time: Optional monotonic timestamp that anchors the
            capture window start.

    Returns:
        Tuple of (reward, result, multiplier):
            reward     — base reward * multiplier, float.
            result     — TrickResult(trick, status) or None.
            multiplier — factor applied to base reward (1.0 if no penalty or failed).
    """
    time.sleep(wait_time)
    result, diagnostics = capture_and_detect_with_diagnostics(
        driver,
        capture_count=capture_count,
        capture_interval=capture_interval,
        action_start_time=action_start_time,
    )
    base = scorer(result)

    multiplier = 1.0
    if penalty is not None and result is not None and result.status == "landed":
        multiplier = penalty.get_multiplier_and_record(result.trick)

    if return_diagnostics:
        return base * multiplier, result, multiplier, diagnostics
    return base * multiplier, result, multiplier


def normalize_trick_name(trick_name: str) -> str:
    normalized = trick_name.upper().strip()
    # Preserve current OCR workaround convention.
    normalized = normalized.replace("540", "360")
    return normalized


def near_miss_multiplier(base: float) -> float:
    """Failure-multiplier util: scales near-misses without going negative.

    Returns max(0, base * (base - 0.1)) so a 1.0-target failure scores 0.9,
    a 0.4 family-member failure scores 0.12, and a 0.05 default-reward
    failure scores 0.0 (clamped — would be negative under raw formula).
    """
    return max(0.0, base * (base - 0.1))


def compute_conditioned_reward(result: TrickResult | None, *, target_trick: str) -> float:
    """Binary reward for trick-conditioned policy training.

    Returns:
        1.0 if a landed detected component matches target_trick exactly
        after normalization, else 0.0.
    """
    if result is None or result.status != "landed":
        return 0.0

    target = normalize_trick_name(target_trick)
    components = [normalize_trick_name(c) for c in result.trick.split(" + ")]
    return 1.0 if any(component == target for component in components) else 0.0


def get_conditioned_reward(
    driver,
    *,
    target_trick: str,
    wait_time: float = 0.0,
    capture_count: int = 14,
    capture_interval: float = 0.15,
    action_start_time: float | None = None,
    return_diagnostics: bool = False,
) -> tuple[float, TrickResult | None] | tuple[float, TrickResult | None, CaptureDiagnostics]:
    """Capture trick text and return conditioned binary reward.

    The capture loop polls continuously during the configured detection window.
    """
    time.sleep(wait_time)
    result, diagnostics = capture_and_detect_with_diagnostics(
        driver,
        capture_count=capture_count,
        capture_interval=capture_interval,
        action_start_time=action_start_time,
    )
    reward = compute_conditioned_reward(result, target_trick=target_trick)
    if return_diagnostics:
        return reward, result, diagnostics
    return reward, result


# ---------------------------------------------------------------------------
# Sanity-check entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== near_miss_multiplier ===")
    nmm_cases = [
        (1.0,  0.9),
        (0.6,  0.30),
        (0.4,  0.12),
        (0.3,  0.06),
        (0.05, 0.0),
        (0.0,  0.0),
    ]
    nmm_passed = True
    for base, expected in nmm_cases:
        actual = near_miss_multiplier(base)
        ok = abs(actual - expected) < 1e-6
        if not ok:
            nmm_passed = False
        print(f"  [{'PASS' if ok else 'FAIL'}] near_miss({base}) = {actual:.4f} (expected {expected:.4f})")
    print("near_miss_multiplier tests passed." if nmm_passed else "near_miss_multiplier FAILURES.")

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
