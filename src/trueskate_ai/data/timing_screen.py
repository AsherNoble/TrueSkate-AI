"""Per-clip onset-error screen from two-anchor calibration rates.

M1-ONSET-VALIDATION-20260924: with a true clock rate of ~1.0, a falsely early
start anchor delays a clip's real onset by ``(rate - 1) * (end - gesture)`` and
a falsely early end anchor by ``(1 - rate) * (gesture - start)`` (WDA
monotonic seconds). Segments with ``abs(rate - 1) <= rate_threshold`` are
treated as ordinary; their spread matches ±1-frame anchor quantisation
(M1-SCREEN-DRAFT-20260925). The screen is opt-in and records its parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

TWO_ANCHOR_METHOD = "wda-submitted-two-centre-controls-v2"
MULTI_ANCHOR_METHOD = "wda-submitted-multi-centre-controls-v1"


@dataclass(frozen=True)
class TimingScreen:
    rate_threshold: float
    max_error_frames: float
    frame_s: float = 1 / 30

    def __post_init__(self) -> None:
        if self.rate_threshold <= 0 or self.max_error_frames < 0 or self.frame_s <= 0:
            raise ValueError("invalid timing screen parameters")

    def describe(self) -> dict[str, float]:
        return {"rate_threshold": self.rate_threshold, "max_error_frames": self.max_error_frames,
                "frame_s": self.frame_s, "source": "M1-SCREEN-DRAFT-20260925"}


def predicted_onset_error_s(meta: Mapping[str, Any], rate_threshold: float) -> tuple[str | None, float]:
    """Return ``(implicated_anchor, predicted_error_s)``; raises if calibration is absent."""
    cal = meta.get("tap_calibration") or {}
    if cal.get("method") == MULTI_ANCHOR_METHOD:
        # Consensus fits are validated only between their inlier anchors; an
        # extrapolated clip (gesture outside the inlier WDA range) is not.
        inlier_times = [float(d["submitted_to_ios_monotonic_s"])
                        for d in cal.get("detections", []) if d.get("inlier")]
        gesture = float(meta["wda_submitted_monotonic_s"])
        if (not cal.get("accepted") or len(inlier_times) < 3
                or not min(inlier_times) <= gesture <= max(inlier_times)):
            return "multi_anchor_unsupported", float("inf")
        return None, 0.0
    if cal.get("method") != TWO_ANCHOR_METHOD:
        raise ValueError(f"clip lacks {TWO_ANCHOR_METHOD} calibration")
    anchors = {d["role"]: float(d["submitted_to_ios_monotonic_s"]) for d in cal["detections"]}
    rate = float(cal["rate"])
    gesture = float(meta["wda_submitted_monotonic_s"])
    if rate - 1.0 > rate_threshold:
        return "start", (rate - 1.0) * (anchors["end"] - gesture)
    if 1.0 - rate > rate_threshold:
        return "end", (1.0 - rate) * (gesture - anchors["start"])
    return None, 0.0


def passes(meta: Mapping[str, Any], screen: TimingScreen) -> bool:
    _, error_s = predicted_onset_error_s(meta, screen.rate_threshold)
    return error_s <= screen.max_error_frames * screen.frame_s + 1e-12


# --- Frozen corpus screen v1 (M1-CORPUS-AUDIT-20260927) ---------------------
# Whole-segment exclusion for two-anchor segments: excluded if the fitted rate
# is off by more than 0.0008, or if any calibration detection lies outside
# 85-210 ms after its control's WDA submission (M1-TIMING-GATE-20260926).
CORPUS_SCREEN_V1 = {"name": "corpus-screen-v1", "rate_threshold": 0.0008,
                    "latency_window_s": (0.085, 0.210),
                    "source": "M1-CORPUS-AUDIT-20260927"}


def anchor_latencies_s(meta: Mapping[str, Any], segment_manifest: Mapping[str, Any]) -> dict[str, float]:
    """Detection time minus command video time, per calibration role."""
    started = float(segment_manifest["started_at_epoch_s"])
    command_s = {e["wda_action_sequence"]: float(e["wda_submitted_epoch_s"]) - started
                 for e in segment_manifest["gestures"] if e.get("calibration_control")}
    return {d["role"]: float(d["onset_video_s"]) - command_s[d["wda_action_sequence"]]
            for d in meta["tap_calibration"]["detections"]}


def passes_corpus_screen_v1(meta: Mapping[str, Any], segment_manifest: Mapping[str, Any]) -> bool:
    cal = meta.get("tap_calibration") or {}
    if cal.get("method") != TWO_ANCHOR_METHOD:
        raise ValueError(f"corpus-screen-v1 applies only to {TWO_ANCHOR_METHOD}")
    if abs(float(cal["rate"]) - 1.0) > CORPUS_SCREEN_V1["rate_threshold"]:
        return False
    low, high = CORPUS_SCREEN_V1["latency_window_s"]
    latencies = anchor_latencies_s(meta, segment_manifest)
    return {"start", "end"} <= latencies.keys() and all(low <= v <= high for v in latencies.values())
