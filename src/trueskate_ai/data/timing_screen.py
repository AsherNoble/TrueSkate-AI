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
