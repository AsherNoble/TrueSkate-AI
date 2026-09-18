"""Versioned True Skate control regions used to protect gesture endpoints.

The exact conditions under which a moving finger activates a control are not yet
known.  Until that is resolved, these regions apply to both the first and last
point of each gesture. Intermediate path points may still cross a control.
Stationary taps/holds have only one point, so that point is checked once.

The bottom navigation row is transient, but this first profile deliberately
blocks it at all times.  We can make that region state-dependent later if the
lost sampling area becomes material.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


CONTROL_START_MAP_VERSION = "xr-414x896-20260914-v1"
CONTROL_MAP_LOGICAL_SIZE = (414.0, 896.0)
CONTROL_SAFETY_MARGIN_POINTS = 16.0


@dataclass(frozen=True)
class ControlHitbox:
    name: str
    rect: tuple[float, float, float, float]
    source: str
    transient: bool = False

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 <= y <= y1


# Effective hitboxes mapped on XR2. One axis was measured directly; the other
# follows the visible control-cell split. The XR1 and XR2 share the same logical
# 414 x 896 layout. The spin right edge also contains a user-confirmed historical
# accidental activation at (0.1832222850, 0.4213833183).
EFFECTIVE_CONTROL_HITBOXES = (
    ControlHitbox("more", (0.0, 0.0, 0.1655, 0.14217578125),
                  "top depth inherited; horizontal split inferred"),
    ControlHitbox("park_editor", (0.1655, 0.0, 0.3865, 0.14217578125),
                  "lower edge measured; horizontal split inferred"),
    ControlHitbox("reset", (0.3865, 0.0, 0.6125, 0.11454296875),
                  "lower edge measured; horizontal split inferred"),
    ControlHitbox("camera", (0.6125, 0.0, 0.8325, 0.11966015625),
                  "lower edge measured; horizontal split inferred"),
    ControlHitbox("rewind", (0.8325, 0.0, 1.0, 0.14217578125),
                  "lower edge measured; horizontal split inferred"),
    ControlHitbox("bolt_challenges", (0.0, 0.14217578125, 0.1478046875, 0.2625),
                  "right edge measured; vertical split inferred"),
    ControlHitbox("fast_forward", (0.0, 0.2625, 0.1251171875, 0.3585),
                  "right edge measured; vertical split inferred"),
    ControlHitbox("spin", (0.0, 0.3585, 0.1832222850, 0.465),
                  "right edge measured/historical activation; vertical split inferred"),
    ControlHitbox("me", (0.0, 0.9, 0.2, 1.0),
                  "upper edge conservatively inferred; horizontal split inferred", True),
    ControlHitbox("skateparks", (0.2, 0.9, 0.4, 1.0),
                  "upper edge conservatively inferred; horizontal split inferred", True),
    ControlHitbox("community", (0.4, 0.9, 0.6, 1.0),
                  "upper edge conservatively inferred; horizontal split inferred", True),
    ControlHitbox("shop", (0.6, 0.9, 0.8, 1.0),
                  "upper edge conservatively inferred; horizontal split inferred", True),
    ControlHitbox("settings", (0.8, 0.9, 1.0, 1.0),
                  "upper edge conservatively inferred; horizontal split inferred", True),
)


def _inflate(hitbox: ControlHitbox) -> ControlHitbox:
    width, height = CONTROL_MAP_LOGICAL_SIZE
    x0, y0, x1, y1 = hitbox.rect
    mx = CONTROL_SAFETY_MARGIN_POINTS / width
    my = CONTROL_SAFETY_MARGIN_POINTS / height
    return ControlHitbox(
        hitbox.name,
        (max(0.0, x0 - mx), max(0.0, y0 - my),
         min(1.0, x1 + mx), min(1.0, y1 + my)),
        hitbox.source,
        hitbox.transient,
    )


CONTROL_START_EXCLUSIONS = tuple(_inflate(hitbox) for hitbox in EFFECTIVE_CONTROL_HITBOXES)
# The geometry is shared by both endpoint checks. Keep the historical name above
# for experiment-artifact compatibility.
CONTROL_ENDPOINT_EXCLUSIONS = CONTROL_START_EXCLUSIONS


def controls_at_point(point: tuple[float, float]) -> tuple[str, ...]:
    """Return every conservatively expanded control containing ``point``."""
    return tuple(hitbox.name for hitbox in CONTROL_ENDPOINT_EXCLUSIONS if hitbox.contains(point))


def point_is_safe(point: tuple[float, float]) -> bool:
    return not controls_at_point(point)


def controls_at_start(point: tuple[float, float]) -> tuple[str, ...]:
    """Compatibility name for callers that only inspect touch-downs."""
    return controls_at_point(point)


def start_is_safe(point: tuple[float, float]) -> bool:
    return point_is_safe(point)


def move_point_out_of_controls(point: tuple[float, float]) -> tuple[float, float]:
    """Move an endpoint minimally toward play space until no control contains it.

    Top controls move downward, left controls move right, and the transient
    bottom row moves upward.  Iteration handles overlap between the top and left
    safety regions.
    """
    x, y = (float(point[0]), float(point[1]))
    for _ in range(len(CONTROL_ENDPOINT_EXCLUSIONS) + 1):
        containing = [h for h in CONTROL_ENDPOINT_EXCLUSIONS if h.contains((x, y))]
        if not containing:
            return x, y
        for hitbox in containing:
            x0, y0, x1, y1 = hitbox.rect
            if hitbox.transient:
                y = math.nextafter(y0, -math.inf)
            elif y0 == 0.0:
                y = math.nextafter(y1, math.inf)
            else:
                x = math.nextafter(x1, math.inf)
    raise RuntimeError(f"could not move gesture endpoint out of control regions: {(x, y)}")


def move_start_out_of_controls(point: tuple[float, float]) -> tuple[float, float]:
    """Compatibility name for callers that only sanitize touch-downs."""
    return move_point_out_of_controls(point)
