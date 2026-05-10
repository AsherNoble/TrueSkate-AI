This file is superseded by [GESTURES.md](../GESTURES.md) at the repo root.

## Tips for Manual Editing

- Points should form a smooth path; large jumps between waypoints create abrupt direction changes
- Easing power tunes **how** the gesture accelerates, not the distance traveled
- Delays should account for visual feedback and control feel; too short = overlapping motion; too long = unresponsive
- Test incrementally — small adjustments (±0.05 easing, ±0.02 duration) are usually sufficient

## Legacy files

The files in `/pre_normalisation` use UIKit logical-point coordinates. They predate the normalised [0,1] coordinate system and will break `execute_gesture_recipe()`.
