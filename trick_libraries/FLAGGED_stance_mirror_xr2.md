# FLAGGED: XR2 stance was REGULAR until 2026-06-12 ~09:00 (now goofy)

Every trick DETECTED on iPhone_XR2 before the stance switch is mirrored:
the game labels tricks relative to skater stance, so the same physical
gesture reads as the mirror trick. XR1 was goofy throughout (labels true).

## Mirror map (regular-stance label ↔ goofy/true-gesture label)
| Detected on XR2 (regular) | Actual gesture (goofy frame) |
|---|---|
| LASER FLIP | 360 FLIP |
| DOUBLE/TRIPLE LASER FLIP | 360 DOUBLE/TRIPLE FLIP |
| INWARD HEELFLIP (+ DOUBLE) | HARD FLIP (+ DOUBLE) |
| HEELFLIP family | KICKFLIP family |
| VARIAL HEELFLIP | VARIAL KICKFLIP |
| POP SHOVE-IT ↔ FS POP SHOVE-IT | (BS/FS swap) |
| BACKSIDE ↔ FRONTSIDE 180/360 | (BS/FS swap) |

## Flagged libraries (mined from XR2 regular-stance detections)
- `laser_flip_20260612_042725.json` — actually a 360 FLIP gesture (the 70% "laser" convergence = a second converged 360-flip recipe)
- `double_laser_flip_20260612_030211.json` — actually 360 DOUBLE FLIP gesture
- `double_laser_flip_demo_20260611.json` — mined from XR1 logs (labels true: 360 DOUBLE FLIP gesture) but demo-validated on regular-XR2 where it read "DOUBLE LASER FLIP" 4/4. The demo finale was a 360 double flip in mirror.
- `triple_laser_flip_*` / `inward_heelflip_*` / `varial_heelflip_*` / heel-family entries in `tail_20260611/`, `tail_may2026/`, `tail_day20260611/` — any whose samples came from iPhone_XR2 rows. (Tail libs mix devices; per-row device_id in source JSONLs disambiguates.)

## Flagged conclusions to revisit
- "Recipes are device-flavored" (journal 2026-06-12) — WRONG mechanism; it was the stance mirror. Cross-device replay is likely fine between same-stance devices.
- "INWARD HEELFLIP failed 0% on XR2" (night 1/2) — it was being asked to land a mirrored trick from an unmirrored prior.
- BACKSIDE 180 0.6% failure — rewards/curriculum referenced BS but detections were FS-mirrored; needs re-judging.

## Not affected
- All XR1 runs/libraries (goofy throughout): 360 flip, hard flip, pop shove-it, 360 double flip, kickflip-descent lineage.
- XR2 runs AFTER the 2026-06-12 stance switch.
