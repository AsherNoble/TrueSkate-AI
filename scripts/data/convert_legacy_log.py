"""Convert a pre-normalisation (April 2026) CMA-ES JSONL log to normalised coords.

Before the normalisation change (PR #13, landed 2026-05-09/10), gesture params
lived in a canonical 375x812 logical-point space and were mapped to each
device at execution time by ``norm_to_device`` (commit 11fcf09):

    scale    = device_w / 375
    y_offset = (device_h - 812 * scale) / 2
    device_x = x * scale
    device_y = y_offset + y * scale

This script rewrites the coordinate params of every eval record into today's
normalised [0,1] space so the unmodified build_trick_library.py can mine them.

Two mappings, because of the April iPhone_11 dims bug (fixed in c972b38 —
the device was configured 414x896 while its real Display-Zoomed space is
375x812, so gestures executed ~10% scaled/offset from canonical):

    canonical          x/375, y/812. Exactly reproduces the PHYSICAL gesture
                       for April XR runs (correct config), and the INTENDED
                       gesture for iPhone_11 runs.
    executed-iphone11  Models what iPhone_11 PHYSICALLY executed under the
                       wrong 414x896 config. Mine April iPhone_11 logs under
                       BOTH mappings and keep whichever replays correctly.

Usage:
    python scripts/data/convert_legacy_log.py --log <april_jsonl> --out <new_jsonl> \
        [--mapping canonical|executed-iphone11]

Refuses logs that already look normalised (all coords <= 1.0).
"""
import argparse
import json
import sys
from pathlib import Path

_CANONICAL_W = 375.0
_CANONICAL_H = 812.0
# The wrong dims iPhone_11 was configured with for all of April.
_MISCONFIG_W = 414.0
_MISCONFIG_H = 896.0

PARAMS_PER_SLOT = 8  # x0,y0,x1,y1,x2,y2,duration,easing — delays follow all slots


def _convert_point(x: float, y: float, mapping: str) -> tuple[float, float]:
    if mapping == "canonical":
        return x / _CANONICAL_W, y / _CANONICAL_H
    # executed-iphone11: canonical -> (wrongly configured) device points,
    # then normalise by the REAL 375x812 space those points landed in.
    scale = _MISCONFIG_W / _CANONICAL_W
    y_offset = (_MISCONFIG_H - _CANONICAL_H * scale) / 2.0
    return (x * scale) / _CANONICAL_W, (y_offset + y * scale) / _CANONICAL_H


def _convert_params(params: list[float], mapping: str) -> list[float]:
    """Convert the 6 coord values in each 8-param slot; pass through the rest."""
    n = len(params)
    if (n + 1) % 9 != 0:
        raise ValueError(f"Unexpected param vector length {n} (expected 9N-1)")
    num_gestures = (n + 1) // 9
    out = list(params)
    for g in range(num_gestures):
        base = g * PARAMS_PER_SLOT
        for p in range(3):
            x, y = params[base + 2 * p], params[base + 2 * p + 1]
            nx, ny = _convert_point(x, y, mapping)
            out[base + 2 * p] = round(nx, 4)
            out[base + 2 * p + 1] = round(ny, 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a pre-normalisation CMA-ES JSONL log to normalised coords."
    )
    parser.add_argument("--log", type=Path, required=True, help="April-era JSONL log")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--mapping", choices=["canonical", "executed-iphone11"],
                        default="canonical")
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"ERROR: log file not found: {args.log}")

    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    eval_rows = [r for r in rows if r.get("type") not in ("run_config", "generation_summary")
                 and "params" in r]
    if not eval_rows:
        sys.exit(f"ERROR: no eval records in {args.log}")

    # Guard: a legacy log has coord values way above 1.0 (canonical points).
    max_coord = max(
        v
        for r in eval_rows
        for g in range((len(r["params"]) + 1) // 9)
        for v in r["params"][g * PARAMS_PER_SLOT : g * PARAMS_PER_SLOT + 6]
    )
    if max_coord <= 1.0:
        sys.exit(
            f"ERROR: {args.log} already looks normalised (max coord {max_coord}) — refusing."
        )

    converted = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            if row.get("type") in ("run_config", "generation_summary"):
                row = dict(row, legacy_converted=args.mapping)
            elif "params" in row:
                row = dict(row, params=_convert_params(row["params"], args.mapping),
                           legacy_converted=args.mapping)
                converted += 1
            f.write(json.dumps(row) + "\n")

    print(f"Converted {converted} eval records ({args.mapping}) → {args.out}")


if __name__ == "__main__":
    main()
