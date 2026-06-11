"""Remap saved trick-library coords into the current RL gesture bounds.

The X crop (X_BOUND_MIN=0.12, added 2026-06-11 to stop gestures opening the
replay-clip UI) means libraries mined before it can hold points the optimizer
would now clamp, distorting gesture shape on warm-start. This applies the
minimal-change fix: per gesture, if any point lies left of X_BOUND_MIN, SHIFT
the whole gesture right by the deficit (shape preserved); only if that would
push past X_BOUND_MAX, compress the x-span into range. Y likewise (was always
bounded, so usually a no-op).

Writes <name>_xfix.json beside each input unless --in-place.

Usage:
    python scripts/data/remap_library_bounds.py <lib.json> [<lib.json> ...] [--in-place]
"""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trueskate_ai.sim.gestures import X_BOUND_MAX, X_BOUND_MIN, Y_BOUND_MAX, Y_BOUND_MIN


def _remap_axis(values: list[float], lo: float, hi: float) -> tuple[list[float], bool]:
    vmin, vmax = min(values), max(values)
    if vmin >= lo and vmax <= hi:
        return values, False
    span = vmax - vmin
    if span <= (hi - lo):
        # Shift only — gesture shape untouched.
        shift = lo - vmin if vmin < lo else hi - vmax
        return [v + shift for v in values], True
    # Span itself exceeds the legal range: compress into [lo, hi].
    return [lo + (v - vmin) * (hi - lo) / span for v in values], True


def _fix_recipe(recipe: dict) -> int:
    changed = 0
    for g in recipe.get("gestures", []):
        xs = [p[0] for p in g["points"]]
        ys = [p[1] for p in g["points"]]
        xs2, cx = _remap_axis(xs, X_BOUND_MIN, X_BOUND_MAX)
        ys2, cy = _remap_axis(ys, Y_BOUND_MIN, Y_BOUND_MAX)
        if cx or cy:
            g["points"] = [[round(x, 4), round(y, 4)] for x, y in zip(xs2, ys2)]
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("libraries", nargs="+", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    for lib in args.libraries:
        data = json.loads(lib.read_text())
        n = 0
        for key in ("median_gestures", "best_gestures"):
            if key in data:
                n += _fix_recipe(data[key])
        if n == 0:
            print(f"{lib.name}: already within bounds — untouched")
            continue
        data["bounds_remap"] = f"shifted {n} gesture(s) into x[{X_BOUND_MIN},{X_BOUND_MAX}]"
        out = lib if args.in_place else lib.with_name(lib.stem + "_xfix.json")
        out.write_text(json.dumps(data, indent=2) + "\n")
        print(f"{lib.name}: {n} gesture(s) remapped → {out.name}")


if __name__ == "__main__":
    main()
