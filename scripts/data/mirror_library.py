"""Mirror a trick library left-right: x -> 1 - x on every gesture point.

Mirroring a gesture swaps the trick's chirality under a fixed stance:
BS <-> FS rotations and kickflip <-> heelflip families. e.g. a converged
360 FLIP recipe (BS 360 + kickflip) mirrors into a LASER FLIP candidate
(FS 360 + heelflip) — the highest-leverage warm start for the mirror trick.

Durations, easing and delays are untouched; y is untouched. After mirroring,
x-points may fall below X_BOUND_MIN (mirror of [0.12, 1] is [0, 0.88]) — the
shift-preserving bounds remap from remap_library_bounds is applied per
gesture, so shapes survive intact.

Usage:
    python scripts/data/mirror_library.py <lib.json> --trick "LASER FLIP" [--out <path>]
"""
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from remap_library_bounds import _fix_recipe


def _mirror_recipe(recipe: dict) -> None:
    for g in recipe.get("gestures", []):
        g["points"] = [[round(1.0 - x, 4), y] for x, y in g["points"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--trick", required=True,
                        help="Trick name for the mirrored recipe (e.g. 'LASER FLIP')")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.library.read_text())
    for key in ("median_gestures", "best_gestures"):
        if key in data:
            _mirror_recipe(data[key])
            _fix_recipe(data[key])  # shift back inside [X_BOUND_MIN, X_BOUND_MAX]
    source_trick = data.get("trick")
    data["trick"] = args.trick
    data["mirrored_from"] = f"{args.library.name} (trick: {source_trick})"
    data["notes"] = (f"x-mirrored from {args.library.name}; chirality swap "
                     f"{source_trick} -> {args.trick}. Bounds-remapped after mirror.")

    out = args.out or args.library.with_name(
        args.library.stem.split("_2026")[0] + "_MIRRORED_" + args.trick.lower().replace(" ", "_") + ".json"
    )
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{args.library.name} → {out}")


if __name__ == "__main__":
    main()
