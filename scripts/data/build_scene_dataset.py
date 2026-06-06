"""Build a train/val manifest for the skatepark scene classifier.

Positives (in-skatepark frames) are auto-mined from existing repo imagery.
Negatives (home screen / menus / other apps) come from a folder you populate.
See experiments/scene_classifier_journal.md.

Usage:
    python scripts/data/build_scene_dataset.py \
        --negatives-dir data/scene/negatives

Writes data/scene/manifest.json: {"train":[{path,label}], "val":[...]} where
label 1 = skatepark, 0 = not. Classes are balanced (down to the smaller class)
so a naive model can't win by predicting the majority.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

_IMG_EXTS = (".png", ".jpg", ".jpeg")

# Default places in-skatepark frames already live (see journal).
_DEFAULT_POSITIVE_GLOBS = [
    "data/extracted_frames/*/img_*.jpg",
    "logs/runs/*/ocr_failures/*/frame_*.png",
    "logs/runs/*/frames/*/frame_*.png",
    "tmp/*.png",
]


def _gather(globs: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in globs:
        found.extend(glob.glob(str(_REPO_ROOT / pattern)))
    # Stable de-dup.
    return sorted(set(found))


def _gather_dir(d: Path) -> list[str]:
    if not d.exists():
        return []
    return sorted(
        str(p) for p in d.rglob("*") if p.suffix.lower() in _IMG_EXTS
    )


def _split(items: list[dict], val_frac: float) -> tuple[list, list]:
    n_val = max(1, int(len(items) * val_frac)) if items else 0
    # Deterministic stride sampling for val (no RNG → reproducible manifests).
    val_idx = set(range(0, len(items), max(1, len(items) // n_val))[:n_val]) if n_val else set()
    train = [it for i, it in enumerate(items) if i not in val_idx]
    val = [it for i, it in enumerate(items) if i in val_idx]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene-classifier manifest.")
    parser.add_argument("--negatives-dir", type=Path, default=_REPO_ROOT / "data" / "scene" / "negatives",
                        help="Folder of not-in-skatepark images (recursed).")
    parser.add_argument("--positive-glob", action="append", default=None,
                        help="Extra positive globs (repeatable). Added to defaults.")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "data" / "scene" / "manifest.json")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=4000,
                        help="Cap each class before balancing (avoid huge manifests).")
    args = parser.parse_args()

    pos_globs = list(_DEFAULT_POSITIVE_GLOBS) + (args.positive_glob or [])
    positives = _gather(pos_globs)
    negatives = _gather_dir(args.negatives_dir)

    print(f"Found {len(positives)} positive, {len(negatives)} negative images.")
    if not negatives:
        print(
            f"\nNo negatives found in {args.negatives_dir}.\n"
            "Collect screenshots of the iOS home screen, the app switcher, the\n"
            "True Skate pause/level-select menus, and other apps, drop them in\n"
            "that folder, and re-run. (See the journal.)"
        )
        sys.exit(1)

    # Cap then balance to the smaller class.
    positives = positives[: args.max_per_class]
    negatives = negatives[: args.max_per_class]
    n = min(len(positives), len(negatives))
    positives, negatives = positives[:n], negatives[:n]
    print(f"Balanced to {n} per class.")

    pos_items = [{"path": p, "label": 1} for p in positives]
    neg_items = [{"path": p, "label": 0} for p in negatives]

    pos_tr, pos_val = _split(pos_items, args.val_frac)
    neg_tr, neg_val = _split(neg_items, args.val_frac)
    # Interleave classes so batches are mixed even without shuffling.
    train = [x for pair in zip(pos_tr, neg_tr) for x in pair]
    val = [x for pair in zip(pos_val, neg_val) for x in pair]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"train": train, "val": val}, indent=2))
    print(f"Wrote {args.out}: {len(train)} train, {len(val)} val.")


if __name__ == "__main__":
    main()
