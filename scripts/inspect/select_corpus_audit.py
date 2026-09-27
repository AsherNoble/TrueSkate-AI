#!/usr/bin/env python3
"""Select the M1-CORPUS-AUDIT blind random sample from a screened corpus.

Applies the frozen ``corpus-screen-v1`` (whole-segment: rate within 0.0008 and
both calibration detections 85-210 ms after their WDA submissions) to every
two-anchor clip, then draws a seeded uniform random sample of kept clips. The
selection file matches ``build_linear_onset_validation_viewer``. Expected
frames (the first stored frame with a non-negative time) go to a separate file
that is never given to the viewer.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from trueskate_ai.data.timing_screen import CORPUS_SCREEN_V1, TWO_ANCHOR_METHOD, passes_corpus_screen_v1


def expected_frame(meta: dict) -> int:
    return next(i for i, t in enumerate(meta["frame_times"]) if t >= 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path, help="Screen every two-anchor clip under this root.")
    source.add_argument("--manifest", type=Path,
                        help="Draw from a frozen cohort manifest (e.g. the final park mix); every "
                             "drawn clip is re-checked against corpus-screen-v1.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    manifests: dict[Path, dict] = {}
    kept, excluded = [], Counter()
    total = Counter()
    if args.manifest is not None:
        from trueskate_ai.data.cohort_manifest import manifest_entries, read_manifest

        cohort = read_manifest(args.manifest)
        args.corpus = Path(cohort["root_hint"])
        meta_paths = [args.corpus / entry["path"] / "meta.json" for entry in manifest_entries(cohort)]
    else:
        meta_paths = sorted(args.corpus.rglob("meta.json"))
    for meta_path in meta_paths:
        meta = json.loads(meta_path.read_text())
        if (meta.get("tap_calibration") or {}).get("method") != TWO_ANCHOR_METHOD:
            raise SystemExit(f"{meta_path}: not a two-anchor clip")
        session_dir = next(p for p in meta_path.parents if p.name == meta["session"])
        manifest_path = session_dir / f"segment_{int(meta['segment_index']):05d}.json"
        if manifest_path not in manifests:
            manifests[manifest_path] = json.loads(manifest_path.read_text())
        total[meta["park"]] += 1
        if passes_corpus_screen_v1(meta, manifests[manifest_path]):
            kept.append((meta_path.parent, meta))
        else:
            excluded[meta["park"]] += 1

    rng = random.Random(args.seed)
    chosen = rng.sample(kept, args.n)
    samples, expected = [], []
    for order, (path, meta) in enumerate(chosen, 1):
        source = str(path.relative_to(args.corpus))
        samples.append({"order": order, "source": source})
        expected.append({"order": order, "source": source, "park": meta["park"], "device": meta["device"],
                         "rate": meta["tap_calibration"]["rate"],
                         "expected_first_trace_frame_0based": expected_frame(meta)})

    args.out_dir.mkdir(parents=True, exist_ok=False)
    stage = args.out_dir / "selected"
    for sample in samples:
        link = stage / f"{sample['order'] - 1:03d}" / sample["source"]
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(args.corpus / sample["source"])
    counts = {"total_by_park": dict(total), "excluded_by_park": dict(excluded),
              "kept": len(kept), "total": sum(total.values())}
    (args.out_dir / "selection.json").write_text(json.dumps({
        "schema": "model1-onset-validation-selection-v1", "seed": args.seed,
        "source_corpus": str(args.corpus), "purpose": "M1-CORPUS-AUDIT random sample of screened corpus",
        "source_manifest": None if args.manifest is None else {
            "path": str(args.manifest), "fingerprint": cohort["fingerprint"]},
        "screen": CORPUS_SCREEN_V1, "counts": counts, "selected_count": len(samples),
        "samples": samples}, indent=1) + "\n")
    (args.out_dir / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")
    print(json.dumps(counts, indent=1))
    print("selected by park:", dict(Counter(e["park"] for e in expected)))


if __name__ == "__main__":
    main()
