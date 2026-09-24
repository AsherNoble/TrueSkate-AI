#!/usr/bin/env python3
"""Select a blind 24-clip check of the *mid-band* calibration-rate screen.

M1-ONSET-VALIDATION-20260924 validated the anchor-error formula only at
extreme rates (>1.004 or <0.995). This selects, from recordings unused by any
earlier audit/validation, all six mid_high and the one mid_low recordings available, plus five ordinary:

- mid_high: 1.0012 < rate <= 1.002 (start anchor implicated),
- mid_low: 0.998 <= rate < 0.9988 (end anchor implicated),
- ordinary: abs(rate - 1) < 0.0003,

taking each recording's earliest and latest admitted clip. Predicted first
visible trace frames are written to a separate predictions file (never given to
the viewer). The selection file matches ``build_linear_onset_validation_viewer``
and a staging tree of symlinks is created for it. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

FRAME_S = 1 / 30
QUOTAS = {"mid_high": 6, "mid_low": 1, "ordinary": 5}


def category(rate: float) -> str | None:
    if 1.0012 < rate <= 1.002:
        return "mid_high"
    if 0.998 <= rate < 0.9988:
        return "mid_low"
    if abs(rate - 1) < 0.0003:
        return "ordinary"
    return None


def predicted_delay(meta: dict) -> float:
    cal = meta["tap_calibration"]
    anchors = {d["role"]: d["submitted_to_ios_monotonic_s"] for d in cal["detections"]}
    g, rate = meta["wda_submitted_monotonic_s"], cal["rate"]
    if rate > 1.0012:
        return (rate - 1) * (anchors["end"] - g)
    if rate < 0.9988:
        return (1 - rate) * (g - anchors["start"])
    return 0.0


def predicted_frame(meta: dict) -> int:
    """First stored frame whose source-relative time is >= the predicted delay (0-based)."""
    delay = predicted_delay(meta)
    times = meta["frame_times"]
    return next(i for i, t in enumerate(times) if t >= delay - 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--exclude-selection", type=Path, action="append", default=[],
                    help="Earlier selection/audit JSON files whose sessions are excluded.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    excluded = set()
    for path in args.exclude_selection:
        text = path.read_text()
        excluded.update(part for part in text.replace('"', " ").replace("/", " ").split()
                        if part.startswith("iPhone_XR") and "_2026" in part)

    sessions: dict[str, list[tuple[int, Path, dict]]] = defaultdict(list)
    for meta_path in args.corpus.rglob("meta.json"):
        meta = json.loads(meta_path.read_text())
        cal = meta.get("tap_calibration") or {}
        if cal.get("method") != "wda-submitted-two-centre-controls-v2":
            continue
        session = str(meta.get("session"))
        if session in excluded or category(cal["rate"]) is None:
            continue
        sessions[session].append((int(meta["gesture_index"]), meta_path.parent, meta))

    by_category: dict[str, list[str]] = defaultdict(list)
    for session, clips in sessions.items():
        if len(clips) >= 2:
            by_category[category(clips[0][2]["tap_calibration"]["rate"])].append(session)

    rng = random.Random(args.seed)
    samples, predictions = [], []
    pair = 0
    for cat, quota in QUOTAS.items():
        pool = sorted(by_category[cat])
        if len(pool) < quota:
            raise SystemExit(f"only {len(pool)} {cat} recordings available")
        for session in rng.sample(pool, quota):
            pair += 1
            clips = sorted(sessions[session], key=lambda c: c[0])
            for role, (gesture, path, meta) in (("early", clips[0]), ("late", clips[-1])):
                source = str(path.relative_to(args.corpus))
                samples.append({"pair_id": pair, "role": role, "category": cat, "source": source,
                                "session": session, "gesture_index": gesture, "park": meta["park"],
                                "device": meta["device"], "rate": meta["tap_calibration"]["rate"]})
                predictions.append({"source": source, "category": cat, "role": role,
                                    "predicted_delay_s": round(predicted_delay(meta), 5),
                                    "predicted_first_trace_frame_0based": predicted_frame(meta),
                                    "nominal_onset_frame_0based": predicted_frame({**meta, "tap_calibration": {**meta["tap_calibration"], "rate": 1.0}})})
    rng.shuffle(samples)
    for order, sample in enumerate(samples, 1):
        sample["order"] = order

    args.out_dir.mkdir(parents=True, exist_ok=False)
    stage = args.out_dir / "selected"
    for sample in samples:
        link = stage / f"{sample['order'] - 1:03d}" / sample["source"]
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(args.corpus / sample["source"])
    (args.out_dir / "selection.json").write_text(json.dumps({
        "schema": "model1-onset-validation-selection-v1", "seed": args.seed,
        "source_corpus": str(args.corpus), "purpose": "mid-band rate screen check",
        "excluded_sessions": len(excluded), "quotas": QUOTAS,
        "selected_count": len(samples), "samples": samples}, indent=1) + "\n")
    (args.out_dir / "predictions.json").write_text(json.dumps(predictions, indent=1) + "\n")
    print(f"selected {len(samples)} clips from {sum(map(len, by_category.values()))} eligible recordings; "
          f"pools: {{k: len(v) for k, v in by_category.items()}}".replace("{k: len(v) for k, v in by_category.items()}",
                                                                          str({k: len(v) for k, v in by_category.items()})))


if __name__ == "__main__":
    main()
