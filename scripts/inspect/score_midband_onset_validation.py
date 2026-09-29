#!/usr/bin/env python3
"""Score the sealed mid-band onset check (M1-SCREEN-DRAFT-20260925), frozen before labels.

Criterion: mid-band lateness is supported if >=4/6 mid-high early clips are
labelled at displayed frame >= 9 and <=1/5 ordinary early clips are; not
supported if <=2/6 mid-high early clips are; otherwise inconclusive. Also
reports exact agreement with the sealed predicted frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LATE_FRAME_1BASED = 9


def score(selection: dict, predictions: list[dict], labels: dict) -> dict:
    predicted = {p["source"]: p for p in predictions}
    rows = []
    for sample in selection["samples"]:
        item_id = f"{sample['order'] - 1:03d}/{sample['source']}"
        label = labels.get(item_id)
        frame = None if not label or label.get("uncertain") else label.get("displayed_frame_1based")
        pred = predicted[sample["source"]]
        rows.append({"category": sample["category"], "role": sample["role"], "labelled_frame": frame,
                     "predicted_frame": pred["predicted_first_trace_frame_0based"] + 1,
                     "uncertain": bool(label and label.get("uncertain")), "missing": label is None})

    def late(category: str, role: str = "early") -> tuple[int, int]:
        chosen = [r for r in rows if r["category"] == category and r["role"] == role and r["labelled_frame"]]
        return sum(r["labelled_frame"] >= LATE_FRAME_1BASED for r in chosen), len(chosen)

    mid_late, mid_n = late("mid_high")
    ord_late, ord_n = late("ordinary")
    if mid_late >= 4 and ord_late <= 1:
        verdict = "supported"
    elif mid_late <= 2:
        verdict = "not_supported"
    else:
        verdict = "inconclusive"
    labelled = [r for r in rows if r["labelled_frame"]]
    return {
        "verdict": verdict,
        "mid_high_early_late": [mid_late, mid_n],
        "ordinary_early_late": [ord_late, ord_n],
        "mid_high_late_clip_late": list(late("mid_high", "late")),
        "exact_prediction_matches": [sum(r["labelled_frame"] == r["predicted_frame"] for r in labelled), len(labelled)],
        "uncertain": sum(r["uncertain"] for r in rows), "missing": sum(r["missing"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    selection = json.loads(args.selection.read_text())
    export = json.loads(args.labels.read_text())
    if str(export.get("selection_seed")) != str(selection["seed"]):
        raise SystemExit("labels were exported for a different selection seed")
    result = score(selection, json.loads(args.predictions.read_text()), export["labels"])
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
