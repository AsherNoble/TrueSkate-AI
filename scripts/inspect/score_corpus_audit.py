#!/usr/bin/env python3
"""Score the M1-CORPUS-AUDIT blind random sample, frozen before labels.

A clip is satisfactory if it is labelled (not unclear) and its labelled first
visible trace is within one displayed frame of the expected frame (the first
stored frame with a non-negative time). The audit passes only if every clip is
satisfactory. Exact matches are reported as a secondary measure.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TOLERANCE_FRAMES = 1


def score(selection: dict, expected: list[dict], labels: dict) -> dict:
    by_source = {e["source"]: e for e in expected}
    rows = []
    for sample in selection["samples"]:
        label = labels.get(f"{sample['order'] - 1:03d}/{sample['source']}")
        exp = by_source[sample["source"]]
        frame = None if not label or label.get("uncertain") else label.get("frame_index_0based")
        diff = None if frame is None else frame - exp["expected_first_trace_frame_0based"]
        rows.append({"order": sample["order"], "source": sample["source"], "park": exp["park"],
                     "device": exp["device"], "rate": exp["rate"],
                     "expected_frame_1based": exp["expected_first_trace_frame_0based"] + 1,
                     "labelled_frame_1based": None if frame is None else frame + 1, "difference": diff,
                     "unclear": bool(label and label.get("uncertain")), "missing": label is None,
                     "note": (label or {}).get("note", ""),
                     "satisfactory": diff is not None and abs(diff) <= TOLERANCE_FRAMES})
    n = len(rows)
    ok = sum(r["satisfactory"] for r in rows)
    return {
        "n": n, "satisfactory": ok, "verdict": "pass" if ok == n else "fail",
        "exact": sum(r["difference"] == 0 for r in rows),
        "difference_counts": dict(sorted(Counter(r["difference"] for r in rows if r["difference"] is not None).items())),
        "unclear": sum(r["unclear"] for r in rows), "missing": sum(r["missing"] for r in rows),
        "rule_of_three_upper_if_pass": round(3 / n, 4),
        "failures": [r for r in rows if not r["satisfactory"]],
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = score(json.loads(args.selection.read_text()), json.loads(args.expected.read_text()),
                   json.loads(args.labels.read_text())["labels"])
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
