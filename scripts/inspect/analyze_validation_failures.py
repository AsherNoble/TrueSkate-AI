#!/usr/bin/env python3
"""Summarise M1-DIAG validation autopsies across seeds (read-only).

Takes the per-seed ``autopsy_failures`` JSON files (validation partition) and
reports, over the failing clips, which component fails, the along/perpendicular
error geometry, rendered-trail evidence at the missed endpoint, where failures
concentrate (park, device, command duration, endpoint near the frame edge), and
how consistently the seeds fail the same clips.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

TOL = 0.03          # recovery endpoint tolerance (normalised)
DUR_TOL = 0.10      # recovery duration tolerance (s)
EDGE = 0.05         # "near the frame edge" margin (normalised)


def component(record: dict) -> str:
    parts = [name for name, failed in (("start", record["start_error"] > TOL),
                                       ("end", record["end_error"] > TOL),
                                       ("duration", record["duration_error"] > DUR_TOL)) if failed]
    return "+".join(parts) or "none"


def rate(failed: int, total: int) -> str:
    return f"{failed}/{total} ({100 * failed / max(total, 1):.1f}%)"


def breakdown(records: list[dict], key) -> dict[str, str]:
    total, failed = Counter(), Counter()
    for record in records:
        group = key(record)
        total[group] += 1
        failed[group] += not record["recovered"]
    return {group: rate(failed[group], total[group]) for group in sorted(total)}


def duration_bin(record: dict) -> str:
    duration = record["gesture_duration"]
    for edge in (0.45, 0.60, 0.75, 0.90, 1.05):
        if duration < edge:
            return f"<{edge:.2f}s"
    return ">=1.05s"


def end_near_edge(record: dict) -> str:
    x, y = record["commanded"][-3], record["commanded"][-2]
    near = min(x, y, 1 - x, 1 - y) < EDGE
    return "end within 0.05 of edge" if near else "end interior"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("autopsy", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report: dict = {"seeds": {}}
    failures_by_sample: dict[str, int] = defaultdict(int)
    all_samples: set[str] = set()
    for path in args.autopsy:
        payload = json.loads(path.read_text())
        if payload["partition"] != "validation":
            raise SystemExit(f"{path}: not a validation autopsy")
        records = payload["all_records"]
        failures = [record for record in records if not record["recovered"]]
        end_failures = [record for record in failures if record["end_error"] > TOL]
        for record in records:
            all_samples.add(record["sample"])
            failures_by_sample[record["sample"]] += not record["recovered"]
        report["seeds"][payload["checkpoint"]] = {
            "recovery": rate(len(records) - len(failures), len(records)),
            "failure_components": dict(Counter(component(r) for r in failures).most_common()),
            "end_failures": {
                "count": len(end_failures),
                "along_short_(<0)": sum(r["end_along"] < 0 for r in end_failures),
                "along_long_(>0)": sum(r["end_along"] > 0 for r in end_failures),
                "along_dominant_(|along|>perp)": sum(abs(r["end_along"]) > r["end_perp"]
                                                     for r in end_failures),
                "median_abs_along": median(abs(r["end_along"]) for r in end_failures),
                "median_perp": median(r["end_perp"] for r in end_failures),
                "trail_gap_at_commanded_end": {
                    "median": median(r["trail_gap_end"] for r in end_failures),
                    "<0.01_(visible)": sum(r["trail_gap_end"] < 0.01 for r in end_failures),
                    ">0.03_(no_evidence)": sum(r["trail_gap_end"] > 0.03 for r in end_failures),
                },
                "median_trail_gap_end_recovered_clips": median(
                    r["trail_gap_end"] for r in records if r["recovered"]),
            },
            "by_park": breakdown(records, lambda r: r["park"]),
            "by_device": breakdown(records, lambda r: r["device"]),
            "by_command_duration": breakdown(records, duration_bin),
            "by_end_position": breakdown(records, end_near_edge),
        }
    seeds = len(args.autopsy)
    report["seed_consistency"] = {
        f"failed_by_{k}_of_{seeds}": sum(count == k for count in failures_by_sample.values())
        for k in range(1, seeds + 1)
    }
    report["seed_consistency"]["clips"] = len(all_samples)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
