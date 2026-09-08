#!/usr/bin/env python3
"""Reproducible phase-1 analysis for EQ-051 training logs.

Only Python stdlib + numpy are used.  Run from the directory containing the
nine eq046_*.log files:
    python3 analyze_eq051.py

Definitions (kept deliberately mechanical):
  * A collapse is an epoch whose recovery is >5.0 percentage points lower than
    the immediately preceding epoch.
  * The pre-collapse local level is recovery at the preceding epoch.  A
    candidate is a spike if next epoch recovers to at least that level; it is
    sustained if collapse and the following two epochs are all below it.
    End-of-log or in-between cases are reported as "other", not forced.
  * Component deltas are collapse epoch minus preceding epoch.  For the three
    error metrics, positive means worse.  "endpoints" means either start or
    end median error worsened; "both" means endpoints and duration worsened.
"""

from __future__ import annotations

import glob
import os
import re
from collections import Counter, defaultdict

import numpy as np

PATTERN = re.compile(
    r"^epoch=(?P<epoch>\d+)\s+"
    r"val_start_med=(?P<start>[0-9.]+)\s+"
    r"val_end_med=(?P<end>[0-9.]+)\s+"
    r"val_duration_mae=(?P<duration>[0-9.]+)\s+"
    r"val_recovery=(?P<recovery>[0-9.]+)%\s+secs=(?P<secs>[0-9.]+)$"
)
NAME = re.compile(r"^eq046_(base|lf001|lf002)_s([123])\.log$")
FIELDS = ("start", "end", "duration", "recovery", "secs")


def parse_logs():
    runs = {}
    for path in sorted(glob.glob("eq046_*.log")):
        match = NAME.match(os.path.basename(path))
        if not match:
            continue
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                found = PATTERN.match(line.strip())
                if found:
                    row = {key: float(found[key]) for key in FIELDS}
                    row["epoch"] = int(found["epoch"])
                    rows.append(row)
        rows.sort(key=lambda row: row["epoch"])
        if [row["epoch"] for row in rows] != list(range(1, 41)):
            raise ValueError(f"{path}: expected exactly epochs 1..40")
        runs[f"{match.group(1)}_s{match.group(2)}"] = rows
    if len(runs) != 9:
        raise ValueError(f"expected 9 named logs, found {len(runs)}")
    return runs


def classify(rows, i):
    """Classify collapse at zero-based index i using preceding recovery level."""
    level = rows[i - 1]["recovery"]
    if i + 1 < len(rows) and rows[i + 1]["recovery"] >= level:
        return "spike"
    if i + 2 < len(rows) and all(rows[j]["recovery"] < level for j in (i, i + 1, i + 2)):
        return "sustained"
    return "other"


def component_label(ds, de, dd):
    endpoints = ds > 0 or de > 0
    duration = dd > 0
    if endpoints and duration:
        return "both"
    if endpoints:
        return "endpoints"
    if duration:
        return "duration"
    return "neither"


def median_iqr(values):
    values = np.asarray(values, dtype=float)
    return np.median(values), np.percentile(values, 25), np.percentile(values, 75)


def auc_binary(feature, y):
    """Mann-Whitney AUC; ties receive half credit. NaN if either class absent."""
    feature, y = np.asarray(feature, float), np.asarray(y, bool)
    pos, neg = feature[y], feature[~y]
    if not len(pos) or not len(neg):
        return np.nan
    return (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))


def fmt_triplet(values, digits=1):
    med, q1, q3 = median_iqr(values)
    return f"{med:.{digits}f} [{q1:.{digits}f},{q3:.{digits}f}]"


def main():
    runs = parse_logs()
    events = []
    per_run = {}
    for name, rows in runs.items():
        recoveries = np.array([row["recovery"] for row in rows])
        found = []
        for i in range(1, len(rows)):
            drop = recoveries[i] - recoveries[i - 1]
            if drop < -5.0:
                current, previous = rows[i], rows[i - 1]
                ds = current["start"] - previous["start"]
                de = current["end"] - previous["end"]
                dd = current["duration"] - previous["duration"]
                event = {
                    "run": name, "arm": name.rsplit("_s", 1)[0], "epoch": current["epoch"],
                    "drop": drop, "class": classify(rows, i), "ds": ds, "de": de, "dd": dd,
                    "component": component_label(ds, de, dd),
                }
                events.append(event)
                found.append(event)
        per_run[name] = {
            "events": found,
            "tail_sd": np.std(recoveries[20:40], ddof=1),
            "tail_mean": np.mean(recoveries[20:40]),
            "min": np.min(recoveries), "maxdrop": np.min(np.diff(recoveries)),
        }

    print("EQ-051 | exact inputs: 9 runs x 40 epochs | collapse: delta recovery < -5.0 pp")
    print("\nRUN                         n  spike sust other  tail_mean tail_sd  min_rec  worst_drop")
    for name in sorted(per_run):
        values = per_run[name]
        counts = Counter(event["class"] for event in values["events"])
        print(f"{name:27s} {len(values['events']):2d} {counts['spike']:6d} {counts['sustained']:4d}"
              f" {counts['other']:5d} {values['tail_mean']:10.2f} {values['tail_sd']:7.2f}"
              f" {values['min']:8.1f} {values['maxdrop']:11.1f}")

    print("\nCOLLAPSE EVENTS (deltas: collapse epoch - previous epoch; positive error delta = worse)")
    print("run                         ep  drop_pp  shape       d_start    d_end    d_dur    component")
    for event in events:
        print(f"{event['run']:27s} {event['epoch']:2d} {event['drop']:8.1f} {event['class']:10s}"
              f" {event['ds']:+8.4f} {event['de']:+8.4f} {event['dd']:+8.4f}  {event['component']}")

    drops = np.array([-event["drop"] for event in events])
    print("\nCOMPONENT CO-MOVEMENT AT 57 COLLAPSES (not a per-clip causal attribution)")
    print("metric     worsened/events  median delta [Q1,Q3]        corr(delta, drop size)")
    for field, label in (("ds", "start"), ("de", "end"), ("dd", "duration")):
        deltas = np.array([event[field] for event in events])
        print(f"{label:9s} {np.sum(deltas > 0):2d}/57           {fmt_triplet(deltas, 4):25s} {np.corrcoef(deltas, drops)[0, 1]:+.2f}")
    component_counts = Counter(event["component"] for event in events)
    print(f"event labels: endpoints-only={component_counts['endpoints']}, duration-only={component_counts['duration']}, "
          f"both={component_counts['both']}, neither={component_counts['neither']}")

    print("\nARM SUMMARY (n=3 each; descriptive only)")
    print("arm     runs  collapses  spike/sust/other  tail-sd median [Q1,Q3]  collapse pp median [Q1,Q3]")
    for arm in ("base", "lf001", "lf002"):
        subset = [per_run[name] for name in sorted(per_run) if name.startswith(arm + "_")]
        arm_events = [event for event in events if event["arm"] == arm]
        counts = Counter(event["class"] for event in arm_events)
        print(f"{arm:7s} {len(subset):4d} {len(arm_events):10d}  {counts['spike']}/{counts['sustained']}/{counts['other']}"
              f"          {fmt_triplet([item['tail_sd'] for item in subset], 2):22s}"
              f" {fmt_triplet([-item['drop'] for item in arm_events], 1) if arm_events else 'NA':s}")

    others = [item for name, item in per_run.items() if name != "base_s1"]
    base_s1 = per_run["base_s1"]
    print("\nBASE_S1 VS OTHER EIGHT (descriptive; one run vs n=8)")
    print(f"base_s1: collapses={len(base_s1['events'])}, tail_mean={base_s1['tail_mean']:.2f}, "
          f"tail_sd={base_s1['tail_sd']:.2f}, min_recovery={base_s1['min']:.1f}, "
          f"worst_drop={base_s1['maxdrop']:.1f}")
    print(f"other 8: tail_mean {fmt_triplet([item['tail_mean'] for item in others], 2)}, "
          f"tail_sd {fmt_triplet([item['tail_sd'] for item in others], 2)}, "
          f"collapse count {fmt_triplet([len(item['events']) for item in others], 1)}, "
          f"worst_drop {fmt_triplet([item['maxdrop'] for item in others], 1)}")

    # One-epoch-ahead, univariate screening. Feature is known at epoch t-1 and
    # label says whether t is a collapse. Use raw values and prior one-epoch deltas.
    feature_rows = []
    for rows in runs.values():
        for i in range(2, len(rows)):
            prev, before, current = rows[i - 1], rows[i - 2], rows[i]
            record = {"collapse_next": current["recovery"] - prev["recovery"] < -5.0}
            for field in FIELDS:
                record[field] = prev[field]
                record["d_" + field] = prev[field] - before[field]
            feature_rows.append(record)
    y = np.array([row["collapse_next"] for row in feature_rows])
    print("\nONE-EPOCH-AHEAD SCREEN (333 transitions; positives=%d; raw error/secs high is worse)" % y.sum())
    print("feature          collapse median [Q1,Q3]       non-collapse median [Q1,Q3]   AUC  direction")
    for field in ("start", "end", "duration", "recovery", "secs", "d_start", "d_end", "d_duration", "d_recovery", "d_secs"):
        x = np.array([row[field] for row in feature_rows])
        raw_auc = auc_binary(x, y)
        auc = max(raw_auc, 1.0 - raw_auc)
        direction = "higher" if raw_auc >= 0.5 else "lower"
        digits = 4 if field not in ("recovery", "secs", "d_recovery", "d_secs") else 1
        print(f"{field:16s} {fmt_triplet(x[y], digits):30s} {fmt_triplet(x[~y], digits):30s}"
              f" {auc:4.2f} {direction}")
    print("AUC is descriptive only: repeated epochs within runs are not independent, and this is an uncorrected 10-feature screen.")


if __name__ == "__main__":
    main()
