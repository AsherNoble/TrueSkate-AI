#!/usr/bin/env python3
"""M1-TIMING-GATE-20260926: held-out, label-free check of the command-latency gate.

For every aligned segment in a corpus, the stored start/end calibration
detections (``onset_video_s``) are compared with each control's command video
time ``wda_submitted_epoch_s - started_at_epoch_s`` from the segment manifest.
The gate accepts latencies in [GATE_LOW_S, GATE_HIGH_S]. Criteria are fixed in
the experiment record. Read-only; standard library only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

GATE_LOW_S = 0.085
GATE_HIGH_S = 0.210
SCREEN_RATE = 0.0008


def category(rate: float) -> str:
    if rate > 1.002:
        return "high"
    if rate < 0.998:
        return "low"
    if abs(rate - 1) < 0.0003:
        return "ordinary"
    return "other"


def segments(corpus: Path):
    """Yield (manifest_path, first clip meta, clip count) per aligned segment."""
    clips: dict[tuple[Path, int], list[dict]] = defaultdict(list)
    for meta_path in corpus.rglob("meta.json"):
        meta = json.loads(meta_path.read_text())
        cal = meta.get("tap_calibration") or {}
        if cal.get("method") != "wda-submitted-two-centre-controls-v2":
            continue
        session_dir = next(p for p in meta_path.parents if p.name == meta["session"])
        clips[(session_dir, int(meta["segment_index"]))].append(meta)
    for (session_dir, index), metas in sorted(clips.items()):
        yield session_dir / f"segment_{index:05d}.json", metas[0], len(metas)


def analyse(corpus: Path) -> dict:
    rows = []
    for manifest_path, meta, n_clips in segments(corpus):
        manifest = json.loads(manifest_path.read_text())
        started = manifest["started_at_epoch_s"]
        command_s = {e["wda_action_sequence"]: e["wda_submitted_epoch_s"] - started
                     for e in manifest["gestures"] if e.get("calibration_control")}
        cal = meta["tap_calibration"]
        latency = {d["role"]: d["onset_video_s"] - command_s[d["wda_action_sequence"]]
                   for d in cal["detections"]}
        rows.append({"segment": str(manifest_path.relative_to(corpus)), "device": meta["device"],
                     "park": meta["park"], "rate": cal["rate"], "category": category(cal["rate"]),
                     "clips": n_clips, "latency_s": {k: round(v, 4) for k, v in latency.items()}})

    def passes(value: float) -> bool:
        return GATE_LOW_S <= value <= GATE_HIGH_S

    def share(items, role, want_pass):
        vals = [r["latency_s"][role] for r in items if role in r["latency_s"]]
        return {"n": len(vals), "k": sum(passes(v) == want_pass for v in vals)}

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    ordinary, high, low = by_cat["ordinary"], by_cat["high"], by_cat["low"]
    c1s, c1e = share(ordinary, "start", True), share(ordinary, "end", True)
    c2s = {"n": len(high), "k": sum(r["latency_s"]["start"] < GATE_LOW_S for r in high)}
    c2e = share(high, "end", True)
    crit1 = c1s["k"] >= 0.95 * c1s["n"] and c1e["k"] >= 0.95 * c1e["n"]
    crit2 = c2s["k"] >= 0.80 * c2s["n"] and c2e["k"] >= 0.90 * c2e["n"]

    def describe(vals):
        vals = sorted(vals)
        if not vals:
            return None
        q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
        return {"n": len(vals), "min": vals[0], "p1": q(0.01), "p5": q(0.05), "median": statistics.median(vals),
                "p95": q(0.95), "p99": q(0.99), "max": vals[-1]}

    distribution = {}
    for key in sorted({(r["device"], r["park"]) for r in rows}):
        group = [r for r in rows if (r["device"], r["park"]) == key and r["category"] == "ordinary"]
        distribution[" / ".join(key)] = {role: describe([r["latency_s"][role] for r in group if role in r["latency_s"]])
                                         for role in ("start", "end")}

    gate_fail = [r for r in rows if not all(passes(v) for v in r["latency_s"].values())]
    screened = [r for r in rows if abs(r["rate"] - 1) > SCREEN_RATE]
    fail_ids, screen_ids = {r["segment"] for r in gate_fail}, {r["segment"] for r in screened}
    clips = lambda ids: sum(r["clips"] for r in rows if r["segment"] in ids)
    return {
        "gate_s": [GATE_LOW_S, GATE_HIGH_S],
        "segments": len(rows), "categories": dict(Counter(r["category"] for r in rows)),
        "criterion_1_ordinary_pass": {"start": c1s, "end": c1e, "met": crit1},
        "criterion_2_high_rate": {"start_fails_early": c2s, "end_passes": c2e, "met": crit2},
        "verdict": "supported" if crit1 and crit2 else ("mapping_not_accurate" if not crit1 else "misses_faults"),
        "low_rate_descriptive": [{"rate": r["rate"], "latency_s": r["latency_s"], "device": r["device"],
                                  "park": r["park"]} for r in low],
        "ordinary_latency_by_device_park": distribution,
        "gate_vs_screen": {
            "gate_fail_segments": len(fail_ids), "gate_fail_clips": clips(fail_ids),
            "screen_0.0008_segments": len(screen_ids), "screen_0.0008_clips": clips(screen_ids),
            "both_segments": len(fail_ids & screen_ids), "both_clips": clips(fail_ids & screen_ids),
            "gate_only_segments": len(fail_ids - screen_ids), "screen_only_segments": len(screen_ids - fail_ids),
            "gate_fail_by_category": dict(Counter(r["category"] for r in gate_fail)),
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = analyse(args.corpus)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "low_rate_descriptive")}, indent=1))


if __name__ == "__main__":
    main()
