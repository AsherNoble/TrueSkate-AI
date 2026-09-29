#!/usr/bin/env python3
"""Score M1-DIE5-COMPARE against blind human labels (protocol as amended).

Frozen before any real label exists. Inputs: the phase 2 analysis JSON, the
viewer selection key, and the viewer's exported labels. Uncertain labels are
excluded and counted. Frame errors use global source-frame indices.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from trueskate_ai.collection.die_five_calibration import classify_error, mcnemar_exact_p

LA = "SLS 2015 Los Angeles"
GROSS = ("gross_early", "gross_late")
FPS = 30.0


def _rate(n: int, d: int) -> float | None:
    return None if d == 0 else round(n / d, 4)


def probe_times(segment: str) -> list[float]:
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "collection" / "align_xctest_traces.py"
    spec = importlib.util.spec_from_file_location("die5_score_aligner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._probe_video_frame_times(Path(segment).with_suffix(".mov"))


def score(analysis: list[dict], key: dict, labels: dict, frame_times=probe_times) -> dict:
    segments = {seg["segment"]: seg for seg in analysis}
    markers = {(seg["segment"], m["wda_action_sequence"]): m for seg in analysis for m in seg["markers"]}
    human: dict[tuple, int] = {}
    uncertain = unlabelled = 0
    items = []
    for item in key["items"]:
        label = labels.get(item["id"])
        if label is None:
            unlabelled += 1
            continue
        if label.get("uncertain") or label.get("frame_index_0based") is None:
            uncertain += 1
            continue
        frame = item["first_global_frame"] + int(label["frame_index_0based"])
        k = (item["segment"], item["wda_action_sequence"])
        human[k] = frame
        m = markers[k]
        seg = segments[item["segment"]]
        items.append({
            **item, "device": seg["device"], "park": seg["park"], "human_frame": frame,
            "a_class": classify_error(m.get("a_frame"), frame),
            "b_class": None if item["marker"] == "single" else classify_error(m.get("b_frame"), frame),
            "a_score": m.get("a_score"),
        })

    # 1. Primary: LA start markers, one per recording.
    starts = [i for i in items if i["role"] == "start" and i["park"] == LA]
    a_only = [i for i in starts if i["a_class"] in GROSS and i["b_class"] not in GROSS]
    b_only = [i for i in starts if i["b_class"] in GROSS and i["a_class"] not in GROSS]
    primary = {
        "n_recordings": len(starts),
        "a_gross": sum(i["a_class"] in GROSS for i in starts),
        "b_gross": sum(i["b_class"] in GROSS for i in starts),
        "a_no_result": sum(i["a_class"] == "no_result" for i in starts),
        "b_no_result": sum(i["b_class"] == "no_result" for i in starts),
        "discordant_a_only": len(a_only),
        "discordant_b_only": len(b_only),
        "mcnemar_exact_p": mcnemar_exact_p(len(a_only), len(b_only)),
        "discordant_recordings": [
            {"segment": i["segment"], "device": i["device"], "a": i["a_class"], "b": i["b_class"]}
            for i in a_only + b_only
        ],
        "a_classes": Counter(i["a_class"] for i in starts),
        "b_classes": Counter(i["b_class"] for i in starts),
    }

    # 2. B gross errors on accepted labelled die-five markers, with rule of three.
    accepted = [i for i in items if i["b_class"] not in (None, "no_result")]
    b_gross_accepted = sum(i["b_class"] in GROSS for i in accepted)
    b_accepted = {
        "n": len(accepted), "gross": b_gross_accepted,
        "rule_of_three_upper": round(3 / len(accepted), 4) if accepted and not b_gross_accepted else None,
    }

    # 3. Rejection: B no-result on start markers per device (≤5% each).
    rejection = {}
    for device in sorted({i["device"] for i in items}):
        dev = [i for i in items if i["device"] == device and i["role"] == "start"]
        rejection[device] = {"starts": len(dev),
                             "b_no_result_rate": _rate(sum(i["b_class"] == "no_result" for i in dev), len(dev))}

    # 4. Die-five mid markers: inverse-probability weighted gross rates.
    pools = Counter()
    for seg in analysis:
        for m in seg["markers"]:
            if m["role"] == "mid" and m["marker"] == "die_five":
                agree = m.get("a_frame") is not None and m.get("a_frame") == m.get("b_frame")
                pools["agree" if agree else "disagree_or_no_result"] += 1
    chosen = Counter(i["stratum"] for i in key["items"] if i["role"] == "mid" and i["marker"] == "die_five")
    weights = {s: pools[s] / chosen[s] for s in chosen if chosen[s]}
    mids = [i for i in items if i["role"] == "mid" and i["marker"] == "die_five"]
    weighted = {}
    for det in ("a", "b"):
        w_total = sum(weights[i["stratum"]] for i in mids)
        w_gross = sum(weights[i["stratum"]] for i in mids if i[f"{det}_class"] in GROSS)
        weighted[det] = None if not w_total else round(w_gross / w_total, 4)

    # 5. Corner check on labelled mids.
    def within(i):
        return i["a_class"] in ("exact", "within_tolerance")
    singles = [i for i in items if i["role"] == "mid" and i["marker"] == "single"]
    corner = {
        "n_die_five": len(mids), "n_single": len(singles),
        "a_within_die_five": _rate(sum(map(within, mids)), len(mids)),
        "a_within_single": _rate(sum(map(within, singles)), len(singles)),
    }
    sd = [i["a_score"] for i in mids if i["a_score"] is not None]
    ss = [i["a_score"] for i in singles if i["a_score"] is not None]
    corner["a_median_score_ratio"] = (round(statistics.median(sd) / statistics.median(ss), 4)
                                      if sd and ss else None)
    corner["pass"] = (None if corner["a_within_die_five"] is None or corner["a_within_single"] is None
                      or corner["a_median_score_ratio"] is None else
                      corner["a_within_die_five"] >= corner["a_within_single"] - 0.10
                      and 0.8 <= corner["a_median_score_ratio"] <= 1.25)

    # 6. Fit replay: human-anchored rate, and mid residuals (frames) vs human labels.
    cache: dict[str, list[float]] = {}

    def times(seg):
        return cache.setdefault(seg["segment"], frame_times(seg["segment"]))

    replay = defaultdict(list)
    residuals = defaultdict(list)
    for seg in analysis:
        start = next((m for m in seg["markers"] if m["role"] == "start"), None)
        end = next((m for m in seg["markers"] if m["role"] == "end"), None)
        if start is None or end is None:
            continue
        hs, he = human.get((seg["segment"], start["wda_action_sequence"])), human.get((seg["segment"], end["wda_action_sequence"]))
        if hs is not None and he is not None:
            rate = (times(seg)[he] - times(seg)[hs]) / (
                end["wda_submitted_monotonic_s"] - start["wda_submitted_monotonic_s"])
            replay["human"].append(abs(rate - 1) > 0.002)
        for det in ("a", "b"):
            fit = seg["fits"].get(det)
            if fit is None or "rate" not in fit:
                replay[det].append(None)
                continue
            replay[det].append(abs(fit["rate"] - 1) > 0.002)
            for seq, predicted_s in fit["mid_predicted_s"].items():
                h = human.get((seg["segment"], int(seq)))
                if h is not None:
                    residuals[det].append(round((times(seg)[h] - predicted_s) * FPS, 3))
    fit_replay = {det: {"segments": len(v), "no_fit": v.count(None),
                        "anomalous": sum(x is True for x in v)} for det, v in replay.items()}
    for det, res in residuals.items():
        fit_replay[det]["mid_residual_frames_abs_max"] = max(map(abs, res)) if res else None
        fit_replay[det]["mid_residual_within_1"] = _rate(sum(abs(r) <= 1.0 for r in res), len(res))

    # 7. False alarms per second (no labels needed).
    fa = {}
    for det in ("a", "b"):
        fa[det] = {
            "steady_windows": sum(len(s["null_windows"]) for s in analysis),
            "steady_alarms": sum(n.get(f"{det}_frame") is not None for s in analysis for n in s["null_windows"]),
            "steady_scanned_s": round(sum(s.get("null_scanned_s", 0) for s in analysis), 3),
            "post_reset_windows": sum(s.get("post_reset_null") is not None for s in analysis),
            "post_reset_alarms": sum((s.get("post_reset_null") or {}).get(f"{det}_frame") is not None for s in analysis),
            "post_reset_scanned_s": round(sum((s.get("post_reset_null") or {}).get("scanned_s", 0) for s in analysis), 3),
        }
        for kind in ("steady", "post_reset"):
            secs = fa[det][f"{kind}_scanned_s"]
            fa[det][f"{kind}_alarms_per_s"] = None if not secs else round(fa[det][f"{kind}_alarms"] / secs, 5)

    return {
        "labelled": len(items), "uncertain": uncertain, "unlabelled": unlabelled,
        "primary_la_start": primary, "b_accepted": b_accepted, "b_start_rejection": rejection,
        "mid_weighted_gross_rate": weighted, "mid_stratum_weights": weights,
        "corner_check": corner, "fit_replay": fit_replay, "false_alarms": fa,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text())
    if labels.get("schema") != "die5-compare-labels-v1":
        raise SystemExit("unexpected label schema")
    key = json.loads(args.key.read_text())
    if str(key["seed"]) != str(labels["selection_seed"]):
        raise SystemExit("labels were made for a different selection")
    result = score(json.loads(args.analysis.read_text()), key, labels["labels"])
    args.out.write_text(json.dumps(result, indent=1, default=dict) + "\n")
    print(json.dumps(result, indent=1, default=dict))


if __name__ == "__main__":
    main()
