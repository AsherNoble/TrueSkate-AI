"""Aggregate per-frame trace-vs-ring lag (vlag-all.json) for round 6. Positive = trace behind ring."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

S = Path(sys.argv[1]); M = S/"audit6-mirror"
rows = json.loads((S/"lag/vlag-all.json").read_text())
manifest = {str(x["id"]): x for x in json.loads((M/"audit-6/manifest.json").read_text())["samples"]}
review = {str(r["id"]): r for r in json.loads(sorted((S/"review-backup").glob("curved-audit-6-*.json"))[0].read_text())["samples"]}
MIN_FRAMES, ALIGNED = 3, 8.0   # ms: about half a native frame

clips = {}
for sid, rs in rows.items():
    smp = manifest[sid]; meta = json.loads((M/smp["sample_dir"]/"meta.json").read_text())
    g = smp["gesture"]; P = np.array(g["command_points"])*[828, 1792]
    seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1)
    pauses = [i for i, l in enumerate(seg_len) if l == 0]
    good = [r for r in rs if r["reliable"]]
    lags = np.array([r["lag_ms"] for r in good])
    by_seg = defaultdict(list)
    for r in good:
        by_seg[r["segment"]].append(r["lag_ms"])
    if len(lags) < MIN_FRAMES:
        direction = "unmeasurable"
    else:
        med = float(np.median(lags))
        direction = "aligned" if abs(med) < ALIGNED else ("trace behind" if med > 0 else "trace ahead")
    clips[sid] = dict(id=int(sid), park=smp["park"], recording=meta["source_recording_id"],
                      heldout_ms=1000*meta["calibration"]["heldout_middle_error_s"],
                      sampler=g["sampler"], profile=g.get("profile"), segments=len(seg_len), pauses=pauses,
                      duration_ms=g["times_ms"][-1], rating=review[sid]["rating"], reason=review[sid]["reason"],
                      onset_mark_offset=None, n=len(lags), direction=direction,
                      median=float(np.median(lags)) if len(lags) else None,
                      p10=float(np.percentile(lags, 10)) if len(lags) else None,
                      p90=float(np.percentile(lags, 90)) if len(lags) else None,
                      seg_median={int(k): float(np.median(v)) for k, v in sorted(by_seg.items())},
                      frames=[(r["nominal_ms"], r["lag_ms"], r["segment"], r["speed_px_ms"]) for r in good])
    if review[sid].get("onset_frame") is not None:
        t = np.asarray(smp["frame_times_s"])
        clips[sid]["onset_mark_offset"] = int(review[sid]["onset_frame"]-np.argmin(abs(t-smp["onset_s"])))

C = list(clips.values()); meas = [c for c in C if c["direction"] != "unmeasurable"]
print(f"clips {len(C)}; measurable {len(meas)} (>= {MIN_FRAMES} reliable frames); reliable frames total {sum(c['n'] for c in C)}")
from collections import Counter
print("direction (all):", dict(Counter(c["direction"] for c in C)))
for rating in ("meet", "fail", "unclear"):
    sub = [c for c in meas if c["rating"] == rating]
    if sub:
        med = np.array([c["median"] for c in sub])
        print(f"  {rating:7s} n={len(sub):2d} direction {dict(Counter(c['direction'] for c in sub))} "
              f"clip-median lag: median {np.median(med):+.1f} ms, IQR [{np.percentile(med,25):+.1f}, {np.percentile(med,75):+.1f}], "
              f"|lag|>=17 ms in {int((abs(med)>=17).sum())}")
allmed = np.array([c["median"] for c in meas])
print(f"all measurable clip medians: mean {allmed.mean():+.1f}, median {np.median(allmed):+.1f}, sd {allmed.std():.1f} ms")

# Within-recording consistency: does a shared calibration explain the per-clip offset?
rec = defaultdict(list)
for c in meas:
    rec[c["recording"]].append(c["median"])
multi = {k: v for k, v in rec.items() if len(v) >= 2}
within = np.concatenate([np.array(v)-np.mean(v) for v in multi.values()]) if multi else np.array([])
grand = np.concatenate(list(multi.values())) if multi else np.array([])
if len(grand):
    print(f"recordings with >=2 measurable clips: {len(multi)}; total sd {grand.std():.1f} ms, within-recording sd {within.std():.1f} ms "
          f"(share of variance between recordings {1-within.var()/grand.var():.2f})")
    for k, v in sorted(multi.items()):
        print(f"   {k[-25:]}: {[round(x) for x in v]}")

# Within-clip structure: change of lag across segments.
growth = []
for c in meas:
    seg = c["seg_median"]
    if len(seg) >= 2:
        ks = sorted(seg); growth.append(seg[ks[-1]]-seg[ks[0]])
if growth:
    g = np.array(growth)
    print(f"within-clip change last-vs-first measured segment (n={len(g)}): median {np.median(g):+.1f} ms, IQR [{np.percentile(g,25):+.1f}, {np.percentile(g,75):+.1f}], |change|>=17 ms in {int((abs(g)>=17).sum())}")

# Holds: lag just before vs after a zero-length segment.
print("pause clips (lag by segment; '*' marks the hold):")
for c in meas + [c for c in C if c["direction"] == "unmeasurable"]:
    if c["pauses"]:
        segs = " ".join(f"{k}{'*' if k-1 in c['pauses'] or k in c['pauses'] else ''}:{v:+.0f}" for k, v in c["seg_median"].items())
        print(f"   {c['id']:3d} {c['rating']:7s} holds at seg {c['pauses']} of {c['segments']}: {segs or '-'}")

# Cap-correction check: residual lag vs 1/speed across all reliable frames (after removing clip medians).
x, y = [], []
for c in meas:
    for nominal, lag, seg, speed in c["frames"]:
        x.append(1/speed); y.append(lag-c["median"])
if x:
    b = np.polyfit(x, y, 1)
    print(f"residual lag vs 1/speed: slope {b[0]:+.1f} ms per (ms/px) -> implied cap error {b[0]:+.1f} px")

print("onset-mark offset (user) vs measured direction:")
for d in ("trace ahead", "aligned", "trace behind"):
    sub = [c["onset_mark_offset"] for c in meas if c["direction"] == d and c["onset_mark_offset"] is not None]
    print(f"   {d:12s} marks {dict(Counter(sub))}")
(S/"research/6-synthesis/lag-summary.json").write_text(json.dumps(clips, indent=1))
