"""M1-EXPAND free analysis: quantity vs diversity vs split, from existing per-clip results."""
import json, math, random, statistics as st, sys, collections
from pathlib import Path

D = Path(sys.argv[1]); ENDPROBE = Path(sys.argv[2]); HOLD = Path(sys.argv[3])
meta = json.load(open(D / "m1_analysis_meta.json"))
sessions = json.load(open(D / "m1_sessions.json"))
counts = meta["train_park_counts"]
vmeta = {r["path"]: r for r in meta["validation_meta"]}
hmeta = {r["path"]: r for r in meta["holdout_meta"]}
PARKS = ["SLS 2015 Los Angeles", "SLS 2013 Kansas City", "SLS 2015 Super Crown", "Skateboard GB 2024", "The Workshop"]
SHORT = {"SLS 2015 Los Angeles": "LA", "SLS 2013 Kansas City": "KC", "SLS 2015 Super Crown": "SC15",
         "Skateboard GB 2024": "GB", "The Workshop": "WS"}

def autopsy(name):
    return {r["sample"]: bool(r["recovered"]) for r in json.load(open(D / name))["all_records"]}

rungs = {"2293": autopsy("basic_linear_n2293_cosine_seed0_autopsy_validation.json"),
         "4585": autopsy("basic_linear_n4585_cosine_seed2_autopsy_validation.json"),
         "9170": autopsy("basic_linear_n9170_cosine_seed2_autopsy_validation.json"),
         "18394": autopsy("basic_linear_expand20260930_cosine_seed1_autopsy_validation.json")}
ep = json.load(open(ENDPROBE))
seeds9170 = {n.split("_seed")[1][0]: {r["path"]: bool(r["baseline"]["recovered"]) for r in ep["per_sample"][n]}
             for n in ep["checkpoints"]}
hold = {r["path"]: bool(r["recovered"]) for r in json.load(open(HOLD))["per_sample"]}

def fail_rate(results, park):
    xs = [not ok for p, ok in results.items() if vmeta[p]["park"] == park]
    return sum(xs) / len(xs), sum(xs), len(xs)

print("=" * 78)
print("A. Per-park validation failure vs that park's own training clips (one seed per rung)")
print(f"{'park':6s}" + "".join(f"{r:>18s}" for r in rungs))
for park in PARKS:
    cells = []
    for r, res in rungs.items():
        f, k, n = fail_rate(res, park)
        cells.append(f"{counts[r][park]:>6d}: {f:6.1%}")
    print(f"{SHORT[park]:6s}" + "".join(f"{c:>18s}" for c in cells))

print("\nconsistency: autopsy n9170 seed2 vs endprobe baseline seed2 agree on",
      sum(rungs['9170'][p] == seeds9170['2'][p] for p in rungs['9170']), "/", len(rungs['9170']), "clips")

print("=" * 78)
print("B. Is each park's 9,170 -> 18,394 change beyond seed noise? (9,170: three cosine seeds)")
print(f"{'park':6s} {'own clips':>14s} {'9170 seeds 0/1/2':>26s} {'18394 s1':>9s}  {'paired gained/lost vs each 9170 seed'}")
for park in PARKS:
    rates = [fail_rate(seeds9170[s], park)[0] for s in "012"]
    f18 = fail_rate(rungs["18394"], park)[0]
    pair = []
    for s in "012":
        g = sum(1 for p, ok in rungs["18394"].items() if vmeta[p]["park"] == park and ok and not seeds9170[s][p])
        l = sum(1 for p, ok in rungs["18394"].items() if vmeta[p]["park"] == park and not ok and seeds9170[s][p])
        pair.append(f"{g}/{l}")
    print(f"{SHORT[park]:6s} {counts['9170'][park]:>5d}->{counts['18394'][park]:<6d}  "
          f"{' / '.join(f'{x:5.1%}' for x in rates):>26s} {f18:8.1%}  {', '.join(pair)}")
# How much did the 9,170 seeds themselves disagree with each other, per park? (seed-only noise)
print("seed-only noise (9170 seed pairs, gained/lost):", end=" ")
for park in PARKS:
    pp = []
    for a, b in (("0", "1"), ("0", "2"), ("1", "2")):
        g = sum(1 for p in seeds9170[a] if vmeta[p]["park"] == park and seeds9170[b][p] and not seeds9170[a][p])
        l = sum(1 for p in seeds9170[a] if vmeta[p]["park"] == park and not seeds9170[b][p] and seeds9170[a][p])
        pp.append(f"{g}/{l}")
    print(f"{SHORT[park]} {','.join(pp)}", end="; ")
print()

print("=" * 78)
print("C. Does sharing a recording with training help? (validation, 18,394 seed1 model)")
# For each validation clip: number of training clips from the same recording (session).
train_sess = sessions["18394"]
rows = []
for p, ok in rungs["18394"].items():
    m = vmeta[p]
    rows.append((m["park"], train_sess.get(m["session"], 0), not ok))
bins = [(0, 4), (5, 7), (8, 9), (10, 11), (12, 99)]
print(f"{'mates':>8s}" + "".join(f"{SHORT[p]:>12s}" for p in PARKS) + f"{'all':>12s}")
for lo, hi in bins:
    line = f"{lo:>3d}-{hi if hi < 99 else '+':<4}"
    for park in PARKS + ["all"]:
        xs = [f for pk, k, f in rows if lo <= k <= hi and (park == "all" or pk == park)]
        line += f"{(f'{sum(xs)/len(xs):.1%} ({len(xs)})' if xs else '-'):>12s}"
    print(line)
# Within-park logistic-free test: Mantel-Haenszel-style comparison of mates <= median vs > median per park.
num = den = 0.0
detail = []
for park in PARKS:
    ks = [k for pk, k, f in rows if pk == park]
    med = st.median(ks)
    lo_ = [f for pk, k, f in rows if pk == park and k <= med]
    hi_ = [f for pk, k, f in rows if pk == park and k > med]
    if lo_ and hi_:
        detail.append(f"{SHORT[park]} med {med:g}: few-mates {sum(lo_)/len(lo_):.1%} (n={len(lo_)}) vs many {sum(hi_)/len(hi_):.1%} (n={len(hi_)})")
print("\n".join(detail))
# Permutation test within park: correlation between mates and failure.
def stat(rs):
    return sum((k - mk[pk]) * (f - mf[pk]) for pk, k, f in rs)
mk = {p: st.mean(k for pk, k, f in rows if pk == p) for p in PARKS}
mf = {p: st.mean(f for pk, k, f in rows if pk == p) for p in PARKS}
obs = stat(rows)
rng = random.Random(0); more = 0; N = 5000
by_park = collections.defaultdict(list)
for r in rows:
    by_park[r[0]].append(r)
for _ in range(N):
    perm = []
    for p, rs in by_park.items():
        fs = [f for _, _, f in rs]; rng.shuffle(fs)
        perm += [(p, k, f) for (_, k, _), f in zip(rs, fs)]
    more += stat(perm) <= obs
print(f"within-park covariance(mates, failure) = {obs:+.2f}; one-sided p (more mates -> fewer failures) = {more/N:.3f}")

print("=" * 78)
print("D. Holdout vs validation at matched gesture duration")
def dbin(d):
    for edge in (.45, .60, .75, .90, 1.05):
        if d < edge:
            return f"<{edge:.2f}"
    return ">=1.05"
vb = collections.defaultdict(lambda: [0, 0]); hb = collections.defaultdict(lambda: [0, 0])
for p, ok in rungs["18394"].items():
    b = vb[dbin(vmeta[p]["duration"])]; b[0] += 1; b[1] += (not ok)
for p, ok in hold.items():
    b = hb[dbin(hmeta[p]["duration"])]; b[0] += 1; b[1] += (not ok)
expected = 0
for b in sorted(vb):
    vr = vb[b][1] / vb[b][0]; hr = hb[b][1] / hb[b][0] if hb[b][0] else float("nan")
    expected += hb[b][0] * vr
    print(f"{b:>7s}  validation {vr:6.1%} (n={vb[b][0]:4d})   holdout {hr:6.1%} (n={hb[b][0]:3d})")
print(f"holdout failure observed {sum(v[1] for v in hb.values())}/{len(hold)} = "
      f"{sum(v[1] for v in hb.values())/len(hold):.1%}; expected at validation rates for its duration mix: "
      f"{expected/len(hold):.1%}")

print("=" * 78)
print("E. Do holdout failures cluster by recording? (whole sessions, ~9 clips each)")
for park in sorted({m["park"] for m in hmeta.values()}):
    sess = collections.defaultdict(list)
    for p, ok in hold.items():
        if hmeta[p]["park"] == park:
            sess[hmeta[p]["session"]].append(not ok)
    allf = [f for v in sess.values() for f in v]; pbar = sum(allf) / len(allf)
    obs_var = sum((sum(v) - len(v) * pbar) ** 2 for v in sess.values())
    exp_var = sum(len(v) * pbar * (1 - pbar) for v in sess.values())
    rng = random.Random(1); ge = 0
    for _ in range(5000):
        shuffled = allf[:]; rng.shuffle(shuffled); i = 0; s = 0.0
        for v in sess.values():
            chunk = shuffled[i:i + len(v)]; i += len(v); s += (sum(chunk) - len(chunk) * pbar) ** 2
        ge += s >= obs_var
    worst = sorted((sum(v) for v in sess.values()), reverse=True)
    top = worst[:max(1, len(worst) // 10)]
    print(f"{park:22s} sessions {len(sess):3d}  failure {pbar:5.1%}  dispersion {obs_var/exp_var:4.2f}x binomial "
          f"(p={ge/5000:.3f}); worst 10% of sessions hold {sum(top)}/{sum(allf)} failures")
