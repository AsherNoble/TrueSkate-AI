"""Leading edge of the trail vs the commanded head, per frame, all clips -> data/headedge.json"""
import json, glob
import numpy as np
from xsec import load, xprofile, OUT, C

rows = []
for p in sorted(glob.glob(str(OUT/"data/*.npz"))):
    sid = int(p.split("/")[-1][:-4]); d = load(sid)
    s, S, Pt = d["s"], d["S"], d["Pt"]
    if S[-1] < 100:
        continue
    el = d["elapsed"]; pre = el < -60
    PK = np.array([xprofile(d["EX"][j])[0] for j in range(len(el))])      # (nk, ns)
    base = PK[pre].max(0) if pre.any() else np.zeros(len(s))               # scenery before onset
    for j in range(len(el)):
        t = el[j]
        if not (17 <= t <= Pt[-1]-5):
            continue
        sc = float(np.interp(t, Pt, S))
        seg = int(np.clip(np.searchsorted(Pt, t, side="right")-1, 0, len(Pt)-2))
        v = (S[seg+1]-S[seg])/(Pt[seg+1]-Pt[seg])
        if v < .3:
            continue
        w = (s >= sc-160) & (s <= sc+260)
        if w.sum() < 20:
            continue
        idx = np.where(w)[0]; pk = PK[j, idx]
        a = int(np.argmax(pk)); M = pk[a]
        e = a
        while e+1 < len(idx) and pk[e+1] >= M/2:
            e += 1
        if e+1 < len(idx):
            frac = (pk[e]-M/2)/max(pk[e]-pk[e+1], 1e-6)
            edge = s[idx[e]]+4*np.clip(frac, 0, 1)
        else:
            edge = np.nan
        near = (s >= edge-48) & (s <= edge+64)
        rows.append(dict(sid=sid, k=int(d["ks"][j]), t=float(t), seg=seg, v=float(v), sc=sc, edge=float(edge),
                         delta=float(edge-sc), peak=float(M),
                         board=bool(d["BOARD"][j][near].any()), inb=bool(d["inb"][near].all()),
                         overlap=bool((d["ov"][near] < 140).any()),
                         vert=float(np.min(np.abs(S[1:-1]-edge))) if len(S) > 2 else 1e9,
                         end=bool(edge > S[-1]-24), scenery=float(base[near].max()) if near.any() else 0.,
                         head_peak_cmd=float(PK[j, np.argmin(np.abs(s-sc))])))
(OUT/"data/headedge.json").write_text(json.dumps(rows))
print(len(rows))
