"""Cross-section and fade-with-age of the moving trail (straight stretches, away from caps/vertices/board)."""
import json, sys
import numpy as np
from common import *

OFFS = np.arange(-80, 81, 4).astype(float)
C = len(OFFS)//2


def load(sid):
    return dict(np.load(OUT/f"data/{sid}.npz"))


def xprofile(ex):
    """ex: (no, n) excess profiles -> peak, ridge offset (px), FWHM (px)."""
    ex = ex.astype(np.float32)
    core = ex[C-3:C+4]                       # |o|<=12
    j = np.argmax(core, 0)+C-3
    n = ex.shape[1]; idx = np.arange(n)
    pk = ex[j, idx]
    a = ex[np.clip(j-1, 0, None), idx]; b = ex[np.clip(j+1, None, len(OFFS)-1), idx]
    den = a-2*pk+b
    frac = np.where(den < 0, 0.5*(a-b)/np.where(den == 0, -1, den), 0)
    ridge = OFFS[j]+4*np.clip(frac, -.5, .5)
    half = pk/2; fw = np.full(n, np.nan)
    for i in range(n):
        if pk[i] <= 0:
            continue
        col = ex[:, i]; L = R = None
        for q in range(j[i], 0, -1):
            if col[q-1] < half[i]:
                L = OFFS[q-1]+4*(half[i]-col[q-1])/(col[q]-col[q-1]); break
        for q in range(j[i], len(OFFS)-1):
            if col[q+1] < half[i]:
                R = OFFS[q]+4*(col[q]-half[i])/(col[q]-col[q+1]); break
        if L is not None and R is not None:
            fw[i] = R-L
    return pk, ridge, fw


def valid_mask(d):
    s, S = d["s"], d["S"]
    moving_ok = (d["speed"] > .3) & (d["vert_d"] > 48) & (d["ov"] > 140) & d["inb"] & (s > 48) & (s < S[-1]-48)
    return moving_ok[None] & ~d["BOARD"]


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1].split(",")]
    for sid in ids:
        d = load(sid); ok = valid_mask(d)
        age = d["elapsed"][:, None]-d["t_first"][None]
        rows = []
        for j in range(len(d["ks"])):
            pk, ridge, fw = xprofile(d["EX"][j])
            for i in np.where(ok[j])[0]:
                rows.append((age[j, i], pk[i], ridge[i], fw[i], d["speed"][i], float(d["NOISE"][j, i])))
        r = np.array(rows)
        print(sid, "n", len(r))
        for a0 in range(-60, 420, 20):
            sel = (r[:, 0] >= a0) & (r[:, 0] < a0+20)
            if sel.sum() < 5: continue
            print(f"  age {a0:4d}..{a0+20:4d} n {sel.sum():4d} peak med {np.median(r[sel,1]):6.1f} p90 {np.percentile(r[sel,1],90):6.1f}  fwhm {np.nanmedian(r[sel,3]):5.1f}  ridge {np.median(r[sel,2]):5.1f} noise {np.median(r[sel,5]):4.1f}")
