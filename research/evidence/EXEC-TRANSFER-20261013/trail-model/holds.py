"""Stationary holds: cap profile, rise and fade at a fixed spot -> data/holds.json"""
import json
import cv2, numpy as np
from common import *

m = manifest()
holds = []
for sid, s in m.items():
    P = Path_(s)
    for i, L in enumerate(P.len):
        if L < 1:
            kind = "first" if i == 0 else ("last" if i == len(P.len)-1 else "mid")
            # direction for the forward ray: away from the trail (arrival dir for mid/last, -departure for first)
            if kind == "first":
                d = P.p[i+1]-P.p[i+2] if len(P.p) > i+2 else np.array([1., 0])
            else:
                d = P.p[i]-P.p[i-1]
            holds.append(dict(sid=sid, kind=kind, i=i, pt=P.p[i].tolist(), t0=float(P.t[i]), t1=float(P.t[i+1]),
                              dir=(d/np.linalg.norm(d)).tolist()))
    if P.S[-1] < 60 and P.t[-1] >= 300:
        d = P.p[-1]-P.p[0]; d = d/np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.array([1., 0])
        holds.append(dict(sid=sid, kind="quasi", i=-1, pt=P.p.mean(0).tolist(), t0=0., t1=float(P.t[-1]),
                          dir=d.tolist(), L=float(P.S[-1])))

R = np.arange(0, 101, 2.)
out = []
for h in holds:
    s = m[h["sid"]]; f = decode(s); e = elapsed_ms(s); P = Path_(s)
    x0, y0 = h["pt"]; dx, dy = h["dir"]; ang = np.arctan2(dy, dx)
    th = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    rel = (th+np.pi) % (2*np.pi)-np.pi            # angle relative to forward direction
    back = np.abs(np.abs(rel)-np.pi) < np.radians(50)  # sector containing the trail (mid/last)
    if h["kind"] == "first":
        back = np.abs(rel) > 10                        # nothing behind before leaving; use all
    gx = (x0+R[:, None]*np.cos(th+ang)[None]).astype(np.float32)
    gy = (y0+R[:, None]*np.sin(th+ang)[None]).astype(np.float32)
    frames = []
    for k in range(len(e)):
        if not (h["t0"]-120 <= e[k] <= h["t1"]+400):
            continue
        img = f[k].astype(np.float32)
        pol = cv2.remap(img, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)   # (nr, nth, 3)
        lum = (pol[..., 2]+pol[..., 1])/2
        ann = (R >= 70) & (R <= 100)
        use = ~back if h["kind"] in ("mid", "last") else np.ones(len(th), bool)
        bg = np.median(lum[ann][:, use]); noise = float(np.std(lum[ann][:, use]))
        ex = lum-bg
        fwd = ex[:, np.abs(rel) < np.radians(20)].mean(1)
        side = ex[:, np.abs(np.abs(rel)-np.pi/2) < np.radians(20)].mean(1)
        rad = ex[:, use].mean(1)
        bm = board_mask(f[k]); yy, xx = int(np.clip(y0, 0, H-1)), int(np.clip(x0, 0, W-1))
        board = bool(bm[max(0, yy-60):yy+60, max(0, xx-60):xx+60].any())
        # centroid of excess within r<40 (all directions)
        wts = np.clip(ex[R < 40], 0, None)*R[R < 40, None]
        cxy = [float((wts*R[R < 40, None]*np.cos(th+ang)[None]).sum()/max(wts.sum(), 1e-6)),
               float((wts*R[R < 40, None]*np.sin(th+ang)[None]).sum()/max(wts.sum(), 1e-6))]
        frames.append(dict(k=k, t=float(e[k]), peak=float(ex[0].mean()), fwd=fwd.round(1).tolist(),
                           side=side.round(1).tolist(), rad=rad.round(1).tolist(), noise=noise, board=board,
                           bg_rgb=pol[ann].reshape(-1, 3).mean(0)[::-1].round(0).tolist(), centroid=cxy))
    h["frames"] = frames
    out.append(h)
    print(h["sid"], h["kind"], round(h["t1"]-h["t0"]), "frames", len(frames), flush=True)
(OUT/"data/holds.json").write_text(json.dumps(out))
