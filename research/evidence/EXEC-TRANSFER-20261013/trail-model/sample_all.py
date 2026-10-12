"""Sample every clip in path coordinates: excess(frame, s, o) with flags. -> data/<sid>.npz"""
import sys, time
import cv2, numpy as np
from common import *

STEP = 4.0
OFFS = np.arange(-80, 81, 4).astype(np.float32)   # perpendicular offsets, px
BGI = np.where(np.abs(OFFS) >= 72)[0]


def geometry(P):
    s = np.arange(0, P.S[-1]+1e-9, STEP) if P.S[-1] > 0 else np.zeros(1)
    seg = np.clip(np.searchsorted(P.S, s, side="right")-1, 0, len(P.len)-1)
    # skip zero-length segments for direction: use nearest moving segment
    nz = np.where(P.len > 0)[0]
    segd = np.array([nz[np.argmin(np.abs(nz-i))] if len(nz) else i for i in seg])
    d = P.p[segd+1]-P.p[segd]; n = np.linalg.norm(d, axis=1, keepdims=True); n[n == 0] = 1
    tang = d/n; normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    pts = np.array([P.at_s(v) for v in s])
    # commanded time of first arrival at s (end of hold => leave time handled separately)
    t_arr = np.interp(s, P.S, P.t, left=0)  # np.interp on non-decreasing S picks first? ensure first arrival:
    t_first = np.array([P.t[np.searchsorted(P.S, v, side="left")] if np.any(np.isclose(P.S, v)) else np.interp(v, P.S, P.t) for v in s])
    vert_d = np.min(np.abs(s[:, None]-P.S[None, 1:-1]), axis=1) if len(P.S) > 2 else np.full(len(s), 1e9)
    speed = P.len[segd]/np.maximum(np.diff(P.t)[segd], 1)
    # self-overlap: nearest path point more than 160 px of arc away
    dd = np.linalg.norm(pts[:, None]-pts[None], axis=2)
    far = np.abs(s[:, None]-s[None]) > 160
    ov = np.where(far, dd, np.inf).min(1)
    return s, pts, normal, tang, t_first, vert_d, speed, ov


def run(sid, s_):
    P = Path_(s_); f = decode(s_); e = elapsed_ms(s_)
    s, pts, normal, tang, t_first, vert_d, speed, ov = geometry(P)
    ks = [k for k in range(len(e)) if -250 <= e[k] <= P.t[-1]+450]
    q = pts[None]+OFFS[:, None, None]*normal[None]               # (no, ns, 2)
    qx = q[..., 0].astype(np.float32); qy = q[..., 1].astype(np.float32)
    inb = (pts[:, 0] > 80) & (pts[:, 0] < W-80) & (pts[:, 1] > 80) & (pts[:, 1] < H-80)
    EX = np.zeros((len(ks), len(OFFS), len(s)), np.float16)    # (R+G)/2 excess
    EB = np.zeros((len(ks), len(OFFS), len(s)), np.float16)    # B excess
    BG = np.zeros((len(ks), len(s), 3), np.float16)
    NOISE = np.zeros((len(ks), len(s)), np.float16)
    BOARD = np.zeros((len(ks), len(s)), bool)
    for j, k in enumerate(ks):
        img = f[k].astype(np.float32)
        smp = cv2.remap(img, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # (no, ns, 3)
        bg = smp[BGI].mean(0)
        dif = smp-bg[None]
        EX[j] = (dif[..., 2]+dif[..., 1])/2; EB[j] = dif[..., 0]
        BG[j] = bg
        side = (smp[BGI][..., 2]+smp[BGI][..., 1])/2
        NOISE[j] = side.std(0)
        bm = board_mask(f[k])
        cx = np.clip(q[(len(OFFS)//2-5):(len(OFFS)//2+6), :, 0], 0, W-1).astype(int)
        cy = np.clip(q[(len(OFFS)//2-5):(len(OFFS)//2+6), :, 1], 0, H-1).astype(int)
        BOARD[j] = bm[cy, cx].any(0)
    np.savez_compressed(OUT/f"data/{sid}.npz", ks=np.array(ks), elapsed=e[ks], s=s, pts=pts, normal=normal,
                        t_first=t_first, vert_d=vert_d, speed=speed, ov=ov, inb=inb, EX=EX, EB=EB, BG=BG,
                        NOISE=NOISE, BOARD=BOARD, Pp=P.p, Pt=P.t, S=P.S)


if __name__ == "__main__":
    m = manifest()
    ids = sorted(m) if sys.argv[1] == "all" else [int(x) for x in sys.argv[1].split(",")]
    for sid in ids:
        t0 = time.time(); run(sid, m[sid]); print(sid, round(time.time()-t0, 1), flush=True)
