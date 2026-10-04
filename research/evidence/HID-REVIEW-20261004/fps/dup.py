"""Per-frame-pair change in the game area of a recording, excluding system
overlays (cursor disc via tile-count tolerance, lower-right button, home
indicator) and the orange touch trail (HSV mask, dilated)."""
import cv2, numpy as np, subprocess, sys, json, os, glob
from multiprocessing import Pool

W, H = 414, 896          # half-res == point grid
TW, TH = 18, 32          # tile size at half res -> 23 x 28 tiles
static = np.zeros((H, W), bool)
static[870:, :] = True                    # home indicator (full-res y>=1740)
static[665:740, 340:] = True              # lower-right ring button (full-res x>=680, 1330..1480)
KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

def orange(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 3) & (h <= 28) & (s >= 90) & (v >= 110)).astype(np.uint8)
    return m

def pts_list(f):
    out = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=pts_time,pict_type,key_frame','-of','csv=p=0',f]).decode().split()
    t=[]; pt=[]
    for line in out:
        p=line.split(',')
        # csv order: key_frame, pts_time, pict_type (ffprobe orders by section field order)
        t.append(p)
    return t

def run(f):
    meta = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=key_frame,pts_time,pict_type','-of','json',f])
    frames = json.loads(meta)['frames']
    pts = np.array([float(x['pts_time']) for x in frames]); key = np.array([int(x['key_frame']) for x in frames])
    ptype = [x.get('pict_type','?') for x in frames]
    cap = cv2.VideoCapture(f)
    prev = None; prevor = None
    tiles_all = []; trail_px = []; orange_frac=[]
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        g = fr.astype(np.int16)
        om = orange(fr)
        if prev is not None:
            d = np.abs(g - prev).max(axis=2).astype(np.float32)
            trailmask = cv2.dilate(np.maximum(om, prevor), KER) > 0
            m = ~(static | trailmask)
            # trail change: mean diff inside orange mask (undilated)
            tm = np.maximum(om, prevor) > 0
            trail_px.append([float(d[tm].mean()) if tm.any() else 0.0, int(tm.sum()), int((d[tm] > 12).sum()) if tm.any() else 0])
            ds = np.where(m, d, 0).reshape(H//TH, TH, W//TW, TW).sum(axis=(1,3))
            cnt = m.reshape(H//TH, TH, W//TW, TW).sum(axis=(1,3))
            tm_ = np.where(cnt > 0.5*TH*TW, ds/np.maximum(cnt,1), np.nan)
            tiles_all.append(tm_.astype(np.float32))
        prev = g; prevor = om; n += 1
    cap.release()
    tiles = np.stack(tiles_all)
    name = f.split('rec/')[-1] if 'rec/' in f else os.path.basename(f)
    out = dict(name=name, n=n, npts=len(pts))
    np.savez_compressed('dup_' + name.replace('/','__') + '.npz', tiles=tiles, pts=pts, key=key, ptype=np.array(ptype), trail=np.array(trail_px))
    return out

if __name__ == '__main__':
    files = sys.argv[1:]
    with Pool(6) as p:
        for r in p.imap_unordered(run, files):
            print(r, flush=True)
