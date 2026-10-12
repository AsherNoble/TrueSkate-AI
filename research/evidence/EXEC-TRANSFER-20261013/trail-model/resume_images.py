"""Predeclared midpoint-frame visual audit and hold calibration sheets."""
import json
import cv2
import numpy as np
from common import *


def one(s,k):
    c=cv2.VideoCapture(str(M/s['sample_dir']/'clip.mp4'));c.set(cv2.CAP_PROP_POS_FRAMES,int(k));ok,f=c.read();c.release()
    assert ok and f.shape[:2]==(H,W)
    return f


def make_tile(s,k,xy=None,half=100):
    f=one(s,k);P=Path_(s);e=elapsed_ms(s)
    xy=P.at_time(e[k]) if xy is None else np.asarray(xy)
    x,y=np.round(xy).astype(int)
    pad=cv2.copyMakeBorder(f,half,half,half,half,cv2.BORDER_CONSTANT,value=(32,32,32))
    crop=pad[y:y+2*half,x:x+2*half].copy()
    cv2.circle(crop,(half,half),3,(255,255,0),-1)
    crop=cv2.resize(crop,(240,240))
    cv2.putText(crop,f"{s['id']} f{k} {e[k]:.0f}ms",(4,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3)
    cv2.putText(crop,f"{s['id']} f{k} {e[k]:.0f}ms",(4,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return crop


def main():
    m=manifest();rows=[];tiles=[]
    for sid,s in sorted(m.items()):
        P=Path_(s);e=elapsed_ms(s);k=int(np.argmin(abs(e-P.t[-1]/2)))
        tiles.append(make_tile(s,k,half=120));rows.append(dict(sid=sid,k=k,elapsed_ms=float(e[k]),xy=P.at_time(e[k]).tolist()))
        if len(tiles)==20:
            batch=(len(rows)-1)//20+1
            cv2.imwrite(str(OUT/f'img/resume-midpoint-{batch}.jpg'),np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,20,5)]),[cv2.IMWRITE_JPEG_QUALITY,92]);tiles=[]
    (OUT/'data/resume-midpoint-frame-selection.json').write_text(json.dumps(rows,indent=2))
    for sid in [55,99,83,131,137,133]:
        s=m[sid];P=Path_(s);e=elapsed_ms(s);i=int(np.argmin(P.len));pt=P.p[i]
        ks=np.array([np.argmin(abs(e-(P.t[i]+v))) for v in [-33,0,33,67,100,133,200,267]])
        cv2.imwrite(str(OUT/f'img/resume-hold-{sid}.jpg'),np.vstack([np.hstack([make_tile(s,int(k),pt,80) for k in ks[j:j+4]]) for j in [0,4]]),[cv2.IMWRITE_JPEG_QUALITY,95])

if __name__=='__main__':main()
