"""Full-frame strip with commanded position dot: python full.py sid k0 k1 [step] [crop x0 y0 x1 y1]"""
import sys
import cv2, numpy as np
from common import *
sid, k0, k1 = map(int, sys.argv[1:4]); step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
crop = list(map(int, sys.argv[5:9])) if len(sys.argv) > 8 else [0, 0, W, H]
m = manifest(); s = m[sid]; f = decode(s); e = elapsed_ms(s); P = Path_(s)
tiles = []
for k in range(k0, k1, step):
    fr = f[k].copy()
    c = P.at_time(e[k])
    cv2.circle(fr, (int(c[0]), int(c[1])), 4, (255, 255, 0), -1)
    for p in P.p:
        cv2.drawMarker(fr, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    t = fr[crop[1]:crop[3], crop[0]:crop[2]]
    sc = 300/t.shape[1]
    t = cv2.resize(t, (300, int(t.shape[0]*sc)))
    cv2.putText(t, f"{k} {e[k]:.0f}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    tiles.append(t)
while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(OUT/f"img/full-{sid}-{k0}.jpg"), np.vstack([np.hstack(tiles[i:i+6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
