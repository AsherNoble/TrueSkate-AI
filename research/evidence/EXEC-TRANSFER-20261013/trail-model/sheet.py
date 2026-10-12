"""Crop sheet around a point: python sheet.py sid x y k0 k1 [half] -> img/sheet-sid-k0.png"""
import sys
import cv2, numpy as np
from common import *
sid, x, y, k0, k1 = map(int, sys.argv[1:6]); half = int(sys.argv[6]) if len(sys.argv) > 6 else 80
m = manifest(); s = m[sid]; f = decode(s); e = elapsed_ms(s); P = Path_(s)
tiles = []
for k in range(k0, k1):
    fr = f[k].copy()
    c = P.at_time(e[k])
    cv2.circle(fr, (int(c[0]), int(c[1])), 3, (255, 255, 0), -1)
    pad = cv2.copyMakeBorder(fr, half, half, half, half, cv2.BORDER_CONSTANT)
    t = pad[y:y+2*half, x:x+2*half].copy()
    t = cv2.resize(t, (240, 240), interpolation=cv2.INTER_NEAREST)
    cv2.putText(t, f"{k} {e[k]:.0f}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    tiles.append(t)
while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(OUT/f"img/sheet-{sid}-{k0}.png"), np.vstack([np.hstack(tiles[i:i+8]) for i in range(0, len(tiles), 8)]))
