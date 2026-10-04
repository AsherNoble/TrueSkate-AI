"""Contact sheet of True Skate's trick banner at the end of each replay (outcome only; no trail reading).

Usage: banner_sheet.py <variance_dir> <out.png>
Rows are runs (v2_01, v3_01, ...); columns are frames 1.6, 1.2 and 0.8 s before the recording ends
(the recording stops 1.5 s after the schedule's last report, so the banner is up by then).
"""
import glob, os, sys
import cv2, numpy as np

root, out = sys.argv[1], sys.argv[2]
rows = []
for d in sorted(glob.glob(os.path.join(root, 'v*_*')), key=lambda p: (os.path.basename(p)[3:], os.path.basename(p)[:2])):
    movs = glob.glob(os.path.join(d, '*.mov'))
    if not movs:
        continue
    cap = cv2.VideoCapture(movs[0])
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT); fps = cap.get(cv2.CAP_PROP_FPS) or 60
    dur = n / fps
    tiles = []
    for back in (1.6, 1.2, 0.8):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, dur - back) * 1000); ok, f = cap.read()
        if not ok:
            f = np.zeros((1792, 828, 3), np.uint8)
        banner = f[230:480, 120:708]                     # y 115-240 pt, x 60-354 pt: trick name and score
        board = cv2.resize(f[560:1500, 200:628], (150, 330))
        b = cv2.resize(banner, (520, 220))
        col = np.zeros((330, 520 + 150, 3), np.uint8); col[:220, :520] = b; col[:, 520:] = board
        tiles.append(col)
    row = np.hstack(tiles)
    label = np.zeros((40, row.shape[1], 3), np.uint8)
    cv2.putText(label, os.path.basename(d), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    rows.append(np.vstack([label, row]))
sheet = np.vstack(rows)
cv2.imwrite(out, cv2.resize(sheet, None, fx=0.5, fy=0.5))
print(len(rows), 'runs ->', out)
