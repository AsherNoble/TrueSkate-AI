"""Draw detected head (green), ring at nominal time (cyan) and path (white) for visual checks."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from trail_lag import decode, Path_, trail_mask, board_mask, W, H

M = Path(sys.argv[1]); OUT = Path(sys.argv[2]); results = json.loads(Path(sys.argv[3]).read_text())
manifest = {str(s["id"]): s for s in json.loads((M/"audit-6/manifest.json").read_text())["samples"]}
for sid in sys.argv[4].split(","):
    s = manifest[sid]; frames = decode(M/s["sample_dir"]/"clip.mp4"); path = Path_(s)
    tiles = []
    for r in results[sid]:
        if not -20 <= r["nominal_ms"] <= path.t[-1]+50:
            continue
        f = frames[r["frame"]].copy()
        b = board_mask(f); f[b] = (f[b]*.6).astype(np.uint8)
        f[trail_mask(f) & ~b] = (255, 0, 255)
        for a, b in zip(path.p[:-1], path.p[1:]):
            cv2.line(f, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 255), 2)
        ring = path.at_s(path.s_at_time(min(max(r["nominal_ms"], 0), path.t[-1])))
        cv2.circle(f, tuple(int(v) for v in ring), 20, (255, 255, 0), 5)
        if "head_xy" in r:
            cv2.circle(f, tuple(int(v) for v in r["head_xy"]), 14, (0, 255, 0), 5)
        lag = r.get("lag_ms")
        cv2.putText(f, f"{r['frame']} {'' if lag is None else round(lag)}{'' if r.get('reliable', True) else '?'}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 7)
        tiles.append(cv2.resize(f, (207, 448)))
    if not tiles:
        continue
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT/f"check-{sid}.jpg"), np.vstack([np.hstack(tiles[i:i+10]) for i in range(0, len(tiles), 10)]),
                [cv2.IMWRITE_JPEG_QUALITY, 75])
