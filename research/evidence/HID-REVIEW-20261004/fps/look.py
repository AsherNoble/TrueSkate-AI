import numpy as np, sys
z = np.load(sys.argv[1])
tiles, pts, key, ptype = z['tiles'], z['pts'], z['key'], z['ptype']
gap = np.round(np.diff(pts)*1000, 1)
nvalid = np.isfinite(tiles).reshape(len(tiles), -1).sum(1)
flat = np.nan_to_num(tiles.reshape(len(tiles), -1), nan=0)
n3 = (flat > 3).sum(1); n1 = (flat > 1).sum(1)
med = np.nanmedian(tiles.reshape(len(tiles), -1), axis=1)
p90 = np.nanpercentile(tiles.reshape(len(tiles), -1), 90, axis=1)
mx = np.nanmax(tiles.reshape(len(tiles), -1), axis=1)
lo, hi = int(sys.argv[2]), int(sys.argv[3])
print('valid tiles', nvalid[0])
for i in range(lo, min(hi, len(tiles))):
    print(f"{i+1:4d} t={pts[i+1]:6.3f} gap={gap[i]:5.1f} {ptype[i+1]} key={key[i+1]} n>3={n3[i]:4d} n>1={n1[i]:4d} med={med[i]:5.2f} p90={p90[i]:6.2f} max={mx[i]:6.1f}")
