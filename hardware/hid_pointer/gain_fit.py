"""Fit the AssistiveTouch pointer curve: points per report D vs count-vector length m.

Usage: gain_fit.py <gain_TAG.mov.hover.json> ...  -> table, fit, gain_points.json, gain_fit.json
The fit is the slow region D = a*m (m < 3) and, above it, min(max(L1, L2), L3) for
three straight lines; the constants go into trueskate_ai.control.hid_pointer.
"""
import json, sys
import numpy as np
rows = []
for f in sys.argv[1:]:
    seg = f.split('gain_')[-1].split('.')[0]
    for r in json.load(open(f)):
        c = np.array([r['dx'], r['dy']], float); n = r['reports']
        mv = np.array(r['moved_pt'])
        m = float(np.hypot(*c)); D = float(np.hypot(*mv) / n)
        ang_err = float(np.degrees(np.arctan2(mv[1], mv[0]) - np.arctan2(c[1], c[0])))
        rows.append((seg, r['dx'], r['dy'], n, m, D, ang_err))
rows.sort(key=lambda r: r[4])
for r in rows:
    print('%s (%4d,%4d) x%3d  m=%7.3f  D=%8.3f  D/m=%.3f  angle err %+.2f deg' % (r[:6] + (r[5] / r[4], r[6])))
json.dump([dict(seg=r[0], dx=r[1], dy=r[2], n=r[3], m=r[4], D=r[5]) for r in rows], open('gain_points.json', 'w'), indent=1)

m, D, n = np.array([(r[4], r[5], r[3]) for r in rows]).T
slow = m < 2.95
a = (m[slow] @ D[slow]) / (m[slow] @ m[slow])
def fit_line(sel):
    A = np.vstack([m[sel], np.ones(sel.sum())]).T
    return np.linalg.lstsq(A, D[sel], rcond=None)[0]
k1, k2 = 25.6, 71.8
for _ in range(10):
    L1 = fit_line((m >= 2.95) & (m <= k1)); L2 = fit_line((m > k1) & (m <= k2)); L3 = fit_line(m > k2)
    k1 = (L2[1] - L1[1]) / (L1[0] - L2[0]); k2 = (L3[1] - L2[1]) / (L2[0] - L3[0])
print('slow gain %.4f' % a)
print('L1 D = %.4f m %+.3f  (to m=%.2f)' % (L1[0], L1[1], k1))
print('L2 D = %.4f m %+.3f  (to m=%.2f)' % (L2[0], L2[1], k2))
print('L3 D = %.4f m %+.3f' % tuple(L3))
def model(mm):
    mm = np.asarray(mm, float)
    fast = np.minimum(np.maximum(L1[0] * mm + L1[1], L2[0] * mm + L2[1]), L3[0] * mm + L3[1])
    return np.where(mm < 2.95, a * mm, fast)
res = D - model(m)
print('per-report residual: rms %.3f pt, max %.3f pt' % (np.sqrt((res ** 2).mean()), np.abs(res).max()))
print('drag-total residual (x n): max %.2f pt' % np.abs(res * n).max())
for mm, dd, rr, nn in zip(m, D, res, n):
    if abs(rr) > 0.25:
        print('  larger residual m=%.2f D=%.3f res %+.3f (n=%d)' % (mm, dd, rr, nn))
json.dump(dict(slow_gain=a, slow_limit=2.95, lines=[list(L1), list(L2), list(L3)], knots=[k1, k2],
               rms_pt=float(np.sqrt((res ** 2).mean())), max_pt=float(np.abs(res).max()), drags=len(rows)),
          open('gain_fit.json', 'w'), indent=1)
