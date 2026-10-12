"""Whole-clip head tracking: most consistent forward-moving head over all frames (Viterbi).

State per frame = commanded time tau (1 ms grid) whose path position the glow currently
ends at. Emission = edge evidence at s(tau)+R_CAP: trail glow behind, none ahead,
from the timing-blind along-path profile (hidden board/overlap stretches are neutral).
Transition: tau may move -4..+80 ms per native frame. Nothing references the ring's
nominal time; lag = nominal - tau afterwards. Positive lag = trace behind the ring.
"""
import json, sys
from pathlib import Path
import numpy as np
from trail_lag import decode, Path_, board_mask, R_CAP
from profile import profile, STEP

L = 48            # px either side of the candidate edge
ON = (10, 40)     # excess mapped to 0..1 "glow present"
FADE_MS = 200     # trail glow lifetime in commanded time (conservative)
STALE_FRAMES = 30 # 500 ms of continuous glow at one path position = scenery


def emissions(sample, frames, ks):
    path = Path_(sample); total = int(path.t[-1])
    taus = np.arange(0, total+1)
    edge_s = np.array([path.s_at_time(t) for t in taus])+R_CAP
    out, vis_out = [], []
    # Glow present continuously for STALE_FRAMES (frames before onset included) is scenery,
    # not trail: the trail at any spot fades within a few hundred ms.
    first = max(0, ks[0]-STALE_FRAMES)
    cache = {}
    for k in range(first, ks[-1]+1):
        e = profile(frames[k], path, board_mask(frames[k]), overlap=False)[1]
        cache[k] = (e, np.where(np.isnan(e), False, e >= ON[0]))
    for k in ks:
        e, _ = cache[k]
        window = [cache[j][1] for j in range(max(first, k-STALE_FRAMES), k+1)]
        stale = np.logical_and.reduce(window) if len(window) > STALE_FRAMES//2 else np.zeros(len(e), bool)
        e = np.where(stale, np.nan, e)
        hidden = np.isnan(e)
        o = np.where(hidden, 0, np.clip((np.nan_to_num(e)-ON[0])/(ON[1]-ON[0]), 0, 1))
        v = (~hidden).astype(float)
        co, cv = np.concatenate([[0], np.cumsum(o)]), np.concatenate([[0], np.cumsum(v)])
        idx = np.clip(np.round(edge_s/STEP).astype(int), 0, len(e))
        a0 = np.clip(idx-L//int(STEP), 0, len(e)); b1 = np.clip(idx+L//int(STEP), 0, len(e))
        n_behind, n_ahead = cv[idx]-cv[a0], cv[b1]-cv[idx]
        behind = np.where(n_behind > 0, (co[idx]-co[a0])/np.maximum(n_behind, 1), .5)
        ahead = np.where(n_ahead > 0, (co[b1]-co[idx])/np.maximum(n_ahead, 1), .5)
        # Past the path's end there is nothing to draw: count it as dark and visible.
        past_end = edge_s >= path.S[-1]-1
        ahead = np.where(past_end, 0, ahead); n_ahead = np.where(past_end, L/STEP, n_ahead)
        # Evidence only where both sides of the candidate edge are seen; unknown = no information.
        out.append(np.where((n_behind > 0) & (n_ahead > 0), behind-ahead, 0.))
        vis_out.append(np.minimum(n_behind, n_ahead)/(L/STEP))
    return taus, np.array(out), np.array(vis_out), path


def viterbi(E, dt, pinned, back=4, fwd=80, rate_cost=.005):
    """Most consistent forward head path. dt: elapsed ms per frame step. pinned: frames well
    before commanded onset, where no touch exists yet (head held at the path start).
    rate_cost per ms of |advance - dt| favours a steady lag of any size over jumps."""
    n, m = E.shape
    score = np.full(m, -np.inf); score[:5] = E[0][:5]; arg = np.zeros((n, m), int)
    for k in range(1, n):
        best = np.full(m, -np.inf); who = np.zeros(m, int)
        for d in range(-back, fwd+1):
            prev = np.full(m, -np.inf); src = np.arange(m)-d
            ok = (src >= 0) & (src < m)
            prev[ok] = score[src[ok]]-rate_cost*abs(d-dt[k])
            better = prev > best
            best[better] = prev[better]; who[better] = src[better]
        score = best+E[k]
        if pinned[k]:
            score[5:] = -np.inf
        arg[k] = who
    path = [int(np.argmax(score))]
    for k in range(n-1, 0, -1):
        path.append(int(arg[k][path[-1]]))
    return path[::-1]


def measure(sample, frames):
    times = np.asarray(sample["frame_times_s"]); onset, rate = sample["onset_s"], sample["clock_rate"]
    nominal = (times-onset)/rate*1000
    p0 = Path_(sample); total = p0.t[-1]
    ks = [k for k in range(len(times)) if -50 <= nominal[k] <= total+100]
    taus, E, V, path = emissions(sample, frames, ks)
    dt = np.diff(nominal[ks], prepend=nominal[ks][0])
    best = viterbi(E, dt, nominal[ks] < -35)
    # Recently drawn trail near the head (still glowing) makes the edge ambiguous.
    grid = np.arange(0, int(total)+1)
    xy_t = np.array([path.at_s(path.s_at_time(t)) for t in grid])
    s_t = np.array([path.s_at_time(t) for t in grid])
    def ambiguous(tau):
        # Recent glow (within FADE_MS) that is far back along the path but spatially close:
        # the path has come back near itself. Contiguous trail just behind the head is expected.
        old = (grid >= tau-FADE_MS) & (s_t <= path.s_at_time(tau)-120)
        if not old.any():
            return False
        here = np.array([path.at_s(path.s_at_time(tau)+d) for d in range(0, R_CAP+L+1, 8)])
        return bool((np.linalg.norm(here[:, None]-xy_t[old][None], axis=2) < 56).any())
    def corner(i):
        """Segment i starts after a hold or a turn sharper than 45 degrees."""
        if i == 0:
            return True
        a, b = path.p[i]-path.p[i-1], path.p[i+1]-path.p[i]
        if path.len[i-1] == 0 or path.len[i] == 0:
            return True
        return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) < np.cos(np.radians(45))
    rows = []
    for k, tau, e_row, v_row in zip(ks, best, E, V):
        s = path.s_at_time(tau)
        i = int(np.clip(np.searchsorted(path.t, tau, side="right")-1, 0, len(path.len)-1))
        speed = path.len[i]/(path.t[i+1]-path.t[i])
        evidence, visible = float(e_row[tau]), float(v_row[tau])
        active = 0 <= nominal[k] <= total
        rows.append(dict(frame=k, t=float(times[k]), nominal_ms=float(nominal[k]), tau_ms=int(tau),
                         lag_ms=float(nominal[k]-tau) if active else None, segment=i, speed_px_ms=float(speed),
                         evidence=evidence, visible=visible, head_xy=[float(v) for v in path.at_s(s)],
                         ambiguous=ambiguous(tau),
                         reliable=bool(active and evidence >= .5 and visible >= .75 and speed >= .4 and not ambiguous(tau)
                                       and (s-path.S[i] >= R_CAP or not corner(i)) and tau < total-5)))
    return rows


if __name__ == "__main__":
    M = Path(sys.argv[1]); OUT = Path(sys.argv[2]); ids = sys.argv[3]
    manifest = json.loads((M/"audit-6/manifest.json").read_text())
    chosen = [s for s in manifest["samples"] if ids == "all" or s["id"] in {int(x) for x in ids.split(",")}]
    results = {}
    for s in chosen:
        rows = measure(s, decode(M/s["sample_dir"]/"clip.mp4")); results[s["id"]] = rows
        lags = [r["lag_ms"] for r in rows if r["reliable"]]
        print(s["id"], s["gesture"]["sampler"], s["gesture"].get("profile"), "reliable", len(lags), "of", len(rows),
              "lag median/min/max", np.round([np.median(lags), min(lags), max(lags)], 1) if lags else "-", flush=True)
    (OUT/f"vlag-{ids if ids != 'all' else 'all'}.json").write_text(json.dumps(results))
