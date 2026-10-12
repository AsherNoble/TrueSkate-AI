"""L0 probe set: name -> dict(payload, cmd=[(t_ms, x, y)...], kind). All points in 414x896 viewport pt."""
import json, math

def chain(pts, durs, pauses=None):
    """pts: [(x,y)...] integer; durs: per-segment ms. pauses: {seg_index_before_which: ms} as W3C 'pause' items."""
    pauses = pauses or {}
    acts = [{'type':'pointerMove','duration':0,'x':pts[0][0],'y':pts[0][1],'origin':'viewport'},{'type':'pointerDown','button':0}]
    cmd = [(0, pts[0][0], pts[0][1])]; t = 0
    for i, (p, d) in enumerate(zip(pts[1:], durs)):
        if i in pauses:
            acts.append({'type':'pause','duration':pauses[i]}); t += pauses[i]; cmd.append((t, cmd[-1][1], cmd[-1][2]))
        acts.append({'type':'pointerMove','duration':d,'x':p[0],'y':p[1],'origin':'viewport'}); t += d; cmd.append((t, p[0], p[1]))
    if len(durs) in pauses:
        acts.append({'type':'pause','duration':pauses[len(durs)]}); t += pauses[len(durs)]; cmd.append((t, cmd[-1][1], cmd[-1][2]))
    acts.append({'type':'pointerUp','button':0})
    return {'actions':[{'type':'pointer','id':'timed_path','parameters':{'pointerType':'touch'},'actions':acts}]}, cmd

def split(total, n):
    base = [total // n] * n
    for i in range(total - sum(base)): base[i] += 1
    return base

def probes():
    P = {}
    A, B = (107, 448), (307, 448)
    # P1 single moves, 200 pt
    for d in (16, 33, 50, 100, 105, 110, 111, 120, 125, 200, 400, 800, 1200):
        P[f'single_{d}'] = (chain([A, B], [d]), 'single')
    P['single_2400_v400'] = (chain([(207, 248), (207, 648)], [2400]), 'single')
    # P2 collinear subdivision of 200 pt
    for total, ns in ((300, (1, 2, 3, 5, 10, 20, 30)), (600, (6, 20, 40))):
        for n in ns:
            xs = [round(107 + 200 * k / n) for k in range(n + 1)]
            P[f'sub_{total}_n{n}'] = (chain([(x, 448) for x in xs], split(total, n)), 'subdiv')
    # P7 non-frame-multiple chains (zig-zag on a line)
    for d in (105, 111, 125):
        pts = [A if k % 2 == 0 else B for k in range(7)]
        P[f'zig6x{d}'] = (chain(pts, [d] * 6), 'quant')
    # P3 corners: 120 pt legs, turn angle a (deviation from straight)
    for a in (30, 90, 150):
        for d in (100, 200):
            v = (207, 448); ux = (1, 0)
            p0 = (round(v[0] - 120), v[1])
            th = math.radians(a)
            p2 = (round(v[0] + 120 * math.cos(th)), round(v[1] - 120 * math.sin(th)))
            P[f'corner{a}_{d}'] = (chain([p0, v, p2], [d, d]), 'corner')
    # P4 holds via zero-length pointerMove (first/mid/last) and W3C pause alternatives
    for h in (50, 100, 200):
        P[f'hold_first_{h}'] = (chain([A, A, B], [h, 300]), 'hold')
        P[f'hold_mid_{h}'] = (chain([A, (207, 448), (207, 448), B], [150, h, 150]), 'hold')
        P[f'hold_last_{h}'] = (chain([A, B, B], [300, h]), 'hold')
    P['pause_first_100'] = (chain([A, B], [300], {0: 100}), 'pause')
    P['pause_mid_100'] = (chain([A, (207, 448), B], [150, 150], {1: 100}), 'pause')
    P['pause_last_100'] = (chain([A, B], [300], {1: 100}), 'pause')
    # P5 segment count at fixed 600 ms, zig-zag in 2D
    zz = [(107, 348), (307, 448), (107, 548), (307, 648), (107, 448), (307, 348), (107, 248), (307, 548), (107, 648), (307, 448), (207, 348)]
    for n in (1, 2, 5, 10):
        P[f'chain{n}_600'] = (chain(zz[:n + 1], split(600, n)), 'chain')
    return P

def r6(path):
    out = {}
    for r in json.load(open(path)):
        acts = r['payload']['actions'][0]['actions']
        cmd = [(0, acts[0]['x'], acts[0]['y'])]; t = 0
        for a in acts[2:-1]:
            t += a['duration']; cmd.append((t, a['x'], a['y']))
        assert [c[0] for c in cmd] == r['times_ms'], r['id']
        out[f"r6_{r['id']:03d}"] = ((r['payload'], cmd), 'r6')
    return out

if __name__ == '__main__':
    P = probes(); print(len(P)); [print(k, v[1], v[0][1][-1][0]) for k, v in P.items()]
