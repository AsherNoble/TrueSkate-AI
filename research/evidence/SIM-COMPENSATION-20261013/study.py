"""Prospective constrained simulator compensation; offline compiler and scoring.

No production modules are changed. run.py is the bounded simulator client.
"""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
sys.path.insert(0, str(WORK / 'src'))
from trueskate_ai.model1.curved.contract import Gesture, digest

TICK = 1000 / 60
CP = np.array([.2620539963245392, .17945900559425354, .7582859992980957, .8902440071105957])
SCALE = np.array([414., 896.])

def ease(x):
    x = np.clip(np.asarray(x), 0, 1)
    lo, hi = np.zeros_like(x), np.ones_like(x)
    for _ in range(40):
        u = (lo + hi) / 2
        bx = 3*(1-u)**2*u*CP[0] + 3*(1-u)*u*u*CP[2] + u**3
        lo, hi = np.where(bx < x, u, lo), np.where(bx < x, hi, u)
    return 3*(1-u)**2*u*CP[1] + 3*(1-u)*u*u*CP[3] + u**3

def gesture(points, times):
    # Quarter-point interior offset avoids floating-point int truncating an exact
    # integer one point low. The payload must compile back to the frozen integers.
    g = Gesture(tuple(tuple((np.array(p)+.25)/SCALE) for p in points), tuple(int(t) for t in times), 'random_chain')
    payload = g.payload()
    actual = [payload['actions'][0]['actions'][0]] + payload['actions'][0]['actions'][2:-1]
    assert [[a['x'], a['y']] for a in actual] == [list(p) for p in points]
    return g

def reference(case, tt):
    return np.column_stack([np.interp(tt, case['times_ms'], np.array(case['points'])[:, k]) for k in range(2)])

def basis(durations, tt):
    # Development approximation: segment clocks ceil to 60 Hz, each leg eases
    # using its requested duration. This is intentionally not called an exact law.
    ends = np.cumsum(np.ceil(np.array(durations)/TICK - 1e-9)*TICK)
    starts = np.r_[0, ends[:-1]]
    mat = np.zeros((len(tt), len(durations)+1))
    for j, t in enumerate(tt):
        i = min(np.searchsorted(ends, t, side='right'), len(durations)-1)
        q = float(ease((t-starts[i])/durations[i]))
        mat[j, i], mat[j, i+1] = 1-q, q
    return mat, ends

def compile_candidate(case):
    points = np.array(case['points'], float)
    times = np.array(case['times_ms'])
    durations = np.diff(times)
    tt = np.arange(0, times[-1]+.1, TICK)
    target = reference(case, tt)/SCALE
    choices = [range(1, max(1, int(d//110))+1) if np.any(a != b) else [1]
               for a, b, d in zip(points[:-1], points[1:], durations)]
    possibilities = []
    for counts in itertools.product(*choices):
        if sum(counts)>10:
            continue
        new_points = [points[0]]
        desired_boundaries = [0.]
        fixed = {0: points[0]}
        free = []
        semantic_indices = []
        for leg, count in enumerate(counts):
            for k in range(1, count+1):
                idx = len(new_points)
                new_points.append(points[leg] + (points[leg+1]-points[leg])*k/count)
                desired_boundaries.append(times[leg]+durations[leg]*k/count)
                if k == count:
                    fixed[idx] = points[leg+1]
                    semantic_indices.append(idx-1)
                else:
                    free.append((idx, leg))
        ticks = np.diff(np.rint(np.array(desired_boundaries)/TICK)).astype(int)
        if np.any(ticks<6):
            continue
        ds = np.maximum(100, np.rint((ticks-.4)*TICK)).astype(int)
        if ds.sum()<300:
            # Respect total minimum rather than silently shortening a contact.
            ds[-1] += 300-int(ds.sum())
        if ds.sum()>1200:
            continue
        mat, ends = basis(ds, tt)
        coords = np.array(new_points)/SCALE
        if free:
            origin = coords.copy()
            columns = []
            for idx, leg in free:
                origin[idx] = points[leg]/SCALE
                delta = (points[leg+1]-points[leg])/SCALE
                columns.append((mat[:, idx, None]*delta).ravel())
            design = np.column_stack(columns)
            lambdas = np.linalg.lstsq(design, (target-mat@origin).ravel(), rcond=None)[0]
            for leg in range(len(durations)):
                selected = [(k, idx) for k, (idx, ll) in enumerate(free) if ll == leg]
                last = 0.
                for k, idx in selected:
                    last = max(last, float(np.clip(lambdas[k], 0, 1)))
                    coords[idx] = (points[leg]+last*(points[leg+1]-points[leg]))/SCALE
        compiled_points = np.rint(coords*SCALE).astype(int).tolist()
        compiled_times = np.r_[0, np.cumsum(ds)].astype(int).tolist()
        try:
            g = gesture(compiled_points, compiled_times)
        except ValueError:
            continue
        pred = mat @ (np.array(compiled_points)/SCALE)
        rms = float(np.sqrt(np.mean(np.sum((pred-target)**2, axis=1))))
        release = float(ends[-1]-times[-1])
        boundary = float(np.mean(np.abs(ends[semantic_indices]-times[1:])))
        objective = rms + .00015*abs(release) + .0001*boundary
        possibilities.append((objective, compiled_points, compiled_times, counts, rms, release, boundary))
    # Identity is an explicit comparator/fallback, never counted as compensation.
    mat, ends = basis(durations, tt)
    pred = mat@(points/SCALE)
    rms = float(np.sqrt(np.mean(np.sum((pred-target)**2, axis=1))))
    identity_score = rms + .00015*abs(ends[-1]-times[-1]) + .0001*float(np.mean(abs(ends-times[1:])))
    possibilities.append((identity_score, points.astype(int).tolist(), times.tolist(), [1]*len(durations), rms, float(ends[-1]-times[-1]), float(np.mean(abs(ends-times[1:])))))
    best = min(possibilities, key=lambda x: (x[0], len(x[1]), x[2]))
    nominal = gesture(case['points'], case['times_ms'])
    adjusted = gesture(best[1], best[2])
    changed = digest(nominal.payload()) != digest(adjusted.payload())
    return {'points': best[1], 'times_ms': best[2], 'payload': adjusted.payload(),
            'payload_sha256': digest(adjusted.payload()), 'changed': changed,
            'status': 'candidate' if changed else 'unchanged constrained fallback',
            'model_objective': best[0], 'model_rms_normalized': best[4],
            'model_release_error_ms': best[5], 'model_boundary_mae_ms': best[6],
            'enumerated_legal_options': len(possibilities)}

def freeze():
    # Independent synthetic command geometries/times, not selected from old R6.
    specs = [
        ('dev_straight', [0,600], [[95,405],[309,405]]),
        ('dev_corner', [0,250,600], [[102,510],[270,510],[270,310]]),
        ('dev_hold', [0,150,450,600], [[110,400],[110,400],[290,400],[290,400]]),
        ('dev_unequal', [0,180,430,700], [[115,610],[281,610],[281,380],[156,380]]),
        ('straight317', [0,317], [[103,362],[295,362]]),
        ('straight463', [0,463], [[276,607],[118,327]]),
        ('straight739', [0,739], [[121,438],[302,532]]),
        ('corner137_241', [0,137,378], [[104,528],[268,528],[268,348]]),
        ('corner219_163', [0,219,382], [[283,360],[126,360],[126,594]]),
        ('corner281_357', [0,281,638], [[113,601],[289,407],[126,287]]),
        ('three123_197_261', [0,123,320,581], [[111,365],[288,365],[288,555],[140,555]]),
        ('four109_157_193_227', [0,109,266,459,686], [[108,565],[243,565],[243,365],[143,365],[143,509]]),
        ('initial_hold117', [0,117,476], [[124,437],[124,437],[289,437]]),
        ('middle_hold131', [0,213,344,621], [[114,571],[266,571],[266,571],[266,345]]),
        ('final_hold149', [0,381,530], [[111,349],[286,538],[286,538]]),
        ('all_holds', [0,109,338,465,638,775], [[118,403],[118,403],[281,403],[281,403],[281,573],[281,573]]),
        ('initial_final_holds', [0,137,556,669], [[133,563],[133,563],[281,354],[281,354]]),
        ('minimum_turns', [0,100,200,300], [[115,541],[275,541],[275,347],[131,347]]),
        ('minimum_holds', [0,100,200,300], [[133,413],[133,413],[279,413],[279,413]]),
        ('ten_minimum_legs', list(range(0,1001,100)), [[120+145*(i%2), 335+26*i] for i in range(11)]),
    ]
    cases = {}
    for i, (name, times, points) in enumerate(specs):
        g = gesture(points, times)
        case = {'split': 'development' if i<4 else 'heldout', 'points': points, 'times_ms': times,
                'nominal_payload': g.payload(), 'nominal_sha256': digest(g.payload())}
        case['adjusted'] = compile_candidate(case)
        cases[name] = case
    out = {'cases': cases, 'request_limit': 160, 'setup_smoke_limit': 10, 'repeats': 4,
           'seed': 2026101303, 'candidate_version': 'legal_ceil_clock_ls_v1',
           'scoring': {'position_rms_max': .005, 'position_max': .01, 'boundary_abs_ms_max': TICK+1,
                       'release_abs_ms_max': TICK+1, 'macro_rms_reduction_min': .25,
                       'required_heldout_cases_passing': 16},
           'script_sha256_at_freeze': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (ROOT/'FROZEN.json').open('x') as stream:
        json.dump(out, stream, indent=2)
    print('Frozen', len(cases), 'cases;', sum(c['adjusted']['changed'] for c in cases.values()), 'changed payloads')

def samples(row):
    result = {}
    for e in row['touches']:
        for v in e.get('co') or [[e['ts'], e['x'], e['y'], e['ph']]]:
            # Last position at a timestamp; original rows and phases remain saved.
            result[v[0]] = v[1:3]
    return sorted(result.items())

def analyze():
    frozen = json.loads((ROOT/'FROZEN.json').read_text())
    raw = WORK/'tmp/sim-compensation/requests.jsonl'
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    metrics = []
    for row in rows:
        if row['stage'] == 'setup':
            continue
        case = frozen['cases'][row['case']]
        b = next(e['ts'] for e in row['touches'] if e['ph']=='B')
        end = next(e['ts'] for e in row['touches'] if e['ph']=='E')
        ss = samples(row)
        ts = (np.array([s[0] for s in ss])-b)*1000
        xy = np.array([s[1] for s in ss])
        tt = np.arange(0, max(case['times_ms'][-1], (end-b)*1000)+.1, TICK)
        observed = np.column_stack([np.interp(tt, ts, xy[:,k]) for k in range(2)])
        desired = reference(case, tt)
        error = np.linalg.norm((observed-desired)/SCALE, axis=1)
        # Earliest approach to each desired waypoint, only near its desired time;
        # repeated endpoints are scored through hold boundaries below.
        arrivals = []
        for i in range(1, len(case['points'])):
            if case['points'][i] == case['points'][i-1]:
                continue
            p = np.array(case['points'][i]); target = case['times_ms'][i]
            mask = (ts>=case['times_ms'][i-1]) & (ts<=target+200)
            near = ts[mask][np.linalg.norm(xy[mask]-p,axis=1)<=1.]
            arrivals.append(float(near[0]-target) if len(near) else 1000.)
        holds = []
        for i in range(len(case['points'])-1):
            if case['points'][i] != case['points'][i+1]:
                continue
            p = np.array(case['points'][i]); start, stop = case['times_ms'][i:i+2]
            # Contiguous endpoint residence enclosing target hold midpoint.
            near = np.linalg.norm(xy-p,axis=1)<=1.
            indices = np.flatnonzero(near)
            groups = np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            relevant = [g for g in groups if len(g) and ts[g[0]]<=stop and ts[g[-1]]>=start]
            if not relevant:
                holds.extend([1000.,1000.]); continue
            group = max(relevant,key=lambda g:ts[g[-1]]-ts[g[0]])
            # Stationary samples can be omitted; infer the departure boundary
            # from the next moving sample, and retain one-frame uncertainty.
            hold_start = ts[group[0]]
            hold_end = ts[min(group[-1]+1,len(ts)-1)]
            holds.extend([float(hold_start-start),float(hold_end-stop)])
        release = (end-b)*1000-case['times_ms'][-1]
        timing = max([abs(release)]+[abs(x) for x in arrivals+holds])
        metrics.append({'case':row['case'],'split':case['split'],'variant':row['variant'],'repeat':row['repeat'],
                        'rms_normalized':float(np.sqrt(np.mean(error**2))),'max_normalized':float(error.max()),
                        'waypoint_arrival_errors_ms':arrivals,'hold_boundary_errors_ms':holds,
                        'release_error_ms':float(release),'max_boundary_abs_ms':float(timing),
                        'contact_duration_ms':float((end-b)*1000),'callbacks':len(row['touches']),
                        'callback_lag_max_ms':max((e['now']-e['ts'])*1000 for e in row['touches'])})
    by = {}
    for name in frozen['cases']:
        by[name] = {}
        for variant in ['nominal','adjusted']:
            rr = [r for r in metrics if r['case']==name and r['variant']==variant]
            assert len(rr)==4, (name,variant,len(rr))
            v = {k:float(np.median([r[k] for r in rr])) for k in ['rms_normalized','max_normalized','release_error_ms','max_boundary_abs_ms','callback_lag_max_ms']}
            v['passes_full_trajectory'] = bool(v['rms_normalized']<=.005 and v['max_normalized']<=.01 and v['max_boundary_abs_ms']<=TICK+1)
            by[name][variant] = v
        by[name]['payload_changed'] = frozen['cases'][name]['adjusted']['changed']
        by[name]['split'] = frozen['cases'][name]['split']
    held = [v for v in by.values() if v['split']=='heldout']
    nominal = float(np.mean([v['nominal']['rms_normalized'] for v in held]))
    adjusted = float(np.mean([v['adjusted']['rms_normalized'] for v in held]))
    result = {'measurement_requests':len(metrics),'setup_requests':sum(r['stage']=='setup' for r in rows),
              'heldout_macro_rms_nominal':nominal,'heldout_macro_rms_adjusted':adjusted,
              'heldout_macro_rms_reduction':1-adjusted/nominal,
              'heldout_full_pass_nominal':sum(v['nominal']['passes_full_trajectory'] for v in held),
              'heldout_full_pass_adjusted':sum(v['adjusted']['passes_full_trajectory'] for v in held),
              'cases':by,'metrics':metrics,'frozen_sha256':hashlib.sha256((ROOT/'FROZEN.json').read_bytes()).hexdigest(),
              'raw_requests_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with gzip.GzipFile(filename=str(ROOT/'requests.jsonl.gz'),mode='wb',mtime=0) as out:
        out.write(raw.read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','metrics']},indent=2))

if __name__=='__main__':
    {'freeze':freeze,'analyze':analyze}[sys.argv[1]]()
