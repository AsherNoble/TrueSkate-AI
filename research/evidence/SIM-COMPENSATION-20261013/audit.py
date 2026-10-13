"""Post-freeze measurement audit and labeled scorer sensitivities; no requests."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import study

ROOT = Path(__file__).resolve().parent

def main():
    frozen = json.loads((ROOT/'FROZEN.json').read_text())
    rows = [json.loads(line) for line in gzip.open(ROOT/'requests.jsonl.gz','rt')]
    assert len(rows)==162
    assert sum(row['stage']=='measurement' for row in rows)==160
    assert sum(row['stage']=='setup' for row in rows)==2
    assert hashlib.sha256((ROOT/'study.py').read_bytes()).hexdigest()==frozen['script_sha256_at_freeze']
    result = json.loads((ROOT/'results.json').read_text())
    primary_missing = []
    timing_failures = []
    sensitivity = []
    phase_counts = {}
    for row in rows:
        timing = row['timing']['value']
        assert timing['dropped_records']==0 and len(timing['records'])==1
        record = timing['records'][0]
        assert record['sequence']==0 and record['outcome']=='success'
        assert abs(record['request_entered']['monotonic_s']-row['http_entered_monotonic'])<1
        for event in row['touches']:
            phase_counts[event['ph']] = phase_counts.get(event['ph'],0)+1
            co = event.get('co',[])
            if co and not any(abs(v[0]-event['ts'])<1e-7 and np.linalg.norm(np.array(v[1:3])-[event['x'],event['y']])<1e-7 for v in co):
                primary_missing.append({'index':row['index'],'case':row['case'],'phase':event['ph'],'sample':[event['ts'],event['x'],event['y']],'co':co})
        if row['stage']=='setup':
            continue
        case = frozen['cases'][row['case']]
        begin = next(e['ts'] for e in row['touches'] if e['ph']=='B')
        union = {v[0]:v[1:3] for event in row['touches'] for v in event.get('co',[])}
        for event in row['touches']:
            union[event['ts']] = [event['x'],event['y']]
        ss = sorted(union.items())
        ts = (np.array([x[0] for x in ss])-begin)*1000
        xy = np.array([x[1] for x in ss])
        ended = next(e['ts'] for e in row['touches'] if e['ph']=='E')
        tt = np.arange(0,max(case['times_ms'][-1],(ended-begin)*1000)+.1,study.TICK)
        observed = np.column_stack([np.interp(tt,ts,xy[:,k]) for k in range(2)])
        error = np.linalg.norm((observed-study.reference(case,tt))/study.SCALE,axis=1)
        # Exploratory audit: sparse stationary input is held, not interpolated
        # toward a later moving sample.25ms exceeds all old regular cadence gaps.
        gap_observed = observed.copy()
        previous = np.clip(np.searchsorted(ts,tt,side='right')-1,0,len(ts)-1)
        following = np.minimum(previous+1,len(ts)-1)
        sparse = (ts[following]-ts[previous]>25.) & (tt<ts[following])
        gap_observed[sparse] = xy[previous[sparse]]
        gap_error = np.linalg.norm((gap_observed-study.reference(case,tt))/study.SCALE,axis=1)
        primary = next(x for x in result['metrics'] if x['case']==row['case'] and x['variant']==row['variant'] and x['repeat']==row['repeat'])
        holds = []
        missing = 0
        for i in range(len(case['points'])-1):
            if case['points'][i]!=case['points'][i+1]:
                continue
            start,stop = case['times_ms'][i:i+2]
            midpoint = (start+stop)/2
            indices = np.flatnonzero(np.linalg.norm(xy-np.array(case['points'][i]),axis=1)<=1.)
            groups = np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            # Sensitivity implements the stricter comment, not frozen intersection rule.
            relevant = [g for g in groups if len(g) and ts[g[0]]<=midpoint<=ts[min(g[-1]+1,len(ts)-1)]]
            if not relevant:
                missing += 1
                holds.extend([None,None])
            else:
                group = max(relevant,key=lambda g:ts[g[-1]]-ts[g[0]])
                holds.extend([float(ts[group[0]]-start),float(ts[min(group[-1]+1,len(ts)-1)]-stop)])
        frozen_holds = primary['hold_boundary_errors_ms']
        same_holds = len(holds)==len(frozen_holds) and all(a is not None and abs(a-b)<1e-6 for a,b in zip(holds,frozen_holds))
        sensitivity.append({'case':row['case'],'variant':row['variant'],'repeat':row['repeat'],
                            'union_rms_normalized':float(np.sqrt(np.mean(error**2))),
                            'union_minus_primary_rms':float(np.sqrt(np.mean(error**2))-primary['rms_normalized']),
                            'hold_midpoint_boundary_errors_ms':holds,'hold_midpoint_missing':missing,
                            'hold_midpoint_same_as_frozen_intersection':same_holds,
                            'frozen_missing_arrivals':sum(v==1000. for v in primary['waypoint_arrival_errors_ms']),
                            'frozen_missing_hold_boundaries':sum(v==1000. for v in frozen_holds),
                            'gap_aware_rms_normalized':float(np.sqrt(np.mean(gap_error**2))),
                            'gap_aware_max_normalized':float(gap_error.max()),
                            'gap_aware_max_boundary_abs_ms':max([abs(primary['release_error_ms'])]+[abs(x) for x in primary['waypoint_arrival_errors_ms']]+[1000. if x is None else abs(x) for x in holds])})
    for case in frozen['cases'].values():
        nominal = study.gesture(case['points'],case['times_ms']).payload()
        adjusted = study.gesture(case['adjusted']['points'],case['adjusted']['times_ms']).payload()
        assert study.digest(nominal)==case['nominal_sha256']
        assert study.digest(adjusted)==case['adjusted']['payload_sha256']
        assert study.compile_candidate(case)==case['adjusted']
    gap_cases = {}
    for name,case in frozen['cases'].items():
        if case['split']!='heldout':
            continue
        gap_cases[name] = {}
        for variant in ['nominal','adjusted']:
            rr = [s for s in sensitivity if s['case']==name and s['variant']==variant]
            vv = {k:float(np.median([s[k] for s in rr])) for k in ['gap_aware_rms_normalized','gap_aware_max_normalized','gap_aware_max_boundary_abs_ms']}
            vv['passes_full_trajectory'] = bool(vv['gap_aware_rms_normalized']<=.005 and vv['gap_aware_max_normalized']<=.01 and vv['gap_aware_max_boundary_abs_ms']<=study.TICK+1)
            gap_cases[name][variant] = vv
    gap_rms = {v:float(np.mean([c[v]['gap_aware_rms_normalized'] for c in gap_cases.values()])) for v in ['nominal','adjusted']}
    out = {'requests':len(rows),'measurement_requests':160,'setup_requests':2,
           'candidate_source_unchanged_since_freeze':True,'all40_payloads_recompile_identically':True,
           'all162_WDA_timing_records_fresh_success':True,'primary_samples_missing_from_nonempty_co':primary_missing,
           'phase_callback_counts':phase_counts,
           'union_max_abs_rms_change':max(abs(s['union_minus_primary_rms']) for s in sensitivity),
           'hold_midpoint_rows_different':sum(not s['hold_midpoint_same_as_frozen_intersection'] for s in sensitivity),
           'missing_arrivals':sum(s['frozen_missing_arrivals'] for s in sensitivity),
           'missing_hold_boundaries':sum(s['frozen_missing_hold_boundaries'] for s in sensitivity),
           'missing_sentinel_interpretation':'Frozen1000 is missing/failure, never a measured1000ms delay',
           'gap_aware_exploratory_sensitivity':{'rule':'Hold previous position across observed gaps>25ms; interpolate ordinary cadence gaps. Uses midpoint hold selector. Post-freeze audit, not primary.',
                                               'heldout_macro_rms':gap_rms,'rms_reduction':1-gap_rms['adjusted']/gap_rms['nominal'],
                                               'full_pass_counts':{v:sum(c[v]['passes_full_trajectory'] for c in gap_cases.values()) for v in ['nominal','adjusted']},'cases':gap_cases},
           'sensitivities':sensitivity,'frozen_sha256':hashlib.sha256((ROOT/'FROZEN.json').read_bytes()).hexdigest()}
    (ROOT/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['sensitivities','primary_samples_missing_from_nonempty_co']},indent=2))

if __name__=='__main__':
    main()
