"""Frozen CURVE-EXEC command generation and offline preparation."""
from __future__ import annotations
import hashlib
import copy
import json
import math
from pathlib import Path
import numpy as np
from trueskate_ai.sim.cubic_curve import CubicInTime, compile_curve, fit_eight_positions

EXPERIMENT = 'CURVE-EXEC-20261002'
SEED = 20261002
FAMILIES = ('straight','arc','s_curve','reversal')
RULES = (10,20,40)
DEVICES = ('iPhone_XR','iPhone_XR2')
PROTOCOL = dict(experiment=EXPERIMENT, seed=SEED, confirmation_seed=SEED+1,
    representation='cubic_in_time_v1', max_segment_duration_ms=list(RULES), min_segment_duration_ms=1,
    offline_n=[2,4,8,16,32], offline_bound=.005, primary_budget=.01, secondary_budget=.03,
    floor_primary_max=.005, floor_stop_above=.025, floor_fallback_budget=.03,
    progress_durations_s=[.6,1.2], progress_tolerance_frames=2, duration_tolerance_s=.10,
    evaluable_overall=.90, evaluable_device_family=.80, max_diagnostics=204,
    pilot_count=36, main_count=144, confirmation_count=24, per_recording_max=8,
    audit_fraction=.15, reannotation_fraction=.10, audit_seed=SEED+2,
    selection='coarsest eligible maximum-duration rule; fresh confirmation required',
    controls={'start_s':1.,'middle_s':30.,'end_s':57.,'hold_s':.05,'point':[.5,.5]},
    diagnostic_slots_s=[6.,12.,18.,24.,36.,42.,48.,54.], stop_recording_s=59.,
    late_tolerance_s=.5, heldout_middle_tolerance_s=2/30,
    measurement_floor='max(audited measurement uncertainty, half maximum paired shape/endpoint discrepancy)',
    uncertainty='max extractor-human discrepancy + max blinded within-human discrepancy',
    orange_hsv={'lower':[5,120,140],'upper':[35,255,255]},
    extraction='blind orange mask minus pre-command orange background; unsnapped skeleton',
    collapse='whole observable trail appears in a single native frame; indeterminate if fading or occlusion hides growth',
    timing_evidence='8 gestures within one frame; held-out middle miss 38 ms; not a finer-accuracy guarantee',
    third_party_hypothesis='gptd-swift PR3: interpolation and sub-16ms collapse unverified on this rig',
    abort=['contamination','recorder failure','incomplete timing','schedule overrun','inadequate pilot measurement'],
    scope_exclusions=['training corpora','cloud training','Model 2','overlap','spin','model curve recovery/generation'])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def seeded_curves(seed=SEED):
    rng = np.random.default_rng(seed)
    templates = [np.array([[-1,0],[-.95,0],[.45,0],[1,0]]),
                 np.array([[-1,0],[-.65,-.9],[.65,-.9],[1,0]]),
                 np.array([[-1,0],[-.3,.95],[.3,-.95],[1,0]]),
                 np.array([[-.5,0],[1,.6],[-1,-.6],[.5,0]])]
    rows = []
    for family,template in zip(FAMILIES,templates):
        for duration in (.3,.6,1.2):
            for _ in range(1000):
                theta = rng.uniform(-math.pi,math.pi)
                rotation = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
                scale = rng.uniform(.13,.25)
                p = template @ rotation * scale + rng.uniform([.54,.46],[.65,.58])
                if rng.random() < .5:
                    p = p[::-1]
                curve = CubicInTime(p,duration)
                try:
                    curve.validate_safe(central=True)
                except ValueError:
                    continue
                rows.append(dict(curve_id=f'{seed}-{family}-{duration:.2f}',family=family,curve=curve.to_dict()))
                break
            else:
                raise RuntimeError('safe coefficient sampling exhausted')
    return rows


def command(row, *, n=None, spacing=None, repeat=0, stage='main'):
    compiled = compile_curve(CubicInTime.from_dict(row['curve']),n=n,max_segment_duration_ms=spacing)
    return dict(**row,stage=stage,repeat=repeat,rule_ms=spacing,requested_n=n,
                compiled=compiled.to_dict(),offline_eligible=compiled.approximation_bound <= .005)


def command_manifest():
    base = seeded_curves()
    fresh = seeded_curves(SEED+1)
    pilot = [r for r in base if r['curve']['duration_s'] in (.3,1.2)]
    stages = {}
    for index,device in enumerate(DEVICES):
        p = [command(r,n=16,repeat=rep,stage='pilot') for rep in range(2) for r in pilot]
        dense = next(r for r in base if r['family']=='s_curve' and r['curve']['duration_s']==.3)
        p += [command(dense,n=32,repeat=rep,stage='pilot_dense') for rep in range(2)]
        main = [command(r,spacing=s,repeat=rep) for r in base for s in RULES for rep in range(2)]
        np.random.default_rng(SEED+index+100).shuffle(main)
        stages[device] = dict(pilot=p,main=main,confirmation={str(s):[command(r,spacing=s,stage='confirmation') for r in fresh] for s in RULES})
        for stage in ('pilot','main'):
            for i,row in enumerate(stages[device][stage]):
                row['command_id']=f'{device}-{stage}-{i:03d}'
        for s,rows in stages[device]['confirmation'].items():
            for i,row in enumerate(rows):
                row['command_id']=f'{device}-confirmation-{s}-{i:03d}'
    root = Path(__file__).resolve().parents[3]
    sources = [
        'src/trueskate_ai/sim/cubic_curve.py', 'src/trueskate_ai/sim/touch_actions.py',
        'src/trueskate_ai/sim/gestures.py', 'src/trueskate_ai/sim/device.py',
        'src/trueskate_ai/data/touch_labels.py', 'src/trueskate_ai/data/control_hitboxes.py',
        'src/trueskate_ai/collection/wda_action_timing.py',
        'src/trueskate_ai/collection/xctest_capture.py',
        'src/trueskate_ai/collection/tap_timing_calibration.py',
        'src/trueskate_ai/collection/scene_settle.py', 'src/trueskate_ai/collection/gameplay_filter.py',
        'src/trueskate_ai/research/curve_protocol.py', 'src/trueskate_ai/research/curve_probe.py',
        'src/trueskate_ai/research/curve_measurement.py', 'src/trueskate_ai/research/curve_report.py',
        'scripts/collection/probe_cubic_curves.py', 'scripts/inspect/prepare_curve_exec.py',
        'scripts/inspect/measure_curve_exec.py', 'scripts/inspect/report_curve_exec.py',
        'scripts/inspect/build_curve_audit.py', 'scripts/inspect/templates/curve_audit.html',
    ]
    manifest = dict(protocol=copy.deepcopy(PROTOCOL),devices=stages,
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    manifest['sha256'] = digest(manifest)
    return manifest


def verify_manifest(manifest,*,check_source=True):
    payload={k:v for k,v in manifest.items() if k!='sha256'}
    expected=command_manifest()
    # Frozen bytes are authoritative. Regeneration is an additional seed check:
    # NumPy/BLAS versions differ by a few ulps across hosts, never device points
    # or ms boundaries. Permit only 1e-12 for floating diagnostic calculations.
    if (manifest.get('protocol') != PROTOCOL or digest(payload)!=manifest.get('sha256')
            or (check_source and manifest.get('source_sha256')!=expected['source_sha256'])
            or not equivalent_commands(manifest.get('devices'),expected['devices'])):
        raise ValueError('manifest differs from frozen protocol/seeded commands')


def equivalent_commands(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equivalent_commands(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equivalent_commands(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return math.isfinite(a) and math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
    return a==b


def offline_report():
    rows=[]
    for row in seeded_curves():
        c=CubicInTime.from_dict(row['curve'])
        for n in (2,4,8,16,32):
            compiled=compile_curve(c,n=n)
            rows.append(dict(curve_id=row['curve_id'],family=row['family'],duration_s=c.duration_s,
                             comparison='fixed_n',**compiled.to_dict()))
        for spacing in RULES:
            compiled=compile_curve(c,max_segment_duration_ms=spacing)
            rows.append(dict(curve_id=row['curve_id'],family=row['family'],duration_s=c.duration_s,
                             comparison='spacing',rule_ms=spacing,**compiled.to_dict()))
    residuals=[]
    u=np.linspace(0,1,8)
    dense=np.linspace(0,1,2001)
    def circle(t):return np.stack([.60+.15*np.cos(2*np.pi*t),.52+.15*np.sin(2*np.pi*t)],axis=-1)
    def corner(t):return np.stack([.4+.4*np.minimum(2*t,1),.35+.3*np.maximum(2*t-1,0)],axis=-1)
    def bends(t):return np.stack([.4+.4*t,.52+.12*np.sin(6*np.pi*t)],axis=-1)
    for name,func in [('circle',circle),('corner',corner),('multiple_bends',bends)]:
        fit=fit_eight_positions(func(u),.6)
        errors=np.linalg.norm(fit.evaluate(dense)-func(dense),axis=1)
        residuals.append(dict(source=name,max_time_aligned_residual=float(errors.max()),
                              rms_time_aligned_residual=float(np.sqrt(np.mean(errors**2))),fit=fit.to_dict()))
    return dict(experiment=EXPERIMENT,rows=rows,representation_residuals=residuals,
                eligible_rules=[s for s in RULES if all(r['approximation_bound']<=.005 for r in rows if r.get('rule_ms')==s)],
                status='offline only; execution fidelity unestablished')
