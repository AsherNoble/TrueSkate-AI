"""Paired synthetic robustness experiment; never trains or executes gestures."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

SEEDS = dict(cohort=2026101301, corruption=2026101302, bootstrap=2026101303)
LEVELS = (.005, .01, .02, .03, .04, .06)
FAMILIES = ("line", "kink", "alternating", "hold", "arch", "near_left", "near_bottom", "near_top")
TYPES = ("independent", "ar1", "drift", "endpoint")

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stable_bytes(x):
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

def fixture_points(family, rng):
    if family == "line":
        return rng.uniform([.28,.24], [.88,.84], (2,2))
    if family == "kink":
        return rng.uniform([.28,.24], [.88,.84], (3,2))
    if family == "alternating":
        n = int(rng.integers(4,8))
        lo, hi = sorted(rng.uniform(.28,.88,2))
        ys = rng.uniform(.25,.82,n)
        return np.column_stack([np.where(np.arange(n)%2,hi,lo),ys])
    if family == "hold":
        p = rng.uniform([.28,.24], [.88,.84], (3,2))
        return np.array([p[0],p[1],p[1],p[2]])
    if family == "arch":
        n = int(rng.integers(5,9))
        x = np.linspace(rng.uniform(.28,.4),rng.uniform(.7,.9),n)
        base, height = rng.uniform(.3,.55), rng.uniform(.04,.25)
        return np.column_stack([x,base+height*np.sin(np.linspace(0,np.pi,n))])
    if family == "near_left":
        n = int(rng.integers(3,7))
        return rng.uniform([.2245,.385], [.234,.475], (n,2))
    if family == "near_bottom":
        n = int(rng.integers(3,7))
        return rng.uniform([.28,.876], [.92,.880], (n,2))
    if family == "near_top":
        n = int(rng.integers(3,7))
        return rng.uniform([.25,.163], [.41,.169], (n,2))
    raise ValueError(family)

def generate(Gesture, segment_is_safe):
    rng = np.random.Generator(np.random.PCG64(SEEDS["cohort"]))
    fixtures, rejected = [], Counter()
    for family in FAMILIES:
        for i in range(24):
            for attempt in range(1000):
                p = fixture_points(family,rng)
                n = len(p)-1
                if n == 1:
                    durations = [int(rng.integers(300,1101))]
                else:
                    # Small slack deliberately challenges frame-locked knots.
                    total = int(rng.integers(max(300,100*n),min(1200,100*n+500)+1))
                    weights = rng.dirichlet(np.ones(n))
                    extra = rng.multinomial(total-100*n,weights)
                    durations = 100+extra
                t = np.r_[0,np.cumsum(durations)].astype(int)
                g = Gesture(tuple(map(tuple,p)),tuple(t.tolist()),"random_chain")
                q = g.command_points()
                if not all(segment_is_safe(a,b) for a,b in zip(q,q[1:])):
                    rejected[family] += 1
                    continue
                onset = .32+float(rng.uniform(0,1/60))
                rate = float(rng.uniform(.999,1.001))
                meta = dict(frame_times_s=(np.arange(120)/60).tolist(),onset_s=onset,
                            release_s=onset+rate*t[-1]/1000,clock_rate=rate,
                            gesture=g.to_dict())
                fixtures.append(dict(id=f"{family}-{i:02d}",family=family,meta=meta,
                                     generation_attempts=attempt+1))
                break
            else:
                raise RuntimeError("fixture generation failed")
    return fixtures,dict(rejected)

def noise_fields(index, a, b):
    # Independent fixture streams remain stable if the run is resumed/chunked.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEEDS["corruption"],index])))
    independent = rng.normal(size=(120,2))/np.sqrt(2)
    z = rng.normal(size=(120,2))/np.sqrt(2)
    ar = np.empty_like(z); ar[0] = z[0]
    for k in range(1,120):
        ar[k] = .9*ar[k-1]+np.sqrt(1-.9**2)*z[k]
    unit = rng.normal(size=2); unit /= np.linalg.norm(unit)
    u = np.clip((np.arange(120)-a)/(b-a),0,1)
    drift = (2*u-1)[:,None]*unit
    dirs = rng.normal(size=(2,2)); dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    weights = np.exp(-abs(np.arange(120)[:,None]-[a,b])/2)
    endpoint = weights@dirs
    return dict(independent=independent,ar1=ar,drift=drift,endpoint=endpoint)

def check_compile(pred, Gesture, segment_is_safe):
    """No fallback or correction; guard the exact proposed command."""
    tick = time.perf_counter_ns()
    if pred is None:
        return None,dict(valid=False,reasons=["decode_failure"],latency_ms=0.)
    p = np.asarray(pred["points"],float)
    t = np.asarray(pred["times_ms"],float)
    reasons = []
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all(): reasons.append("normalized_bounds")
    if not np.array_equal(t,np.rint(t)): reasons.append("integer_ms")
    if not 300<=t[-1]<=1200: reasons.append("duration")
    if not 1<=len(t)-1<=10: reasons.append("segment_cap")
    if np.any(np.diff(t)<100): reasons.append("segment_interval")
    out = dict(pred)
    # A compiler may reject protected-control paths. Their diagnostic score
    # still uses the intended quantized coordinates, never float coordinates
    # merely because compilation failed. Admissibility remains false.
    intended = np.trunc(p*[414,896])/[414,896]
    out["points"] = intended.tolist()
    unsafe = [i for i,(a,b) in enumerate(zip(intended,intended[1:])) if not segment_is_safe(a,b)]
    if unsafe: reasons.append("control_map")
    try:
        command = Gesture(tuple(map(tuple,p)),tuple(map(int,t)),"random_chain")
        payload = command.payload()
        q = command.command_points()
        assert np.array_equal(q,intended)
        out["points"] = q.tolist()
        # Validate every segment after actual viewport truncation.
        result = dict(valid=not reasons,reasons=reasons,unsafe_segments=unsafe,
                      payload_sha256=hashlib.sha256(stable_bytes(payload)).hexdigest())
    except (ValueError,TypeError) as exc:
        reasons.append("compiler")
        result = dict(valid=False,reasons=reasons,error=str(exc),unsafe_segments=unsafe)
    result["latency_ms"] = (time.perf_counter_ns()-tick)/1e6
    return out,result

def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),
                p99=float(np.quantile(a,.99)),max=float(a.max()))

def aggregate(rows):
    def group(rr):
        n=len(rr); x=np.array([r["candidate"]["admissible_pass"]-r["baseline"]["admissible_pass"] for r in rr],float)
        result=dict(n=n,baseline_metric_pass=sum(r["baseline"]["metric_pass"] for r in rr),
                    candidate_metric_pass=sum(r["candidate"]["metric_pass"] for r in rr),
                    baseline_admissible_pass=sum(r["baseline"]["admissible_pass"] for r in rr),
                    candidate_admissible_pass=sum(r["candidate"]["admissible_pass"] for r in rr),
                    gains=int((x>0).sum()),regressions=int((x<0).sum()),difference_pp=float(x.mean()*100))
        for variant in ("baseline","candidate"):
            result[variant+"_safety_failures"] = dict(Counter(reason for r in rr for reason in r[variant]["guard"]["reasons"]))
            result[variant+"_mean_error_median"] = float(np.median([r[variant]["score"]["mean_path_error"] for r in rr if r[variant]["score"] is not None]))
        clusters=defaultdict(list)
        for r,v in zip(rr,x): clusters[r["id"]].append(v)
        clustermeans=np.array([np.mean(v) for v in clusters.values()])
        rng=np.random.Generator(np.random.PCG64(SEEDS["bootstrap"]))
        boot=clustermeans[rng.integers(0,len(clustermeans),(1000,len(clustermeans)))].mean(1)*100
        result["paired_fixture_bootstrap_95_pp"] = np.quantile(boot,[.025,.975]).tolist()
        return result
    result=dict(overall=group(rows),by_condition={},by_condition_boundary={},by_family={})
    buckets=defaultdict(list)
    for r in rows: buckets[(r["noise"],r["level"])].append(r)
    for (typ,level),rr in buckets.items():
        result["by_condition"][f"{typ}:{level:g}"]=group(rr)
    buckets=defaultdict(list)
    for r in rows: buckets[(r["noise"],r["level"],r["onset_shift"],r["release_shift"])].append(r)
    for (typ,level,da,db),rr in buckets.items():
        result["by_condition_boundary"][f"{typ}:{level:g}:{da:+d}:{db:+d}"]=group(rr)
    for family in FAMILIES:
        result["by_family"][family]=group([r for r in rows if r["family"]==family])
    result["latency_ms"]={key:stats([r[key] for r in rows]) for key in ("decode_ms","refit_ms","baseline_pipeline_ms","candidate_pipeline_ms")}
    result["actual_class_deviation"]={key:dict(Counter(str(r[key]) for r in rows)) for key in ("selected_onset_minus_intended","selected_release_minus_intended")}
    result["clipped_values"] = sum(r["clipped_values"] for r in rows)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source/"src"))
    from trueskate_ai.model1.curved.contract import Gesture
    from trueskate_ai.model1.curved.dataset import targets
    from trueskate_ai.model1.curved.decoder import decode,evaluate_path
    from trueskate_ai.data.control_hitboxes import segment_is_safe
    previous=args.source/"research/evidence/EXEC-TRANSFER-20261013/decoder-followup/decoder_followup.py"
    spec=importlib.util.spec_from_file_location("frozen_research_refit",previous);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    fixtures,rejected=generate(Gesture,segment_is_safe)
    fixture_bytes=stable_bytes(fixtures)
    (args.out/"fixtures.json.gz").write_bytes(gzip.compress(fixture_bytes,mtime=0))
    save(args.out/"cohort-frozen.json",dict(seeds=SEEDS,count=len(fixtures),families=FAMILIES,
         fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),generation_rejections=rejected,
         plan_sha256=sha(args.out/"PLAN-FROZEN.md"),levels=LEVELS,corruptions=TYPES))
    start=time.monotonic(); rows=[]; repeats=[]
    for ci,case in enumerate(fixtures):
        meta=case["meta"]; label=targets(meta); times=np.asarray(meta["frame_times_s"])
        clean=np.asarray(label["xy"],float);a,b=int(label["onset"]),int(label["release"])
        fields=noise_fields(ci,a,b)
        conditions=[("clean",0.,np.zeros_like(clean))]+[(typ,level,fields[typ]*level) for typ in TYPES for level in LEVELS]
        for typ,level,noise in conditions:
            unclipped=clean+noise;clipped=int(np.count_nonzero((unclipped<0)|(unclipped>1)))
            xy=np.clip(unclipped,0,1).astype(np.float32)
            for da in (-1,0,1):
                for db in (-1,0,1):
                    onset=-.5*((np.arange(120)-(a+da))/.75)**2
                    release=-.5*((np.arange(120)-(b+db))/.75)**2
                    tick=time.perf_counter_ns();error=None
                    try: pred=decode(xy,onset,release,times)
                    except ValueError as exc: pred=None;error=str(exc)
                    decode_ms=(time.perf_counter_ns()-tick)/1e6
                    tick=time.perf_counter_ns()
                    candidate=(mod.refit_grid_coordinates(xy,times,pred["onset_frame"],pred["release_frame"],pred) if pred is not None else None)
                    refit_ms=(time.perf_counter_ns()-tick)/1e6
                    base,bguard=check_compile(pred,Gesture,segment_is_safe)
                    cand,cguard=check_compile(candidate,Gesture,segment_is_safe)
                    record=dict(id=case["id"],family=case["family"],noise=typ,level=level,
                                onset_shift=da,release_shift=db,clipped_values=clipped,
                                decode_ms=decode_ms,refit_ms=refit_ms,
                                baseline_pipeline_ms=decode_ms+bguard["latency_ms"],
                                candidate_pipeline_ms=decode_ms+refit_ms+cguard["latency_ms"],
                                selected_onset_minus_intended=None if pred is None else pred["onset_frame"]-a-da,
                                selected_release_minus_intended=None if pred is None else pred["release_frame"]-b-db,
                                decode_error=error)
                    for name,p,guard in (("baseline",base,bguard),("candidate",cand,cguard)):
                        score=None if p is None else evaluate_path(p,meta)
                        record[name]=dict(metric_pass=bool(score and score["passed"]),
                                          admissible_pass=bool(score and score["passed"] and guard["valid"]),
                                          score=score,guard=guard,segment_count=None if p is None else p["segment_count"],
                                          selected_boundaries=None if p is None else [p["onset_frame"],p["release_frame"]])
                    rows.append(record)
                    if ci%24==0 and typ=="clean" and da==db==0:
                        again=decode(xy.copy(),onset.copy(),release.copy(),times.copy())
                        again_fit=mod.refit_grid_coordinates(xy.copy(),times.copy(),again["onset_frame"],again["release_frame"],again)
                        repeats.append(dict(id=case["id"],baseline_equal=again==pred,candidate_equal=again_fit==candidate))
        if ci%12==0:print(f"fixtures {ci+1}/{len(fixtures)}, paired cases {len(rows)}, elapsed {time.monotonic()-start:.1f}s",flush=True)
    result=aggregate(rows)
    save(args.out/"summary.json",result)
    (args.out/"paired-results.json.gz").write_bytes(gzip.compress(stable_bytes(rows),mtime=0))
    save(args.out/"repeat-checks.json",repeats)
    files=["src/trueskate_ai/model1/curved/decoder.py","src/trueskate_ai/model1/curved/dataset.py", "src/trueskate_ai/model1/curved/contract.py", "src/trueskate_ai/sim/timed_waypoints.py", "src/trueskate_ai/data/control_hitboxes.py"]
    save(args.out/"provenance.json",dict(source_path=str(args.source),source_sha=subprocess.check_output(["git","-C",str(args.source),"rev-parse","HEAD"],text=True).strip(),
         source_hashes={f:sha(args.source/f) for f in files},refit_sha256=sha(previous),script_sha256=sha(__file__),
         python=sys.version,numpy=np.__version__,platform=platform.platform(),argv=sys.argv,
         elapsed_s=time.monotonic()-start,fixture_count=len(fixtures),paired_case_count=len(rows),
         notes=["Independent generated fixtures only; no corpus, trained model, simulator or physical device.",
                "Same corrupted float32 XY/logits to both variants; true commands enter only scoring.",
                "Unchanged scoring and compiler; invalid commands count as admissibility failures without fallback.",
                "Noise levels are hypothetical stress, not observed error estimates."] ))
    print(json.dumps(result["overall"],indent=2),flush=True)

if __name__=="__main__":main()
