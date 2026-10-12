"""Summarize buffered UIKit batches, keeping delivery and callback clocks separate.

The receipts must explicitly bind each frozen request to a unique logger batch.
No video pixels or glow enter input-delivery measurements. Linear resampling is
an analysis assumption; native samples and same-timestamp conflicts are retained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def summary(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return dict(n=0)
    return dict(n=len(a),median=float(np.median(a)),mean=float(np.mean(a)),
        sd_population=float(np.std(a)),min=float(np.min(a)),max=float(np.max(a)),
        p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)))


def read_batches(log):
    header=None; batches={}; current=None
    for n,line in enumerate(Path(log).read_text().splitlines(),1):
        row=json.loads(line)
        if row.get("kind")=="launch":
            if header is not None: raise ValueError("one app launch per input log required")
            header=row
        elif row.get("kind")=="batch":
            current=row["batch"]
            if current in batches: raise ValueError("duplicate batch")
            batches[current]=dict(marker=row,events=[])
        else:
            if current is None: raise ValueError(f"sample before batch at line {n}")
            batches[current]["events"].append(row)
    if not header or header.get("schema")!="physical-touch-v1":
        raise ValueError("wrong/missing logger schema")
    if [header.get("w"),header.get("h")] != [414,896]:
        raise ValueError("wrong device geometry")
    return header,batches


def measure(events,path):
    native=[e for e in events if e["kind"]=="T"]
    begins=[e for e in native if e["phase"]=="B"]
    ends=[e for e in native if e["phase"]=="E"]
    if len(begins)!=1 or len(ends)!=1 or any(e["phase"]=="C" for e in native):
        raise ValueError("requires one complete, uncancelled contact")
    begin,end=begins[0],ends[0]
    if end["ts"] < begin["ts"]: raise ValueError("negative duration")
    if len({e["tid"] for e in native})!=1: raise ValueError("multiple contacts")
    raw=[e for e in events if e["kind"] in ("T","C")]
    for e in raw:
        if not np.isfinite([e[k] for k in ("ts","now","x","y")]).all():
            raise ValueError("non-finite sample")
        if not begin["ts"]-1e-9 <= e["ts"] <= end["ts"]+1e-9:
            raise ValueError("sample outside B/E timestamp span")
    # T and coalesced arrays may repeat a sample. Do not inflate its weight.
    unique={(float(e["ts"]),float(e["x"]),float(e["y"])):e for e in raw}
    rows=sorted(unique)
    ts=np.asarray([r[0]-begin["ts"] for r in rows])
    xy=np.asarray([r[1:] for r in rows])
    conflicts=len(ts)-len(set(ts.tolist()))
    command=np.asarray(path["gesture"]["command_points"])*[414,896]
    times=np.asarray(path["gesture"]["times_ms"])/1000
    wanted=np.stack([np.interp(ts,times,command[:,k]) for k in range(2)],axis=1)
    deviations=np.linalg.norm(xy-wanted,axis=1)
    callback_age=[1000*(e["now"]-e["ts"]) for e in native]
    waypoints=[]
    for i,(point,time) in enumerate(zip(command,times)):
        distances=np.linalg.norm(xy-point,axis=1)
        near=np.flatnonzero(distances<=.5)
        closest=int(np.argmin(distances))
        waypoints.append(dict(index=i,command_time_ms=float(time*1000),
            nearest_observed_time_ms=float(ts[closest]*1000),
            nearest_distance_pt=float(distances[closest]),
            first_within_half_point_ms=float(ts[near[0]]*1000) if len(near) else None,
            last_within_half_point_ms=float(ts[near[-1]]*1000) if len(near) else None,
            interpretation="sampled occupancy only; holds/repeated coordinates are ambiguous"))
    result=dict(delivered_duration_ms=1000*(end["ts"]-begin["ts"]),
        requested_duration_ms=int(times[-1]*1000),
        release_error_ms=1000*(end["ts"]-begin["ts"]-times[-1]),
        callback_span_ms=1000*(end["now"]-begin["now"]),
        callback_minus_touch_timestamp_ms=summary(callback_age),
        start_error_pt=float(np.linalg.norm(np.array([begin["x"],begin["y"]])-command[0])),
        end_error_pt=float(np.linalg.norm(np.array([end["x"],end["y"]])-command[-1])),
        callback_count=len(native),unique_touch_samples=len(rows),coalesced_rows=len(raw)-len(native),
        same_timestamp_extra_positions=conflicts,
        sampled_command_deviation_pt=summary(deviations),waypoints=waypoints,
        touch_times_from_B_ms=(ts*1000).tolist(),positions_pt=xy.tolist(),
        display_callback_count=sum(e["kind"]=="D" for e in events),
        display_interpretation="CADisplayLink scheduling only; no measured photon onset")
    return result


def analyze(plan,receipts,header,batches):
    expected={r["request_id"]:r for r in plan["helper_requests"]}
    paths={p["path_id"]:p for p in plan["paths"]}
    output=[]; used_batches=set(); used_requests=set()
    for receipt in receipts:
        rid=receipt["request_id"]
        if rid not in expected or rid in used_requests: raise ValueError("unknown/repeated request")
        used_requests.add(rid)
        batch=receipt["log_batch"]
        if batch not in batches or batch in used_batches: raise ValueError("missing/reused logger batch")
        used_batches.add(batch)
        marker=batches[batch]["marker"]
        base=dict(**expected[rid],log_batch=batch,receipt=receipt)
        if marker.get("overflow") or marker.get("reason")!="release":
            output.append(dict(**base,accepted=False,error="overflow or incomplete contact")); continue
        try: value=measure(batches[batch]["events"],paths[expected[rid]["path_id"]])
        except ValueError as exc:
            output.append(dict(**base,accepted=False,error=str(exc))); continue
        output.append(dict(**base,accepted=True,**value))
    groups={}
    for name in paths:
        runs=[r for r in output if r["path_id"]==name and r["accepted"]]
        group=dict(planned_repeats=10,attempted=sum(r["path_id"]==name for r in output),
            admitted=len(runs),delivered_duration_ms=summary([r["delivered_duration_ms"] for r in runs]),
            release_error_ms=summary([r["release_error_ms"] for r in runs]),
            start_error_pt=summary([r["start_error_pt"] for r in runs]),
            end_error_pt=summary([r["end_error_pt"] for r in runs]))
        eligible=[r for r in runs if not r["same_timestamp_extra_positions"]]
        if len(eligible)>=2:
            stop=min(r["touch_times_from_B_ms"][-1] for r in eligible)
            grid=np.arange(0,stop+1e-7,1.)
            profiles=[]
            for run in eligible:
                points=np.asarray(run["positions_pt"])
                profiles.append(np.stack([np.interp(grid,run["touch_times_from_B_ms"],points[:,k])
                                         for k in range(2)],axis=1))
            profiles=np.asarray(profiles)
            sd=np.std(profiles,axis=0)
            radius=np.sqrt(np.sum(sd**2,axis=1))
            median_profile=np.median(profiles,axis=0)
            repeat_deviation=np.linalg.norm((profiles-median_profile)/[414,896],axis=2)
            group["repeatability_interpolated_common_time"]=dict(n=len(eligible),
                grid_step_ms=1,shared_end_ms=float(grid[-1]),
                rms_axis_sd_pt=float(np.sqrt(np.mean(sd**2))),
                max_spatial_sd_pt=float(np.max(radius)),
                p95_repeat_deviation_normalized=float(np.quantile(repeat_deviation,.95)),
                worst_repeat_deviation_normalized=float(np.max(repeat_deviation)),
                warning="piecewise-linear interpolation of observed samples; no duration normalization")
        if runs:
            release=np.asarray([r["delivered_duration_ms"] for r in runs])
            group["p95_absolute_release_deviation_from_median_ms"]=float(
                np.quantile(np.abs(release-np.median(release)),.95))
        groups[name]=group
    failures=[]
    if set(batches)-used_batches:failures.append("unbound logger batches require reconciliation")
    criteria=plan["pilot_gate"]
    for name,path in paths.items():
        group=groups[name]
        runs=[r for r in output if r["path_id"]==name and r["accepted"]]
        if group["attempted"]!=10 or group["admitted"]!=10:
            failures.append(name+": requires10/10 complete attempts")
        comparison=group.get("repeatability_interpolated_common_time",{})
        if comparison.get("n")!=10:
            failures.append(name+": complete conflict-free common-time profiles unavailable")
        else:
            for actual,limit in (("p95_repeat_deviation_normalized","p95_repeat_position_deviation_normalized_max"),
                                 ("worst_repeat_deviation_normalized","worst_repeat_position_deviation_normalized_max")):
                if comparison[actual]>criteria[limit]:failures.append(name+": "+actual+" exceeds frozen gate")
        if group.get("p95_absolute_release_deviation_from_median_ms",float("inf"))>criteria["p95_absolute_release_deviation_from_path_median_ms_max"]:
            failures.append(name+": release variability exceeds frozen gate")
        points=path["gesture"]["command_points"]
        holds=[i for i,(a,b) in enumerate(zip(points,points[1:])) if a==b]
        for run in runs:
            if any(w["nearest_distance_pt"]>criteria["required_corner_endpoint_nearest_sample_distance_pt_max"]
                   for w in run["waypoints"]):
                failures.append(run["request_id"]+": waypoint not observed within frozen distance")
            for i in holds:
                w=run["waypoints"][i]
                first,last=w["first_within_half_point_ms"],w["last_within_half_point_ms"]
                if first is None or last-first < criteria["required_hold_observed_distinct_timestamp_span_ms_min"]:
                    failures.append(run["request_id"]+": hold occupancy unmeasured/too short")
    helper_gate=dict(passed=not failures,failures=failures,
        interpretation="frozen pragmatic diagnostic thresholds; n10/path does not bound rare failure rates",
        learning_pilot_gate="closed until helper AND separate True Skate observation/context gate pass")
    return dict(schema="physical-transfer-analysis-v1",logger=header,requests=output,groups=groups,
        attempted=len(receipts),planned=80,unbound_batches=sorted(set(batches)-used_batches),
        missing_requests=sorted(set(expected)-used_requests),
        helper_gate=helper_gate,
        limitations=["touch timestamp and callback clocks reported separately",
            "callback age assumes raw clocks are comparable; it is not panel latency",
            "waypoint occupancy is sample-limited and ambiguous during holds",
            "interpolated repeatability is not a guarantee between observed samples",
            "no True Skate delivery timestamps can be inferred from this helper"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest","log","receipts","out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.manifest.read_text())
    canonical={k:v for k,v in plan.items() if k!="sha256"}
    digest=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()
    if plan["sha256"]!=digest: raise ValueError("changed manifest")
    header,batches=read_batches(args.log)
    result=analyze(plan,json.loads(args.receipts.read_text()),header,batches)
    result["input_hashes"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (args.manifest,args.log,args.receipts)}
    with args.out.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write("\n")
