"""Exercise scientific clock/identity/measurement failure modes with synthetic logs."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

from analyze import analyze, measure, read_batches
from freeze import make_plan


def events(path,*,lag=0.,offset=0.,age=.002):
    points=np.asarray(path["gesture"]["command_points"])*[414,896]
    times=np.asarray(path["gesture"]["times_ms"])/1000
    end=times[-1]+lag
    out=[]
    for i,t in enumerate(np.linspace(0,end,49)):
        nominal=t*times[-1]/end
        xy=np.array([np.interp(nominal,times,points[:,k]) for k in range(2)])+[offset,0]
        out.append(dict(kind="T",phase="B" if i==0 else "E" if i==48 else "M",
            seq=i+1,cb=i+1,tid=1,ts=100+t,now=100+t+age,uptime=100+t+age,
            x=float(xy[0]),y=float(xy[1]),target=0))
    return out


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
plan=make_plan()
assert len(plan["helper_requests"])==80
assert len({r["request_id"] for r in plan["helper_requests"]})==80
assert plan["budget"]["total_touch_requests"]==112
assert len(plan["game_recordings"])==2
assert sum(c["kind"]=="scientific" for r in plan["game_recordings"] for c in r["commands"])==20
assert [r["commands"][0]["slot_s"] for r in plan["game_recordings"]]==[1.5,1.5]
path=plan["paths"][0]
raw=events(path)
ideal=measure(raw,path)
assert abs(ideal["delivered_duration_ms"]-400)<1e-8
assert ideal["sampled_command_deviation_pt"]["max"]<1e-8
assert ideal["callback_minus_touch_timestamp_ms"]["median"]>1.99
late=measure(events(path,age=1.2),path)
assert abs(late["delivered_duration_ms"]-ideal["delivered_duration_ms"])<1e-8
assert late["callback_minus_touch_timestamp_ms"]["median"]>1199
# A hidden coalesced sample contributes; a duplicate does not inflate count.
co=copy.deepcopy(raw[12]);co.update(kind="C",ts=100.102,x=180,y=480)
with_co=measure(raw+[co,copy.deepcopy(raw[12])],path)
assert with_co["unique_touch_samples"]==50
cancelled=copy.deepcopy(raw);cancelled[-1]["phase"]="C"
try:measure(cancelled,path)
except ValueError:pass
else:raise AssertionError("cancelled contact accepted")
conflict=copy.deepcopy(raw[12]);conflict.update(kind="C",x=conflict["x"]+5)
assert measure(raw+[conflict],path)["same_timestamp_extra_positions"]==1
header=dict(kind="launch",schema="physical-touch-v1",w=414,h=896)
requests=[r for r in plan["helper_requests"] if r["path_id"]==path["path_id"]][:2]
receipts=[dict(request_id=r["request_id"],log_batch=i+1) for i,r in enumerate(requests)]
batches={1:dict(marker=dict(reason="release",overflow=False),events=raw),
         2:dict(marker=dict(reason="release",overflow=False),events=events(path,lag=.02,offset=2))}
result=analyze(plan,receipts,header,batches)
group=result["groups"][path["path_id"]]
assert abs(group["delivered_duration_ms"]["sd_population"]-10)<1e-8
assert abs(group["p95_absolute_release_deviation_from_median_ms"]-10)<1e-8
assert group["repeatability_interpolated_common_time"]["shared_end_ms"]==400
assert not result["helper_gate"]["passed"]  # partial data never opens the gate
full_receipts=[];full_batches={}
for i,r in enumerate(plan["helper_requests"],1):
    wanted=next(p for p in plan["paths"] if p["path_id"]==r["path_id"])
    full_receipts.append(dict(request_id=r["request_id"],log_batch=i))
    full_batches[i]=dict(marker=dict(reason="release",overflow=False),events=events(wanted))
assert analyze(plan,full_receipts,header,full_batches)["helper_gate"]["passed"]
# Lost data fails closed, and batch/request reuse is rejected.
bad=copy.deepcopy(batches);bad[1]["marker"]["overflow"]=True
assert not analyze(plan,receipts,header,bad)["requests"][0]["accepted"]
try:analyze(plan,receipts+[receipts[0]],header,batches)
except ValueError:pass
else:raise AssertionError("reused request accepted")
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/"fixture.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in
        [header,dict(kind="batch",batch=1,reason="release",overflow=False),*raw])+"\n")
    h,b=read_batches(p)
    assert len(b[1]["events"])==49
report=dict(synthetic_only=True,passed=True,
    checks=["112-request schedule and unique80 helper IDs", "current guard-aware slots",
        "exact linear geometry/duration", "callback stall does not alter touch duration",
        "coalesced sample inclusion and duplicate removal", "cancel rejection",
        "same-timestamp spatial conflict detected", "known20ms duration spread",
        "common-time comparison has no duration normalization", "overflow rejection",
        "request/batch identity protection", "buffered JSONL parsing",
        "partial cohort closes gate; full80 ideal synthetic cohort opens helper-only gate"],
    no_physical_requests=True,no_physical_repeatability_claim=True)
args.out.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
