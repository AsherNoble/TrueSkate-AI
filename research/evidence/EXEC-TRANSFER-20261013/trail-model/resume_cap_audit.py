"""Summarise held-cap calibration; record visually reviewed midpoint labels."""
import json
import numpy as np
from common import OUT, manifest
from resume_summary import stats


def radius(a,level):
    a=np.asarray(a);r=np.arange(0,101,2.)
    ix=np.where(a<level)[0]
    if not len(ix):return None
    i=int(ix[0])
    return 0. if i==0 else float(np.interp(level,[a[i],a[i-1]],[r[i],r[i-1]]))


holds=json.loads((OUT/'data/holds.json').read_text());rows=[]
for sid,lo,hi in [(55,250,750),(99,60,130)]:
    h=next(x for x in holds if x['sid']==sid and x['kind']=='first')
    f=[x for x in h['frames'] if lo<=x['t']<=hi]
    row=dict(sid=sid,t_range_ms=[lo,hi],n_frames=len(f))
    for name in ['rad','side','fwd']:
        row[name]={key:stats([radius(x[name],level(x)) for x in f]) for key,level in [
            ('r_half_px',lambda x:np.asarray(x[name])[0]/2),('r25_px',lambda x:25),('r10_px',lambda x:10)]}
    rows.append(row)
(OUT/'data/resume-cap.json').write_text(json.dumps(rows,indent=2))
board={62,73,77,122,125,128,134,137,155}
scenery={52,54,56,57,58,63,64,65,72,78,85,119}
possible={60,89,95,111,120,136,140,157}
audit=dict(method='Manual review of every midpoint crop in img/resume-midpoint-{1..5}.jpg. Deck intersection includes silhouette edge; scenery means a salient hard edge/HUD within about30native px of command centre. These are static contact-neighbourhood risks, not measured invisible-head events or per-clip any-frame prevalence.',
           total=100,deck_intersections=9,definite_scenery_intersections=12,possible_scenery_intersections=8,
           per_clip=[dict(sid=sid,deck_intersection=sid in board,scenery='definite' if sid in scenery else 'possible' if sid in possible else 'none') for sid in sorted(manifest())])
(OUT/'data/resume-manual-midpoint-audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(rows,indent=2));print({k:v for k,v in audit.items() if k!='per_clip'})
