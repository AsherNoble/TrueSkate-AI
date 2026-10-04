"""Import explicit human assessments; unreviewed defaults never count as ratings."""
import argparse,json,hashlib
from pathlib import Path
from trueskate_ai.research.curved_audit import save_new, verify_manifest, digest
from trueskate_ai.research.review_provenance import curved_execution, legacy_provenance, verify_bundle
from trueskate_ai.sim.timed_waypoints import TimedWaypoints
RATINGS=('good','minor','major','unclear')

def _verify_provenance(identity):
    provenance=identity['provenance'];root=provenance['frozen_manifest'];verify_manifest(root)
    mapping=identity['mapping'];proofs=provenance['recordings']
    if mapping['identity']!=root['identity'] or len(proofs)!=10 or {p['segment'] for p in proofs}!=set(range(1,11)):
        raise ValueError('incomplete curved execution provenance')
    paths={p['path_id']:p for p in root['paths']};expected={}
    for proof in proofs:
        n=proof['segment'];frozen=proof['frozen_manifest']
        curved_execution(frozen,n,proof['planned'],proof['execution'],proof['wda_timing'])
        actual=frozen['recordings'][n-1];original=root['recordings'][n-1]
        samples=lambda r:[{k:v for k,v in c.items() if k!='slot_s'} for c in r['commands'] if c['kind']=='sample']
        if frozen['paths']!=root['paths'] or any(actual[k]!=original[k] for k in ('device','park')) or digest(samples(actual))!=digest(samples(original)):
            raise ValueError('replacement sample conditions differ')
        cal=proof['calibration']
        if cal.get('accepted') is not True:raise ValueError('unaccepted recording calibration')
        fit=cal['fit']
        for i,command in enumerate(actual['commands']):
            if command['kind']!='sample':continue
            token=hashlib.sha256(f'{root["identity"]}:{n}:{command["path_id"]}:blind'.encode()).hexdigest()[:20]
            onset=fit['intercept_s']+fit['rate']*proof['wda_timing']['records'][i]['submitted_to_ios']['monotonic_s']
            expected[token]=dict(segment=n,source_manifest=frozen['identity'],source_manifest_sha256=frozen['sha256'],
                device=actual['device'],park=actual['park'],**paths[command['path_id']],onset_s=onset)
    items=mapping['items'];clips={c['id']:c for c in identity['clips']}
    if len(items)!=100 or len(clips)!=100 or {x['id'] for x in items}!=set(expected) or set(clips)!=set(expected):
        raise ValueError('curved review clip coverage differs from manifest')
    for item in items:
        if any(digest(item.get(k))!=digest(v) for k,v in expected[item['id']].items()):
            raise ValueError('curved review condition/source differs from executed command')
        frames=clips[item['id']]['frames'];times=item['source_pts_s'];indices=item['source_frames']
        if len(frames)!=len(times) or len(indices)!=len(times) or any(b<=a for a,b in zip(indices,indices[1:])):
            raise ValueError('curved review source frame mismatch')
        rate=next(p['calibration']['fit']['rate'] for p in proofs if p['segment']==item['segment'])
        timed=TimedWaypoints(tuple(map(tuple,item['points'])),tuple(item['times_ms']))
        for frame,t in zip(frames,times):
            if frame['time_s']!=t-times[0] or digest(frame['target'])!=digest(timed.position((t-item['onset_s'])/rate*1000)):
                raise ValueError('curved review target/source timing mismatch')
    return mapping


def report(mapping,export,*,allow_legacy=False,media_root=None):
    if export.get('schema')=='blind-curved-execution-v2':
        identity=verify_bundle(mapping,export,media_root)
        mapping=_verify_provenance(identity)
        provenance='v2: frozen ordered commands, successful execution receipts and displayed JPEG bytes bound'
    else:
        provenance=legacy_provenance(allow_legacy)
        if export.get('schema')!='blind-curved-execution-v1' or export.get('bundle_sha256')!=mapping['bundle_sha256']:raise ValueError('wrong assessment dataset')
    items={x['id']:x for x in mapping['items']};marks=export['assessments']
    if set(marks)-set(items):raise ValueError('unknown clip IDs')
    rows=[]
    for token,mark in marks.items():
        if mark.get('rating') not in RATINGS or not isinstance(mark.get('comments',''),str):raise ValueError('invalid rating/comments')
        rows.append(dict(**items[token],assessment=mark))
    def counts(rows):return dict(n=len(rows),**{r:sum(x['assessment']['rating']==r for x in rows) for r in RATINGS})
    result=dict(identity=mapping['identity'],provenance=provenance,expected=len(items),reviewed=len(rows),missing=len(items)-len(rows),overall=counts(rows),
                source='human blinded assessments only',device_park_interpretation='device and park are confounded',
                comments=[dict(id=r['id'],comment=r['assessment']['comments']) for r in rows if r['assessment'].get('comments')])
    for field in ('family','duration_ms','waypoint_count'):
        result['by_'+field]={str(v):counts([r for r in rows if r[field]==v]) for v in sorted({r[field] for r in items.values()})}
    result['by_device_park']={f'{d} / {p}':counts([r for r in rows if (r['device'],r['park'])==(d,p)]) for d,p in sorted({(r['device'],r['park']) for r in items.values()})}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mapping','assessments','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--media-root',type=Path)
    p.add_argument('--allow-legacy',action='store_true',help='read historical v1 with explicitly weaker provenance')
    a=p.parse_args()
    with open(a.mapping) as f:m=json.load(f)
    with open(a.assessments) as f:e=json.load(f)
    save_new(a.out,report(m,e,allow_legacy=a.allow_legacy,media_root=a.media_root))
