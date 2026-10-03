"""Import explicit human assessments; unreviewed defaults never count as ratings."""
import argparse,json
from collections import Counter
from trueskate_ai.research.curved_audit import save_new
RATINGS=('good','minor','major','unclear')

def report(mapping,export):
    if export.get('schema')!='blind-curved-execution-v1' or export.get('bundle_sha256')!=mapping['bundle_sha256']:raise ValueError('wrong assessment dataset')
    items={x['id']:x for x in mapping['items']};marks=export['assessments']
    if set(marks)-set(items):raise ValueError('unknown clip IDs')
    rows=[]
    for token,mark in marks.items():
        if mark.get('rating') not in RATINGS or not isinstance(mark.get('comments',''),str):raise ValueError('invalid rating/comments')
        rows.append(dict(**items[token],assessment=mark))
    def counts(rows):return dict(n=len(rows),**{r:sum(x['assessment']['rating']==r for x in rows) for r in RATINGS})
    result=dict(identity=mapping['identity'],expected=len(items),reviewed=len(rows),missing=len(items)-len(rows),overall=counts(rows),
                source='human blinded assessments only',device_park_interpretation='device and park are confounded',
                comments=[dict(id=r['id'],comment=r['assessment']['comments']) for r in rows if r['assessment'].get('comments')])
    for field in ('family','duration_ms','waypoint_count'):
        result['by_'+field]={str(v):counts([r for r in rows if r[field]==v]) for v in sorted({r[field] for r in items.values()})}
    result['by_device_park']={f'{d} / {p}':counts([r for r in rows if (r['device'],r['park'])==(d,p)]) for d,p in sorted({(r['device'],r['park']) for r in items.values()})}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mapping','assessments','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    with open(a.mapping) as f:m=json.load(f)
    with open(a.assessments) as f:e=json.load(f)
    save_new(a.out,report(m,e))
