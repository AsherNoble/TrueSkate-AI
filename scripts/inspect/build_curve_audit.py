"""Export command-blind native-frame annotation bundles and import their labels."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from trueskate_ai.research.curve_report import audit_selection
from trueskate_ai.research.curve_protocol import digest,save_new,PROTOCOL
from trueskate_ai.research.curve_measurement import frame_pts,read_native_frames
from trueskate_ai.research.review_provenance import bundle,file_sha256,curve_row_provenance,verify_bundle,legacy_provenance


def build(rows,out):
    proofs=[curve_row_provenance(row) for row in rows]
    if out.exists():raise ValueError('preserve previous audit exports')
    out.mkdir(parents=True)
    choice=audit_selection(rows)
    cases=[(i,'annotations') for i in sorted(set(choice['audit']+choice['review']))]
    cases += [(i,'repeat_annotations') for i in choice['repeat']]
    rng=np.random.default_rng(PROTOCOL['audit_seed']+1);rng.shuffle(cases)
    private={};clips=[]
    for order,(i,kind) in enumerate(cases):
        token=digest(dict(index=i,kind=kind,order=order,seed=PROTOCOL['audit_seed']))[:16]
        row=rows[i];video=Path(row['original_video']);pts=frame_pts(video)
        frames,times,indices=read_native_frames(video,pts,start=row['onset_s']-.4,end=row['onset_s']+1.7)
        directory=out/token;directory.mkdir();items=[]
        for frame,t,source_index in zip(frames,times,indices):
            name=f'{source_index:06d}.jpg'
            if not cv2.imwrite(str(directory/name),frame,[cv2.IMWRITE_JPEG_QUALITY,90]):raise ValueError('native frame write failed')
            items.append(dict(path=f'{token}/{name}',pts_s=t,source_frame=source_index,sha256=file_sha256(directory/name)))
        clips.append(dict(token=token,frames=items))
        private[token]=dict(row_index=i,kind=kind,original_video=str(video),source_frames=indices,pts_s=times)
    private_bundle=bundle('blind-curve-v2',clips,dict(rows_sha256=digest(rows),executions=proofs),private)
    save_new(out.parent/(out.name+'-private.json'),private_bundle)
    (out/'clips.js').write_text('const CLIPS='+json.dumps(clips)+';const BUNDLE_SHA256='+json.dumps(private_bundle['bundle_sha256'])+';const BUNDLE_SCHEMA="blind-curve-v2";\n')
    shutil.copyfile(Path(__file__).parent/'templates/curve_audit.html',out/'index.html')
    (out/'review_integrity.js').write_bytes((Path(__file__).with_name('templates')/'review_integrity.js').read_bytes())
    return out/'index.html'


def import_marks(private_path,export_path,out,*,allow_legacy=False,media_root=None):
    private=json.loads(private_path.read_text());export=json.loads(export_path.read_text())
    if private.get('version')==2:
        identity=verify_bundle(private,export,media_root)
        mapping=identity['mapping'];rows_sha256=identity['provenance']['rows_sha256']
        provenance='v2: execution and JPEG bytes bound'
    else:
        provenance=legacy_provenance(allow_legacy)
        if private['bundle_sha256']!=export.get('bundle_sha256'):raise ValueError('audit bundle identity mismatch')
        mapping=private['mapping'];rows_sha256=private['rows_sha256']
    result=dict(rows_sha256=rows_sha256,provenance=provenance,annotations={},repeat_annotations={})
    for token,marks in export['annotations'].items():
        if token not in mapping:raise ValueError('unknown blinded token')
        item=mapping[token]
        for point in marks.get('centreline',[])+[c['xy'] for c in marks.get('contacts',[])]+([marks['endpoint']] if marks.get('endpoint') else []):
            if len(point)!=2 or not all(np.isfinite(v) and 0<=v<=1 for v in point):raise ValueError('invalid normalized annotation')
        allowed=dict(zip(item['source_frames'],item['pts_s']))
        for contact in marks.get('contacts',[]):
            if allowed.get(contact['source_frame'])!=contact['pts_s']:raise ValueError('contact PTS differs from original source frame')
        for boundary in ('motion_start_s','motion_end_s'):
            if marks.get(boundary) is not None and marks[boundary] not in item['pts_s']:raise ValueError('motion boundary is not source PTS')
        result[item['kind']][str(item['row_index'])]=marks
    save_new(out,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--measurements',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--private-map',type=Path);p.add_argument('--export',type=Path);p.add_argument('--allow-legacy',action='store_true');p.add_argument('--media-root',type=Path);a=p.parse_args()
    if a.export:
        if not a.private_map:p.error('--export requires --private-map')
        import_marks(a.private_map,a.export,a.out,allow_legacy=a.allow_legacy,media_root=a.media_root)
    else:
        if not a.measurements:p.error('provide measurement files')
        rows=[r for path in a.measurements for r in json.loads(path.read_text())['rows']]
        print(build(rows,a.out))
if __name__=='__main__':main()
