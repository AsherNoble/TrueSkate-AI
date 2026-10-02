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


def build(rows,out):
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
            name=f'{source_index:06d}.png';cv2.imwrite(str(directory/name),frame)
            items.append(dict(path=f'{token}/{name}',pts_s=t,source_frame=source_index))
        clips.append(dict(token=token,frames=items))
        private[token]=dict(row_index=i,kind=kind,original_video=str(video),source_frames=indices,pts_s=times)
    bundle_hash=digest(dict(rows_sha256=digest(rows),private=private))
    save_new(out.parent/(out.name+'-private.json'),dict(bundle_sha256=bundle_hash,rows_sha256=digest(rows),mapping=private,audit_selection=choice))
    (out/'clips.js').write_text('const CLIPS='+json.dumps(clips)+';const BUNDLE_SHA256='+json.dumps(bundle_hash)+';\n')
    shutil.copyfile(Path(__file__).parent/'templates/curve_audit.html',out/'index.html')
    return out/'index.html'


def import_marks(private_path,export_path,out):
    private=json.loads(private_path.read_text());export=json.loads(export_path.read_text())
    if private['bundle_sha256']!=export.get('bundle_sha256'):raise ValueError('audit bundle identity mismatch')
    result=dict(rows_sha256=private['rows_sha256'],annotations={},repeat_annotations={})
    for token,marks in export['annotations'].items():
        if token not in private['mapping']:raise ValueError('unknown blinded token')
        mapping=private['mapping'][token]
        for point in marks.get('centreline',[])+[c['xy'] for c in marks.get('contacts',[])]+([marks['endpoint']] if marks.get('endpoint') else []):
            if len(point)!=2 or not all(np.isfinite(v) and 0<=v<=1 for v in point):raise ValueError('invalid normalized annotation')
        allowed=dict(zip(mapping['source_frames'],mapping['pts_s']))
        for contact in marks.get('contacts',[]):
            if allowed.get(contact['source_frame'])!=contact['pts_s']:raise ValueError('contact PTS differs from original source frame')
        for boundary in ('motion_start_s','motion_end_s'):
            if marks.get(boundary) is not None and marks[boundary] not in mapping['pts_s']:raise ValueError('motion boundary is not source PTS')
        result[mapping['kind']][str(mapping['row_index'])]=marks
    save_new(out,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--measurements',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--private-map',type=Path);p.add_argument('--export',type=Path);a=p.parse_args()
    if a.export:
        if not a.private_map:p.error('--export requires --private-map')
        import_marks(a.private_map,a.export,a.out)
    else:
        if not a.measurements:p.error('provide measurement files')
        rows=[r for path in a.measurements for r in json.loads(path.read_text())['rows']]
        print(build(rows,a.out))
if __name__=='__main__':main()
