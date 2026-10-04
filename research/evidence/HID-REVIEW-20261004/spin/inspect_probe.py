import cv2
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/claude-review-resume-20261004/spin')
p=ROOT/sys.argv[1]
r=json.loads((p/'run.json').read_text())
probe=json.loads((p/'ffprobe.json').read_text())
pts=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']])
cap=cv2.VideoCapture(str(p/'recording.mov'),cv2.CAP_FFMPEG)
overview=np.linspace(0,len(pts)-1,16,dtype=int)
sets={'overview':overview}
if 'pointer' in r:
    target=r['pointer']['go_epoch']-r['recording']['started_at_epoch_s']+r['schedule_info']['first_press_slot']*.015
    center=int(np.argmin(abs(pts-target)))
    # Wide enough to absorb the host/video clock offset; label source PTS.
    sets['pointer-burst']=list(range(max(0,center-8),min(len(pts),center+16)))
needed=set(int(i) for s in sets.values() for i in s)
images={};count=0
while True:
    ok,bgr=cap.read()
    if not ok:break
    if count in needed:
        images[count]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((207,448))
    count+=1
cap.release()
assert count==len(pts),(count,len(pts))
for name,indices in sets.items():
    rows=(len(indices)+3)//4
    sheet=Image.new('RGB',(828,rows*472),(30,30,30));draw=ImageDraw.Draw(sheet)
    for k,i in enumerate(indices):
        x=k%4*207;y=k//4*472
        sheet.paste(images[int(i)],(x,y+24));draw.text((x+3,y+4),f'f {i} PTS {pts[int(i)]:.3f}s',fill='white')
    sheet.save(p/f'{name}.png')
dt=np.diff(pts)
summary={'decoded_frames':count,'source_pts_first':float(pts[0]),'source_pts_last':float(pts[-1]),
         'source_frame_step_median':float(np.median(dt)),'source_step_min':float(dt.min()),
         'source_step_max':float(dt.max()),'source_gaps_gt_1_5_frame':int((dt>.025).sum()),
         'completed':r.get('completed'),'release':r.get('release'),'pointer_late_us':[r.get('pointer',{}).get('min_late_us'),r.get('pointer',{}).get('max_late_us')],
         'hold':r.get('hold'),'streams':probe['streams']}
(p/'inspection-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
