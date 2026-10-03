"""Native PTS decoding shared with the preserved 135-clip audit."""
import json
import subprocess
import numpy as np

def frame_pts(video):
    result=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
                                    '-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)])
    frames=json.loads(result)['frames']
    pts=np.asarray([float(f['best_effort_timestamp_time']) for f in frames])
    if len(pts)<2 or not np.isfinite(pts).all() or np.any(np.diff(pts)<=0):
        raise ValueError('missing or nonchronological original frame PTS')
    if abs(float(np.median(np.diff(pts)))-1/30)>.005:
        raise ValueError('recording is not native ~30fps')
    return pts


def read_native_frames(video,pts,*,start=-np.inf,end=np.inf):
    # OpenCV emitted 1773 frames for an XCTest file with 1772 source PTS.
    # Use the same FFmpeg decoding/edit-list semantics as ffprobe and the
    # canonical exact-source-frame extractor. Never truncate to make counts fit.
    return _decode_source_frames(video,pts,start=start,end=end)


def _decode_source_frames(video,pts,*,start=-np.inf,end=np.inf,keep=True,inspect=None):
    metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                        '-show_entries','stream=width,height','-of','json',str(video)]))['streams'][0]
    width,height=metadata['width'],metadata['height']
    selected=[i for i,t in enumerate(pts) if start<=t<=end]
    if not selected:return [],[],[]
    args=['ffmpeg','-v','error','-i',str(video),'-map','0:v:0']
    if selected!=list(range(len(pts))):
        args+=['-vf',f'select=between(n\\,{selected[0]}\\,{selected[-1]})']
    args+=['-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
    # Bound stderr separately so a full diagnostic pipe cannot deadlock.
    import tempfile
    frames=[];size=width*height*3;count=0
    with tempfile.TemporaryFile() as stderr:
        process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=stderr)
        try:
            while True:
                raw=process.stdout.read(size)
                if not raw:break
                if len(raw)!=size:raise ValueError('incomplete decoded source frame')
                if count>=len(selected):raise ValueError('decoded frame count exceeds original PTS selection')
                if keep or inspect is not None:
                    image=np.frombuffer(raw,np.uint8).reshape(height,width,3)
                    if keep:frames.append(image.copy())
                    if inspect is not None:inspect(image)
                count+=1
            status=process.wait()
            if status:
                stderr.seek(0);raise ValueError('FFmpeg source decode failed: '+stderr.read().decode(errors='replace'))
            if count!=len(selected):raise ValueError('decoded frame count differs from original PTS selection')
        finally:
            process.stdout.close()
            if process.poll() is None:process.kill();process.wait()
    return frames,[float(pts[i]) for i in selected],selected


