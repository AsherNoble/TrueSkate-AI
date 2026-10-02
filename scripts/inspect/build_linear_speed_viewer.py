"""Add preserved linear-speed attempts to the existing native-frame video viewer."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from trueskate_ai.research.curve_measurement import frame_pts, _decode_source_frames
from trueskate_ai.research.curve_protocol import save_new
from trueskate_ai.research.linear_speed_probe import verify_manifest


def build(manifest_path, recordings, out):
    frozen = json.loads(manifest_path.read_text())
    verify_manifest(frozen)
    out.mkdir(parents=True, exist_ok=False)
    videos, clips = [], []
    for repeat, commands in enumerate(frozen['recordings'], 1):
        run = recordings/f'recording_{repeat}'
        video_index = None
        executed = {}
        if (run/'original.mov').exists():
            execution = json.loads((run/'execution.json').read_text())
            report = json.loads((run/'wda-timing.json').read_text())
            pts = frame_pts(run/'original.mov')
            _decode_source_frames(run/'original.mov', pts, keep=False)
            destination = out/f'recording_{repeat}.mp4'
            subprocess.run(['ffmpeg','-v','error','-copyts','-i',str(run/'original.mov'),'-map','0:v:0',
                            '-c','copy','-video_track_timescale','600','-movie_timescale','600',
                            '-movflags','+faststart',str(destination)], check=True)
            remux_pts = frame_pts(destination)
            if len(pts) != len(remux_pts) or any(abs(a-b)>1e-6 for a,b in zip(pts,remux_pts)):
                raise ValueError('viewer remux changed native source PTS')
            video_index = len(videos)
            videos.append(dict(src=destination.name, repeat=repeat, pts=pts.tolist(),
                               error=execution['error'], decoded_frame_count=len(pts),
                               source_pts_preserved=True,
                               original_sha256=hashlib.sha256((run/'original.mov').read_bytes()).hexdigest()))
            admission_path = run/'admission.json'
            admission = json.loads(admission_path.read_text()) if admission_path.exists() else {}
            fit = admission.get('fit') if admission.get('accepted') else None
            for i, event in enumerate(execution['events']):
                spec = event['spec']
                if spec['kind'] != 'diagnostic':
                    continue
                if i < len(report['records']):
                    stamp = report['records'][i]['submitted_to_ios']
                    onset = (fit['intercept_s']+fit['rate']*stamp['monotonic_s'] if fit
                             else stamp['epoch_s']-execution['video']['started_at_epoch_s'])
                    executed[spec['command_id']] = dict(onset=onset, start=max(float(pts[0]),onset-.7),
                        end=min(float(pts[-1]),onset+spec['duration_ms']/1000+1.5),
                        timing='accepted start/end calibration' if fit else 'approximate WDA epoch; calibration not admitted')
        for spec in commands:
            if spec['kind'] == 'diagnostic':
                info = executed.get(spec['command_id'])
                clips.append(dict(id=spec['command_id'], duration_ms=spec['duration_ms'], repeat=repeat,
                                  title=f"{spec['duration_ms']} ms · repeat {repeat}", video=video_index,
                                  executed=info is not None, **(info or {})))
    data = dict(manifest_sha256=frozen['sha256'], videos=videos, clips=clips)
    save_new(out/'provenance.json', data)
    (out/'data.js').write_text('const DATA='+json.dumps(data).replace('<','\\u003c')+';\n')
    template = Path(__file__).with_name('templates')/'linear_speed.html'
    (out/'index.html').write_text(template.read_text())
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','recordings','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    data = build(args.manifest, args.recordings, args.out)
    print(f"{sum(c['executed'] for c in data['clips'])}/16 executed clips; {len(data['videos'])} preserved recordings")
