"""Bounded offline FFmpeg benchmarks on existing experimental recordings."""
from pathlib import Path
import json
import platform
import statistics
import subprocess
import time

SOURCE = Path('/Users/ashernoble/.claude/jobs/4e7efedf/tmp/agent-fps/rec')
DEST = Path('/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/claude-review-resume-20261004/fps')
FILES = {
    '30': SOURCE/'demo-replay-20261004-pointer-v3/01_pointer_immediate_r1.mov',
    '60': SOURCE/'demo-replay-20261004-variance/v3_01/01_pointer_immediate_r1.mov',
}
MODES = ['decode', 'scale512', 'encode512', 'encode64']
rows = []
for mode in MODES:
    for rate, path in FILES.items():
        results = []
        for repeat in range(2):
            command=['ffmpeg','-nostdin','-hide_banner','-v','error','-y',
                     '-threads','2','-i',str(path),'-an']
            output = None
            if mode == 'decode':
                command += ['-f','null','-']
            elif mode == 'scale512':
                command += ['-vf','scale=512:-2','-fps_mode','passthrough','-f','null','-']
            else:
                width = '512' if mode == 'encode512' else '64'
                output=DEST/f'benchmark-{rate}-{mode}.mp4'
                command += ['-vf',f'scale={width}:-2','-fps_mode','passthrough',
                            '-c:v','libx264','-threads','2','-preset','medium',
                            '-crf','20','-pix_fmt','yuv420p',str(output)]
            start=time.perf_counter()
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            results.append(time.perf_counter()-start)
        row=dict(rate=rate,mode=mode,source=str(path),elapsed_s=results,
                 median_s=statistics.median(results),bytes=output.stat().st_size if output else None)
        rows.append(row)
        print(json.dumps(row),flush=True)
(DEST/'decode-benchmark.json').write_text(json.dumps(dict(
    machine=platform.machine(),system=platform.platform(),
    ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
    threads=2,rows=rows),indent=2))
