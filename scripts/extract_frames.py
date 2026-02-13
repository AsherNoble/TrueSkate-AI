import os
import subprocess
import sys


if len(sys.argv) < 2:
    print("Usage: python3 extract_frames.py <video_file>")
    sys.exit(1)

video = sys.argv[1]

fps = 60  # Screen Recordings are at 60fps so no additional benefit from fps > 60

video_path = f'/Users/ashernoble/Projects/Training_Data/Sorted/{video}'
base = os.path.basename(video_path).rsplit('.', 1)[0]
output_video_path = f'/Users/ashernoble/Projects/TrueSkate-AI/data/extracted_frames/{base}_{fps}fps'
os.makedirs(output_video_path, exist_ok=True)


cmd = [
    'ffmpeg',
    '-i', video_path,
    '-vf', f'fps={fps},format=yuv420p',
    '-q:v', '2',
    os.path.join(output_video_path, 'img_%05d.jpg')
]

subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
