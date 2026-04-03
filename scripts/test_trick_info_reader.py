"""Randomly sample 50 frames from data/extracted_frames/ and test detect_trick."""

import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trueskate_ai.sim.trick_info_reader import detect_trick

EXTRACTED_FRAMES = Path(__file__).parent.parent / "data" / "extracted_frames"
SAMPLE_SIZE = 50

random.seed(42)

all_frames = [
    (p.parent.name, p.name, p)
    for p in EXTRACTED_FRAMES.rglob("img_*.jpg")
]

if len(all_frames) < SAMPLE_SIZE:
    print(f"Warning: only {len(all_frames)} frames available, sampling all.")
    sample = all_frames
else:
    sample = random.sample(all_frames, SAMPLE_SIZE)

detected = 0
not_detected = 0

for subfolder, filename, path in sample:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"{subfolder}/{filename}: ERROR reading frame")
        continue
    trick = detect_trick(frame)
    if trick:
        print(f"{subfolder}/{filename}: {trick}")
        detected += 1
    else:
        print(f"{subfolder}/{filename}: No trick detected")
        not_detected += 1

print(f"\n{detected}/{SAMPLE_SIZE} detected, {not_detected}/{SAMPLE_SIZE} no trick detected")
