import cv2
import sys
sys.path.insert(0, "src")
from trueskate_ai.sim.trick_info_reader import detect_trick

frame = cv2.imread("./data/IMG_FC049D83DA34-1.jpeg")
result = detect_trick(frame)

print(result)