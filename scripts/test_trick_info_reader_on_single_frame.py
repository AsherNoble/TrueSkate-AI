import cv2
import sys
sys.path.insert(0, "src")
from trueskate_ai.sim.trick_info_reader import detect_trick, _ocr_above_anchor, _find_anchor

frame = cv2.imread("./data/<path_to_image>")

search = frame[250:600, :]
anchor = _find_anchor(search)
if anchor:
    mask, status = anchor
    print(f"Anchor: {status}")
else:
    print("No anchor found")

result = detect_trick(frame)
print(f"Result: {result}")