import sys
import cv2
from trueskate_ai.sim.score_reader import detect_trick


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_score_reader.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: could not read image at {image_path}")
        sys.exit(1)

    result = detect_trick(frame)
    print(result if result is not None else "No trick detected")


if __name__ == "__main__":
    main()
