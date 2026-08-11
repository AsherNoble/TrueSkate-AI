#!/bin/bash
# Await a guarded basic-hold corpus, verify it, then launch the strict Modal run.
# Nothing here deletes source footage; a failed verification exits before upload.
set -eu

OUT="${1:?usage: finalize_basic_hold_run.sh OUT_DIR RUN_LABEL [target] [collector_pid_file]}"
RUN_LABEL="${2:?usage: finalize_basic_hold_run.sh OUT_DIR RUN_LABEL [target] [collector_pid_file]}"
TARGET="${3:-1000}"
PID_FILE="${4:-tmp/basic_hold_diverse_xr1.pid}"
REPO=/Users/training-server/trueskate-ai

cd "$REPO"
accepted_count() {
  PYTHONPATH=src .venv/bin/python - "$OUT" <<'PY'
import sys
from pathlib import Path
from trueskate_ai.vision.basic_hold_dataset import discover_basic_hold_samples
print(len(discover_basic_hold_samples(Path(sys.argv[1]))[0]))
PY
}

while :; do
  accepted=$(accepted_count)
  echo "[basic-hold-finalizer] accepted=$accepted target=$TARGET"
  if [ "$accepted" -ge "$TARGET" ]; then
    break
  fi
  sleep 60
done

# The target guard terminates the parent after the threshold. Let its bounded
# child/aligner finish before scanning or uploading the directory.
while [ -s "$PID_FILE" ] && kill -0 "$(tr -d '[:space:]' < "$PID_FILE")" 2>/dev/null; do
  echo "[basic-hold-finalizer] collector still exiting"
  sleep 30
done
sleep 30

# Compact clips carry frames.mp4 rather than PNGs. Mark, never delete, any menu
# clip so the strict loader excludes it before the final count is checked.
PYTHONPATH=src .venv/bin/python - "$OUT" <<'PY'
import sys
from pathlib import Path
import cv2
from trueskate_ai.vision.gameplay_filter import is_menu_frame
from trueskate_ai.vision.basic_hold_dataset import discover_basic_hold_samples

root = Path(sys.argv[1])
paths, _ = discover_basic_hold_samples(root)
marked = 0
for sample in paths:
    video = sample / "frames.mp4"
    if not video.is_file():
        continue
    cap = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_menu_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
                (sample / ".menu").write_text("replay/menu frame, not gameplay\n")
                marked += 1
                break
    finally:
        cap.release()
print(f"[basic-hold-finalizer] menu_marked={marked}")
PY

PYTHONPATH=src .venv/bin/python scripts/cloud/upload_basic_hold_corpus.py \
  --source "$OUT" --volume trueskate-mvp --remote-subdir basic_hold_diverse_xctest \
  --min-samples "$TARGET"
env MODAL_CORPUS_VOLUME=trueskate-mvp .venv/bin/modal run scripts/cloud/train_basic_hold_modal.py \
  --data-subdir basic_hold_diverse_xctest --run-label "$RUN_LABEL" \
  --epochs 40 --batch-size 8 --lr 1e-3 --seed 0 --base-channels 16 --split-strategy command
