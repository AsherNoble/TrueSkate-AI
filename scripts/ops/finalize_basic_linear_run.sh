#!/bin/bash
# Await a guarded MVP-2 corpus, clean menu clips, then run strict Modal training.
# Source captures are never deleted by this script; only .menu exclusion markers
# may be added before the loader allow-list is uploaded.
set -eu

OUT="${1:?usage: finalize_basic_linear_run.sh OUT_DIR RUN_LABEL [target] [collector_pid_file ...]}"
RUN_LABEL="${2:?usage: finalize_basic_linear_run.sh OUT_DIR RUN_LABEL [target] [collector_pid_file ...]}"
TARGET="${3:-1000}"
shift 3
PID_FILES=("${@:-tmp/basic_linear_xr1.pid}")
REPO=/Users/training-server/trueskate-ai

cd "$REPO"
accepted_count() {
  PYTHONPATH=src .venv/bin/python - "$OUT" <<'PY'
import sys
from pathlib import Path
from trueskate_ai.vision.basic_linear_dataset import discover_basic_linear_samples
print(len(discover_basic_linear_samples(Path(sys.argv[1]))[0]))
PY
}

while :; do
  accepted=$(accepted_count)
  echo "[basic-linear-finalizer] accepted=$accepted target=$TARGET"
  if [ "$accepted" -ge "$TARGET" ]; then break; fi
  sleep 60
done

while :; do
  alive=0
  for pid_file in "${PID_FILES[@]}"; do
    [ -s "$pid_file" ] && kill -0 "$(tr -d '[:space:]' < "$pid_file")" 2>/dev/null && alive=1
  done
  [ "$alive" -eq 0 ] && break
  echo "[basic-linear-finalizer] collector still exiting"
  sleep 30
done
sleep 30

PYTHONPATH=src .venv/bin/python scripts/data/flag_menu_samples.py --data "$OUT"
accepted=$(accepted_count)
if [ "$accepted" -lt "$TARGET" ]; then
  echo "[basic-linear-finalizer] only $accepted strict clips remain after menu scan; refusing upload" >&2
  exit 2
fi

PYTHONPATH=src .venv/bin/python scripts/cloud/upload_basic_linear_corpus.py \
  --source "$OUT" --volume trueskate-mvp --remote-subdir basic_linear_xctest \
  --min-samples "$TARGET"
env MODAL_CORPUS_VOLUME=trueskate-mvp .venv/bin/modal run scripts/cloud/train_basic_linear_modal.py \
  --data-subdir basic_linear_xctest --run-label "$RUN_LABEL" \
  --epochs 40 --batch-size 8 --lr 1e-3 --seed 0 --base-channels 16 --split-strategy command
