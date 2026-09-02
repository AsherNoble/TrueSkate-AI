#!/bin/bash
# Finish the device-balanced Stage 1 linear tranche without a Modal call.
#
# Waits for BOTH strict target watchers, allows their collectors/aligners to
# exit, flags replay/menu samples non-destructively, then writes the final
# strict corpus audit and sends one completion/failure notification. It never
# deletes clips, restarts services, or starts training.
set -eu

REPO=/Users/training-server/trueskate-ai
OUT_ROOT=/Users/training-server/trueskate-ai/data/basic_linear_stage1_20260831
TARGET=1100
DEVICES=(iPhone_XR iPhone_XR2)
LOG="$REPO/logs/basic_linear_stage1_finalize.log"
AUDIT_JSON="$REPO/tmp/basic_linear_stage1_audit.json"

cd "$REPO"

strict_count() {
  PYTHONPATH=src .venv/bin/python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path
from trueskate_ai.vision.basic_linear_dataset import discover_basic_linear_samples

root, device = Path(sys.argv[1]), sys.argv[2]
samples, _stats = discover_basic_linear_samples(root)
print(sum(json.loads((sample / "meta.json").read_text()).get("device") == device
          for sample in samples))
PY
}

notify() {
  PYTHONPATH=src .venv/bin/python - "$1" "$2" <<'PY'
import sys
from trueskate_ai.utils.notify import notify
notify(sys.argv[1], title="TrueSkate Model 1 Stage 1", priority="high", tags=[sys.argv[2]])
PY
}

echo "[stage1-finalize] waiting for $TARGET strict clips/device" | tee -a "$LOG"
while :; do
  complete=1
  for device in "${DEVICES[@]}"; do
    count=$(strict_count "$OUT_ROOT/$device" "$device")
    echo "[stage1-finalize] $device strict=$count target=$TARGET" | tee -a "$LOG"
    if [ "$count" -lt "$TARGET" ]; then complete=0; fi
  done
  [ "$complete" -eq 1 ] && break
  sleep 60
done

# The strict guards signal the outer collector shell.  Give any in-flight
# aligner time to finish, then require no device-specific collector process
# before the corpus sweep reads its files.
while pgrep -f 'collect_sls_xctest.py.*--devices (iPhone_XR|iPhone_XR2)' >/dev/null; do
  echo "[stage1-finalize] collector/aligner still exiting" | tee -a "$LOG"
  sleep 15
done

for device in "${DEVICES[@]}"; do
  PYTHONPATH=src .venv/bin/python scripts/data/flag_menu_samples.py \
    --data "$OUT_ROOT/$device" >>"$LOG" 2>&1
done

if PYTHONPATH=src .venv/bin/python scripts/data/audit_basic_linear_corpus.py \
  --data "$OUT_ROOT" --json-out "$AUDIT_JSON" \
  --require-device iPhone_XR --require-device iPhone_XR2 \
  --require-park "The Workshop" --min-per-device 1000 \
  --require-unique-commands >>"$LOG" 2>&1; then
  notify "Stage 1 collection audited successfully; JSON: $AUDIT_JSON" white_check_mark
  echo "[stage1-finalize] AUDIT PASSED" | tee -a "$LOG"
else
  notify "Stage 1 collection finished, but its audit FAILED; inspect $LOG" warning
  echo "[stage1-finalize] AUDIT FAILED" | tee -a "$LOG"
  exit 1
fi
