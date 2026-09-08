#!/bin/bash
# Stop a set of explicitly identified MVP-2 collectors at one strict target.
# Arguments after OUT/TARGET are DEVICE:PID_FILE pairs.  Every process is checked
# before signalling, so this cannot terminate an unrelated XR workload.
set -eu

OUT="${1:?usage: stop_basic_linear_fleet_at_target.sh OUT_DIR [target] DEVICE:PID_FILE...}"
TARGET="${2:-1000}"
shift 2
if [ "$#" -lt 1 ]; then
  echo "need at least one DEVICE:PID_FILE pair" >&2
  exit 2
fi
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
  echo "[basic-linear-fleet-target] accepted=$accepted target=$TARGET"
  if [ "$accepted" -lt "$TARGET" ]; then sleep 30; continue; fi
  for pair in "$@"; do
    device=${pair%%:*}
    pid_file=${pair#*:}
    [ -s "$pid_file" ] || { echo "missing pid file: $pid_file" >&2; continue; }
    pid=$(tr -d '[:space:]' < "$pid_file")
    command=$(ps -p "$pid" -o command= 2>/dev/null || true)
    expected="mvp_collect_linear.sh $device $OUT"
    if [[ "$command" == *"$expected"* ]]; then
      echo "[basic-linear-fleet-target] stopping pid=$pid ($expected)"
      kill -TERM "$pid"
    else
      echo "[basic-linear-fleet-target] pid=$pid does not identify expected collector" >&2
    fi
  done
  exit 0
done
