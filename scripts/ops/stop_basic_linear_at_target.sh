#!/bin/bash
# Stop one explicitly identified MVP-2 linear collector after strict admission.
# This only terminates the named collector; it never deletes raw recordings.
set -eu

OUT="${1:?usage: stop_basic_linear_at_target.sh OUT_DIR [target] [device] [pid_file]}"
TARGET="${2:-1000}"
DEVICE="${3:-iPhone_XR}"
PID_FILE="${4:-tmp/basic_linear_xr1.pid}"
REPO=/Users/training-server/trueskate-ai

cd "$REPO"
case "$TARGET" in
  ''|*[!0-9]*) echo "target must be a non-negative integer: $TARGET" >&2; exit 2 ;;
esac

while :; do
  accepted=$(PYTHONPATH=src .venv/bin/python - "$OUT" "$DEVICE" <<'PY'
import sys
import json
from pathlib import Path
from trueskate_ai.vision.basic_linear_dataset import discover_basic_linear_samples

root, device = Path(sys.argv[1]), sys.argv[2]
samples, _stats = discover_basic_linear_samples(root)
# The source directory may be shared by multiple collectors.  Count only the
# explicit device so two independent target guards cannot accidentally stop
# one phone early because the other supplied most of the corpus.
print(sum(
    json.loads((sample / "meta.json").read_text()).get("device") == device
    for sample in samples
))
PY
)
  echo "[basic-linear-target] accepted=$accepted target=$TARGET"
  if [ "$accepted" -ge "$TARGET" ]; then
    if [ -s "$PID_FILE" ]; then
      pid=$(tr -d '[:space:]' < "$PID_FILE")
      command=$(ps -p "$pid" -o command= 2>/dev/null || true)
      expected="mvp_collect_linear.sh $DEVICE $OUT"
      if [[ "$command" == *"$expected"* ]]; then
        echo "[basic-linear-target] target reached; stopping pid=$pid ($expected)"
        kill -TERM "$pid"
      else
        echo "[basic-linear-target] target reached, but pid file does not identify expected collector" >&2
      fi
    else
      echo "[basic-linear-target] target reached, but no pid file at $PID_FILE" >&2
    fi
    exit 0
  fi
  sleep 30
done
